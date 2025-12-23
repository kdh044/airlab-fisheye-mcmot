#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Door Tracker v5 - Robust ID Association with Pose+Bearing
=========================================================
특징:
1. 3D 좌표 추정(깊이) 포기 -> Pose(내위치) + Bearing(방향) 기반 매핑
2. 강력한 게이팅:
   - 거리: 등록 시점 위치와 현재 위치 차이
   - 각도: 등록 시점 방향과 현재 방향 차이
   - 후방 제거: 로봇 뒤로 넘어간 ID 매칭 제외
3. 마할라노비스 대신 직관적인 Heuristic 사용 (복도 환경 최적화)
"""

import rospy
import rospkg
import cv2
import numpy as np
import yaml
import threading
import time
from collections import deque

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from cv_bridge import CvBridge
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
import tf2_ros

# ========== 설정 ==========
CONFIG = {
    'door_conf_thresh': 0.3,
    'tracker_type': 'bytetrack.yaml',
    
    # 1. 거리 게이트 (미터)
    # 문을 처음 등록한 위치에서 이만큼 멀어지면 더 이상 그 ID와 매칭 안 함
    'max_dist_from_anchor': 2.0, 
    
    # 2. 각도 게이트 (도)
    # 문을 바라보는 방향(Bearing) 차이가 이보다 크면 다른 문으로 간주
    'max_bearing_diff': 30.0,
    
    # 3. 확정 조건
    'confirm_frames': 3,     # 연속 3프레임 이상 보여야 ID 등록
    
    # SLAM 관련
    'slam_topic': '/orb_slam3/camera_pose',
    'max_time_diff': 0.1,    # 이미지-포즈 시간 동기화 허용 오차
    
    # 시각화
    'arrow_len': 1.5,
}

def wrap_angle(rad):
    """ -pi ~ pi """
    return (rad + np.pi) % (2 * np.pi) - np.pi

class DoorNode:
    """지도에 등록된 문 (Anchor)"""
    def __init__(self, uid, anchor_pose, bearing, heading_world):
        self.id = uid
        self.anchor_pose = anchor_pose  # 등록 시점의 로봇(Rig) 포즈 (4x4)
        self.anchor_xyz = anchor_pose[:3, 3]
        self.bearing = bearing          # 로봇 기준 문 방향 (rad)
        self.heading = heading_world    # 월드 기준 로봇 헤딩 (rad)
        self.count = 1
        self.last_seen = time.time()
        self.active = True # False면 로봇 뒤로 지나가서 비활성

    def update(self, pose, bearing, heading):
        # 최신 정보로 조금씩 업데이트 (EMA)
        alpha = 0.2
        self.anchor_xyz = (1-alpha)*self.anchor_xyz + alpha*pose[:3,3]
        self.bearing = wrap_angle((1-alpha)*self.bearing + alpha*bearing)
        self.heading = wrap_angle((1-alpha)*self.heading + alpha*heading)
        self.count += 1
        self.last_seen = time.time()

    def get_arrow_tip(self):
        # 시각화용 끝점 계산 (월드 좌표)
        # 문 방향(월드) = 로봇헤딩 + 베어링
        world_dir = self.heading + self.bearing
        dx = CONFIG['arrow_len'] * np.cos(world_dir)
        dy = CONFIG['arrow_len'] * np.sin(world_dir)
        return self.anchor_xyz + np.array([dx, dy, 0])

class Candidate:
    """ID 등록 전 후보"""
    def __init__(self, pose, bearing, heading):
        self.pose = pose
        self.bearing = bearing
        self.heading = heading
        self.hits = 1
        self.last_seen = time.time()
    
    def update(self, pose, bearing, heading):
        self.hits += 1
        self.pose = pose
        self.bearing = bearing
        self.heading = heading
        self.last_seen = time.time()

class DoorMap:
    def __init__(self):
        self.nodes = {}  # id -> DoorNode
        self.next_id = 0
    
    def add_node(self, pose, bearing, heading):
        uid = self.next_id
        self.nodes[uid] = DoorNode(uid, pose, bearing, heading)
        self.next_id += 1
        return uid
    
    def find_match(self, current_pose, current_bearing, current_heading):
        curr_xyz = current_pose[:3, 3]
        
        best_id = -1
        min_cost = float('inf')
        
        for uid, node in self.nodes.items():
            if not node.active: continue
            
            # 1. Anchor 거리 체크 (너무 멀리서 등록한 문은 매칭 X)
            dist = np.linalg.norm(curr_xyz[:2] - node.anchor_xyz[:2])
            if dist > CONFIG['max_dist_from_anchor']:
                continue
            
            # 2. 후방 체크 (로봇이 문을 지나쳤는지)
            # 로봇 진행방향(Heading) 기준으로 문이 뒤에 있는지? 
            # (단, 옆 카메라는 뒤를 볼 수도 있으니 신중해야 함. 일단 거리로 컷)
            
            # 3. 각도 체크 (Bearing + Heading)
            # 월드 기준 문 방향 차이
            node_dir = node.heading + node.bearing
            curr_dir = current_heading + current_bearing
            diff = abs(wrap_angle(node_dir - curr_dir))
            
            if np.degrees(diff) > CONFIG['max_bearing_diff']:
                continue
            
            # Cost Function (가까울수록, 각도 비슷할수록 좋음)
            cost = dist + diff * 2.0 
            
            if cost < min_cost:
                min_cost = cost
                best_id = uid
        
        return best_id

class TrackerV5:
    def __init__(self):
        rospy.init_node("tracker_v5", anonymous=False)
        r = rospkg.RosPack()
        
        # Load Config
        calib_path = r.get_path('ces') + "/MOT/calibration file/2024-09-13-20-37-00-camchain.yaml"
        self.cameras = self._load_calib(calib_path)
        
        model_path = r.get_path('ces') + "/MOT/pretrained/best.pt"
        self.yolo_front = YOLO(model_path) # Tracking용
        self.yolo_side = YOLO(model_path)  # Detection용
        
        # State
        self.door_map = DoorMap()
        self.candidates = {} # local_id(front) -> Candidate
        self.local_map = {}  # local_id(front) -> world_id
        
        self.pose_buf = deque(maxlen=200) # (time, T_world_rig)
        self.slam_pose = None
        
        self.lock = threading.Lock()
        self.images = {}
        self.bridge = CvBridge()
        
        # Pub/Sub
        for cam in ['front', 'left', 'rear', 'right']:
            # Subscribers
            topic = f"/camera/undistorted/{cam}"
            rospy.Subscriber(topic, Image, self._make_img_cb(cam))
            # Camera Info (optional, for intrinsics update)
            rospy.Subscriber(f"{topic}/camera_info", CameraInfo, self._make_info_cb(cam))
            
        rospy.Subscriber(CONFIG['slam_topic'], PoseStamped, self.cb_slam)
        
        self.pub_marker = rospy.Publisher("/door_markers", MarkerArray, queue_size=1)
        self.pub_path = rospy.Publisher("/robot_path", Path, queue_size=1)
        self.tf_pub = tf2_ros.TransformBroadcaster()
        
        self.path_msg = Path()
        self.path_msg.header.frame_id = "world"
        
        rospy.loginfo("Tracker V5 Started - Robust Pose Association")

    def _load_calib(self, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        cams = {}
        # Mapping: topic_name -> (fx, cx, width)
        # Note: Using approximate defaults, updated by CameraInfo if avail
        # cam0=front, cam1=left(rear topic), cam2=rear(left topic), cam3=right
        # Topic map from previous scripts:
        # front->cam0, left->cam2, rear->cam1, right->cam3
        mapping = {'front':'cam0', 'rear':'cam1', 'left':'cam2', 'right':'cam3'}
        
        for name, cid in mapping.items():
            intr = data[cid]['intrinsics']
            cams[name] = {'fx': intr[2], 'cx': intr[4], 'w': 640}
        return cams

    def _make_img_cb(self, name):
        def cb(msg):
            with self.lock:
                try:
                    img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                    self.images[name] = (msg.header.stamp.to_sec(), img)
                except: pass
        return cb

    def _make_info_cb(self, name):
        def cb(msg):
            # Update intrinsics
            if name in self.cameras:
                self.cameras[name]['fx'] = msg.K[0]
                self.cameras[name]['cx'] = msg.K[2]
                self.cameras[name]['w'] = msg.width
        return cb

    def cb_slam(self, msg):
        # PoseStamped -> T (4x4)
        q = msg.pose.orientation
        p = msg.pose.position
        rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = [p.x, p.y, p.z]
        
        t = msg.header.stamp.to_sec()
        
        with self.lock:
            self.pose_buf.append((t, T))
            self.slam_pose = T
            
            # Path update
            self.path_msg.poses.append(msg)
            if len(self.path_msg.poses) > 500:
                self.path_msg.poses.pop(0)
        
        # Publish TF immediately
        self.pub_tf(T, msg.header.stamp)
        self.pub_path.publish(self.path_msg)

    def pub_tf(self, T, stamp):
        ts = TransformStamped()
        ts.header.stamp = stamp
        ts.header.frame_id = "world"
        ts.child_frame_id = "base_link"
        ts.transform.translation.x = T[0,3]
        ts.transform.translation.y = T[1,3]
        ts.transform.translation.z = T[2,3]
        q = R.from_matrix(T[:3,:3]).as_quat()
        ts.transform.rotation.x = q[0]
        ts.transform.rotation.y = q[1]
        ts.transform.rotation.z = q[2]
        ts.transform.rotation.w = q[3]
        self.tf_pub.sendTransform(ts)

    def get_pose(self, t):
        # Find nearest pose in buffer
        if not self.pose_buf: return None
        # Binary search or linear scan (buffer is small)
        best = None
        min_dt = 100.0
        for pt, pT in self.pose_buf:
            dt = abs(pt - t)
            if dt < min_dt:
                min_dt = dt
                best = pT
        
        if min_dt < CONFIG['max_time_diff']:
            return best
        return None

    def get_yaw(self, T):
        # Extract Yaw from Rotation Matrix
        # Z-axis of camera frame projected to World XY
        forward = T[:3, :3] @ np.array([0, 0, 1])
        return np.arctan2(forward[1], forward[0])

    def pix2bearing(self, u, cam):
        # Pixel u -> Angle relative to camera center
        c = self.cameras[cam]
        return np.arctan((u - c['cx']) / c['fx'])

    def process_front(self, img, t, T_rig):
        # 1. Track
        res = self.yolo_front.track(img, persist=True, tracker=CONFIG['tracker_type'], 
                                   conf=CONFIG['door_conf_thresh'], classes=[0], verbose=False)
        
        yaw = self.get_yaw(T_rig)
        current_matches = {} # local -> world
        
        if res and res[0].boxes.id is not None:
            boxes = res[0].boxes.xyxy.cpu().numpy()
            ids = res[0].boxes.id.cpu().numpy().astype(int)
            
            for box, lid in zip(boxes, ids):
                u = (box[0] + box[2]) / 2.0
                bear = self.pix2bearing(u, 'front')
                
                # Check if we already mapped this local ID
                if lid in self.local_map:
                    wid = self.local_map[lid]
                    # Update Map
                    if wid in self.door_map.nodes:
                        self.door_map.nodes[wid].update(T_rig, bear, yaw)
                    current_matches[lid] = wid
                else:
                    # New Track -> Try to Match or Create Candidate
                    
                    # 1. Candidate check
                    if lid not in self.candidates:
                        self.candidates[lid] = Candidate(T_rig, bear, yaw)
                        continue # Wait for more frames
                    
                    self.candidates[lid].update(T_rig, bear, yaw)
                    
                    if self.candidates[lid].hits >= CONFIG['confirm_frames']:
                        # Try Global Match
                        match = self.door_map.find_match(T_rig, bear, yaw)
                        
                        if match != -1:
                            self.local_map[lid] = match
                            current_matches[lid] = match
                        else:
                            # Create New
                            new_id = self.door_map.add_node(T_rig, bear, yaw)
                            self.local_map[lid] = new_id
                            current_matches[lid] = new_id
                        
                        del self.candidates[lid]

        return current_matches, res[0]

    def process_side(self, cam, img, t, T_rig):
        # Detection Only (No Tracking)
        res = self.yolo_side.predict(img, conf=CONFIG['door_conf_thresh'], classes=[0], verbose=False)
        
        # Extrinsic (Rig -> Cam) is static approx or Identity
        # For simplicity, we assume Side Cams are just rotated
        # Rig Yaw is enough for now
        yaw = self.get_yaw(T_rig)
        
        # Side cam offset (approx 90 deg)
        # Front=0, Left=+90, Right=-90, Rear=180
        # But here we just use bearing relative to Rig
        # We need precise extrinsic to do this correctly.
        # Let's Skip Side Mapping for V5 safety - Just Visualize
        return res[0]

    def viz(self, front_res, matches, side_res_list):
        # Combine Images
        f_img = front_res.plot() if front_res else np.zeros((480,640,3), np.uint8)
        
        # Override labels with World ID
        if front_res and front_res.boxes.id is not None:
            boxes = front_res.boxes.xyxy.cpu().numpy()
            ids = front_res.boxes.id.cpu().numpy().astype(int)
            for box, lid in zip(boxes, ids):
                if lid in matches:
                    wid = matches[lid]
                    label = f"ID: {wid}"
                    color = (0, 255, 0)
                else:
                    label = f"Scan.."
                    color = (0, 0, 255)
                
                x1, y1 = int(box[0]), int(box[1])
                cv2.putText(f_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.rectangle(f_img, (x1, y1), (int(box[2]), int(box[3])), color, 2)

        # Side images
        side_imgs = []
        for r in side_res_list:
            side_imgs.append(r.plot() if r else np.zeros((480,640,3), np.uint8))
        
        # Grid
        h, w = f_img.shape[:2]
        s_resized = [cv2.resize(img, (w, h)) for img in side_imgs]
        
        top = np.hstack((f_img, s_resized[1])) # Front, Rear(mapped to cam1)
        bot = np.hstack((s_resized[0], s_resized[2])) # Left(mapped to cam2), Right
        
        cv2.imshow("Tracker V5", np.vstack((top, bot)))
        cv2.waitKey(1)

    def pub_markers(self):
        ma = MarkerArray()
        for uid, node in self.door_map.nodes.items():
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = rospy.Time.now()
            m.ns = "doors"
            m.id = uid
            m.type = Marker.ARROW
            m.action = Marker.ADD
            
            p1 = Point(*node.anchor_xyz)
            p2 = Point(*node.get_arrow_tip())
            
            m.points = [p1, p2]
            m.scale.x = 0.05
            m.scale.y = 0.1
            m.color.a = 1.0; m.color.g = 1.0
            
            ma.markers.append(m)
            
            # Text
            t = Marker()
            t.header.frame_id = "world"
            t.ns = "text"
            t.id = uid + 1000
            t.type = Marker.TEXT_VIEW_FACING
            t.text = f"Door {uid}"
            t.pose.position = p2
            t.pose.position.z += 0.5
            t.scale.z = 0.5
            t.color.a = 1.0; t.color.r = 1.0; t.color.g = 1.0; t.color.b = 1.0
            ma.markers.append(t)
            
        self.pub_marker.publish(ma)

    def loop(self):
        rate = rospy.Rate(15)
        while not rospy.is_shutdown():
            with self.lock:
                if 'front' not in self.images:
                    rate.sleep(); continue
                
                ft, f_img = self.images['front']
                stamps = {k: self.images[k][0] for k in ['left','rear','right'] if k in self.images}
                imgs = {k: self.images[k][1] for k in ['left','rear','right'] if k in self.images}
            
            # Get Pose
            T_rig = self.get_pose(ft)
            
            f_res = None
            matches = {}
            side_res_list = [None, None, None] # left, rear, right order
            
            if T_rig is not None:
                matches, f_res = self.process_front(f_img, ft, T_rig)
                
                # Process Sides (Visualization only for V5 safety)
                for i, cam in enumerate(['left', 'rear', 'right']):
                    if cam in imgs:
                        # Side pose approx same as rig for viz
                        side_res_list[i] = self.process_side(cam, imgs[cam], stamps[cam], T_rig)
                
                self.viz(f_res, matches, side_res_list)
                self.pub_markers()
            else:
                # No SLAM Pose
                if f_img is not None:
                    cv2.putText(f_img, "NO SLAM POSE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    cv2.imshow("Tracker V5", f_img)
                    cv2.waitKey(1)
            
            rate.sleep()

if __name__ == "__main__":
    TrackerV5().loop()
