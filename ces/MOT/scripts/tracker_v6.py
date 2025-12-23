#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Door Tracker V6 - Mahalanobis Matching & Handle Detection
=========================================================
1. 마할라노비스 거리 기반 ID 매칭 (위치+방향 공분산 고려)
2. 손잡이(Handle) 검출 및 시각화 복구
3. 화면 끊김 방지 (비동기 렌더링)
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
    'door_conf_thresh': 0.25,
    'handle_conf_thresh': 0.15,
    'tracker_type': 'bytetrack.yaml',
    
    # 매칭 파라미터
    'mahalanobis_gate': 4.0, # Chi-square 분포 상위 95% (자유도 2~3)
    'max_dist_hard': 3.0,    # 하드 리미트 (미터)
    
    # 확정 조건
    'confirm_frames': 3,
    
    # SLAM
    'slam_topic': '/orb_slam3/camera_pose',
    'max_time_diff': 0.2,    # 넉넉하게
    
    'arrow_len': 1.5,
}

def wrap_angle(rad):
    return (rad + np.pi) % (2 * np.pi) - np.pi

class DoorNode:
    def __init__(self, uid, pose, bearing, heading):
        self.id = uid
        self.anchor_pose = pose
        self.anchor_xyz = pose[:3, 3]
        self.bearing = bearing
        self.heading = heading
        
        # 공분산 (Covariance) 초기화
        # [x, y, theta] 불확실성
        self.cov = np.diag([0.5, 0.5, 0.2]) 
        
        self.count = 1
        self.last_seen = time.time()
        self.active = True

    def update(self, pose, bearing, heading):
        # 업데이트 (Measurement Update 유사)
        alpha = 0.3
        self.anchor_xyz = (1-alpha)*self.anchor_xyz + alpha*pose[:3,3]
        self.bearing = wrap_angle((1-alpha)*self.bearing + alpha*bearing)
        self.heading = wrap_angle((1-alpha)*self.heading + alpha*heading)
        self.last_seen = time.time()

    def get_mahalanobis_dist(self, curr_xyz, curr_angle):
        # State vector diff: [dx, dy, d_theta]
        # d_theta = (heading + bearing) 차이
        
        node_angle = self.heading + self.bearing
        d_angle = wrap_angle(node_angle - curr_angle)
        
        diff = np.array([
            self.anchor_xyz[0] - curr_xyz[0],
            self.anchor_xyz[1] - curr_xyz[1],
            d_angle
        ])
        
        # D = sqrt( diff.T * inv(Cov) * diff )
        try:
            inv_cov = np.linalg.inv(self.cov)
            dist_sq = diff.T @ inv_cov @ diff
            return np.sqrt(dist_sq)
        except:
            return float('inf')

class DoorMap:
    def __init__(self):
        self.nodes = {}
        self.next_id = 0
    
    def add_node(self, pose, bearing, heading):
        uid = self.next_id
        self.nodes[uid] = DoorNode(uid, pose, bearing, heading)
        self.next_id += 1
        return uid

    def find_match(self, pose, bearing, heading):
        curr_xyz = pose[:3, 3]
        curr_angle = heading + bearing
        
        best_id = -1
        min_m_dist = float('inf')
        
        for uid, node in self.nodes.items():
            if not node.active: continue
            
            # 1. Hard Gate (Euclidean) - 너무 멀면 계산 안함
            e_dist = np.linalg.norm(curr_xyz[:2] - node.anchor_xyz[:2])
            if e_dist > CONFIG['max_dist_hard']: continue
            
            # 2. Mahalanobis Distance
            m_dist = node.get_mahalanobis_dist(curr_xyz, curr_angle)
            
            if m_dist < CONFIG['mahalanobis_gate']:
                if m_dist < min_m_dist:
                    min_m_dist = m_dist
                    best_id = uid
        
        return best_id

class TrackerV6:
    def __init__(self):
        rospy.init_node("tracker_v6", anonymous=False)
        r = rospkg.RosPack()
        
        calib_path = r.get_path('ces') + "/MOT/calibration file/2024-09-13-20-37-00-camchain.yaml"
        self.cameras = self._load_calib(calib_path)
        
        model_path = r.get_path('ces') + "/MOT/pretrained/best.pt"
        self.yolo_front = YOLO(model_path)
        self.yolo_side = YOLO(model_path)
        
        self.door_map = DoorMap()
        self.candidates = {} 
        self.local_map = {}
        
        self.pose_buf = deque(maxlen=300)
        self.lock = threading.Lock()
        self.images = {}
        self.bridge = CvBridge()
        
        # Subscribers
        for cam in ['front', 'left', 'rear', 'right']:
            topic = f"/camera/undistorted/{cam}"
            rospy.Subscriber(topic, Image, self._make_img_cb(cam))
            rospy.Subscriber(f"{topic}/camera_info", CameraInfo, self._make_info_cb(cam))
        
        rospy.Subscriber(CONFIG['slam_topic'], PoseStamped, self.cb_slam)
        
        # Publishers
        self.pub_marker = rospy.Publisher("/door_markers", MarkerArray, queue_size=1)
        self.pub_path = rospy.Publisher("/robot_path", Path, queue_size=1)
        self.tf_pub = tf2_ros.TransformBroadcaster()
        
        self.path_msg = Path()
        self.path_msg.header.frame_id = "world"
        
        rospy.loginfo("Tracker V6 (Mahalanobis) Started")

    def _load_calib(self, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        cams = {}
        # Mapping: topic_name -> (fx, cx, width)
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
            if name in self.cameras:
                self.cameras[name]['fx'] = msg.K[0]
                self.cameras[name]['cx'] = msg.K[2]
                self.cameras[name]['w'] = msg.width
        return cb

    def cb_slam(self, msg):
        q = msg.pose.orientation
        p = msg.pose.position
        rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = [p.x, p.y, p.z]
        t = msg.header.stamp.to_sec()
        
        with self.lock:
            self.pose_buf.append((t, T))
            self.path_msg.poses.append(msg)
            if len(self.path_msg.poses) > 500: self.path_msg.poses.pop(0)
        
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
        if not self.pose_buf: return None
        best = None; min_dt = 100.0
        for pt, pT in self.pose_buf:
            dt = abs(pt - t)
            if dt < min_dt: min_dt = dt; best = pT
        return best if min_dt < CONFIG['max_time_diff'] else None

    def get_yaw(self, T):
        forward = T[:3, :3] @ np.array([0, 0, 1])
        return np.arctan2(forward[1], forward[0])

    def pix2bearing(self, u, cam):
        c = self.cameras[cam]
        return np.arctan((u - c['cx']) / c['fx'])

    def process_front(self, img, t, T_rig):
        # Class 0: Door, 1: Handle
        res = self.yolo_front.track(img, persist=True, tracker=CONFIG['tracker_type'], 
                                   conf=CONFIG['handle_conf_thresh'], classes=[0,1], verbose=False)
        
        yaw = self.get_yaw(T_rig) if T_rig is not None else 0
        current_matches = {} # local -> world
        
        if res and res[0].boxes.id is not None:
            boxes = res[0].boxes.xyxy.cpu().numpy()
            ids = res[0].boxes.id.cpu().numpy().astype(int)
            clss = res[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, lid, cls_id in zip(boxes, ids, clss):
                if cls_id == 1: continue # Handle은 매핑 안함
                
                if T_rig is not None:
                    u = (box[0] + box[2]) / 2.0
                    bear = self.pix2bearing(u, 'front')
                    
                    if lid in self.local_map:
                        wid = self.local_map[lid]
                        if wid in self.door_map.nodes:
                            self.door_map.nodes[wid].update(T_rig, bear, yaw)
                        current_matches[lid] = wid
                    else:
                        if lid not in self.candidates:
                            self.candidates[lid] = {'pose': T_rig, 'bear': bear, 'yaw': yaw, 'hits': 1}
                            continue
                        
                        cand = self.candidates[lid]
                        cand['hits'] += 1
                        cand['pose'] = T_rig; cand['bear'] = bear; cand['yaw'] = yaw
                        
                        if cand['hits'] >= CONFIG['confirm_frames']:
                            match = self.door_map.find_match(T_rig, bear, yaw)
                            if match != -1:
                                self.local_map[lid] = match; current_matches[lid] = match
                            else:
                                new_id = self.door_map.add_node(T_rig, bear, yaw)
                                self.local_map[lid] = new_id; current_matches[lid] = new_id
                            del self.candidates[lid]

        return current_matches, res[0]

    def viz(self, front_res, matches, side_res_list, imgs, has_pose):
        # Front
        if imgs.get('front') is not None:
            f_img = imgs['front'].copy()
            if front_res and front_res.boxes.id is not None:
                boxes = front_res.boxes.xyxy.cpu().numpy()
                ids = front_res.boxes.id.cpu().numpy().astype(int)
                clss = front_res.boxes.cls.cpu().numpy().astype(int)
                
                for box, lid, cls_id in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = map(int, box)
                    if cls_id == 0: # Door
                        if lid in matches:
                            label = f"Door {matches[lid]}"
                            color = (0, 255, 0) # Green (Locked)
                        else:
                            label = "Scan.."
                            color = (0, 165, 255) # Orange (Scanning)
                    else: # Handle
                        label = "Handle"
                        color = (0, 0, 255) # Red
                    
                    cv2.rectangle(f_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(f_img, label, (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            f_img = np.zeros((480, 640, 3), np.uint8)
        
        # Overlay Status
        status = "SLAM: OK" if has_pose else "SLAM: NO POSE (Disp Only)"
        col = (0, 255, 0) if has_pose else (0, 0, 255)
        cv2.putText(f_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        # Side Images (Detection Only)
        side_imgs = []
        for i, cam in enumerate(['left', 'rear', 'right']):
            img = imgs.get(cam)
            res = side_res_list[i]
            if img is not None:
                s_img = img.copy()
                if res and res.boxes is not None:
                    for box, cls_id in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                        if cls_id == 0:
                            cv2.rectangle(s_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (200, 200, 200), 2)
                side_imgs.append(s_img)
            else:
                side_imgs.append(np.zeros((480, 640, 3), np.uint8))
        
        # Grid
        h, w = f_img.shape[:2]
        s_resized = [cv2.resize(img, (w, h)) for img in side_imgs]
        top = np.hstack((f_img, s_resized[1])) 
        bot = np.hstack((s_resized[0], s_resized[2]))
        
        cv2.imshow("Tracker V6 (Mahalanobis)", np.vstack((top, bot)))
        cv2.waitKey(1)

    def pub_markers(self):
        ma = MarkerArray()
        for uid, node in self.door_map.nodes.items():
            if not node.active: continue
            
            # Arrow
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = rospy.Time.now()
            m.ns = "doors"; m.id = uid; m.type = Marker.ARROW; m.action = Marker.ADD
            
            # Calculate arrow points
            world_dir = node.heading + node.bearing
            dx = CONFIG['arrow_len'] * np.cos(world_dir)
            dy = CONFIG['arrow_len'] * np.sin(world_dir)
            p1 = Point(*node.anchor_xyz); p2 = Point(node.anchor_xyz[0]+dx, node.anchor_xyz[1]+dy, node.anchor_xyz[2])
            
            m.points = [p1, p2]
            m.scale.x = 0.05; m.scale.y = 0.1; m.scale.z = 0.1
            m.color.a = 1.0; m.color.g = 1.0 # Green
            ma.markers.append(m)
            
            # Text
            t = Marker()
            t.header.frame_id = "world"; t.ns = "text"; t.id = uid+1000
            t.type = Marker.TEXT_VIEW_FACING; t.text = f"D{uid}"
            t.pose.position = p2; t.pose.position.z += 0.2
            t.scale.z = 0.3; t.color.a = 1.0; t.color.r = 1.0; t.color.g = 1.0; t.color.b = 1.0
            ma.markers.append(t)
            
        self.pub_marker.publish(ma)

    def loop(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            with self.lock:
                # Copy images to avoid threading issues
                imgs = {k: (v[1].copy() if v else None) for k, v in self.images.items()}
                stamps = {k: (v[0] if v else 0) for k, v in self.images.items()}
            
            if imgs.get('front') is None:
                rate.sleep(); continue
            
            T_rig = self.get_pose(stamps['front'])
            
            f_res = None
            matches = {}
            side_res_list = [None, None, None]
            
            # Front Process
            matches, f_res = self.process_front(imgs['front'], stamps['front'], T_rig)
            
            # Side Process (Detect only)
            for i, cam in enumerate(['left', 'rear', 'right']):
                if imgs.get(cam) is not None:
                    res = self.yolo_side.predict(imgs[cam], conf=CONFIG['door_conf_thresh'], classes=[0,1], verbose=False)
                    side_res_list[i] = res[0]
            
            self.viz(f_res, matches, side_res_list, imgs, T_rig is not None)
            
            if T_rig is not None:
                self.pub_markers()
            
            rate.sleep()

if __name__ == "__main__":
    TrackerV6().loop()
