#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Door Tracker v4 - Pose + Bearing 기반 (3D xyz 없음)
====================================================
핵심:
1. 문을 3D 점으로 안 박음 → pose + bearing으로 저장
2. 매칭: 포즈거리 + 헤딩 + 베어링 게이트
3. RViz: 화살표로 "문 방향" 표시
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
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R
import tf2_ros

# ========== 설정 ==========
CONFIG = {
    'door_conf_thresh': 0.3,
    'tracker_type': 'bytetrack.yaml',
    
    # 매칭 게이트
    'pose_dist_gate': 1.5,      # 포즈 거리 게이트 (미터)
    'heading_gate_deg': 40.0,   # 헤딩 차이 게이트 (도)
    'bearing_gate_deg': 15.0,   # 베어링 차이 게이트 (도)
    
    # 확정 조건
    'confirm_frames': 5,        # N 프레임 연속 보여야 확정
    
    # 시각화
    'arrow_length': 2.0,        # RViz 화살표 길이
    
    # SLAM
    'slam_topic': '/orb_slam3/camera_pose',
    'max_time_diff': 0.2,
}


def wrap_angle(a):
    """각도를 -π ~ π로 wrap"""
    return (a + np.pi) % (2 * np.pi) - np.pi


class DoorNode:
    """문 관측 노드 (3D 점이 아닌 pose + bearing)"""
    def __init__(self, nid, anchor_pose, bearing, heading):
        self.id = nid
        self.anchor_pose = anchor_pose.copy()  # T_world_front (4x4)
        self.anchor_xyz = anchor_pose[:3, 3].copy()
        self.bearing = bearing      # 카메라 기준 수평각 (rad)
        self.heading = heading      # 카메라가 바라보는 방향 (yaw, rad)
        self.count = 1
        self.last_seen = time.time()
    
    def update(self, anchor_pose, bearing, heading, alpha=0.3):
        # EMA로 스무딩
        self.anchor_xyz = (1 - alpha) * self.anchor_xyz + alpha * anchor_pose[:3, 3]
        self.bearing = wrap_angle((1 - alpha) * self.bearing + alpha * bearing)
        self.heading = wrap_angle((1 - alpha) * self.heading + alpha * heading)
        self.count += 1
        self.last_seen = time.time()
    
    def get_arrow_endpoint(self, length=2.0):
        """문 방향 화살표 끝점 계산"""
        # 문 방향 = 헤딩 + 베어링
        direction = self.heading + self.bearing
        dx = length * np.cos(direction)
        dy = length * np.sin(direction)
        x, y, z = self.anchor_xyz
        return (x + dx, y + dy, z)


class Candidate:
    """확정 전 후보 (N 프레임 연속 보여야 확정)"""
    def __init__(self, local_id, pose, bearing, heading):
        self.local_id = local_id
        self.anchor_pose = pose.copy()
        self.bearing = bearing
        self.heading = heading
        self.hit_count = 1
        self.last_seen = time.time()
    
    def update(self, pose, bearing, heading):
        self.anchor_pose = pose.copy()
        self.bearing = bearing
        self.heading = heading
        self.hit_count += 1
        self.last_seen = time.time()


class DoorMap:
    """문 노드 관리"""
    def __init__(self):
        self.nodes = {}  # id -> DoorNode
        self.next_id = 0
    
    def add_node(self, anchor_pose, bearing, heading):
        nid = self.next_id
        self.nodes[nid] = DoorNode(nid, anchor_pose, bearing, heading)
        self.next_id += 1
        rospy.loginfo(f"[DoorMap] New Door #{nid} (bearing: {np.degrees(bearing):.1f}°)")
        return nid
    
    def find_match(self, pose, bearing, heading, exclude_ids=set()):
        """포즈 + 헤딩 + 베어링으로 매칭"""
        xyz = pose[:3, 3]
        
        best_id = -1
        best_score = float('inf')
        
        for nid, node in self.nodes.items():
            if nid in exclude_ids:
                continue
            
            # 1. 포즈 거리
            dist = np.linalg.norm(xyz[:2] - node.anchor_xyz[:2])  # XY만
            if dist > CONFIG['pose_dist_gate']:
                continue
            
            # 2. 헤딩 차이
            heading_diff = abs(wrap_angle(heading - node.heading))
            if np.degrees(heading_diff) > CONFIG['heading_gate_deg']:
                continue
            
            # 3. 베어링 차이
            bearing_diff = abs(wrap_angle(bearing - node.bearing))
            if np.degrees(bearing_diff) > CONFIG['bearing_gate_deg']:
                continue
            
            # 스코어: 거리 + 각도 차이
            score = dist + 0.5 * heading_diff + 0.5 * bearing_diff
            if score < best_score:
                best_score = score
                best_id = nid
        
        return best_id


class Camera:
    def __init__(self, name, data, T_rig_cam):
        self.name = name
        self.T_rig_cam = T_rig_cam
        intr = data['intrinsics']
        self.fx, self.fy = intr[2], intr[3]
        self.cx, self.cy = intr[4], intr[5]
        self.img_width = 640
        self.img_height = 480
        self.K_valid = False
    
    def update_from_caminfo(self, msg):
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        self.img_width = msg.width
        self.img_height = msg.height
        self.K_valid = True


class DoorTrackerV4:
    def __init__(self):
        rospy.init_node("door_tracker_v4", anonymous=False)
        r = rospkg.RosPack()
        
        # Calibration
        calib_path = r.get_path('ces') + "/MOT/calibration file/2024-09-13-20-37-00-camchain.yaml"
        self.cameras = self._load_cameras(calib_path)
        
        # YOLO
        model_path = r.get_path('ces') + "/MOT/pretrained/best.pt"
        self.front_tracker = YOLO(model_path)
        self.detector = YOLO(model_path)
        rospy.loginfo("YOLO loaded")
        
        # State
        self.door_map = DoorMap()
        self.candidates = {}  # local_id -> Candidate
        self.local_to_world = {}  # local_id -> world_id
        
        self.slam_pose = None
        self.pose_buf = deque(maxlen=200)
        self.path_poses = []
        
        self.images = {"front": None, "left": None, "rear": None, "right": None}
        self.lock = threading.Lock()
        self.bridge = CvBridge()
        
        # Subscribers (드라이버 버그: left/rear 스왑)
        rospy.Subscriber("/camera/undistorted/front", Image, lambda m: self._update_image("front", m))
        rospy.Subscriber("/camera/undistorted/left", Image, lambda m: self._update_image("rear", m))
        rospy.Subscriber("/camera/undistorted/rear", Image, lambda m: self._update_image("left", m))
        rospy.Subscriber("/camera/undistorted/right", Image, lambda m: self._update_image("right", m))
        
        # CameraInfo
        rospy.Subscriber("/camera/undistorted/front/camera_info", CameraInfo, lambda m: self._update_caminfo("front", m))
        rospy.Subscriber("/camera/undistorted/left/camera_info", CameraInfo, lambda m: self._update_caminfo("rear", m))
        rospy.Subscriber("/camera/undistorted/rear/camera_info", CameraInfo, lambda m: self._update_caminfo("left", m))
        rospy.Subscriber("/camera/undistorted/right/camera_info", CameraInfo, lambda m: self._update_caminfo("right", m))
        
        # SLAM
        rospy.Subscriber(CONFIG['slam_topic'], PoseStamped, self.cb_slam_pose)
        
        # Publishers
        self.marker_pub = rospy.Publisher("/door_markers", MarkerArray, queue_size=1)
        self.path_pub = rospy.Publisher("/robot_path", Path, queue_size=1)
        
        # TF Broadcaster (RViz용)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        rospy.loginfo("Door Tracker v4 (Pose+Bearing) Started")
    
    def _load_cameras(self, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        
        cams = {}
        cam_ids = ['cam0', 'cam2', 'cam1', 'cam3']
        cam_names = ['front', 'rear', 'left', 'right']
        
        T_chain = [np.eye(4)]
        for i in range(1, 4):
            T_cn_cnm1 = np.array(data[f'cam{i}'].get('T_cn_cnm1', np.eye(4).tolist()))
            T_rel = np.linalg.inv(T_cn_cnm1)
            T_chain.append(T_chain[-1] @ T_rel)
        
        for i, (cid, name) in enumerate(zip(cam_ids, cam_names)):
            cams[name] = Camera(name, data[cid], T_chain[i])
            rospy.loginfo(f"[{name}] Loaded from {cid}")
        
        return cams
    
    def _update_image(self, key, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            stamp = msg.header.stamp.to_sec()
            with self.lock:
                self.images[key] = (stamp, img)
        except:
            pass
    
    def _update_caminfo(self, key, msg):
        with self.lock:
            if key in self.cameras:
                self.cameras[key].update_from_caminfo(msg)
    
    def cb_slam_pose(self, msg):
        tx = msg.pose.position.x
        ty = msg.pose.position.y
        tz = msg.pose.position.z
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        
        R_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = [tx, ty, tz]
        stamp = msg.header.stamp.to_sec()
        
        with self.lock:
            self.pose_buf.append((stamp, T))
            self.slam_pose = T
            self.path_poses.append((tx, ty, tz))
            if len(self.path_poses) > 1000:
                self.path_poses = self.path_poses[-500:]
        
        # Publish TF immediately for smooth RViz
        self.publish_tf(T, msg.header.stamp)
    
    def get_pose_nearest(self, stamp_sec):
        with self.lock:
            if not self.pose_buf:
                return None
            best = min(self.pose_buf, key=lambda x: abs(x[0] - stamp_sec))
            if abs(best[0] - stamp_sec) > CONFIG['max_time_diff']:
                return None
            return best[1]
    
    def get_heading(self, T):
        """Pose에서 yaw(헤딩) 추출"""
        R_mat = T[:3, :3]
        # 카메라 Z축 (전방)이 월드에서 어디를 향하는지
        forward = R_mat @ np.array([0, 0, 1])
        return np.arctan2(forward[1], forward[0])
    
    def pixel_to_bearing(self, u, cam='front'):
        """픽셀 u → 수평 베어링 (rad)"""
        c = self.cameras[cam]
        x = (u - c.cx) / c.fx
        return np.arctan(x)
    
    def run_front_tracking(self, img):
        results = self.front_tracker.track(
            img, persist=True, tracker=CONFIG['tracker_type'],
            conf=CONFIG['door_conf_thresh'], classes=[0], verbose=False
        )
        
        tracks = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            tids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, tid, conf in zip(xyxy, tids, confs):
                if conf < CONFIG['door_conf_thresh']:
                    continue
                x1, y1, x2, y2 = box
                u = (x1 + x2) / 2.0
                tracks.append({
                    'local_id': int(tid),
                    'box': box,
                    'u': u,
                    'conf': conf
                })
        return tracks
    
    def update_door_map(self, front_tracks, T_world_front):
        """Front 트랙에서 문 노드 등록/업데이트"""
        heading = self.get_heading(T_world_front)
        
        # 이미 할당된 world_id들
        assigned_this_frame = set()
        
        for track in front_tracks:
            tid = track['local_id']
            u = track['u']
            bearing = self.pixel_to_bearing(u)
            
            # 이미 world_id 있으면 업데이트만
            if tid in self.local_to_world:
                wid = self.local_to_world[tid]
                if wid in self.door_map.nodes:
                    self.door_map.nodes[wid].update(T_world_front, bearing, heading)
                track['world_id'] = wid
                assigned_this_frame.add(wid)
                continue
            
            # 후보 관리
            if tid not in self.candidates:
                self.candidates[tid] = Candidate(tid, T_world_front, bearing, heading)
                track['world_id'] = None
                continue
            
            cand = self.candidates[tid]
            cand.update(T_world_front, bearing, heading)
            
            # 확정 조건
            if cand.hit_count < CONFIG['confirm_frames']:
                track['world_id'] = None
                continue
            
            # 기존 노드 매칭 시도
            matched_id = self.door_map.find_match(
                T_world_front, bearing, heading, 
                exclude_ids=assigned_this_frame
            )
            
            if matched_id >= 0:
                self.door_map.nodes[matched_id].update(T_world_front, bearing, heading)
                self.local_to_world[tid] = matched_id
                track['world_id'] = matched_id
            else:
                # 새 노드 등록
                new_id = self.door_map.add_node(T_world_front, bearing, heading)
                self.local_to_world[tid] = new_id
                track['world_id'] = new_id
            
            assigned_this_frame.add(self.local_to_world[tid])
            del self.candidates[tid]  # 후보에서 제거
    
    def get_T_world_cam(self, cam_name, T_world_front):
        T_rig_front = self.cameras['front'].T_rig_cam
        T_rig_cam = self.cameras[cam_name].T_rig_cam
        T_front_cam = np.linalg.inv(T_rig_front) @ T_rig_cam
        return T_world_front @ T_front_cam
    
    def assign_side_ids(self, cam, dets, T_world_cam):
        """Side 카메라: 현재 포즈 근처 노드로 매칭"""
        if len(dets) == 0 or len(self.door_map.nodes) == 0:
            return dets
        
        heading = self.get_heading(T_world_cam)
        
        for det in dets:
            u = det['u']
            bearing = self.pixel_to_bearing(u, cam)
            
            matched_id = self.door_map.find_match(T_world_cam, bearing, heading)
            det['world_id'] = matched_id if matched_id >= 0 else None
        
        return dets
    
    def run_detection_only(self, cam, img):
        results = self.detector.predict(
            img, conf=CONFIG['door_conf_thresh'], classes=[0], verbose=False
        )
        
        dets = []
        if results and results[0].boxes is not None:
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(xyxy, confs):
                if conf < CONFIG['door_conf_thresh']:
                    continue
                x1, y1, x2, y2 = box
                u = (x1 + x2) / 2.0
                dets.append({'box': box, 'u': u, 'world_id': None})
        return dets
    
    def process_frame(self):
        with self.lock:
            front_data = self.images.get('front')
            if front_data is None:
                return
            front_stamp, front_img = front_data
            
            imgs = {}
            stamps = {}
            for k in ['front', 'left', 'rear', 'right']:
                if self.images[k] is not None:
                    stamps[k], imgs[k] = self.images[k]
                else:
                    imgs[k] = None
                    stamps[k] = None
        
        T_world_front = self.get_pose_nearest(front_stamp)
        if T_world_front is None:
            self.draw_no_pose(imgs)
            return
        
        # 1. Front tracking
        front_tracks = self.run_front_tracking(front_img)
        self.update_door_map(front_tracks, T_world_front)
        
        # 2. Side cameras
        side_results = {}
        for cam in ['left', 'rear', 'right']:
            if imgs[cam] is None or stamps[cam] is None:
                continue
            T_w_front_side = self.get_pose_nearest(stamps[cam])
            if T_w_front_side is None:
                continue
            T_world_cam = self.get_T_world_cam(cam, T_w_front_side)
            dets = self.run_detection_only(cam, imgs[cam])
            dets = self.assign_side_ids(cam, dets, T_world_cam)
            side_results[cam] = dets
        
        # 3. 시각화
        self.draw_results(imgs, front_tracks, side_results)
        self.publish_markers()
        self.publish_path()
        # publish_tf call moved to cb_slam_pose for lower latency
    
    def publish_tf(self, T_world_front, stamp):
        """TF 발행 (world -> base_link)"""
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "world"
        t.child_frame_id = "base_link"
        
        t.transform.translation.x = T_world_front[0, 3]
        t.transform.translation.y = T_world_front[1, 3]
        t.transform.translation.z = T_world_front[2, 3]
        
        # Rotation matrix to quaternion
        rot = R.from_matrix(T_world_front[:3, :3])
        q = rot.as_quat()  # [x, y, z, w]
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        
        self.tf_broadcaster.sendTransform(t)
    
    def draw_no_pose(self, imgs):
        if imgs.get('front') is not None:
            img = imgs['front'] if isinstance(imgs['front'], np.ndarray) else None
            if img is not None:
                cv2.putText(img, "WAITING FOR SLAM POSE...", (50, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Door Tracker v4", img)
                cv2.waitKey(1)
    
    def draw_results(self, imgs, front_tracks, side_results):
        # Front
        if imgs['front'] is not None:
            for t in front_tracks:
                x1, y1, x2, y2 = map(int, t['box'])
                wid = t.get('world_id')
                if wid is not None:
                    color = (0, 255, 0)
                    label = f"Door #{wid}"
                else:
                    color = (0, 165, 255)
                    label = f"Scanning... (L{t['local_id']})"
                cv2.rectangle(imgs['front'], (x1, y1), (x2, y2), color, 2)
                cv2.putText(imgs['front'], label, (x1, max(y1-5, 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Side
        for cam, dets in side_results.items():
            if imgs[cam] is None:
                continue
            for det in dets:
                x1, y1, x2, y2 = map(int, det['box'])
                wid = det.get('world_id')
                if wid is not None:
                    color = (0, 255, 0)
                    label = f"Door #{wid}"
                else:
                    color = (128, 128, 128)
                    label = "Unknown"
                cv2.rectangle(imgs[cam], (x1, y1), (x2, y2), color, 2)
                cv2.putText(imgs[cam], label, (x1, max(y1-5, 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Status
        status = f"Doors: {len(self.door_map.nodes)} | Candidates: {len(self.candidates)}"
        if imgs['front'] is not None:
            cv2.putText(imgs['front'], status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Grid
        ref = imgs.get('front')
        if ref is None:
            return
        h, w = ref.shape[:2]
        grid_imgs = []
        for cam in ['front', 'rear', 'left', 'right']:
            if imgs[cam] is not None:
                resized = cv2.resize(imgs[cam], (w, h))
            else:
                resized = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(resized, f"NO: {cam}", (50, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            grid_imgs.append(resized)
        
        for i, name in enumerate(['FRONT', 'REAR', 'LEFT', 'RIGHT']):
            cv2.putText(grid_imgs[i], name, (10, h-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        top = np.hstack((grid_imgs[0], grid_imgs[1]))
        bot = np.hstack((grid_imgs[2], grid_imgs[3]))
        cv2.imshow("Door Tracker v4", np.vstack((top, bot)))
        cv2.waitKey(1)
    
    def publish_markers(self):
        ma = MarkerArray()
        
        for nid, node in self.door_map.nodes.items():
            x, y, z = node.anchor_xyz
            ex, ey, ez = node.get_arrow_endpoint(CONFIG['arrow_length'])
            
            # Arrow (문 방향)
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = rospy.Time.now()
            m.ns = "door_arrows"
            m.id = nid
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.points = [Point(x, y, z), Point(ex, ey, ez)]
            m.scale.x = 0.1  # shaft diameter
            m.scale.y = 0.2  # head diameter
            m.color.a = 1.0
            m.color.r = 0.2
            m.color.g = 0.8
            m.color.b = 0.2
            m.lifetime = rospy.Duration(0)
            ma.markers.append(m)
            
            # Text
            t = Marker()
            t.header.frame_id = "map"
            t.header.stamp = rospy.Time.now()
            t.ns = "door_text"
            t.id = 10000 + nid
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = ex
            t.pose.position.y = ey
            t.pose.position.z = ez + 0.3
            t.pose.orientation.w = 1.0
            t.scale.z = 0.3
            t.color.a = 1.0
            t.color.r = t.color.g = t.color.b = 1.0
            t.text = f"Door #{nid}"
            t.lifetime = rospy.Duration(0)
            ma.markers.append(t)
        
        self.marker_pub.publish(ma)
    
    def publish_path(self):
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()
        
        for (x, y, z) in self.path_poses:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        
        self.path_pub.publish(path)
    
    def cleanup(self):
        now = time.time()
        # 오래된 후보 제거
        to_remove = [tid for tid, c in self.candidates.items()
                     if now - c.last_seen > 3.0]
        for tid in to_remove:
            del self.candidates[tid]
    
    def spin(self):
        rate = rospy.Rate(10)
        counter = 0
        while not rospy.is_shutdown():
            self.process_frame()
            counter += 1
            if counter >= 30:
                self.cleanup()
                counter = 0
            rate.sleep()


if __name__ == "__main__":
    try:
        node = DoorTrackerV4()
        node.spin()
    except rospy.ROSInterruptException:
        pass
