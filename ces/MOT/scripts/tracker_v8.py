#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Door Tracker V8 - Kalman Filter & Mahalanobis Gating (Spatial Veto)
===================================================================
핵심 기능:
1. Spatial Veto (공간 거부):
   - 2D Tracker가 같은 ID라고 우겨도, 3D 위치(마할라노비스 거리)가 멀면 강제로 ID 분리.
   - "떨어져 있는 다른 문인데 같은 ID를 주는 문제" 해결.
2. Kalman Filter:
   - 문의 3D 위치(x,y,z)와 불확실성(Covariance) 추정.
   - 정지 물체(Static Object) 모델 사용.
3. Height-based Depth:
   - 문 높이(2.1m) 가정을 통해 단일 프레임에서 3D 좌표 즉시 획득.
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
from cv_bridge import CvBridge
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
import tf2_ros

# ========== 설정 ==========
CONFIG = {
    'door_conf_thresh': 0.25,
    'handle_conf_thresh': 0.15,
    'tracker_type': 'bytetrack.yaml',
    
    # 1. 깊이 추정
    'door_real_height': 2.1,  # 문 실제 높이 (미터)
    
    # 2. 칼만 필터
    'P_init': 1.0,  # 초기 위치 불확실성 (Variance)
    'Q_proc': 0.01, # 프로세스 노이즈 (문은 고정되어 있으므로 작게)
    'R_meas': 0.5,  # 측정 노이즈 (센서/인식 오차)
    
    # 3. 매칭 게이트 (핵심)
    # 마할라노비스 거리가 이 값보다 크면 "같은 ID라도" 거부함 (Chi-square 99% ~= 11.3)
    'mahalanobis_gate': 10.0, 
    
    # SLAM
    'slam_topic': '/orb_slam3/camera_pose',
    'max_time_diff': 0.2, # 시간 동기화 허용 오차
}

class KalmanDoor:
    """3D Kalman Filter for a Static Object"""
    def __init__(self, uid, z_meas):
        self.id = uid
        # State [x, y, z]
        self.x = np.array(z_meas, dtype=np.float64)
        
        # Covariance P (3x3)
        self.P = np.eye(3) * CONFIG['P_init']
        
        self.last_seen = time.time()
        self.hits = 1

    def predict(self):
        # Static Model: x_k = x_{k-1}
        # P_k = P_{k-1} + Q
        Q = np.eye(3) * CONFIG['Q_proc']
        self.P += Q

    def update(self, z_meas):
        # Kalman Gain K = P * inv(P + R)
        R_mat = np.eye(3) * CONFIG['R_meas']
        S = self.P + R_mat
        
        try:
            K = self.P @ np.linalg.inv(S)
        except:
            K = np.zeros((3,3))
            
        # State Update
        y = z_meas - self.x # Innovation
        self.x += K @ y
        
        # Covariance Update
        self.P = (np.eye(3) - K) @ self.P
        
        self.last_seen = time.time()
        self.hits += 1

    def get_mahalanobis(self, z_meas):
        """측정값 z가 현재 상태 x와 얼마나 다른지(표준편차 단위)"""
        y = z_meas - self.x
        R_mat = np.eye(3) * CONFIG['R_meas']
        S = self.P + R_mat # Innovation Covariance
        
        try:
            inv_S = np.linalg.inv(S)
            dist_sq = y.T @ inv_S @ y
            return np.sqrt(dist_sq)
        except:
            return float('inf')

class TrackerV8:
    def __init__(self):
        rospy.init_node("tracker_v8", anonymous=False)
        r = rospkg.RosPack()
        
        calib_path = r.get_path('ces') + "/MOT/calibration file/2024-09-13-20-37-00-camchain.yaml"
        self.cameras = self._load_calib(calib_path)
        
        model_path = r.get_path('ces') + "/MOT/pretrained/best.pt"
        self.yolo = YOLO(model_path)
        
        # Global Map
        self.doors = {}      # global_id -> KalmanDoor
        self.local_map = {}  # local_id -> global_id
        self.next_gid = 0
        
        self.pose_buf = deque(maxlen=300)
        self.lock = threading.Lock()
        self.images = {}
        self.bridge = CvBridge()
        
        # Subs
        for cam in ['front', 'left', 'rear', 'right']:
            topic = f"/camera/undistorted/{cam}"
            rospy.Subscriber(topic, Image, self._make_img_cb(cam))
            rospy.Subscriber(f"{topic}/camera_info", CameraInfo, self._make_info_cb(cam))
        
        rospy.Subscriber(CONFIG['slam_topic'], PoseStamped, self.cb_slam)
        
        # Pubs
        self.pub_marker = rospy.Publisher("/door_markers", MarkerArray, queue_size=1)
        self.tf_pub = tf2_ros.TransformBroadcaster()
        
        rospy.loginfo("Tracker V8 (Kalman + Mahalanobis Veto) Started")

    def _load_calib(self, path):
        with open(path) as f: data = yaml.safe_load(f)
        cams = {}
        mapping = {'front':'cam0', 'rear':'cam1', 'left':'cam2', 'right':'cam3'}
        for name, cid in mapping.items():
            intr = data[cid]['intrinsics']
            cams[name] = {'fx': intr[2], 'fy': intr[3], 'cx': intr[4], 'cy': intr[5]}
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
                self.cameras[name]['fx'] = msg.K[0]; self.cameras[name]['fy'] = msg.K[4]
                self.cameras[name]['cx'] = msg.K[2]; self.cameras[name]['cy'] = msg.K[5]
        return cb

    def cb_slam(self, msg):
        q = msg.pose.orientation; p = msg.pose.position
        rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot; T[:3, 3] = [p.x, p.y, p.z]
        t = msg.header.stamp.to_sec()
        with self.lock: self.pose_buf.append((t, T))
        
        # TF Broadcast
        ts = TransformStamped()
        ts.header = msg.header
        ts.header.frame_id = "world"; ts.child_frame_id = "base_link"
        ts.transform.translation = msg.pose.position
        ts.transform.rotation = msg.pose.orientation
        self.tf_pub.sendTransform(ts)

    def get_pose(self, t):
        if not self.pose_buf: return None
        best = None; min_dt = 100.0
        for pt, pT in self.pose_buf:
            dt = abs(pt - t)
            if dt < min_dt: min_dt = dt; best = pT
        return best if min_dt < CONFIG['max_time_diff'] else None

    def estimate_3d(self, box, T_rig):
        """Bounding Box Height -> 3D World Coord"""
        x1, y1, x2, y2 = box
        h_pix = y2 - y1
        if h_pix < 10: return None # 너무 작음
        
        u, v = (x1+x2)/2.0, (y1+y2)/2.0
        c = self.cameras['front']
        
        # Depth = (f * H_real) / H_pix
        depth = (c['fy'] * CONFIG['door_real_height']) / h_pix
        depth = np.clip(depth, 0.5, 20.0) # 안전장치 (0.5m ~ 20m)
        
        # Cam Coord
        X_c = (u - c['cx']) * depth / c['fx']
        Y_c = (v - c['cy']) * depth / c['fy']
        Z_c = depth
        
        # World Coord
        P_cam = np.array([X_c, Y_c, Z_c, 1.0])
        P_world = T_rig @ P_cam
        return P_world[:3]

    def process_front(self, img, t, T_rig):
        # 1. Predict (Kalman Prediction Step)
        for d in self.doors.values(): d.predict()
        
        # 2. Track 2D (YOLO + ByteTrack)
        res = self.yolo.track(img, persist=True, tracker=CONFIG['tracker_type'], 
                              conf=CONFIG['handle_conf_thresh'], classes=[0,1], verbose=False)
        matches = {} # lid -> gid
        
        if res and res[0].boxes.id is not None:
            boxes = res[0].boxes.xyxy.cpu().numpy()
            ids = res[0].boxes.id.cpu().numpy().astype(int)
            clss = res[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, lid, cls_id in zip(boxes, ids, clss):
                if cls_id == 1: continue # Handle skip
                
                # SLAM 없으면 측정 불가 -> 트래킹만 하고 매핑은 안 함
                if T_rig is None: continue 
                
                # 3. Measure (2D -> 3D)
                z_meas = self.estimate_3d(box, T_rig)
                if z_meas is None: continue
                
                matched_gid = -1
                
                # [Spatial Veto Logic]
                # A. 2D Tracker가 "이거 아까 그 놈(Local ID)"이라고 함
                if lid in self.local_map:
                    candidate_gid = self.local_map[lid]
                    if candidate_gid in self.doors:
                        # 검증: 진짜 가까운가? (Mahalanobis check)
                        dist = self.doors[candidate_gid].get_mahalanobis(z_meas)
                        
                        if dist < CONFIG['mahalanobis_gate']:
                            matched_gid = candidate_gid # 통과!
                        else:
                            # 기각! (ID Switch 발생한 것으로 간주)
                            # matched_gid = -1 유지 (새로 검색하게 됨)
                            pass
                
                # B. 기각됐거나 새로운 놈이면 -> 전역 검색 (Global Match)
                if matched_gid == -1:
                    best_dist = float('inf')
                    for gid, door in self.doors.items():
                        dist = door.get_mahalanobis(z_meas)
                        if dist < CONFIG['mahalanobis_gate'] and dist < best_dist:
                            best_dist = dist
                            matched_gid = gid
                
                # 4. Update or Create
                if matched_gid != -1:
                    self.doors[matched_gid].update(z_meas)
                    self.local_map[lid] = matched_gid
                    matches[lid] = matched_gid
                else:
                    # 완전 새로운 문
                    new_gid = self.next_gid
                    self.next_gid += 1
                    self.doors[new_gid] = KalmanDoor(new_gid, z_meas)
                    self.local_map[lid] = new_gid
                    matches[lid] = new_gid
                    
        return res[0], matches

    def viz(self, res, matches, img, has_pose):
        viz_img = img.copy()
        if res and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int)
            clss = res.boxes.cls.cpu().numpy().astype(int)
            
            for box, lid, cls_id in zip(boxes, ids, clss):
                x1, y1, x2, y2 = map(int, box)
                if cls_id == 0: # Door
                    if lid in matches:
                        gid = matches[lid]
                        label = f"Door #{gid}"
                        color = (0, 255, 0) # Green (Locked & Verified)
                    else:
                        label = f"New/Rej"
                        color = (0, 165, 255) # Orange (Unmapped or Rejected)
                else: # Handle
                    label = "Handle"
                    color = (0, 0, 255) # Red
                
                cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(viz_img, label, (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        status = "SLAM: OK (KF Running)" if has_pose else "SLAM: NO POSE"
        col = (0, 255, 0) if has_pose else (0, 0, 255)
        cv2.putText(viz_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.imshow("Tracker V8 (Kalman)", viz_img)
        cv2.waitKey(1)

    def pub_viz_markers(self):
        ma = MarkerArray()
        for gid, door in self.doors.items():
            m = Marker()
            m.header.frame_id = "world"; m.header.stamp = rospy.Time.now()
            m.ns = "kf_doors"; m.id = gid; m.type = Marker.CUBE; m.action = Marker.ADD
            m.pose.position = Point(*door.x)
            m.scale.x = 0.5; m.scale.y = 1.0; m.scale.z = 2.1 # 문 모양
            m.color.a = 0.6; m.color.g = 1.0; m.color.r = 0.2
            ma.markers.append(m)
            
            t = Marker()
            t.header.frame_id = "world"; t.ns = "text"; t.id = gid+1000
            t.type = Marker.TEXT_VIEW_FACING; t.text = f"D{gid}"
            t.pose.position = Point(door.x[0], door.x[1], door.x[2]+1.2)
            t.scale.z = 0.4; t.color.a = 1.0; t.color.r = 1.0; t.color.g = 1.0; t.color.b = 1.0
            ma.markers.append(t)
            
        self.pub_marker.publish(ma)

    def loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            with self.lock:
                if 'front' not in self.images: rate.sleep(); continue
                t, img = self.images['front']
            
            T_rig = self.get_pose(t)
            res, matches = self.process_front(img, t, T_rig)
            
            self.viz(res, matches, img, T_rig is not None)
            if T_rig is not None: self.pub_viz_markers()
            rate.sleep()

if __name__ == "__main__":
    TrackerV8().loop()
