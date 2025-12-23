#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Door Tracker V7 - Linear Corridor Optimized (Simple & Fast)
===========================================================
복잡한 수학 제거. "직진 복도" 시나리오에 최적화.

1. 기본적으로 YOLO+ByteTrack의 2D 트래킹 성능을 신뢰.
2. 트래킹이 끊겨서 ID가 바뀌는 경우(ID Switch)에만 SLAM 좌표로 복구.
3. 로직:
   - "지금 보이는 문"의 위치를 지도에 저장.
   - "새로운 ID"가 뜨면? -> 지도에 있는 문이랑 가까운지 확인.
   - 가까우면? -> "아, 아까 걔네(Old ID)"로 강제 변경.
   - 멀면? -> "진짜 새 문(New ID)" 등록.
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
from geometry_msgs.msg import PoseStamped, Point
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
    
    # ID 복구 게이트 (이 범위 안에 있던 문이면 같은 ID로 간주)
    'recovery_dist': 1.5,    # 미터 (위치 차이)
    'recovery_angle': 20.0,  # 도 (방향 차이)
    
    # SLAM
    'slam_topic': '/orb_slam3/camera_pose',
    'max_time_diff': 0.2,    # 시간 동기화 오차 허용
}

def wrap_angle(rad):
    return (rad + np.pi) % (2 * np.pi) - np.pi

class SimpleDoor:
    """단순 문 객체"""
    def __init__(self, uid, pose, bearing, heading):
        self.id = uid
        self.pose = pose
        self.xyz = pose[:3, 3]
        self.bearing = bearing
        self.heading = heading
        self.last_seen = time.time()
    
    def update(self, pose, bearing, heading):
        # 위치 최신화 (이동평균)
        alpha = 0.3
        self.xyz = (1-alpha)*self.xyz + alpha*pose[:3,3]
        self.bearing = wrap_angle((1-alpha)*self.bearing + alpha*bearing)
        self.heading = wrap_angle((1-alpha)*self.heading + alpha*heading)
        self.last_seen = time.time()

class TrackerV7:
    def __init__(self):
        rospy.init_node("tracker_v7", anonymous=False)
        r = rospkg.RosPack()
        
        # 설정 로드
        calib_path = r.get_path('ces') + "/MOT/calibration file/2024-09-13-20-37-00-camchain.yaml"
        self.cameras = self._load_calib(calib_path)
        
        # 모델 로드
        model_path = r.get_path('ces') + "/MOT/pretrained/best.pt"
        self.yolo = YOLO(model_path)
        
        # 데이터 관리
        self.doors = {}      # global_id -> SimpleDoor
        self.local_map = {}  # local_id(ByteTrack) -> global_id
        self.next_gid = 0
        
        self.pose_buf = deque(maxlen=300)
        self.lock = threading.Lock()
        self.images = {}
        self.bridge = CvBridge()
        
        # 구독
        for cam in ['front', 'left', 'rear', 'right']:
            topic = f"/camera/undistorted/{cam}"
            rospy.Subscriber(topic, Image, self._make_img_cb(cam))
            rospy.Subscriber(f"{topic}/camera_info", CameraInfo, self._make_info_cb(cam))
        
        rospy.Subscriber(CONFIG['slam_topic'], PoseStamped, self.cb_slam)
        
        self.pub_marker = rospy.Publisher("/door_markers", MarkerArray, queue_size=1)
        self.tf_pub = tf2_ros.TransformBroadcaster()
        
        rospy.loginfo("Tracker V7 (Linear Optimized) Started")

    def _load_calib(self, path):
        with open(path) as f: data = yaml.safe_load(f)
        cams = {}
        mapping = {'front':'cam0', 'rear':'cam1', 'left':'cam2', 'right':'cam3'}
        for name, cid in mapping.items():
            intr = data[cid]['intrinsics']
            cams[name] = {'fx': intr[2], 'cx': intr[4]}
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
        return cb

    def cb_slam(self, msg):
        q = msg.pose.orientation; p = msg.pose.position
        rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot; T[:3, 3] = [p.x, p.y, p.z]
        t = msg.header.stamp.to_sec()
        with self.lock:
            self.pose_buf.append((t, T))

    def get_pose(self, t):
        if not self.pose_buf: return None
        best = None; min_dt = 100.0
        for pt, pT in self.pose_buf:
            dt = abs(pt - t)
            if dt < min_dt: min_dt = dt; best = pT
        return best if min_dt < CONFIG['max_time_diff'] else None

    def find_global_match(self, pose, bearing, heading):
        """새로 나타난 문이 기존 문과 겹치는지 확인"""
        curr_xyz = pose[:3, 3]
        curr_angle = heading + bearing
        
        best_id = -1
        min_dist = float('inf')
        
        for gid, door in self.doors.items():
            # 1. 위치 거리 (Euclidean)
            dist = np.linalg.norm(curr_xyz[:2] - door.xyz[:2])
            if dist > CONFIG['recovery_dist']: continue
            
            # 2. 각도 차이
            door_angle = door.heading + door.bearing
            angle_diff = abs(wrap_angle(door_angle - curr_angle))
            if np.degrees(angle_diff) > CONFIG['recovery_angle']: continue
            
            # 매칭 성공
            if dist < min_dist:
                min_dist = dist
                best_id = gid
        
        return best_id

    def process_front(self, img, t, T_rig):
        # Tracking (Door=0, Handle=1)
        res = self.yolo.track(img, persist=True, tracker=CONFIG['tracker_type'], 
                              conf=CONFIG['handle_conf_thresh'], classes=[0,1], verbose=False)
        
        if T_rig is None: return res[0], {} # SLAM 없으면 그냥 보여주기만 함
        
        yaw = np.arctan2(T_rig[0,1], T_rig[0,0]) # Simple Yaw
        matches = {} # local -> global
        
        if res and res[0].boxes.id is not None:
            boxes = res[0].boxes.xyxy.cpu().numpy()
            ids = res[0].boxes.id.cpu().numpy().astype(int)
            clss = res[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, lid, cls_id in zip(boxes, ids, clss):
                if cls_id == 1: continue # Handle 패스
                
                # 1. Bearing 계산
                u = (box[0] + box[2]) / 2.0
                cx = self.cameras['front']['cx']; fx = self.cameras['front']['fx']
                bearing = np.arctan((u - cx) / fx)
                
                # 2. ID 매핑 로직
                global_id = -1
                
                # Case A: 이미 아는 놈 (Local ID가 살아있음)
                if lid in self.local_map:
                    global_id = self.local_map[lid]
                    # 위치 업데이트
                    if global_id in self.doors:
                        self.doors[global_id].update(T_rig, bearing, yaw)
                
                # Case B: 새로운 놈 (Local ID가 새로 뜸 -> Re-ID 시도)
                else:
                    matched_gid = self.find_global_match(T_rig, bearing, yaw)
                    
                    if matched_gid != -1:
                        # "아, 너 아까 걔(Old Global ID)구나!"
                        global_id = matched_gid
                    else:
                        # "진짜 새로운 문이네"
                        global_id = self.next_gid
                        self.next_gid += 1
                        self.doors[global_id] = SimpleDoor(global_id, T_rig, bearing, yaw)
                    
                    self.local_map[lid] = global_id
                
                matches[lid] = global_id
        
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
                        color = (0, 255, 0) # Green (Registered)
                    else:
                        label = f"New {lid}"
                        color = (0, 165, 255) # Orange (Unmapped)
                else: # Handle
                    label = "Handle"
                    color = (0, 0, 255) # Red
                
                cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(viz_img, label, (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        status = "SLAM: OK" if has_pose else "SLAM: NO POSE"
        col = (0, 255, 0) if has_pose else (0, 0, 255)
        cv2.putText(viz_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        
        cv2.imshow("Tracker V7 (Linear)", viz_img)
        cv2.waitKey(1)

    def pub_viz_markers(self):
        ma = MarkerArray()
        for gid, door in self.doors.items():
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = rospy.Time.now()
            m.ns = "doors"; m.id = gid; m.type = Marker.ARROW; m.action = Marker.ADD
            
            # 화살표 방향
            angle = door.heading + door.bearing
            dx = 1.5 * np.cos(angle); dy = 1.5 * np.sin(angle)
            
            m.points = [Point(*door.xyz), Point(door.xyz[0]+dx, door.xyz[1]+dy, door.xyz[2])]
            m.scale.x = 0.05; m.scale.y = 0.1; m.scale.z = 0.1
            m.color.a = 1.0; m.color.g = 1.0
            ma.markers.append(m)
            
            t = Marker()
            t.header.frame_id = "world"; t.ns = "text"; t.id = gid+1000
            t.type = Marker.TEXT_VIEW_FACING; t.text = f"D{gid}"
            t.pose.position = Point(door.xyz[0]+dx, door.xyz[1]+dy, door.xyz[2]+0.2)
            t.scale.z = 0.3; t.color.a = 1.0; t.color.r = 1.0; t.color.g = 1.0; t.color.b = 1.0
            ma.markers.append(t)
            
        self.pub_marker.publish(ma)

    def loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            with self.lock:
                if 'front' not in self.images:
                    rate.sleep(); continue
                t, img = self.images['front']
            
            T_rig = self.get_pose(t)
            res, matches = self.process_front(img, t, T_rig)
            
            self.viz(res, matches, img, T_rig is not None)
            if T_rig is not None: self.pub_viz_markers()
            
            rate.sleep()

if __name__ == "__main__":
    TrackerV7().loop()
