#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Door Mapper - 앞 카메라 + Odometry 기반 문 매핑
==============================================
- 앞 카메라로만 문 검출
- SLAM pose로 경로 + 문 위치 기록
- RViz에서 시각화
"""

import rospy
import rospkg
import cv2
import numpy as np
import yaml
import threading
import time
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from cv_bridge import CvBridge
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

# ========== 설정 ==========
CONFIG = {
    'door_conf_thresh': 0.3,
    'door_default_depth': 3.0,      # 문까지 기본 거리 (미터)
    'merge_distance': 1.5,          # 같은 문으로 볼 거리 (미터)
    'slam_topic': '/orb_slam3/camera_pose',
}


class DoorLandmark:
    def __init__(self, lid, pos, bearing):
        self.id = lid
        self.pos = np.array(pos, dtype=np.float64)  # World [x, y, z]
        self.bearing = bearing  # 처음 발견 시 방향
        self.count = 1
        self.last_seen = time.time()
    
    def update(self, pos, alpha=0.2):
        self.pos = (1 - alpha) * self.pos + alpha * np.array(pos)
        self.count += 1
        self.last_seen = time.time()


class DoorMapper:
    def __init__(self):
        rospy.init_node("door_mapper", anonymous=False)
        r = rospkg.RosPack()
        
        # YOLO
        model_path = r.get_path('ces') + "/MOT/pretrained/best.pt"
        self.detector = YOLO(model_path)
        rospy.loginfo("YOLO loaded")
        
        # Calibration (front camera only)
        calib_path = r.get_path('ces') + "/MOT/calibration file/2024-09-13-20-37-00-camchain.yaml"
        self.fx, self.fy, self.cx, self.cy = self._load_intrinsics(calib_path)
        
        # State
        self.slam_pose = None  # T_world_cam (4x4)
        self.landmarks = {}    # id -> DoorLandmark
        self.next_id = 0
        self.path_poses = []   # 경로 기록
        
        self.image = None
        self.lock = threading.Lock()
        self.bridge = CvBridge()
        
        # Subscribers
        rospy.Subscriber("/camera/undistorted/front", Image, self.cb_image)
        rospy.Subscriber(CONFIG['slam_topic'], PoseStamped, self.cb_slam)
        
        # Publishers
        self.marker_pub = rospy.Publisher("/door_markers", MarkerArray, queue_size=1)
        self.path_pub = rospy.Publisher("/robot_path", Path, queue_size=1)
        
        rospy.loginfo("Door Mapper Started")
    
    def _load_intrinsics(self, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        intr = data['cam0']['intrinsics']
        return intr[2], intr[3], intr[4], intr[5]  # fx, fy, cx, cy
    
    def cb_image(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.image = img
        except:
            pass
    
    def cb_slam(self, msg):
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
        
        with self.lock:
            self.slam_pose = T
            self.path_poses.append((tx, ty, tz))
            if len(self.path_poses) > 1000:
                self.path_poses = self.path_poses[-500:]
    
    def pixel_to_bearing(self, u, v):
        """픽셀 → 카메라 좌표계 방향 벡터"""
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        d = np.array([x, y, 1.0], dtype=np.float64)
        return d / np.linalg.norm(d)
    
    def detection_to_world(self, box, T_world_cam):
        """BBox → World 좌표 (대략적 위치)"""
        x1, y1, x2, y2 = box
        u = (x1 + x2) / 2.0  # 중앙
        v = y2               # 바닥
        
        # 카메라 좌표계 방향
        ray_cam = self.pixel_to_bearing(u, v)
        
        # 카메라 위치 & 방향
        R_wc = T_world_cam[:3, :3]
        t_wc = T_world_cam[:3, 3]
        
        # World 좌표계 방향
        ray_world = R_wc @ ray_cam
        
        # 대략적 거리로 위치 추정 (실제 깊이 모르니까)
        depth = CONFIG['door_default_depth']
        pos_world = t_wc + depth * ray_world
        
        # Bearing angle (yaw)
        bearing = np.arctan2(ray_world[1], ray_world[0])
        
        return pos_world, bearing
    
    def find_or_create_landmark(self, pos, bearing):
        """기존 landmark 찾거나 새로 생성"""
        for lid, lm in self.landmarks.items():
            dist = np.linalg.norm(lm.pos[:2] - pos[:2])  # XY 거리만
            if dist < CONFIG['merge_distance']:
                lm.update(pos)
                return lid
        
        # 새 landmark
        lid = self.next_id
        self.next_id += 1
        self.landmarks[lid] = DoorLandmark(lid, pos, bearing)
        rospy.loginfo(f"[Mapper] New Door #{lid} at ({pos[0]:.2f}, {pos[1]:.2f})")
        return lid
    
    def process_frame(self):
        with self.lock:
            if self.image is None or self.slam_pose is None:
                return
            img = self.image.copy()
            T_world_cam = self.slam_pose.copy()
        
        # YOLO detection
        results = self.detector.predict(
            img, conf=CONFIG['door_conf_thresh'], classes=[0], verbose=False
        )
        
        detections = []
        if results and results[0].boxes is not None:
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(xyxy, confs):
                if conf >= CONFIG['door_conf_thresh']:
                    pos, bearing = self.detection_to_world(box, T_world_cam)
                    lid = self.find_or_create_landmark(pos, bearing)
                    detections.append({
                        'box': box,
                        'conf': conf,
                        'world_id': lid
                    })
        
        # 시각화
        self.draw_results(img, detections)
        self.publish_markers()
        self.publish_path()
    
    def draw_results(self, img, detections):
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            wid = det['world_id']
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Door #{wid}", (x1, max(y1-5, 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 상태 표시
        cv2.putText(img, f"Doors: {len(self.landmarks)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.imshow("Door Mapper", img)
        cv2.waitKey(1)
    
    def publish_markers(self):
        ma = MarkerArray()
        
        for lid, lm in self.landmarks.items():
            x, y, z = lm.pos
            
            # Door sphere
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = rospy.Time.now()
            m.ns = "doors"
            m.id = lid
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z / 2
            m.pose.orientation.w = 1.0
            m.scale.x = 0.5
            m.scale.y = 0.5
            m.scale.z = max(z, 0.5)
            m.color.a = 0.8
            m.color.r = 0.2
            m.color.g = 0.8
            m.color.b = 0.2
            m.lifetime = rospy.Duration(0)  # 영구
            ma.markers.append(m)
            
            # Text
            t = Marker()
            t.header.frame_id = "world"
            t.header.stamp = rospy.Time.now()
            t.ns = "door_text"
            t.id = 1000 + lid
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = x
            t.pose.position.y = y
            t.pose.position.z = z + 0.5
            t.pose.orientation.w = 1.0
            t.scale.z = 0.4
            t.color.a = 1.0
            t.color.r = t.color.g = t.color.b = 1.0
            t.text = f"Door #{lid}"
            t.lifetime = rospy.Duration(0)
            ma.markers.append(t)
        
        self.marker_pub.publish(ma)
    
    def publish_path(self):
        path = Path()
        path.header.frame_id = "world"
        path.header.stamp = rospy.Time.now()
        
        for (x, y, z) in self.path_poses:
            pose = PoseStamped()
            pose.header.frame_id = "world"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        
        self.path_pub.publish(path)
    
    def spin(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.process_frame()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = DoorMapper()
        node.spin()
    except rospy.ROSInterruptException:
        pass
