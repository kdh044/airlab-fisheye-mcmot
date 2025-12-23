#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Door Tracker v2 - 개선된 버전
1. Bbox 기반 단일뷰 3D 점 추정 (Scanning 제거)
2. Hungarian 1:1 매칭 (같은 ID 중복 할당 방지)
3. Overlap 쌍만 Ray union
4. EMA Smoothing + RViz Marker
"""

import rospy
import rospkg
import cv2
import numpy as np
import yaml
import threading
import time
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R

# ========== 설정 ==========
CONFIG = {
    'door_conf_thresh': 0.25,
    'handle_conf_thresh': 0.15,
    'tracker_type': 'bytetrack.yaml',
    'ray_match_thresh': 0.3,       # Ray 매칭 거리 (rig 좌표)
    
    # Bbox 기반 depth 추정
    'door_height_m': 2.0,          # 실내 문 높이 가정
    'min_depth_m': 0.5,
    'max_depth_m': 20.0,
    
    # ID 매칭 설정
    'assign_thresh_m': 1.5,        # 월드 좌표 매칭 거리 (너무 작으면 ID 끊김)
    
    # EMA 스무딩
    'ema_alpha': 0.3,
    
    # SLAM 토픽
    'slam_topic': '/orb_slam3/camera_pose',
}

CLASS_COLORS = {
    0: (0, 255, 255),   # door - cyan
    1: (0, 0, 255),     # handle - red
}

# 겹치는 카메라 쌍만 Ray union 허용
OVERLAP_PAIRS = [
    ('front', 'left'),
    ('front', 'right'),
    ('rear', 'left'),
    ('rear', 'right'),
]

# ========== 유틸리티 ==========

def ray_ray_distance(o1, d1, o2, d2):
    """두 Ray 사이 최단 거리"""
    w0 = o1 - o2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)
    denom = a * c - b * b
    if abs(denom) < 1e-8:
        return float('inf')
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    if s < 0 or t < 0:
        return float('inf')
    p1 = o1 + s * d1
    p2 = o2 + t * d2
    return np.linalg.norm(p1 - p2)

def triangulate_midpoint(o1, d1, o2, d2):
    """두 Ray의 교차 중점"""
    w0 = o1 - o2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)
    denom = a * c - b * b
    if abs(denom) < 1e-8:
        return None
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    if s < 0 or t < 0:
        return None
    p1 = o1 + s * d1
    p2 = o2 + t * d2
    return (p1 + p2) / 2.0


class Landmark:
    """월드 좌표계 문 랜드마크"""
    def __init__(self, landmark_id, pos_3d):
        self.id = landmark_id
        self.pos = np.array(pos_3d, dtype=float)
        self.count = 1
        self.last_seen = time.time()
    
    def update(self, pos_3d, alpha=0.3):
        """EMA 스무딩"""
        self.pos = (1 - alpha) * self.pos + alpha * np.array(pos_3d)
        self.count += 1
        self.last_seen = time.time()


class WorldMap:
    """전역 랜드마크 관리"""
    def __init__(self):
        self.landmarks = {}
        self.next_id = 0
    
    def add_landmark(self, pos_3d):
        lid = self.next_id
        self.landmarks[lid] = Landmark(lid, pos_3d)
        self.next_id += 1
        rospy.loginfo(f"[WorldMap] New landmark #{lid} at ({pos_3d[0]:.2f}, {pos_3d[1]:.2f}, {pos_3d[2]:.2f})")
        return lid


class UnionFind:
    """프레임 내 멀티카메라 통합"""
    def __init__(self):
        self.parent = {}
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py


class Camera:
    """카메라 intrinsic + extrinsic"""
    def __init__(self, name, data, T_rig_cam):
        self.name = name
        self.T_rig_cam = T_rig_cam
        
        intr = data['intrinsics']
        # Double Sphere: [xi, alpha, fx, fy, cx, cy]
        self.fx, self.fy = intr[2], intr[3]
        self.cx, self.cy = intr[4], intr[5]
        
        self.origin_rig = np.linalg.inv(T_rig_cam)[:3, 3]
        self.R_rig_cam = T_rig_cam[:3, :3]
    
    def update_size(self, w, h):
        """Undistorted 이미지용 principal point 업데이트"""
        self.cx = w / 2.0
        self.cy = h / 2.0
    
    def pixel_to_ray_rig(self, u, v):
        """픽셀 → Rig 좌표계 Ray"""
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        d_cam = np.array([x, y, 1.0])
        d_cam = d_cam / np.linalg.norm(d_cam)
        R_cam_rig = self.R_rig_cam.T
        d_rig = R_cam_rig @ d_cam
        return d_rig
    
    def bbox_to_point_rig(self, box, assume_height_m=2.0, min_z=0.5, max_z=20.0):
        """
        Bbox 높이로 depth 추정 → Rig 좌표 점 생성
        핵심: 단일 카메라에서도 3D 점을 만들어 Scanning 제거
        """
        x1, y1, x2, y2 = box
        hpx = max(1.0, (y2 - y1))
        
        # Pinhole depth approximation: Z = fy * H_real / h_pixel
        Z = (self.fy * assume_height_m) / hpx
        Z = float(np.clip(Z, min_z, max_z))
        
        # Bbox 바닥 중앙
        u = (x1 + x2) / 2.0
        v = y2
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        
        # Camera 좌표
        p_cam = np.array([x * Z, y * Z, Z])
        
        # Cam → Rig
        R_cam_rig = self.R_rig_cam.T
        p_rig = R_cam_rig @ p_cam + self.origin_rig
        return p_rig


class DoorTrackerV2:
    def __init__(self):
        rospy.init_node("door_tracker_v2", anonymous=False)
        r = rospkg.RosPack()
        
        # 1. Load Calibration
        calib_path = r.get_path('ces') + "/MOT/calibration file/2024-09-13-20-37-00-camchain.yaml"
        self.cameras = self._load_cameras(calib_path)
        
        # 2. YOLO Trackers (4개 카메라)
        model_path = r.get_path('ces') + "/MOT/pretrained/best.pt"
        self.trackers = {}
        for cam in ['front', 'left', 'rear', 'right']:
            self.trackers[cam] = YOLO(model_path)
        rospy.loginfo("YOLO trackers loaded")
        
        # 3. State
        self.world_map = WorldMap()
        self.local_union = UnionFind()
        self.slam_pose = None
        
        self.images = {"front": None, "left": None, "rear": None, "right": None}
        self.lock = threading.Lock()
        self.bridge = CvBridge()
        
        # 4. Subscribers (토픽 left/rear 스왑 주의)
        rospy.Subscriber("/camera/undistorted/front", Image, lambda m: self._update_image("front", m))
        rospy.Subscriber("/camera/undistorted/left", Image, lambda m: self._update_image("rear", m))
        rospy.Subscriber("/camera/undistorted/rear", Image, lambda m: self._update_image("left", m))
        rospy.Subscriber("/camera/undistorted/right", Image, lambda m: self._update_image("right", m))
        rospy.Subscriber(CONFIG['slam_topic'], PoseStamped, self.cb_slam_pose)
        
        # 5. Publishers
        self.marker_pub = rospy.Publisher("/door_markers", MarkerArray, queue_size=1)
        
        # Overlap set for fast lookup
        self.overlap_set = set()
        for a, b in OVERLAP_PAIRS:
            self.overlap_set.add((a, b))
            self.overlap_set.add((b, a))
        
        rospy.loginfo("Door Tracker v2 (Bbox Depth + Hungarian) Started")
    
    def _load_cameras(self, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        
        cams = {}
        # 토픽 left/rear 스왑에 맞춰 cam1↔cam2 스왑
        cam_ids = ['cam0', 'cam2', 'cam1', 'cam3']
        cam_names = ['front', 'rear', 'left', 'right']
        
        T_chain = [np.eye(4)]
        for i in range(1, 4):
            cid = f'cam{i}'
            T_cn_cnm1 = np.array(data[cid].get('T_cn_cnm1', np.eye(4)))
            T_rel = np.linalg.inv(T_cn_cnm1)
            T_chain.append(T_chain[-1] @ T_rel)
        
        for i, (cid, name) in enumerate(zip(cam_ids, cam_names)):
            cams[name] = Camera(name, data[cid], T_chain[i])
            rospy.loginfo(f"[{name}] Loaded from {cid}")
        
        return cams
    
    def _update_image(self, key, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h, w = img.shape[:2]
            if key in self.cameras:
                self.cameras[key].update_size(w, h)
            with self.lock:
                self.images[key] = img
        except Exception:
            pass
    
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
        
        with self.lock:
            self.slam_pose = T
    
    def run_tracking(self, cam, img):
        """카메라별 트래킹"""
        results = self.trackers[cam].track(
            img, persist=True, tracker=CONFIG['tracker_type'],
            conf=CONFIG['handle_conf_thresh'], classes=[0, 1], verbose=False
        )
        
        tracks = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            res = results[0]
            xyxy = res.boxes.xyxy.cpu().numpy()
            tids = res.boxes.id.cpu().numpy().astype(int)
            clss = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            
            for j in range(len(tids)):
                cls_id = int(clss[j])
                conf = float(confs[j])
                
                if cls_id == 0 and conf < CONFIG['door_conf_thresh']:
                    continue
                if cls_id == 1 and conf < CONFIG['handle_conf_thresh']:
                    continue
                
                x1, y1, x2, y2 = xyxy[j]
                u, v = (x1 + x2) / 2, y2
                
                ray_dir = self.cameras[cam].pixel_to_ray_rig(u, v)
                ray_org = self.cameras[cam].origin_rig
                
                tracks.append({
                    'cam': cam,
                    'local_id': int(tids[j]),
                    'box': xyxy[j],
                    'cls': cls_id,
                    'conf': conf,
                    'ray_org_rig': ray_org,
                    'ray_dir_rig': ray_dir,
                    'world_id': None,
                })
        
        return tracks
    
    def process_frame(self):
        with self.lock:
            if all(v is None for v in self.images.values()):
                return
            
            imgs = {}
            valid_cams = []
            for k in ['front', 'left', 'rear', 'right']:
                if self.images[k] is not None:
                    imgs[k] = self.images[k].copy()
                    valid_cams.append(k)
                else:
                    imgs[k] = np.zeros((480, 640, 3), dtype=np.uint8)
            
            current_pose = self.slam_pose
        
        # 1. 트래킹
        all_tracks = []
        for cam in valid_cams:
            tracks = self.run_tracking(cam, imgs[cam])
            all_tracks.extend(tracks)
        
        # 2. Door만 처리
        doors = [t for t in all_tracks if t['cls'] == 0]
        
        # 3. Ray Union (겹치는 카메라쌍만!)
        self.local_union = UnionFind()
        for i in range(len(doors)):
            for j in range(i + 1, len(doors)):
                t1, t2 = doors[i], doors[j]
                if t1['cam'] == t2['cam']:
                    continue
                if (t1['cam'], t2['cam']) not in self.overlap_set:
                    continue  # 겹치지 않는 쌍은 무시
                
                dist = ray_ray_distance(
                    t1['ray_org_rig'], t1['ray_dir_rig'],
                    t2['ray_org_rig'], t2['ray_dir_rig']
                )
                if dist < CONFIG['ray_match_thresh']:
                    self.local_union.union((t1['cam'], t1['local_id']), (t2['cam'], t2['local_id']))
        
        # 4. 클러스터 그룹화
        clusters = {}
        for t in doors:
            key = (t['cam'], t['local_id'])
            root = self.local_union.find(key)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(t)
        
        # 5. World ID 할당 (Hungarian 1:1)
        if current_pose is not None:
            R_wr = current_pose[:3, :3]
            t_wr = current_pose[:3, 3]
            
            # 5-1. 각 클러스터의 월드 좌표 추정
            obs = []
            for root, group in clusters.items():
                pos_3d_rig = None
                
                # 멀티뷰: triangulation 시도 (각도 큰 쌍 선택)
                if len(group) >= 2:
                    best_pair = None
                    best_angle = -1.0
                    for a in range(len(group)):
                        for b in range(a + 1, len(group)):
                            da = group[a]['ray_dir_rig']
                            db = group[b]['ray_dir_rig']
                            ang = np.arccos(np.clip(np.dot(da, db), -1.0, 1.0))
                            if ang > best_angle:
                                best_angle = ang
                                best_pair = (group[a], group[b])
                    
                    if best_pair and best_angle > np.deg2rad(2.0):
                        t1, t2 = best_pair
                        pos_3d_rig = triangulate_midpoint(
                            t1['ray_org_rig'], t1['ray_dir_rig'],
                            t2['ray_org_rig'], t2['ray_dir_rig']
                        )
                
                # 단일뷰: bbox 기반 depth 추정 (핵심!)
                if pos_3d_rig is None:
                    rep = group[0]
                    pos_3d_rig = self.cameras[rep['cam']].bbox_to_point_rig(
                        rep['box'],
                        assume_height_m=CONFIG['door_height_m'],
                        min_z=CONFIG['min_depth_m'],
                        max_z=CONFIG['max_depth_m'],
                    )
                
                pos_3d_world = R_wr @ pos_3d_rig + t_wr
                obs.append({'root': root, 'group': group, 'p_world': pos_3d_world})
            
            # 5-2. Hungarian 매칭 (1:1)
            lm_ids = list(self.world_map.landmarks.keys())
            M = len(obs)
            N = len(lm_ids)
            
            assigned = {}
            
            if M > 0 and N > 0:
                cost = np.full((M, N), 1e6, dtype=np.float32)
                for i in range(M):
                    p = obs[i]['p_world']
                    for j, lid in enumerate(lm_ids):
                        d = np.linalg.norm(self.world_map.landmarks[lid].pos - p)
                        if d < CONFIG['assign_thresh_m']:
                            cost[i, j] = d
                
                row_ind, col_ind = linear_sum_assignment(cost)
                for r, c in zip(row_ind, col_ind):
                    if cost[r, c] < CONFIG['assign_thresh_m']:
                        assigned[r] = lm_ids[c]
            
            # 5-3. 업데이트 or 새 랜드마크 생성
            for i in range(M):
                if i in assigned:
                    lid = assigned[i]
                    self.world_map.landmarks[lid].update(obs[i]['p_world'], CONFIG['ema_alpha'])
                else:
                    lid = self.world_map.add_landmark(obs[i]['p_world'])
                    assigned[i] = lid
                
                # 그룹 내 모든 트랙에 world_id 할당
                for t in obs[i]['group']:
                    t['world_id'] = assigned[i]
        
        # 6. 시각화
        self.draw_results(imgs, all_tracks, current_pose is not None)
        self.publish_markers()
    
    def draw_results(self, imgs, tracks, slam_ok):
        # 각 이미지에 detection 그리기
        for t in tracks:
            img = imgs[t['cam']]
            x1, y1, x2, y2 = map(int, t['box'])
            cls_id = t['cls']
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            if cls_id == 0:
                wid = t.get('world_id')
                if wid is not None:
                    label = f"Door #{wid}"
                    label_color = (0, 255, 0)
                else:
                    label = "Scanning..."
                    label_color = (0, 165, 255)
            else:
                label = "Handle"
                label_color = color
            
            cv2.putText(img, label, (x1, max(y1-5, 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
        
        # 상태 표시
        status = f"SLAM: {'OK' if slam_ok else 'NO'} | Doors: {len(self.world_map.landmarks)}"
        cv2.putText(imgs['front'], status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if slam_ok else (0, 0, 255), 2)
        
        # 2x2 그리드
        h, w = imgs['front'].shape[:2]
        resized = [cv2.resize(imgs[k], (w, h)) for k in ['front', 'rear', 'left', 'right']]
        
        for i, name in enumerate(['FRONT', 'REAR', 'LEFT', 'RIGHT']):
            cv2.putText(resized[i], name, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        top = np.hstack((resized[0], resized[1]))
        bot = np.hstack((resized[2], resized[3]))
        grid = np.vstack((top, bot))
        
        cv2.imshow("Door Tracker v2", grid)
        cv2.waitKey(1)
    
    def publish_markers(self):
        ma = MarkerArray()
        
        for lid, lm in self.world_map.landmarks.items():
            x, y, z = lm.pos
            
            # Sphere
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = rospy.Time.now()
            m.ns = "doors"
            m.id = lid
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color.a = 1.0
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.lifetime = rospy.Duration(0.5)
            ma.markers.append(m)
            
            # Text
            t = Marker()
            t.header.frame_id = "world"
            t.header.stamp = rospy.Time.now()
            t.ns = "door_text"
            t.id = 10000 + lid
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = x
            t.pose.position.y = y
            t.pose.position.z = z + 0.5
            t.pose.orientation.w = 1.0
            t.scale.z = 0.3
            t.color.a = 1.0
            t.color.r = t.color.g = t.color.b = 1.0
            t.text = f"Door #{lid}"
            t.lifetime = rospy.Duration(0.5)
            ma.markers.append(t)
        
        self.marker_pub.publish(ma)
    
    def spin(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.process_frame()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = DoorTrackerV2()
        node.spin()
    except rospy.ROSInterruptException:
        pass
