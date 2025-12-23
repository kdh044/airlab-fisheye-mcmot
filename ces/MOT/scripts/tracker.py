#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Camera Door Tracker with Global ID
- 카메라별 ByteTrack (로컬 ID)
- 겹침 영역에서 ray 기반 매칭
- Union-Find로 전역 ID 통합

카메라 매핑 (토픽 ↔ 캘리브):
- front: cam0 (정면)  ← image_raw_front
- rear:  cam1 (후면)  ← image_raw_left 
- left:  cam2 (좌측)  ← image_raw_rear 
- right: cam3 (우측)  ← image_raw_right
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
from cv_bridge import CvBridge
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R

# ========== 설정 ==========
CONFIG = {
    'door_conf_thresh': 0.15,
    'handle_conf_thresh': 0.1,
    'tracker_type': 'bytetrack.yaml',
    'ray_match_thresh': 0.3,       # 로컬(Rig) Ray 매칭 거리
    'world_match_thresh': 1.0,     # 월드 좌표 매칭 거리 (미터)
    'slam_topic': '/orb_slam3/camera_pose', # SLAM 포즈 토픽
    'cam_height': 1.0,             # 바닥 가정용 (단일 뷰일 때) - 필요시 사용
}

CLASS_COLORS = {
    0: (0, 255, 255),   # door
    1: (0, 0, 255),     # handle
}

OVERLAP_PAIRS = [
    ('front', 'left'),
    ('front', 'right'),
    ('rear', 'left'),
    ('rear', 'right'),
]

class Landmark:
    """월드 좌표계상의 문(Door) 랜드마크"""
    def __init__(self, landmark_id, pos_3d):
        self.id = landmark_id
        self.pos = np.array(pos_3d) # [x, y, z]
        self.count = 1
        self.last_seen = time.time()
    
    def update(self, pos_3d):
        # 간단한 이동평균 업데이트
        alpha = 0.2
        self.pos = (1 - alpha) * self.pos + alpha * np.array(pos_3d)
        self.count += 1
        self.last_seen = time.time()

class WorldMap:
    """전역 랜드마크 관리"""
    def __init__(self):
        self.landmarks = {} # id -> Landmark
        self.next_id = 0
    
    def find_match(self, ray_origin, ray_dir, thresh=1.0):
        """
        Ray와 가장 가까운 랜드마크 찾기 (Point-to-Line Distance)
        dist = || (P - O) - ((P - O) . d) * d ||
        """
        best_id = -1
        min_dist = float('inf')
        
        for lid, lm in self.landmarks.items():
            P = lm.pos
            O = ray_origin
            d = ray_dir
            
            # 벡터 P-O
            w = P - O
            # 투영
            proj = np.dot(w, d)
            if proj < 0: continue # 뒤에 있는 랜드마크 제외
            
            # 수직 거리
            perp = w - proj * d
            dist = np.linalg.norm(perp)
            
            if dist < min_dist:
                min_dist = dist
                best_id = lid
        
        if min_dist < thresh:
            return best_id, min_dist
        return -1, min_dist

    def add_landmark(self, pos_3d):
        lid = self.next_id
        self.landmarks[lid] = Landmark(lid, pos_3d)
        self.next_id += 1
        return lid

class UnionFind:
    """프레임 내 멀티카메라 통합용"""
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
    def __init__(self, cid, data, T_rig_cam):
        self.cid = cid
        self.T_rig_cam = T_rig_cam
        intr = data['intrinsics']
        self.xi, self.alpha, self.fx, self.fy, self.cx, self.cy = intr[:6]
        self.origin_rig = np.linalg.inv(T_rig_cam)[:3, 3]
        self.R_rig_cam = T_rig_cam[:3, :3]
        self.img_width = None
        self.img_height = None
    
    def update_size(self, w, h):
        """Update intrinsics for Undistorted/Pinhole image (centered principal point)"""
        if self.img_width == w and self.img_height == h: return
        self.img_width = w
        self.img_height = h
        self.cx = w / 2.0
        self.cy = h / 2.0
        # fx, fy kept from original (approximation used by undistort node)
    
    def pixel_to_ray_rig(self, u, v):
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        z = 1.0
        d_cam = np.array([x, y, z])
        d_cam = d_cam / np.linalg.norm(d_cam)
        R_cam_rig = self.R_rig_cam.T
        d_rig = R_cam_rig @ d_cam
        return d_rig

def ray_ray_distance(o1, d1, o2, d2):
    w0 = o1 - o2
    a = np.dot(d1, d1); b = np.dot(d1, d2); c = np.dot(d2, d2)
    d = np.dot(d1, w0); e = np.dot(d2, w0)
    denom = a * c - b * b
    if abs(denom) < 1e-8: return float('inf')
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    if s < 0 or t < 0: return float('inf')
    p1 = o1 + s * d1
    p2 = o2 + t * d2
    return np.linalg.norm(p1 - p2)

def triangulate_midpoint(o1, d1, o2, d2):
    w0 = o1 - o2
    a = np.dot(d1, d1); b = np.dot(d1, d2); c = np.dot(d2, d2)
    d = np.dot(d1, w0); e = np.dot(d2, w0)
    denom = a * c - b * b
    if abs(denom) < 1e-8: return None
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    if s < 0 or t < 0: return None
    p1 = o1 + s * d1
    p2 = o2 + t * d2
    return (p1 + p2) / 2.0

class MultiCamTracker:
    def __init__(self):
        rospy.init_node("tracker_slam", anonymous=False)
        r = rospkg.RosPack()
        
        # 1. Load Calibration
        calib_path = r.get_path('ces') + "/MOT/calibration file/2024-09-13-20-37-00-camchain.yaml"
        self.cameras = self._load_cameras(calib_path)
        
        # 2. YOLO Trackers
        model_path = r.get_path('ces') + "/MOT/pretrained/best.pt"
        self.trackers = {}
        for cam in ['front', 'left', 'rear', 'right']:
            self.trackers[cam] = YOLO(model_path)
        
        # 3. State
        self.world_map = WorldMap()
        self.local_union = UnionFind()
        self.slam_pose = None # T_world_rig (4x4)
        
        self.images = {"front": None, "left": None, "rear": None, "right": None}
        self.lock = threading.Lock()
        self.bridge = CvBridge()
        
        # 4. Subscribers (Undistorted Topics)
        # Note: Maintaining the topic swap logic (Left <-> Rear) from previous config
        rospy.Subscriber("/camera/undistorted/front", Image, lambda m: self._update_image("front", m))
        rospy.Subscriber("/camera/undistorted/left", Image, lambda m: self._update_image("rear", m)) 
        rospy.Subscriber("/camera/undistorted/rear", Image, lambda m: self._update_image("left", m))
        rospy.Subscriber("/camera/undistorted/right", Image, lambda m: self._update_image("right", m))
        
        # SLAM Pose Subscriber
        rospy.Subscriber(CONFIG['slam_topic'], PoseStamped, self.cb_slam_pose)
        
        rospy.loginfo("Door Tracker with SLAM-based ID Persistence (Undistorted Input) Started")

    def _load_cameras(self, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        cams = {}
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
        return cams

    def cb_slam_pose(self, msg):
        # PoseStamped -> T_world_rig (assume tracking frame is consistent with rig origin)
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

    def _update_image(self, key, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h, w = img.shape[:2]
            # Update camera principal point for undistorted image (center)
            if key in self.cameras:
                self.cameras[key].update_size(w, h)
            
            with self.lock:
                self.images[key] = img
        except Exception:
            pass

    def run_tracking(self, cam, img):
        results = self.trackers[cam].track(
            img, persist=True, tracker=CONFIG['tracker_type'],
            conf=CONFIG['handle_conf_thresh'], classes=[0, 1], verbose=False
        )
        tracks = []
        if results and results[0].boxes.id is not None:
            res = results[0]
            xyxy = res.boxes.xyxy.cpu().numpy()
            tids = res.boxes.id.cpu().numpy().astype(int)
            clss = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            
            for j in range(len(tids)):
                if clss[j] == 0 and confs[j] < CONFIG['door_conf_thresh']: continue
                
                # Center-bottom point
                x1, y1, x2, y2 = xyxy[j]
                u, v = (x1 + x2) / 2, y2 
                
                # Rig Ray
                ray_dir_rig = self.cameras[cam].pixel_to_ray_rig(u, v)
                ray_org_rig = self.cameras[cam].origin_rig
                
                track = {
                    'cam': cam, 'local_id': tids[j], 'box': xyxy[j], 'cls': clss[j],
                    'ray_org_rig': ray_org_rig, 'ray_dir_rig': ray_dir_rig,
                    'world_id': None # To be filled
                }
                tracks.append(track)
        return tracks

    def process_frame(self):
        # 1. Capture Data
        with self.lock:
            # Check if we have at least one image
            if all(v is None for v in self.images.values()):
                return
            
            # Copy images, fill missing with black
            imgs = {}
            valid_cams = []  # Store which cameras have real images
            current_pose = self.slam_pose # T_world_rig
            
            # Determine reference size from any available image
            ref_h, ref_w = 480, 640
            for v in self.images.values():
                if v is not None:
                    ref_h, ref_w = v.shape[:2]
                    break
            
            for k in ['front', 'left', 'rear', 'right']:
                if self.images[k] is not None:
                    imgs[k] = self.images[k].copy()
                    valid_cams.append(k)
                else:
                    # Create black placeholder
                    imgs[k] = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
                    cv2.putText(imgs[k], f"WAITING: {k}", (50, ref_h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 2. Local Tracking (Only on valid images)
        all_tracks = []
        for cam in valid_cams:
            tracks = self.run_tracking(cam, imgs[cam])
            all_tracks.extend(tracks)
        
        # 3. Resolve Multi-View & World ID
        self.local_union = UnionFind()
        
        # (A) Local Multi-View Binding (Rig space)
        # 같은 물체를 보는 카메라끼리 묶기
        doors = [t for t in all_tracks if t['cls'] == 0]
        # Compare all pairs (naive n^2, but n is small)
        for i in range(len(doors)):
            for j in range(i + 1, len(doors)):
                t1, t2 = doors[i], doors[j]
                if t1['cam'] == t2['cam']: continue # Same camera
                
                dist = ray_ray_distance(
                    t1['ray_org_rig'], t1['ray_dir_rig'],
                    t2['ray_org_rig'], t2['ray_dir_rig']
                )
                if dist < CONFIG['ray_match_thresh']:
                    self.local_union.union((t1['cam'], t1['local_id']), (t2['cam'], t2['local_id']))
        
        # (B) Group by Local Cluster
        clusters = {}
        for t in doors:
            key = (t['cam'], t['local_id'])
            root = self.local_union.find(key)
            if root not in clusters: clusters[root] = []
            clusters[root].append(t)
            
        # (C) Assign World ID
        # SLAM 포즈가 있으면 World 좌표로 변환하여 매칭/등록
        if current_pose is not None:
            R_wr = current_pose[:3, :3]
            t_wr = current_pose[:3, 3]
            
            for root, group in clusters.items():
                # 1. Try to triangulate within group
                pos_3d_rig = None
                if len(group) >= 2:
                    # Pick best pair (max parallax) - simple: just first two
                    t1, t2 = group[0], group[1]
                    pos_3d_rig = triangulate_midpoint(
                        t1['ray_org_rig'], t1['ray_dir_rig'],
                        t2['ray_org_rig'], t2['ray_dir_rig']
                    )
                
                # 2. Convert Rays/Point to World
                # Group Representative Ray (from first cam)
                rep = group[0]
                ray_org_world = R_wr @ rep['ray_org_rig'] + t_wr
                ray_dir_world = R_wr @ rep['ray_dir_rig']
                
                assigned_id = -1
                
                # 3. Match with Map
                # 3-1. If we have 3D point (from triangulation), match point-to-point
                if pos_3d_rig is not None:
                    pos_3d_world = R_wr @ pos_3d_rig + t_wr
                    # Find closest landmark
                    best_id = -1
                    min_d = float('inf')
                    for lid, lm in self.world_map.landmarks.items():
                        d = np.linalg.norm(lm.pos - pos_3d_world)
                        if d < min_d: min_d = d; best_id = lid
                    
                    if min_d < CONFIG['world_match_thresh']:
                        assigned_id = best_id
                        self.world_map.landmarks[best_id].update(pos_3d_world)
                    else:
                        # New Landmark
                        assigned_id = self.world_map.add_landmark(pos_3d_world)
                
                else:
                    # 3-2. Single view: Match Ray to Landmarks
                    mid, mdist = self.world_map.find_match(ray_org_world, ray_dir_world, thresh=0.5)
                    if mid != -1:
                        assigned_id = mid
                    else:
                        # Cannot create new 3D landmark from single ray (unbounded depth)
                        # But we could assume ground plane?
                        # For now, just leave it as None or Temporary
                        pass
                
                # Assign to tracks
                if assigned_id != -1:
                    for t in group: t['world_id'] = assigned_id

        # 4. Visualization
        grid = self.draw_results(imgs, all_tracks, current_pose is not None)
        cv2.imshow("SLAM Door Tracker", grid)
        cv2.waitKey(1)

    def draw_results(self, imgs, tracks, slam_ok):
        # ... Drawing Logic ...
        # Camera names for display
        cam_map = {'front': 0, 'rear': 1, 'left': 2, 'right': 3} # Grid positions
        grids = [imgs['front'], imgs['rear'], imgs['left'], imgs['right']] # Reordered for display
        
        # Draw on original images first (mapped back)
        for t in tracks:
            if self.images[t['cam']] is None: continue # Skip if placeholder
            
            img = imgs[t['cam']]
            x1, y1, x2, y2 = map(int, t['box'])
            color = CLASS_COLORS.get(t['cls'], (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            label = ""
            if t['cls'] == 0: # Door
                if t['world_id'] is not None:
                    label = f"ID: {t['world_id']}"
                    color = (0, 255, 0) # Green for locked ID
                else:
                    label = "Scanning..."
            else:
                label = "Handle"
            
            cv2.putText(img, label, (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Make Grid
        h, w = grids[0].shape[:2]
        resized = [cv2.resize(g, (w, h)) for g in grids]
        
        # Status Text
        status = "SLAM: OK" if slam_ok else "SLAM: NO POSE"
        cv2.putText(resized[0], status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if not slam_ok else (0,255,0), 2)
        
        top = np.hstack((resized[0], resized[2])) # Front, Left
        bot = np.hstack((resized[1], resized[3])) # Rear, Right
        return np.vstack((top, bot))

    def spin(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.process_frame()
            rate.sleep()

if __name__ == "__main__":
    node = MultiCamTracker()
    node.spin()