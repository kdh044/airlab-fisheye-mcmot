#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Camera Door Tracker with Global ID
- 카메라별 ByteTrack (로컬 ID)
- 겹침 영역에서 ray 기반 매칭
- Union-Find로 전역 ID 통합

카메라 매핑 (토픽 ↔ 캘리브) - 토픽 이름이 드라이버에서 스왑되어 있음:
- front: cam0 (정면)  ← image_raw_front
- rear:  cam1 (후면)  ← image_raw_left  (토픽명 잘못됨)
- left:  cam2 (좌측)  ← image_raw_rear  (토픽명 잘못됨)
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
from cv_bridge import CvBridge
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# ========== 설정 ==========
CONFIG = {
    'door_conf_thresh': 0.15,    # Door 임계값
    'handle_conf_thresh': 0.1,   # Handle 임계값
    'tracker_type': 'bytetrack.yaml',
    'ray_match_thresh': 0.3,   # ray 최단거리 임계값 (미터)
}

CLASS_COLORS = {
    0: (0, 255, 255),   # door
    1: (0, 0, 255),     # handle
}

# 인접 카메라 쌍 (겹침이 있는)
OVERLAP_PAIRS = [
    ('front', 'left'),
    ('front', 'right'),
    ('rear', 'left'),
    ('rear', 'right'),
]
# ==========================


class UnionFind:
    """전역 ID 관리용 Union-Find"""
    def __init__(self):
        self.parent = {}
        self.next_global_id = 0
    
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
    
    def get_global_id(self, cam, local_id):
        """(cam, local_id) → global_id"""
        key = (cam, local_id)
        root = self.find(key)
        return root


class Camera:
    """카메라 캘리브레이션"""
    def __init__(self, cid, data, T_rig_cam):
        self.cid = cid
        self.T_rig_cam = T_rig_cam  # rig → camera 변환
        
        intr = data['intrinsics']
        if len(intr) >= 6:
            self.xi, self.alpha, self.fx, self.fy, self.cx, self.cy = intr[:6]
        else:
            raise ValueError(f"Invalid intrinsics for {cid}")
        
        # Camera origin in rig frame
        self.origin_rig = np.linalg.inv(T_rig_cam)[:3, 3]
        # Camera rotation (rig → cam)
        self.R_rig_cam = T_rig_cam[:3, :3]
    
    def pixel_to_ray(self, u, v):
        """픽셀 좌표 → rig 좌표계에서의 광선 방향 (단위 벡터)"""
        # Undistorted 좌표 → normalized camera coords
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        z = 1.0
        
        # 카메라 좌표계에서의 방향
        d_cam = np.array([x, y, z])
        d_cam = d_cam / np.linalg.norm(d_cam)
        
        # Rig 좌표계로 변환
        R_cam_rig = self.R_rig_cam.T  # cam → rig
        d_rig = R_cam_rig @ d_cam
        
        return d_rig


def ray_ray_distance(o1, d1, o2, d2):
    """두 광선 사이의 최단 거리"""
    w0 = o1 - o2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)
    
    denom = a * c - b * b
    if abs(denom) < 1e-8:
        # 평행
        return np.linalg.norm(w0 - (np.dot(w0, d1) / a) * d1)
    
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    
    # 둘 다 양수여야 카메라 앞쪽
    if s < 0 or t < 0:
        return float('inf')
    
    p1 = o1 + s * d1
    p2 = o2 + t * d2
    
    return np.linalg.norm(p1 - p2)


class MultiCamTracker:
    def __init__(self):
        rospy.init_node("multi_cam_tracker", anonymous=False)
        r = rospkg.RosPack()
        
        # 카메라 로드
        calib_path = r.get_path('ces') + "/MOT/calibration file/2024-09-13-20-37-00-camchain.yaml"
        self.cameras = self._load_cameras(calib_path)
        
        # 카메라별 YOLO 트래커
        model_path = r.get_path('ces') + "/MOT/pretrained/best.pt"
        self.trackers = {}
        for cam in ['front', 'left', 'rear', 'right']:
            self.trackers[cam] = YOLO(model_path)
        rospy.loginfo("YOLO trackers initialized")
        
        # Global ID 관리
        self.global_id_map = UnionFind()
        self.global_id_cache = {}  # (cam, local_id) → display_id (숫자)
        self.next_display_id = 0
        
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        
        self.images = {"front": None, "left": None, "rear": None, "right": None}
        self.last_tracks = {"front": [], "left": [], "rear": [], "right": []}
        
        # 토픽 ↔ 카메라 매핑 (토픽명이 드라이버에서 left/rear 스왑되어 있어 보정)
        rospy.Subscriber("/camera/image_raw_front", Image, lambda m: self._update_image("front", m))
        rospy.Subscriber("/camera/image_raw_left", Image, lambda m: self._update_image("rear", m))   # 토픽 left → 실제 rear
        rospy.Subscriber("/camera/image_raw_rear", Image, lambda m: self._update_image("left", m))   # 토픽 rear → 실제 left
        rospy.Subscriber("/camera/image_raw_right", Image, lambda m: self._update_image("right", m))
        
        rospy.loginfo("Multi-Camera Door Tracker with Global ID started")

    def _load_cameras(self, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        
        cams = {}
        # cam1(rear캘리브)↔cam2(left캘리브) 스왑 - 토픽명 오류 보정
        cam_ids = ['cam0', 'cam2', 'cam1', 'cam3']
        cam_names = ['front', 'rear', 'left', 'right']
        
        # Rig 기준 변환 행렬 계산
        T_chain = [np.eye(4)]  # cam0 = identity
        for i in range(1, 4):
            cid = f'cam{i}'
            T_cn_cnm1 = np.array(data[cid].get('T_cn_cnm1', np.eye(4)))
            T_rel = np.linalg.inv(T_cn_cnm1)
            T_chain.append(T_chain[-1] @ T_rel)
        
        for i, (cid, name) in enumerate(zip(cam_ids, cam_names)):
            cams[name] = Camera(name, data[cid], T_chain[i])
            pos = cams[name].origin_rig
            rospy.loginfo(f"[{name}] origin in rig: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        
        return cams

    def _update_image(self, key, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.images[key] = img
        except Exception as e:
            rospy.logerr(e)

    def run_tracking(self, cam, img):
        """카메라별 ByteTrack → 로컬 트랙"""
        # YOLO 실행은 더 낮은 임계값(Handle: 0.1)으로 수행
        results = self.trackers[cam].track(
            img, persist=True, tracker=CONFIG['tracker_type'],
            conf=CONFIG['handle_conf_thresh'], classes=[0, 1], verbose=False
        )
        
        tracks = []
        if results and len(results) > 0:
            res = results[0]
            if res.boxes is not None and res.boxes.id is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                tids = res.boxes.id.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                
                masks = None
                if res.masks is not None:
                    masks = res.masks.data.cpu().numpy()
                
                for j in range(len(tids)):
                    cls_id = int(clss[j])
                    conf_val = float(confs[j])
                    
                    # 클래스별 임계값 필터링
                    if cls_id == 0 and conf_val < CONFIG['door_conf_thresh']:
                        continue
                    if cls_id == 1 and conf_val < CONFIG['handle_conf_thresh']:
                        continue

                    # 대표점: bbox 바닥 중앙
                    x1, y1, x2, y2 = xyxy[j]
                    u, v = (x1 + x2) / 2, y2  # 바닥 중앙
                    
                    # Rig 좌표계에서의 광선
                    ray_dir = self.cameras[cam].pixel_to_ray(u, v)
                    ray_origin = self.cameras[cam].origin_rig
                    
                    track = {
                        'cam': cam,
                        'local_id': int(tids[j]),
                        'box': xyxy[j],
                        'conf': conf_val,
                        'cls': cls_id,
                        'ray_origin': ray_origin,
                        'ray_dir': ray_dir,
                        'uv': (u, v),
                    }
                    if masks is not None and j < len(masks):
                        track['mask'] = masks[j]
                    tracks.append(track)
        
        return tracks

    def match_overlap_tracks(self, tracks_a, tracks_b):
        """겹침 영역에서 두 카메라 트랙 매칭"""
        if not tracks_a or not tracks_b:
            return []
        
        n, m = len(tracks_a), len(tracks_b)
        cost = np.full((n, m), 1e6)
        
        for i, ta in enumerate(tracks_a):
            for j, tb in enumerate(tracks_b):
                dist = ray_ray_distance(
                    ta['ray_origin'], ta['ray_dir'],
                    tb['ray_origin'], tb['ray_dir']
                )
                if dist < CONFIG['ray_match_thresh']:
                    cost[i, j] = dist
        
        row_idx, col_idx = linear_sum_assignment(cost)
        matches = []
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < CONFIG['ray_match_thresh']:
                matches.append((tracks_a[r], tracks_b[c]))
        
        return matches

    def update_global_ids(self):
        """모든 겹침 쌍에서 매칭 → Union-Find 업데이트 (Door만 매칭)"""
        for cam_a, cam_b in OVERLAP_PAIRS:
            # Handle(1)은 매칭에서 제외, Door(0)만 매칭
            tracks_a = [t for t in self.last_tracks[cam_a] if t['cls'] == 0]
            tracks_b = [t for t in self.last_tracks[cam_b] if t['cls'] == 0]
            
            matches = self.match_overlap_tracks(tracks_a, tracks_b)
            for ta, tb in matches:
                key_a = (ta['cam'], ta['local_id'])
                key_b = (tb['cam'], tb['local_id'])
                self.global_id_map.union(key_a, key_b)

    def get_display_id(self, cam, local_id):
        """(cam, local_id) → 표시용 global ID (정수)"""
        root = self.global_id_map.find((cam, local_id))
        if root not in self.global_id_cache:
            self.global_id_cache[root] = self.next_display_id
            self.next_display_id += 1
        return self.global_id_cache[root]

    def draw_tracks(self, img, tracks):
        """트랙 시각화 (Global ID 표시)"""
        h, w = img.shape[:2]
        for t in tracks:
            x1, y1, x2, y2 = map(int, t['box'])
            cls_id = t['cls']
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            
            # Mask overlay
            if 'mask' in t:
                mask = t['mask']
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                overlay = np.zeros_like(img, dtype=np.uint8)
                overlay[mask > 0.5] = color
                img = cv2.addWeighted(img, 1.0, overlay, 0.4, 0)
            
            # Bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label
            if cls_id == 0: # Door: Show Global ID
                gid = self.get_display_id(t['cam'], t['local_id'])
                label = f"Door G{gid}"
            else: # Handle: Detection only
                label = "Handle"
            
            cv2.putText(img, label, (x1, max(y1-5, 15)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return img

    def make_2x2_grid(self, imgs):
        h, w, _ = imgs[0].shape
        resized = [cv2.resize(img, (w, h)) for img in imgs]
        
        # 디버그 오버레이: 카메라명, 실제토픽, 사용캘리브
        # 순서: front, rear, left, right (imgs 순서)
        debug_info = [
            ('FRONT', 'topic:front', 'cam0'),
            ('REAR',  'topic:left',  'cam1'),  # left 토픽 → rear(cam1 캘리브)
            ('LEFT',  'topic:rear',  'cam2'),  # rear 토픽 → left(cam2 캘리브)
            ('RIGHT', 'topic:right', 'cam3'),
        ]
        
        for i, (name, topic, cid) in enumerate(debug_info):
            cv2.putText(resized[i], name, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(resized[i], f"{topic} -> {cid}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        top = np.hstack((resized[0], resized[1]))
        bottom = np.hstack((resized[2], resized[3]))
        return np.vstack((top, bottom))

    def spin(self):
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            with self.lock:
                if any(self.images[k] is None for k in self.images):
                    rate.sleep()
                    continue
                
                front = self.images["front"].copy()
                left = self.images["left"].copy()
                rear = self.images["rear"].copy()
                right = self.images["right"].copy()
            
            # 1. 카메라별 트래킹
            self.last_tracks["front"] = self.run_tracking("front", front)
            self.last_tracks["left"] = self.run_tracking("left", left)
            self.last_tracks["rear"] = self.run_tracking("rear", rear)
            self.last_tracks["right"] = self.run_tracking("right", right)
            
            # 2. 겹침 쌍에서 Global ID 통합
            self.update_global_ids()
            
            # 3. 시각화
            front = self.draw_tracks(front, self.last_tracks["front"])
            left = self.draw_tracks(left, self.last_tracks["left"])
            rear = self.draw_tracks(rear, self.last_tracks["rear"])
            right = self.draw_tracks(right, self.last_tracks["right"])
            
            grid = self.make_2x2_grid([front, rear, left, right])
            cv2.imshow("Door Tracker (Global ID)", grid)
            cv2.waitKey(1)
            
            rate.sleep()


if __name__ == "__main__":
    node = MultiCamTracker()
    node.spin()