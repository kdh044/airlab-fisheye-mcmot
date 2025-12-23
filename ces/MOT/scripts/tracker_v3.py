#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Door Tracker v3.1 - 핵심 문제 수정 버전
========================================
수정 사항:
1. CameraInfo에서 K 매트릭스 실시간 갱신
2. 이미지 timestamp 기반 pose 동기화
3. 재투영 오차 검증 + 병합 threshold 축소
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
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R

# ========== 설정 ==========
CONFIG = {
    'door_conf_thresh': 0.25,
    'tracker_type': 'bytetrack.yaml',
    
    # 삼각측량 설정
    'min_observations': 5,
    'min_baseline': 0.15,
    'max_obs_history': 30,
    
    # 매칭 설정
    'projection_gate_px': 80,
    'world_merge_thresh': 0.4,      # 0.8 → 0.4로 축소!
    'max_reproj_error': 10.0,       # 재투영 오차 threshold (픽셀)
    
    # 시간 동기화
    'max_time_diff': 0.2,           # pose-image 최대 시간 차이 (초)
    
    # SLAM 토픽
    'slam_topic': '/orb_slam3/camera_pose',
}


class Tracklet:
    """Front 카메라에서의 로컬 트랙 + 관측 누적"""
    def __init__(self, local_id):
        self.local_id = local_id
        self.obs = []
        self.world_id = None
        self.last_seen = time.time()
    
    def add_obs(self, u, v, T_world_front):
        self.obs.append((float(u), float(v), T_world_front.copy()))
        self.last_seen = time.time()
        if len(self.obs) > CONFIG['max_obs_history']:
            self.obs = self.obs[-CONFIG['max_obs_history']:]
    
    def baseline_ok(self):
        if len(self.obs) < CONFIG['min_observations']:
            return False
        centers = np.array([T[:3, 3] for (_, _, T) in self.obs])
        return np.linalg.norm(centers[-1] - centers[0]) > CONFIG['min_baseline']


class Landmark:
    def __init__(self, lid, pos):
        self.id = lid
        self.pos = np.array(pos, dtype=np.float64)
        self.count = 1
        self.last_seen = time.time()
    
    def update(self, pos, alpha=0.3):
        self.pos = (1 - alpha) * self.pos + alpha * np.array(pos)
        self.count += 1
        self.last_seen = time.time()


class WorldMap:
    def __init__(self):
        self.landmarks = {}
        self.next_id = 0
        self.assigned_ids = set()
    
    def add_landmark(self, pos):
        lid = self.next_id
        self.landmarks[lid] = Landmark(lid, pos)
        self.next_id += 1
        self.assigned_ids.add(lid)
        rospy.loginfo(f"[WorldMap] New Landmark #{lid} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        return lid
    
    def find_close_unassigned(self, pos, thresh, exclude_ids):
        best_id, best_dist = -1, float('inf')
        for lid, lm in self.landmarks.items():
            if lid in exclude_ids:
                continue
            d = np.linalg.norm(lm.pos - pos)
            if d < best_dist:
                best_dist, best_id = d, lid
        if best_dist < thresh:
            return best_id, best_dist
        return -1, best_dist


class Camera:
    def __init__(self, name, data, T_rig_cam):
        self.name = name
        self.T_rig_cam = T_rig_cam
        
        # 초기값 (CameraInfo로 덮어씌워질 것)
        intr = data['intrinsics']
        self.fx, self.fy = intr[2], intr[3]
        self.cx, self.cy = intr[4], intr[5]
        self.img_width = 640
        self.img_height = 480
        self.K_valid = False  # CameraInfo 받으면 True
    
    def update_from_caminfo(self, msg):
        """CameraInfo에서 K 업데이트"""
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        self.img_width = msg.width
        self.img_height = msg.height
        self.K_valid = True
    
    def pixel_to_ray(self, u, v):
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        d = np.array([x, y, 1.0], dtype=np.float64)
        return d / (np.linalg.norm(d) + 1e-12)
    
    def project(self, X_cam):
        if X_cam[2] <= 0.1:
            return None
        u = self.fx * (X_cam[0] / X_cam[2]) + self.cx
        v = self.fy * (X_cam[1] / X_cam[2]) + self.cy
        return (u, v)


def triangulate_rays_least_squares(origins, dirs):
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    I = np.eye(3)
    
    for O, d in zip(origins, dirs):
        d = d / (np.linalg.norm(d) + 1e-12)
        M = I - np.outer(d, d)
        A += M
        b += M @ O
    
    if np.linalg.cond(A) > 1e6:
        return None
    try:
        return np.linalg.solve(A, b)
    except:
        return None


class DoorTrackerV3:
    def __init__(self):
        rospy.init_node("door_tracker_v3", anonymous=False)
        r = rospkg.RosPack()
        
        # Calibration
        calib_path = r.get_path('ces') + "/MOT/calibration file/2024-09-13-20-37-00-camchain.yaml"
        self.cameras = self._load_cameras(calib_path)
        
        # YOLO
        model_path = r.get_path('ces') + "/MOT/pretrained/best.pt"
        self.front_tracker = YOLO(model_path)
        self.detector = YOLO(model_path)
        rospy.loginfo("YOLO models loaded")
        
        # State
        self.world_map = WorldMap()
        self.front_tracklets = {}
        self.slam_pose = None
        
        # 이미지: (timestamp, image) 튜플로 저장
        self.images = {"front": None, "left": None, "rear": None, "right": None}
        
        # SLAM pose 버퍼 (시간 동기화용)
        self.pose_buf = deque(maxlen=200)
        
        self.lock = threading.Lock()
        self.bridge = CvBridge()
        
        # Image Subscribers (드라이버 버그: left/rear 스왑)
        rospy.Subscriber("/camera/undistorted/front", Image, lambda m: self._update_image("front", m))
        rospy.Subscriber("/camera/undistorted/left", Image, lambda m: self._update_image("rear", m))
        rospy.Subscriber("/camera/undistorted/rear", Image, lambda m: self._update_image("left", m))
        rospy.Subscriber("/camera/undistorted/right", Image, lambda m: self._update_image("right", m))
        
        # CameraInfo Subscribers (K 매트릭스 갱신)
        rospy.Subscriber("/camera/undistorted/front/camera_info", CameraInfo, lambda m: self._update_caminfo("front", m))
        rospy.Subscriber("/camera/undistorted/left/camera_info", CameraInfo, lambda m: self._update_caminfo("rear", m))
        rospy.Subscriber("/camera/undistorted/rear/camera_info", CameraInfo, lambda m: self._update_caminfo("left", m))
        rospy.Subscriber("/camera/undistorted/right/camera_info", CameraInfo, lambda m: self._update_caminfo("right", m))
        
        # SLAM pose
        rospy.Subscriber(CONFIG['slam_topic'], PoseStamped, self.cb_slam_pose)
        
        # Publishers
        self.marker_pub = rospy.Publisher("/door_markers", MarkerArray, queue_size=1)
        
        rospy.loginfo("Door Tracker v3.1 (Fixed) Started")
    
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
        except Exception as e:
            rospy.logerr_throttle(5, f"Image error: {e}")
    
    def _update_caminfo(self, key, msg):
        """CameraInfo에서 K 매트릭스 업데이트"""
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
    
    def get_pose_nearest(self, stamp_sec):
        """이미지 timestamp에 가장 가까운 pose 반환"""
        with self.lock:
            if not self.pose_buf:
                return None
            best = min(self.pose_buf, key=lambda x: abs(x[0] - stamp_sec))
            if abs(best[0] - stamp_sec) > CONFIG['max_time_diff']:
                return None
            return best[1]
    
    def get_T_world_cam(self, cam_name, T_world_front):
        T_rig_front = self.cameras['front'].T_rig_cam
        T_rig_cam = self.cameras[cam_name].T_rig_cam
        T_front_cam = np.linalg.inv(T_rig_front) @ T_rig_cam
        return T_world_front @ T_front_cam
    
    def reprojection_error(self, X_world, obs_list):
        """재투영 오차 계산 (중앙값)"""
        cam = self.cameras['front']
        errs = []
        for (u, v, T) in obs_list:
            R_wf = T[:3, :3]
            t_wf = T[:3, 3]
            X_cam = R_wf.T @ (X_world - t_wf)
            proj = cam.project(X_cam)
            if proj is None:
                continue
            pu, pv = proj
            errs.append(np.hypot(pu - u, pv - v))
        if not errs:
            return float('inf')
        return float(np.median(errs))
    
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
                u, v = (x1 + x2) / 2.0, y2
                tracks.append({
                    'local_id': int(tid),
                    'box': box,
                    'u': u,
                    'v': v,
                    'conf': conf
                })
        return tracks
    
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
                u, v = (x1 + x2) / 2.0, y2
                dets.append({
                    'box': box,
                    'u': u,
                    'v': v,
                    'world_id': None
                })
        return dets
    
    def update_front_landmarks(self, front_tracks, T_world_front):
        # 이미 할당된 World ID들
        already_assigned = set()
        for tid, trk in self.front_tracklets.items():
            if trk.world_id is not None:
                already_assigned.add(trk.world_id)
        
        for track in front_tracks:
            tid = track['local_id']
            u, v = track['u'], track['v']
            
            if tid not in self.front_tracklets:
                self.front_tracklets[tid] = Tracklet(tid)
            trk = self.front_tracklets[tid]
            
            trk.add_obs(u, v, T_world_front)
            
            if trk.world_id is not None:
                track['world_id'] = trk.world_id
                continue
            
            if not trk.baseline_ok():
                track['world_id'] = None
                continue
            
            # 삼각측량
            origins, dirs = [], []
            for (uu, vv, T) in trk.obs[::3]:
                ray_cam = self.cameras['front'].pixel_to_ray(uu, vv)
                R_wf = T[:3, :3]
                t_wf = T[:3, 3]
                ray_world = R_wf @ ray_cam
                origins.append(t_wf)
                dirs.append(ray_world)
            
            if len(origins) < 3:
                track['world_id'] = None
                continue
            
            X_world = triangulate_rays_least_squares(np.array(origins), np.array(dirs))
            
            if X_world is None:
                track['world_id'] = None
                continue
            
            # 재투영 오차 검증!
            reproj_err = self.reprojection_error(X_world, trk.obs[::2])
            if reproj_err > CONFIG['max_reproj_error']:
                rospy.logwarn_throttle(2, f"[L{tid}] Reproj error too high: {reproj_err:.1f}px")
                track['world_id'] = None
                continue
            
            # 기존 landmark 매칭 (이미 할당된 것 제외)
            matched_id, dist = self.world_map.find_close_unassigned(
                X_world, CONFIG['world_merge_thresh'], already_assigned
            )
            
            if matched_id >= 0:
                self.world_map.landmarks[matched_id].update(X_world)
                trk.world_id = matched_id
            else:
                trk.world_id = self.world_map.add_landmark(X_world)
            
            already_assigned.add(trk.world_id)
            track['world_id'] = trk.world_id
    
    def assign_ids_by_projection(self, cam, dets, T_world_cam):
        if len(dets) == 0 or len(self.world_map.landmarks) == 0:
            return dets
        
        R_wc = T_world_cam[:3, :3]
        t_wc = T_world_cam[:3, 3]
        
        lmk_ids = []
        lmk_uv = []
        
        for lid, lm in self.world_map.landmarks.items():
            X_cam = R_wc.T @ (lm.pos - t_wc)
            proj = self.cameras[cam].project(X_cam)
            if proj is None:
                continue
            u, v = proj
            if 0 <= u < self.cameras[cam].img_width and 0 <= v < self.cameras[cam].img_height:
                lmk_ids.append(lid)
                lmk_uv.append((u, v))
        
        if len(lmk_ids) == 0:
            return dets
        
        C = np.full((len(lmk_ids), len(dets)), 1e6, dtype=np.float64)
        for i, (pu, pv) in enumerate(lmk_uv):
            for j, det in enumerate(dets):
                dist = np.hypot(pu - det['u'], pv - det['v'])
                if dist < CONFIG['projection_gate_px']:
                    C[i, j] = dist
        
        row_ind, col_ind = linear_sum_assignment(C)
        for r, c in zip(row_ind, col_ind):
            if C[r, c] < CONFIG['projection_gate_px']:
                dets[c]['world_id'] = lmk_ids[r]
        
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
        
        # Front pose (시간 동기화)
        T_world_front = self.get_pose_nearest(front_stamp)
        if T_world_front is None:
            self.draw_no_pose(imgs)
            return
        
        # 1. Front tracking
        front_tracks = self.run_front_tracking(front_img)
        self.update_front_landmarks(front_tracks, T_world_front)
        
        # 2. Side cameras (각자 시간에 맞는 pose 사용)
        side_results = {}
        for cam in ['left', 'rear', 'right']:
            if imgs[cam] is None or stamps[cam] is None:
                continue
            
            # Side 카메라 시간에 맞는 pose
            T_w_front_side = self.get_pose_nearest(stamps[cam])
            if T_w_front_side is None:
                continue
            
            T_world_cam = self.get_T_world_cam(cam, T_w_front_side)
            dets = self.run_detection_only(cam, imgs[cam])
            dets = self.assign_ids_by_projection(cam, dets, T_world_cam)
            side_results[cam] = dets
        
        # 3. 시각화
        self.draw_results(imgs, front_tracks, side_results)
        self.publish_markers()
    
    def draw_no_pose(self, imgs):
        if imgs.get('front') is not None:
            img = imgs['front'].copy() if isinstance(imgs['front'], np.ndarray) else None
            if img is not None:
                cv2.putText(img, "WAITING FOR SLAM POSE...", (50, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Door Tracker v3.1", img)
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
        
        # Side cameras
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
        
        # K 상태 표시
        k_status = " | ".join([f"{k}:{'OK' if c.K_valid else 'NO'}" for k, c in self.cameras.items()])
        
        # 상태
        status = f"Landmarks: {len(self.world_map.landmarks)} | {k_status}"
        if imgs['front'] is not None:
            cv2.putText(imgs['front'], status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Grid
        ref_img = imgs['front']
        if ref_img is None:
            return
        h, w = ref_img.shape[:2]
        grid_imgs = []
        for cam in ['front', 'rear', 'left', 'right']:
            if imgs[cam] is not None:
                resized = cv2.resize(imgs[cam], (w, h))
            else:
                resized = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(resized, f"NO IMAGE: {cam}", (50, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            grid_imgs.append(resized)
        
        for i, name in enumerate(['FRONT', 'REAR', 'LEFT', 'RIGHT']):
            cv2.putText(grid_imgs[i], name, (10, h-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        top = np.hstack((grid_imgs[0], grid_imgs[1]))
        bot = np.hstack((grid_imgs[2], grid_imgs[3]))
        grid = np.vstack((top, bot))
        
        cv2.imshow("Door Tracker v3.1", grid)
        cv2.waitKey(1)
    
    def publish_markers(self):
        ma = MarkerArray()
        for lid, lm in self.world_map.landmarks.items():
            x, y, z = lm.pos
            
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
            m.color.g = 1.0
            m.lifetime = rospy.Duration(0.5)
            ma.markers.append(m)
            
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
    
    def cleanup_tracklets(self):
        now = time.time()
        to_remove = [tid for tid, trk in self.front_tracklets.items()
                     if now - trk.last_seen > 5.0]
        for tid in to_remove:
            del self.front_tracklets[tid]
    
    def spin(self):
        rate = rospy.Rate(10)
        cleanup_counter = 0
        while not rospy.is_shutdown():
            self.process_frame()
            cleanup_counter += 1
            if cleanup_counter >= 50:
                self.cleanup_tracklets()
                cleanup_counter = 0
            rate.sleep()


if __name__ == "__main__":
    try:
        node = DoorTrackerV3()
        node.spin()
    except rospy.ROSInterruptException:
        pass
