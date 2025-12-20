#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import rospkg
import cv2
import numpy as np
import yaml
import threading
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_matrix, translation_from_matrix
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class SnapSpaceTracker:
    def __init__(self):
        rospy.init_node('snapspace_tracker', anonymous=False)

        # ---------------------------------------------------------
        # 1. 설정 및 모델 로드
        # ---------------------------------------------------------
        self.ground_height = rospy.get_param("~ground_height", 1.0)
        self.merge_dist_thresh = rospy.get_param("~match_max_dist", 1.0)
        self.cluster_dist_thresh = rospy.get_param("~cluster_max_dist", self.merge_dist_thresh)
        self.track_ttl = rospy.get_param("~track_ttl", 5.0)
        self.conf_thresh = rospy.get_param("~conf_thresh", 0.1)
        self.use_kalman = rospy.get_param("~use_kalman", True)
        self.kalman_q_pos = rospy.get_param("~kalman_q_pos", 0.5)
        self.kalman_q_vel = rospy.get_param("~kalman_q_vel", 1.0)
        self.kalman_r = rospy.get_param("~kalman_r", 0.5)
        self.kalman_init_var = rospy.get_param("~kalman_init_var", 1.0)

        r = rospkg.RosPack()
        self.pkg_path = r.get_path('ces')
        
        # 모델 경로
        model_path = self.pkg_path + "/MOT/pretrained/3_27_witd_model.pt"
        self.model = YOLO(model_path)
        rospy.loginfo(f"Model loaded: {model_path}")

        # 클래스 정보 (fisheye_detection.py 기준)
        self.class_names = {
            0: "building",
            1: "glass_door",
            2: "handle",
            3: "metal_door",
            4: "wood_door"
        }
        self.class_colors = {
            1: (0, 255, 255),   # glass_door
            2: (0, 0, 255),     # handle
            3: (0, 255, 0),     # metal_door
            4: (255, 0, 255)    # wood_door
        }
        self.track_class_ids = {1, 2, 3, 4}
        self.door_class_ids = {1, 3, 4}
        self.handle_class_id = 2

        # 캘리브레이션 로드
        calib_path = self.pkg_path + "/MOT/calibration file/2024-09-13-20-37-00-camchain.yaml"
        self.cameras = self.load_calibration(calib_path)
        rospy.loginfo("Camera calibration loaded.")

        # ---------------------------------------------------------
        # 2. ROS 통신 설정
        # ---------------------------------------------------------
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # 최신 이미지 저장소
        self.current_images = {
            'cam0': None, 'cam1': None, 'cam2': None, 'cam3': None
        }

        # 토픽 구독 (하드코딩된 토픽 이름 - 필요시 수정)
        # cam0: Front, cam1: Rear, cam2: Left, cam3: Right (yaml 기준)
        rospy.Subscriber("/camera/image_raw_front", Image, lambda m: self.img_cb(m, 'cam0'))
        rospy.Subscriber("/camera/image_raw_rear",  Image, lambda m: self.img_cb(m, 'cam1'))
        rospy.Subscriber("/camera/image_raw_left",  Image, lambda m: self.img_cb(m, 'cam2'))
        rospy.Subscriber("/camera/image_raw_right", Image, lambda m: self.img_cb(m, 'cam3'))

        # 결과 발행
        self.marker_pub = rospy.Publisher("/snapspace/markers", MarkerArray, queue_size=10)

        # ---------------------------------------------------------
        # 3. 트래킹 상태 변수
        # ---------------------------------------------------------
        # global_tracks: { global_id: { 'pos': [x, y, z], 'last_seen': time } }
        self.global_tracks = {}
        self.next_global_id = 0
        # merge_dist_thresh is now a param (match_max_dist)

    def load_calibration(self, path):
        """ YAML 파일에서 Intrinsic/Extrinsic 로드 """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        cams = {}
        # cam0 (Front) 기준 좌표계로 통일하기 위한 누적 변환 행렬 계산
        # 체인: cam0 <--- cam1 <--- cam2 <--- cam3 (yaml 구조에 따라 다름, 확인 필요)
        # 일단 단순하게 각 카메라의 T_cn_cnm1을 읽어서 저장
        
        T_global = np.eye(4) # cam0 기준
        
        # cam0 (Front)
        cam0_intr = data['cam0']['intrinsics']
        cam0_model = data['cam0'].get('camera_model', 'pinhole')
        cam0_xi, cam0_alpha, cam0_fx, cam0_fy, cam0_cx, cam0_cy = self._parse_intrinsics(cam0_intr, cam0_model)
        K0_pinhole = np.array([
            [cam0_fx, 0, cam0_cx],
            [0, cam0_fy, cam0_cy],
            [0, 0, 1]
        ])
        H0 = self.compute_ground_homography(K0_pinhole, np.eye(4), self.ground_height)
        cams['cam0'] = {
            'K': np.array(cam0_intr).reshape(-1), # [xi, alpha, fx, fy, cx, cy] for ds model?
            # YOLO용 단순 Pinhole K (fx, fy, cx, cy) 근사 추출
            'K_pinhole': K0_pinhole,
            'T': np.eye(4), # 자기 자신이 기준
            'model': cam0_model,
            'xi': cam0_xi,
            'alpha': cam0_alpha,
            'fx': cam0_fx,
            'fy': cam0_fy,
            'cx': cam0_cx,
            'cy': cam0_cy,
            'H': H0,
            'H_inv': np.linalg.inv(H0)
        }

        # 나머지 카메라 (Extrinsics 반영)
        # 주의: camchain.yaml의 T_cn_cnm1은 "현재 카메라 -> 이전 카메라" 변환임.
        # cam1 -> cam0
        T_1_0 = np.array(data['cam1']['T_cn_cnm1'])
        cams['cam1'] = self.parse_cam_data(data['cam1'], T_1_0) # Rear

        # cam2 -> cam1 -> cam0
        T_2_1 = np.array(data['cam2']['T_cn_cnm1'])
        T_2_0 = np.dot(T_1_0, T_2_1)
        cams['cam2'] = self.parse_cam_data(data['cam2'], T_2_0) # Left

        # cam3 -> cam2 -> cam1 -> cam0
        T_3_2 = np.array(data['cam3']['T_cn_cnm1'])
        T_3_0 = np.dot(T_2_0, T_3_2)
        cams['cam3'] = self.parse_cam_data(data['cam3'], T_3_0) # Right

        return cams

    def _parse_intrinsics(self, intrinsics, model):
        if model == 'ds' and len(intrinsics) >= 6:
            xi, alpha, fx, fy, cx, cy = intrinsics[:6]
            return xi, alpha, fx, fy, cx, cy
        if len(intrinsics) >= 4:
            fx, fy, cx, cy = intrinsics[:4]
            return 0.0, 0.5, fx, fy, cx, cy
        raise ValueError("Invalid intrinsics format")

    def parse_cam_data(self, cam_data, T_global):
        intr = cam_data['intrinsics']
        model = cam_data.get('camera_model', 'pinhole')
        xi, alpha, fx, fy, cx, cy = self._parse_intrinsics(intr, model)
        K_pinhole = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        H = self.compute_ground_homography(K_pinhole, T_global, self.ground_height)
        H_inv = np.linalg.inv(H)
        return {
            'K_pinhole': K_pinhole,
            'T': T_global,
            'model': model,
            'xi': xi,
            'alpha': alpha,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'H': H,
            'H_inv': H_inv
        }

    def compute_ground_homography(self, K, T_cam_to_world, ground_height):
        R = T_cam_to_world[:3, :3]
        t = T_cam_to_world[:3, 3]

        # World -> Camera
        R_cw = R.T
        t_cw = -R_cw @ t

        r1 = R_cw[:, 0]  # X axis
        r3 = R_cw[:, 2]  # Z axis
        t_plane = R_cw[:, 1] * ground_height + t_cw

        H = K @ np.column_stack((r1, r3, t_plane))
        return H

    def img_cb(self, msg, cam_id):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.current_images[cam_id] = img
        except Exception as e:
            rospy.logerr(e)

    def _init_track(self, pos, now):
        state = np.array([pos[0], pos[2], 0.0, 0.0], dtype=float)
        track = {
            'state': state,
            'P': np.eye(4, dtype=float) * self.kalman_init_var,
            'last_update': now,
            'last_seen': now
        }
        track['pos'] = np.array([state[0], self.ground_height, state[1]])
        return track

    def _predict_track(self, track, now):
        dt = (now - track['last_update']).to_sec()
        if dt <= 0.0:
            return
        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=float)
        Q = np.diag([self.kalman_q_pos, self.kalman_q_pos,
                     self.kalman_q_vel, self.kalman_q_vel]) * max(dt, 1e-3)
        track['state'] = F @ track['state']
        track['P'] = F @ track['P'] @ F.T + Q
        track['last_update'] = now
        track['pos'] = np.array([track['state'][0], self.ground_height, track['state'][1]])

    def _update_track(self, track, meas_pos):
        z = np.array([meas_pos[0], meas_pos[2]], dtype=float)
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=float)
        R = np.diag([self.kalman_r, self.kalman_r])
        y = z - (H @ track['state'])
        S = H @ track['P'] @ H.T + R
        K = track['P'] @ H.T @ np.linalg.inv(S)
        track['state'] = track['state'] + K @ y
        I = np.eye(4, dtype=float)
        track['P'] = (I - K @ H) @ track['P']
        track['pos'] = np.array([track['state'][0], self.ground_height, track['state'][1]])

    def _purge_tracks(self, now):
        rem_ids = []
        for gid, track in self.global_tracks.items():
            if (now - track['last_seen']).to_sec() > self.track_ttl:
                rem_ids.append(gid)
        for gid in rem_ids:
            del self.global_tracks[gid]

    def pixel_to_ground(self, u, v, cam_id):
        H_inv = self.cameras[cam_id]['H_inv']
        uv_homo = np.array([u, v, 1.0])
        gp = H_inv @ uv_homo
        if abs(gp[2]) < 1e-6:
            return None
        return np.array([gp[0] / gp[2], gp[1] / gp[2]])

    def pixel_to_ray_ds(self, u, v, cam):
        fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']
        xi, alpha = cam['xi'], cam['alpha']
        mx = (u - cx) / fx
        my = (v - cy) / fy
        r2 = mx * mx + my * my

        denom_sqrt = 1 - (2 * alpha - 1) * r2
        if denom_sqrt < 0:
            return None
        sqrt_term = np.sqrt(denom_sqrt)
        denom = alpha * sqrt_term + (1 - alpha)
        if abs(denom) < 1e-6:
            return None
        mz = (1 - alpha * alpha * r2) / denom

        tmp = mz * mz + (1 - xi * xi) * r2
        if tmp < 0:
            return None
        k = (mz * xi + np.sqrt(tmp)) / (mz * mz + r2)
        ray = np.array([k * mx, k * my, k * mz - xi], dtype=float)
        norm = np.linalg.norm(ray)
        if norm < 1e-6:
            return None
        return ray / norm

    def pixel_to_world(self, u, v, cam_id):
        cam = self.cameras[cam_id]
        if cam.get('model') == 'ds':
            ray_cam = self.pixel_to_ray_ds(u, v, cam)
        else:
            K = cam['K_pinhole']
            uv_homo = np.array([u, v, 1.0])
            K_inv = np.linalg.inv(K)
            ray_cam = K_inv @ uv_homo
            norm = np.linalg.norm(ray_cam)
            if norm < 1e-6:
                ray_cam = None
            else:
                ray_cam = ray_cam / norm

        if ray_cam is None:
            return None

        T = cam['T']
        R = T[:3, :3]
        t = T[:3, 3]

        ray_world = R @ ray_cam
        origin_world = t

        if abs(ray_world[1]) < 1e-3:
            return None
        lam = (self.ground_height - origin_world[1]) / ray_world[1]
        if lam < 0:
            return None
        intersect = origin_world + lam * ray_world
        return intersect

    def broadcast_camera_tfs(self):
        """ 카메라 TF 송출 (cam0 기준) """
        timestamp = rospy.Time.now()
        frame_map = {'cam1': 'rear', 'cam2': 'left', 'cam3': 'right'}
        
        for cam_id, cam_data in self.cameras.items():
            if cam_id == 'cam0':
                continue
            
            # T is transform from cam to cam0 (T_global)
            T = cam_data['T']
            t = translation_from_matrix(T)
            q = quaternion_from_matrix(T)
            
            ts = TransformStamped()
            ts.header.stamp = timestamp
            ts.header.frame_id = "camera_link_front"
            suffix = frame_map.get(cam_id, cam_id)
            ts.child_frame_id = f"camera_link_{suffix}"
            
            ts.transform.translation.x = t[0]
            ts.transform.translation.y = t[1]
            ts.transform.translation.z = t[2]
            ts.transform.rotation.x = q[0]
            ts.transform.rotation.y = q[1]
            ts.transform.rotation.z = q[2]
            ts.transform.rotation.w = q[3]
            
            self.tf_broadcaster.sendTransform(ts)

    def filter_valid_pairs(self, detections):
        """ 
        문 안에 손잡이가 있는 경우만 남기는 필터링
        detections: list of dict
        """
        # 카메라별 그룹화
        cam_groups = {}
        for d in detections:
            c = d['cam']
            if c not in cam_groups: cam_groups[c] = []
            cam_groups[c].append(d)
            
        valid_dets = []
        
        for cid, group in cam_groups.items():
            doors = [d for d in group if d['cls'] in self.door_class_ids]
            handles = [d for d in group if d['cls'] == self.handle_class_id]
            
            # 이번 프레임에서 유효한 것으로 판명된 객체들
            valid_door_ids = set() # using object id in list logic
            valid_handle_ids = set()

            for i, door in enumerate(doors):
                dx1, dy1, dx2, dy2 = door['box']
                has_handle = False
                for j, h in enumerate(handles):
                    hx1, hy1, hx2, hy2 = h['box']
                    hcx, hcy = (hx1+hx2)/2, (hy1+hy2)/2
                    
                    # 문 박스 안에 손잡이 중심이 있는지 확인
                    if dx1 <= hcx <= dx2 and dy1 <= hcy <= dy2:
                        has_handle = True
                        valid_handle_ids.add(j)
                
                if has_handle:
                    valid_door_ids.add(i)
            
            # 다시 리스트로 복원
            for i, door in enumerate(doors):
                if i in valid_door_ids:
                    valid_dets.append(door)
            for j, h in enumerate(handles):
                if j in valid_handle_ids:
                    valid_dets.append(h)

        return valid_dets

    def publish_markers(self, valid_gids=None):
        arr = MarkerArray()
        timestamp = rospy.Time.now()
        
        for gid, track in self.global_tracks.items():
            if track['pos'] is None:
                continue
            
            # 필터링: valid_gids가 제공되었다면, 그 안에 있는 ID만 표시
            if valid_gids is not None and gid not in valid_gids:
                continue

            # 1. Marker 추가
            m = Marker()
            m.header.frame_id = "camera_link_front"
            m.header.stamp = timestamp
            m.ns = "doors"
            m.id = gid
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = track['pos'][0]
            m.pose.position.y = track['pos'][1]
            m.pose.position.z = track['pos'][2]
            m.scale.x = 0.5; m.scale.y = 2.0; m.scale.z = 0.1
            m.color.a = 0.8; m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0
            m.lifetime = rospy.Duration(0.5)
            
            t = Marker()
            t.header = m.header
            t.ns = "ids"
            t.id = gid
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = track['pos'][0]
            t.pose.position.y = track['pos'][1] - 1.2
            t.pose.position.z = track['pos'][2]
            t.scale.z = 0.5
            t.color.a = 1.0; t.color.r = 1.0; t.color.g = 1.0; t.color.b = 1.0
            t.text = f"Door {gid}"
            
            arr.markers.append(m)
            arr.markers.append(t)

            # 2. TF Broadcasting (Object)
            ts = TransformStamped()
            ts.header.stamp = timestamp
            ts.header.frame_id = "camera_link_front"
            ts.child_frame_id = f"door_{gid}"
            ts.transform.translation.x = track['pos'][0]
            ts.transform.translation.y = track['pos'][1]
            ts.transform.translation.z = track['pos'][2]
            ts.transform.rotation.w = 1.0 # No rotation info yet
            self.tf_broadcaster.sendTransform(ts)
        
        self.marker_pub.publish(arr)

    def hungarian(self, cost):
        cost = np.asarray(cost, dtype=float)
        if cost.size == 0:
            return []
        n, m = cost.shape
        transpose = False
        if n > m:
            cost = cost.T
            n, m = cost.shape
            transpose = True

        u = np.zeros(n + 1)
        v = np.zeros(m + 1)
        p = np.zeros(m + 1, dtype=int)
        way = np.zeros(m + 1, dtype=int)

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = np.full(m + 1, np.inf)
            used = np.zeros(m + 1, dtype=bool)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = np.inf
                j1 = 0
                for j in range(1, m + 1):
                    if used[j]:
                        continue
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
                for j in range(m + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        assignment = [-1] * n
        for j in range(1, m + 1):
            if p[j] != 0:
                assignment[p[j] - 1] = j - 1

        pairs = []
        for i, j in enumerate(assignment):
            if j >= 0:
                if transpose:
                    pairs.append((j, i))
                else:
                    pairs.append((i, j))
        return pairs

    def cluster_detections(self, detections, valid_det_indices):
        clusters = []
        for det_idx in valid_det_indices:
            det = detections[det_idx]
            pos = det['pos']
            if pos is None:
                continue
            assigned = False
            for cl in clusters:
                dist = np.linalg.norm(pos[[0, 2]] - cl['pos'][[0, 2]])
                if dist < self.cluster_dist_thresh:
                    cl['indices'].append(det_idx)
                    n = len(cl['indices'])
                    cl['pos'] = (cl['pos'] * (n - 1) + pos) / n
                    assigned = True
                    break
            if not assigned:
                clusters.append({'indices': [det_idx], 'pos': pos.copy()})
        return clusters

    def merge_detections(self, detections):
        """
        여러 카메라에서 온 Detection들을 하나로 합침 (거리 기반)
        반환값: 각 detection에 'gid' (Global ID)가 추가된 리스트
        """
        # detections: list of {'cam': 'cam0', 'pos': [x,y,z], 'cls': 1, 'box': [x1,y1,x2,y2]}
        now = rospy.Time.now()

        if self.use_kalman:
            for track in self.global_tracks.values():
                self._predict_track(track, now)

        for det in detections:
            det['gid'] = -1

        if not detections:
            self._purge_tracks(now)
            return detections

        valid_det_indices = [i for i, det in enumerate(detections) if det.get('pos') is not None]
        track_ids = list(self.global_tracks.keys())

        if not valid_det_indices and track_ids:
            self._purge_tracks(now)
            return detections

        clusters = self.cluster_detections(detections, valid_det_indices)
        if not clusters:
            self._purge_tracks(now)
            return detections

        if not track_ids:
            for cl in clusters:
                new_gid = self.next_global_id
                if self.use_kalman:
                    self.global_tracks[new_gid] = self._init_track(cl['pos'], now)
                else:
                    self.global_tracks[new_gid] = {
                        'pos': cl['pos'],
                        'last_seen': now
                    }
                self.next_global_id += 1
                for det_idx in cl['indices']:
                    detections[det_idx]['gid'] = new_gid
            return detections

        cost = np.full((len(clusters), len(track_ids)), 1e6, dtype=float)
        for r, cl in enumerate(clusters):
            cl_pos = cl['pos']
            for c, gid in enumerate(track_ids):
                track_pos = self.global_tracks[gid]['pos']
                if track_pos is None:
                    continue
                dist = np.linalg.norm(cl_pos[[0, 2]] - track_pos[[0, 2]])
                cost[r, c] = dist

        pairs = self.hungarian(cost)
        matched_tracks = set()

        for r, c in pairs:
            cl = clusters[r]
            gid = track_ids[c]
            dist = cost[r, c]
            if dist > self.merge_dist_thresh:
                continue
            if self.use_kalman:
                self._update_track(self.global_tracks[gid], cl['pos'])
                self.global_tracks[gid]['last_seen'] = now
            else:
                alpha = 0.7
                self.global_tracks[gid]['pos'] = alpha * self.global_tracks[gid]['pos'] + (1 - alpha) * cl['pos']
                self.global_tracks[gid]['last_seen'] = now
            for det_idx in cl['indices']:
                detections[det_idx]['gid'] = gid
            matched_tracks.add(gid)

        for cl in clusters:
            if any(detections[det_idx]['gid'] != -1 for det_idx in cl['indices']):
                continue
            new_gid = self.next_global_id
            if self.use_kalman:
                self.global_tracks[new_gid] = self._init_track(cl['pos'], now)
            else:
                self.global_tracks[new_gid] = {
                    'pos': cl['pos'],
                    'last_seen': now
                }
            self.next_global_id += 1
            for det_idx in cl['indices']:
                detections[det_idx]['gid'] = new_gid
        
        self._purge_tracks(now)
            
        return detections

    def _gid_color(self, gid):
        if gid < 0:
            return (255, 255, 255)
        r = (37 * gid + 89) % 255
        g = (17 * gid + 203) % 255
        b = (53 * gid + 47) % 255
        return (int(b), int(g), int(r))

    def show_visualization(self, imgs, cam_ids, merged_results):
        """ 4분할 화면에 마스크/박스와 ID 그리기 """
        vis_imgs = {}
        # 이미지 복사 및 리사이즈 준비
        for i, img in enumerate(imgs):
            vis_imgs[cam_ids[i]] = img.copy()
        
        # 마스크/박스 그리기
        for res in merged_results:
            cid = res['cam']
            if cid not in vis_imgs: continue
            
            x1, y1, x2, y2 = map(int, res['box'])
            gid = res['gid']
            cls = res['cls']

            mask = res.get('mask')
            if mask is not None:
                color = self.class_colors.get(cls, (0, 255, 0))
                overlay = np.zeros_like(vis_imgs[cid], dtype=np.uint8)
                overlay[mask > 0.5] = color
                vis_imgs[cid] = cv2.addWeighted(vis_imgs[cid], 1.0, overlay, 0.4, 0)

            box_color = self._gid_color(gid)
            cv2.rectangle(vis_imgs[cid], (x1, y1), (x2, y2), box_color, 2)
            class_name = self.class_names.get(cls, cls)
            label = f"{class_name} ID:{gid}" if gid >= 0 else f"{class_name} ID:?"
            cv2.putText(vis_imgs[cid], label, (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        # 4분할 합치기 - fisheye_detection.py 참고
        # 순서: Front, Left, Rear, Right
        img_list = [
            vis_imgs.get('cam0', np.zeros_like(imgs[0])),
            vis_imgs.get('cam2', np.zeros_like(imgs[0])),
            vis_imgs.get('cam1', np.zeros_like(imgs[0])),
            vis_imgs.get('cam3', np.zeros_like(imgs[0]))
        ]
        
        if not all(i is not None for i in img_list): return
        h, w, _ = img_list[0].shape
        resized = [cv2.resize(img, (w, h)) for img in img_list]
        
        top = np.hstack((resized[0], resized[1]))
        bottom = np.hstack((resized[2], resized[3]))
        final_grid = np.vstack((top, bottom))
        
        cv2.imshow("SnapSpace Multi-Cam Tracker", final_grid)
        cv2.waitKey(1)

    def spin(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            # 0. 카메라 TF 송출
            self.broadcast_camera_tfs()

            # 1. 이미지 수집 확인
            imgs = []
            valid_cams = []
            with self.lock:
                # 순서 보장
                cam_order = ['cam0', 'cam1', 'cam2', 'cam3']
                for cid in cam_order:
                    if self.current_images[cid] is not None:
                        imgs.append(self.current_images[cid])
                        valid_cams.append(cid)
            
            if len(imgs) < 4: # 4개 다 들어올 때까지 기다림
                rate.sleep()
                continue

            # 2. 배치 추론 (YOLO)
            results = self.model(imgs, verbose=False)

            # 3. 결과 파싱 및 3D 변환
            detections_3d = [] 
            
            for i, res in enumerate(results):
                cam_id = valid_cams[i]
                boxes = res.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                masks = None
                if res.masks is not None and res.masks.data is not None:
                    masks = res.masks.data.cpu().numpy()

                for idx in range(len(boxes)):
                    cls = int(boxes.cls[idx])
                    if cls not in self.track_class_ids:
                        continue

                    conf = float(boxes.conf[idx])
                    if conf < self.conf_thresh:
                        continue

                    x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy()
                    u = (x1 + x2) / 2
                    v = y2

                    pos_3d = self.pixel_to_world(u, v, cam_id)
                    det = {
                        'cam': cam_id,
                        'pos': pos_3d,
                        'cls': cls,
                        'box': [x1, y1, x2, y2]
                    }
                    if masks is not None and idx < len(masks):
                        det['mask'] = masks[idx]
                    detections_3d.append(det)

            # 4. ID 통합 (Tracking) - 모든 Detection에 대해 Tracking 수행
            merged_results = self.merge_detections(detections_3d)

            # 5. 시각화 필터링 (문+손잡이 쌍만 남김)
            vis_results = self.filter_valid_pairs(merged_results)

            # 6. 시각화 (Video Overlay)
            self.show_visualization(imgs, valid_cams, vis_results)

            # 7. Rviz 마커 및 TF 발행
            # vis_results에 있는 gid만 유효한 것으로 간주
            valid_gids = set(d['gid'] for d in vis_results if d['gid'] != -1)
            self.publish_markers(valid_gids)

            rate.sleep()


if __name__ == "__main__":
    try:
        tracker = SnapSpaceTracker()
        tracker.spin()
    except rospy.ROSInterruptException:
        pass
