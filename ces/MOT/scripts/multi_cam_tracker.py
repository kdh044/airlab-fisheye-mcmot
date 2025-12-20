#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import rospkg
import cv2
import numpy as np
import yaml
import threading
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray

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
        
        # Kalman Filter Params
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

        # 클래스 정보
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

        # 캘리브레이션 로드
        calib_path = self.pkg_path + "/MOT/calibration file/2024-09-13-20-37-00-camchain.yaml"
        self.cameras = self.load_calibration(calib_path)
        rospy.loginfo("Camera calibration loaded.")

        # ---------------------------------------------------------
        # 2. ROS 통신 설정
        # ---------------------------------------------------------
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        
        # 최신 이미지 저장소 (Undistorted images will be stored here)
        self.current_images = {
            'cam0': None, 'cam1': None, 'cam2': None, 'cam3': None
        }

        # 토픽 구독
        rospy.Subscriber("/camera/image_raw_front", Image, lambda m: self.img_cb(m, 'cam0'))
        rospy.Subscriber("/camera/image_raw_rear",  Image, lambda m: self.img_cb(m, 'cam1'))
        rospy.Subscriber("/camera/image_raw_left",  Image, lambda m: self.img_cb(m, 'cam2'))
        rospy.Subscriber("/camera/image_raw_right", Image, lambda m: self.img_cb(m, 'cam3'))

        # 결과 발행
        self.marker_pub = rospy.Publisher("/snapspace/markers", MarkerArray, queue_size=10)

        # ---------------------------------------------------------
        # 3. 트래킹 상태 변수
        # ---------------------------------------------------------
        self.global_tracks = {}
        self.next_global_id = 0

    def load_calibration(self, path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        cams = {}
        T_global = np.eye(4)
        
        # cam0 (Front)
        cams['cam0'] = self.parse_cam_data(data['cam0'], np.eye(4))

        # cam1 -> cam0
        T_1_0 = np.array(data['cam1']['T_cn_cnm1'])
        cams['cam1'] = self.parse_cam_data(data['cam1'], T_1_0)

        # cam2 -> cam1 -> cam0
        T_2_1 = np.array(data['cam2']['T_cn_cnm1'])
        T_2_0 = np.dot(T_1_0, T_2_1)
        cams['cam2'] = self.parse_cam_data(data['cam2'], T_2_0)

        # cam3 -> cam2 -> cam1 -> cam0
        T_3_2 = np.array(data['cam3']['T_cn_cnm1'])
        T_3_0 = np.dot(T_2_0, T_3_2)
        cams['cam3'] = self.parse_cam_data(data['cam3'], T_3_0)

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
        
        return {
            'T': T_global,
            'model': model,
            'xi': xi, 'alpha': alpha,
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'map_x': None, 'map_y': None, # Maps for undistortion
            'K_new': None, # New Camera Matrix for undistorted image
            'H_new_inv': None # Homography inverse for ground projection
        }

    def init_undistort_maps(self, cam_id, w, h):
        cam = self.cameras[cam_id]
        if cam['map_x'] is not None:
            return

        rospy.loginfo(f"Initializing undistort maps for {cam_id} ({w}x{h})...")
        
        # 1. Define new Camera Matrix (K_new) for the rectified (pinhole) view
        # Scale focal length to zoom in/out (0.5-0.6 is good for fisheye)
        scale = 0.5 
        nfx, nfy = cam['fx'] * scale, cam['fy'] * scale
        ncx, ncy = w / 2.0, h / 2.0
        
        K_new = np.array([[nfx, 0, ncx], [0, nfy, ncy], [0, 0, 1]])
        K_new_inv = np.linalg.inv(K_new)
        cam['K_new'] = K_new
        
        # 2. Compute Homography Inverse for Ground Projection (using K_new)
        # This replaces the complex 'pixel_to_ray_ds' logic for projection
        cam['H_new_inv'] = self.compute_ground_homography_inv(K_new, cam['T'], self.ground_height)

        # 3. Generate Undistortion Maps (Inverse Mapping)
        # Target: Rectified Image (u_new, v_new) -> Ray -> Source: Distorted Image (u_raw, v_raw)
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        
        # Back-project pixels to rays using K_new
        uy = (grid_x - ncx) / nfx
        vy = (grid_y - ncy) / nfy
        r2 = uy**2 + vy**2
        norm = np.sqrt(r2 + 1)
        xs, ys, zs = uy/norm, vy/norm, 1.0/norm # Unit rays in camera frame
        
        # Project rays to distorted pixel coordinates (Double Sphere Model)
        xi, alpha = cam['xi'], cam['alpha']
        
        # World(Sphere) -> Camera projection
        # Ray (xs, ys, zs) -> move to sphere center offset by xi
        zs_prime = zs + xi
        d2 = np.sqrt(xs**2 + ys**2 + zs_prime**2)
        denom = alpha * d2 + (1 - alpha) * zs_prime
        
        u_ds = xs / denom
        v_ds = ys / denom
        
        # To Pixel
        map_x = (cam['fx'] * u_ds + cam['cx']).astype(np.float32)
        map_y = (cam['fy'] * v_ds + cam['cy']).astype(np.float32)
        
        cam['map_x'] = map_x
        cam['map_y'] = map_y
        rospy.loginfo(f"Maps initialized for {cam_id}.")

    def compute_ground_homography_inv(self, K, T_cam_to_world, ground_height):
        # Compute H that maps Ground(X, Z) to Pixel(u, v)
        # We need H_inv to map Pixel(u, v) -> Ground(X, Z)
        
        R = T_cam_to_world[:3, :3]
        t = T_cam_to_world[:3, 3]

        # World -> Camera
        R_cw = R.T
        t_cw = -R_cw @ t

        r1 = R_cw[:, 0]  # X axis
        r3 = R_cw[:, 2]  # Z axis
        t_plane = R_cw[:, 1] * ground_height + t_cw

        H = K @ np.column_stack((r1, r3, t_plane))
        return np.linalg.inv(H)

    def img_cb(self, msg, cam_id):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h, w = img.shape[:2]
            
            # Lazy initialization of maps
            if self.cameras[cam_id]['map_x'] is None:
                self.init_undistort_maps(cam_id, w, h)
                
            # UNDISTORT IMAGE immediately
            cam = self.cameras[cam_id]
            img_u = cv2.remap(img, cam['map_x'], cam['map_y'], cv2.INTER_LINEAR)
            
            with self.lock:
                self.current_images[cam_id] = img_u
        except Exception as e:
            pass # Silently ignore errors to prevent spam

    # ------------------------------------------------------------------
    # Tracking Logic (Same as before, but updated to use K_new projection)
    # ------------------------------------------------------------------
    def _init_track(self, pos, now):
        state = np.array([pos[0], pos[2], 0.0, 0.0], dtype=float)
        track = {'state': state, 'P': np.eye(4)*self.kalman_init_var, 'last_update': now, 'last_seen': now}
        track['pos'] = np.array([state[0], self.ground_height, state[1]])
        return track

    def _predict_track(self, track, now):
        dt = (now - track['last_update']).to_sec()
        if dt <= 0.0: return
        F = np.eye(4); F[0,2]=dt; F[1,3]=dt
        Q = np.diag([self.kalman_q_pos]*2 + [self.kalman_q_vel]*2) * max(dt, 1e-3)
        track['state'] = F @ track['state']
        track['P'] = F @ track['P'] @ F.T + Q
        track['last_update'] = now
        track['pos'] = np.array([track['state'][0], self.ground_height, track['state'][1]])

    def _update_track(self, track, meas_pos):
        z = np.array([meas_pos[0], meas_pos[2]])
        H = np.zeros((2,4)); H[0,0]=1; H[1,1]=1
        R = np.eye(2)*self.kalman_r
        y = z - H @ track['state']
        S = H @ track['P'] @ H.T + R
        K = track['P'] @ H.T @ np.linalg.inv(S)
        track['state'] += K @ y
        track['P'] = (np.eye(4) - K @ H) @ track['P']
        track['pos'] = np.array([track['state'][0], self.ground_height, track['state'][1]])

    def _purge_tracks(self, now):
        rem = [gid for gid, t in self.global_tracks.items() if (now - t['last_seen']).to_sec() > self.track_ttl]
        for gid in rem: del self.global_tracks[gid]

    def pixel_to_world(self, u, v, cam_id):
        # Simplified projection: Since image is undistorted, use standard homography
        H_inv = self.cameras[cam_id]['H_new_inv']
        if H_inv is None: return None
        
        uv_homo = np.array([u, v, 1.0])
        gp = H_inv @ uv_homo
        if abs(gp[2]) < 1e-6: return None
        return np.array([gp[0]/gp[2], self.ground_height, gp[1]/gp[2]])

    def hungarian(self, cost):
        # Simple implementation
        try:
            from scipy.optimize import linear_sum_assignment
            r, c = linear_sum_assignment(cost)
            return list(zip(r, c))
        except ImportError:
            pairs = []
            used_c = set()
            for r in range(cost.shape[0]):
                c = np.argmin(cost[r])
                if c not in used_c:
                    pairs.append((r, c))
                    used_c.add(c)
            return pairs

    def cluster_detections(self, detections, valid_indices):
        clusters = []
        for idx in valid_indices:
            pos = detections[idx]['pos']
            assigned = False
            for cl in clusters:
                if np.linalg.norm(pos[[0,2]] - cl['pos'][[0,2]]) < self.cluster_dist_thresh:
                    cl['indices'].append(idx)
                    cl['pos'] = (cl['pos']*(len(cl['indices'])-1) + pos)/len(cl['indices'])
                    assigned = True; break
            if not assigned: clusters.append({'indices': [idx], 'pos': pos.copy()})
        return clusters

    def merge_detections(self, detections):
        now = rospy.Time.now()
        if self.use_kalman:
            for t in self.global_tracks.values(): self._predict_track(t, now)

        for det in detections: det['gid'] = -1
        valid_idxs = [i for i, d in enumerate(detections) if d['pos'] is not None]
        
        if not valid_idxs:
            self._purge_tracks(now)
            return detections
            
        clusters = self.cluster_detections(detections, valid_idxs)
        track_ids = list(self.global_tracks.keys())
        
        if clusters and track_ids:
            cost = np.zeros((len(clusters), len(track_ids)))
            for i, cl in enumerate(clusters):
                for j, gid in enumerate(track_ids):
                    cost[i,j] = np.linalg.norm(cl['pos'][[0,2]] - self.global_tracks[gid]['pos'][[0,2]])
            
            pairs = self.hungarian(cost)
            for r, c in pairs:
                if cost[r, c] > self.merge_dist_thresh: continue
                gid = track_ids[c]
                cl = clusters[r]
                if self.use_kalman:
                    self._update_track(self.global_tracks[gid], cl['pos'])
                    self.global_tracks[gid]['last_seen'] = now
                else:
                    self.global_tracks[gid]['pos'] = 0.7*self.global_tracks[gid]['pos'] + 0.3*cl['pos']
                    self.global_tracks[gid]['last_seen'] = now
                for idx in cl['indices']: detections[idx]['gid'] = gid
        
        for cl in clusters:
            if any(detections[idx]['gid'] != -1 for idx in cl['indices']): continue
            gid = self.next_global_id; self.next_global_id += 1
            if self.use_kalman: self.global_tracks[gid] = self._init_track(cl['pos'], now)
            else: self.global_tracks[gid] = {'pos': cl['pos'], 'last_seen': now}
            for idx in cl['indices']: detections[idx]['gid'] = gid
            
        self._purge_tracks(now)
        return detections

    def publish_markers(self):
        arr = MarkerArray()
        for gid, track in self.global_tracks.items():
            if track['pos'] is None: continue
            m = Marker()
            m.header.frame_id = "camera_link_front"
            m.header.stamp = rospy.Time.now()
            m.ns, m.id, m.type, m.action = "doors", gid, Marker.CUBE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = track['pos']
            m.scale.x, m.scale.y, m.scale.z = 0.5, 2.0, 0.1
            m.color.a, m.color.r = 0.8, 1.0
            m.lifetime = rospy.Duration(0.5)
            
            t = Marker()
            t.header, t.ns, t.id, t.type, t.action = m.header, "ids", gid, Marker.TEXT_VIEW_FACING, Marker.ADD
            t.pose.position.x, t.pose.position.y, t.pose.position.z = track['pos']; t.pose.position.y -= 1.2
            t.scale.z, t.color.a, t.color.r, t.color.g, t.color.b, t.text = 0.5, 1.0, 1.0, 1.0, 1.0, f"Door {gid}"
            arr.markers.append(m); arr.markers.append(t)
        self.marker_pub.publish(arr)

    def _gid_color(self, gid):
        if gid < 0: return (255, 255, 255)
        return (int((53*gid+47)%255), int((17*gid+203)%255), int((37*gid+89)%255))

    def show_visualization(self, imgs, cam_ids, merged_results):
        vis_imgs = {cid: img.copy() for cid, img in zip(cam_ids, imgs)}
        
        for res in merged_results:
            cid = res['cam']
            if cid not in vis_imgs: continue
            x1, y1, x2, y2 = map(int, res['box'])
            gid, cls = res['gid'], res['cls']
            
            color = self.class_colors.get(cls, (0, 255, 0))
            if 'mask' in res and res['mask'] is not None:
                mask = res['mask'] > 0.5
                overlay = np.zeros_like(vis_imgs[cid], dtype=np.uint8)
                overlay[mask] = color
                vis_imgs[cid] = cv2.addWeighted(vis_imgs[cid], 1.0, overlay, 0.4, 0)
            
            box_col = self._gid_color(gid)
            cv2.rectangle(vis_imgs[cid], (x1, y1), (x2, y2), box_col, 2)
            label = f"{self.class_names.get(cls, cls)} {gid}" if gid >= 0 else self.class_names.get(cls, str(cls))
            cv2.putText(vis_imgs[cid], label, (x1, max(y1-10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_col, 2)

        # 4-Split View
        def get_img(k):
            # Ensure images are same size for stacking
            target_h, target_w = imgs[0].shape[:2]
            return cv2.resize(vis_imgs.get(k, np.zeros((target_h, target_w, 3), np.uint8)), (target_w, target_h))
            
        final = np.vstack((np.hstack((get_img('cam0'), get_img('cam2'))),
                           np.hstack((get_img('cam1'), get_img('cam3')))))
        cv2.imshow("SnapSpace Tracker (Undistorted)", final)
        cv2.waitKey(1)

    def spin(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            imgs, valid_cams = [], []
            with self.lock:
                for cid in ['cam0', 'cam1', 'cam2', 'cam3']:
                    if self.current_images[cid] is not None:
                        imgs.append(self.current_images[cid])
                        valid_cams.append(cid)
            
            if len(imgs) < 4:
                rate.sleep()
                continue
                
            results = self.model(imgs, verbose=False)
            detections = []
            
            for i, res in enumerate(results):
                cam_id = valid_cams[i]
                if res.boxes:
                    masks = res.masks.data.cpu().numpy() if res.masks else None
                    for j, box in enumerate(res.boxes):
                        cls = int(box.cls)
                        if cls not in self.track_class_ids or float(box.conf) < self.conf_thresh: continue
                        
                        x1,y1,x2,y2 = box.xyxy.cpu().numpy()[0]
                        pos = self.pixel_to_world((x1+x2)/2, y2, cam_id)
                        
                        det = {'cam': cam_id, 'pos': pos, 'cls': cls, 'box': [x1,y1,x2,y2]}
                        if masks is not None: det['mask'] = masks[j]
                        detections.append(det)
            
            merged = self.merge_detections(detections)
            self.show_visualization(imgs, valid_cams, merged)
            self.publish_markers()
            rate.sleep()

if __name__ == "__main__":
    try: SnapSpaceTracker().spin()
    except rospy.ROSInterruptException: pass