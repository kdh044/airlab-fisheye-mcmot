#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter/union if union > 0 else 0.0

def non_max_suppression(boxes, scores, iou_threshold):
    if len(boxes) == 0: return []
    indices = np.argsort(scores)[::-1]
    keep = []
    while len(indices) > 0:
        i = indices[0]; keep.append(i)
        ious = np.array([compute_iou(boxes[i], boxes[j]) for j in indices[1:]])
        remaining_indices = np.where(ious < iou_threshold)[0]
        indices = indices[remaining_indices + 1]
    return keep

def skew(t):
    tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
    return np.array([[0, -tz,  ty], [tz,  0, -tx], [-ty, tx,  0]], dtype=float)

class Camera:
    def __init__(self, cid, data, T_global, ground_height):
        self.cid = cid; self.T = T_global; self.ground_height = ground_height
        intr = data['intrinsics']
        self.model = data.get('camera_model', 'ds')
        if self.model == 'ds' and len(intr) >= 6: self.xi, self.alpha, self.fx, self.fy, self.cx, self.cy = intr[:6]
        else: raise ValueError(f"Invalid intrinsics for {cid}")
        self.map_x, self.map_y, self.K_new, self.H_inv = None, None, None, None
        self._lock = threading.Lock()
        raw_overlaps = data.get('cam_overlaps', [])
        self.overlaps = [f"cam{i}" if isinstance(i, int) else str(i) for i in raw_overlaps]

    def init_maps(self, w, h):
        with self._lock:
            if self.map_x is not None: return
            rospy.loginfo(f"[{self.cid}] Initializing maps ({w}x{h})...")
            scale = 0.5
            nfx, nfy = self.fx * scale, self.fy * scale
            ncx, ncy = w / 2.0, h / 2.0
            self.K_new = np.array([[nfx, 0, ncx], [0, nfy, ncy], [0, 0, 1]])
            R,t = self.T[:3, :3], self.T[:3, 3]
            R_cw, t_cw = R.T, -R.T @ t
            r1, r3 = R_cw[:, 0], R_cw[:, 2]
            t_plane = R_cw[:, 1] * self.ground_height + t_cw
            H = self.K_new @ np.column_stack((r1, r3, t_plane))
            self.H_inv = np.linalg.inv(H)
            grid_y, grid_x = np.mgrid[0:h, 0:w]
            uy, vy = (grid_x-ncx)/nfx, (grid_y-ncy)/nfy
            r2 = uy**2 + vy**2; norm = np.sqrt(r2 + 1)
            xs, ys, zs = uy/norm, vy/norm, 1.0/norm
            zs_prime = zs + self.xi
            d2 = np.sqrt(xs**2 + ys**2 + zs_prime**2)
            denom = self.alpha*d2 + (1-self.alpha)*zs_prime
            u_ds, v_ds = xs/denom, ys/denom
            self.map_x = (self.fx*u_ds + self.cx).astype(np.float32)
            self.map_y = (self.fy*v_ds + self.cy).astype(np.float32)

    def undistort(self, img):
        return cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR) if self.map_x is not None else img

    def pixel_to_world(self, u, v):
        if self.H_inv is None: return None
        gp = self.H_inv @ np.array([u, v, 1.0])
        return np.array([gp[0]/gp[2], self.ground_height, gp[1]/gp[2]]) if abs(gp[2]) > 1e-6 else None

class KalmanTracker:
    def __init__(self, cameras):
        self.tracks = {}; self.next_gid = 0
        self.max_dist = 1.5; self.ttl = 3.0; self.iou_gate = 0.1
        self.high_thresh = 0.25; self.n_init = 3
        self.q_diag = [0.1, 0.1, 0.2, 0.2]; self.r_val = 0.1
        self.F_cache = {}
        self.epipolar_thresh = 15.0
        self.cameras = cameras # Store cameras reference

    def compute_F(self, cam1, cam2, cameras):
        if (cam1, cam2) in self.F_cache: return self.F_cache[(cam1, cam2)]
        c1, c2 = cameras[cam1], cameras[cam2]
        if c1.K_new is None or c2.K_new is None: return None
        T_rel = np.linalg.inv(c2.T) @ c1.T
        R, t = T_rel[:3, :3], T_rel[:3, 3]
        E = skew(t) @ R
        K1_inv, K2_inv = np.linalg.inv(c1.K_new), np.linalg.inv(c2.K_new)
        F = K2_inv.T @ E @ K1_inv
        self.F_cache[(cam1, cam2)] = F; self.F_cache[(cam2, cam1)] = F.T
        return F

    def check_epipolar(self, track, det, cameras):
        cam1, cam2 = track['last_cam'], det['cam']
        if cam1 == cam2 or cam2 not in cameras[cam1].overlaps: return False
        F = self.compute_F(cam1, cam2, cameras)
        if F is None: return False
        uv1 = np.array([*track['last_uv'], 1.0])
        uv2 = np.array([*det['uv'], 1.0])
        l2 = F @ uv1
        dist = abs(np.dot(uv2, l2)) / np.sqrt(l2[0]**2 + l2[1]**2 + 1e-9)
        return dist < self.epipolar_thresh

    def update(self, detections, cameras):
        now = rospy.Time.now()
        for t in self.tracks.values(): self.predict(t, now)

        dets_high = [d for d in detections if d['conf'] >= self.high_thresh]
        dets_low = [d for d in detections if d['conf'] < self.high_thresh]
        for det in detections: det['gid'] = -1
        
        track_ids = list(self.tracks.keys())
        matched_gids = set()

        # 1. Match High Score
        if dets_high and track_ids:
            self._match(dets_high, track_ids, matched_gids, now, cameras)
        
        # 2. Match Low Score
        unmatched_gids = [gid for gid in track_ids if gid not in matched_gids]
        if dets_low and unmatched_gids:
            self._match(dets_low, unmatched_gids, matched_gids, now, cameras, is_low_score=True)
        
        # 3. New Tracks
        for det in dets_high:
            if det['gid'] == -1:
                gid = self.next_gid; self.next_gid += 1; det['gid'] = gid
                self.tracks[gid] = self._init_track(det, now)

        self._manage_states(now)
        return detections

    def _match(self, dets, gids, matched_gids, now, cameras, is_low_score=False):
        if not dets or not gids: return
        cost = np.full((len(dets), len(gids)), 1e6)
        for i, det in enumerate(dets):
            for j, gid in enumerate(gids):
                track = self.tracks[gid]
                # Handover Gating
                if track['last_cam'] != det['cam']:
                    if track['track_state'] != 'Lost' or not self.check_epipolar(track, det, cameras):
                        continue
                cost[i,j] = np.linalg.norm(det['pos'][[0,2]] - track['pos'][[0,2]])

        row, col = linear_sum_assignment(cost)
        for r, c in zip(row, col):
            if cost[r,c] > self.max_dist: continue
            det, gid = dets[r], gids[c]
            if det['cam'] == self.tracks[gid]['last_cam']:
                if compute_iou(det['box'], self.tracks[gid]['last_box']) < self.iou_gate: continue
            
            det['gid'] = gid; matched_gids.add(gid)
            self._update_track(self.tracks[gid], det, now)
            
    def _init_track(self, det, now):
        pos = det['pos']
        return {'pos':pos,'last_seen':now,'state':np.array([pos[0],pos[2],0.0,0.0]),
                'P':np.eye(4),'track_state':'New','hits':1,'last_cam':det['cam'],
                'last_uv':det['uv'],'last_box':det['box'],'display_cls':det['cls']}

    def predict(self, track, now):
        dt = (now - track['last_seen']).to_sec()
        if dt <= 0: return
        F = np.eye(4); F[0,2]=dt; F[1,3]=dt
        Q = np.diag(self.q_diag) * dt
        track['state'] = F @ track['state']
        track['P'] = F @ track['P'] @ F.T + Q
        track['pos'][0], track['pos'][2] = track['state'][0], track['state'][1]

    def _update_track(self, track, det, now):
        z = det['pos'][[0,2]]
        H = np.array([[1,0,0,0],[0,1,0,0]])
        R = np.eye(2) * self.r_val
        y = z - H @ track['state']
        S = H @ track['P'] @ H.T + R
        K = track['P'] @ H.T @ np.linalg.inv(S)
        track['state'] += K @ y
        track['P'] = (np.eye(4) - K @ H) @ track['P']
        track['pos'][0], track['pos'][2] = track['state'][0], track['state'][1]
        track['last_seen'] = now; track['last_cam'] = det['cam']
        track['last_uv'] = det['uv']; track['last_box'] = det['box']
        track['display_cls'] = det['cls']; track['hits'] += 1
        if track['track_state'] == 'New' and track['hits'] >= self.n_init: track['track_state'] = 'Tracked'
        elif track['track_state'] == 'Lost': track['track_state'] = 'Tracked'
    
    def _manage_states(self, now):
        rem = [gid for gid, t in self.tracks.items() if (now - t['last_seen']).to_sec() > self.ttl]
        for gid in rem: del self.tracks[gid]
        for t in self.tracks.values():
            if t['track_state'] == 'Tracked' and (now - t['last_seen']).to_sec() > 0.8:
                t['track_state'] = 'Lost'

class SnapSpaceNode:
    def __init__(self):
        rospy.init_node('snapspace_tracker', anonymous=False)
        r = rospkg.RosPack()
        calib_path = r.get_path('ces') + "/MOT/calibration file/2024-09-13-20-37-00-camchain.yaml"
        self.cameras = self._load_cameras(calib_path)
        self.tracker = KalmanTracker(self.cameras)
        self.model = YOLO(r.get_path('ces') + "/MOT/pretrained/3_27_witd_model.pt")
        self.bridge = CvBridge(); self.lock = threading.Lock()
        self.imgs = {k: None for k in ['cam0', 'cam1', 'cam2', 'cam3']}
        self.last_timestamps = {k: None for k in ['cam0', 'cam1', 'cam2', 'cam3']}
        
        self.class_names = {0:"b", 1:"Glass", 2:"Handle", 3:"Metal", 4:"Wood"}
        self.class_colors = {1:(0,255,255), 3:(0,255,0), 4:(255,0,255)}
        for c in ['front','rear','left','right']:
            cid = {'front':'cam0','rear':'cam1','left':'cam2','right':'cam3'}[c]
            rospy.Subscriber(f"/camera/image_raw_{c}", Image, self._img_cb, callback_args=cid)

    def _load_cameras(self, path):
        with open(path) as f: data = yaml.safe_load(f)
        ground_h = rospy.get_param("~ground_height", 1.0)
        cams = {}
        for i in range(4):
            cid = f'cam{i}'
            T = np.eye(4)
            if i > 0: # Simple chain T_0->1->2->3
                prev_T = cams[f'cam{i-1}'].T
                T_rel = np.array(data[cid].get('T_cn_cnm1', np.eye(4)))
                T = prev_T @ T_rel
            cams[cid] = Camera(cid, data[cid], T, ground_h)
        return cams

    def _img_cb(self, msg, cid):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.cameras[cid].map_x is None: self.cameras[cid].init_maps(img.shape[1], img.shape[0])
            img_u = self.cameras[cid].undistort(img)
            with self.lock: 
                self.imgs[cid] = (img_u, msg.header.stamp) # Store with timestamp
        except: pass

    def spin(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            frames, active_cams, current_stamps = [], [], {}
            with self.lock:
                for cid in ['cam0', 'cam1', 'cam2', 'cam3']:
                    if self.imgs[cid] is not None:
                        frame, stamp = self.imgs[cid]
                        frames.append(frame)
                        active_cams.append(cid)
                        current_stamps[cid] = stamp

            if len(frames) < 4: rate.sleep(); continue
            
            is_new_frame = False
            for cid, stamp in current_stamps.items():
                if self.last_timestamps[cid] is None or stamp > self.last_timestamps[cid]:
                    is_new_frame = True
                    break
            
            if not is_new_frame:
                rate.sleep()
                continue

            self.last_timestamps = current_stamps
            
            processed_results = []
            times = {}
            for i, frame in enumerate(frames):
                start_t = time.perf_counter()
                result = self.model(frame, verbose=False)[0]
                end_t = time.perf_counter()
                
                processed_results.append(result)
                cam_name = {'cam0':'front', 'cam1':'rear', 'cam2':'left', 'cam3':'right'}[active_cams[i]]
                times[cam_name] = (end_t - start_t) * 1000.0

            log_msg = "[Segmentation Time] " + " | ".join([f"{k}: {v:.1f} ms" for k, v in times.items()])
            rospy.loginfo_throttle(1.0, log_msg)

            detections = self._process_yolo(processed_results, active_cams)
            tracked_dets = self.tracker.update(detections, self.cameras)
            self._vis(frames, active_cams, tracked_dets)
            
            rate.sleep()

    def _process_yolo(self, results, active_cams):
        all_dets_by_cam = {cid: [] for cid in active_cams}
        for i, res in enumerate(results):
            cid = active_cams[i]
            if not res.boxes: continue
            for j, box in enumerate(res.boxes):
                raw_cls, conf = int(box.cls), float(box.conf)
                if conf < 0.05 or raw_cls not in {1,3,4}: continue
                det_data = {'cam':cid,'box':box.xyxy.cpu().numpy()[0],'conf':conf,'cls':raw_cls}
                if res.masks and j < len(res.masks.data): det_data['mask'] = res.masks.data[j].cpu().numpy()
                all_dets_by_cam[cid].append(det_data)
        final_detections = []
        for cid, dets in all_dets_by_cam.items():
            if not dets: continue
            boxes = np.array([d['box'] for d in dets])
            scores = np.array([d['conf'] for d in dets])
            indices = non_max_suppression(boxes, scores, iou_threshold=0.6)
            for idx in indices:
                det = dets[idx]; xyxy = det['box']
                uv = [(xyxy[0]+xyxy[2])/2, xyxy[3]]
                pos = self.cameras[cid].pixel_to_world(uv[0], uv[1])
                if pos is None: continue
                det['pos'] = pos; det['gid'] = -1; det['uv'] = uv
                final_detections.append(det)
        return final_detections

    def _vis(self, imgs, cams, dets):
        vis = {c: i.copy() for c, i in zip(cams, imgs)}
        for d in dets:
            cid, gid = d.get('cam'), d.get('gid', -1)
            if cid not in vis: continue
            x1, y1, x2, y2 = map(int, d['box'])
            color = (0, 255, 255) # Unified Yellow
            cv2.rectangle(vis[cid], (x1, y1), (x2, y2), color, 2)
            label = f"Door {gid if gid!=-1 else '?'} ({d['conf']:.2f})"
            cv2.putText(vis[cid], label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        h, w = imgs[0].shape[:2]
        def get_img(k): return cv2.resize(vis.get(k, np.zeros((h,w,3),np.uint8)), (w,h))
        top = np.hstack((get_img('cam0'), get_img('cam2')))
        bottom = np.hstack((get_img('cam1'), get_img('cam3')))
        final = np.vstack((top, bottom))
        cv2.imshow("SnapSpace Tracker (v14)", final)
        cv2.waitKey(1)

if __name__ == '__main__':
    try: SnapSpaceNode().spin()
    except rospy.ROSInterruptException: pass