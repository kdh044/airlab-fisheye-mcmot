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

# ========== 튜닝 파라미터 ==========
CONFIG = {
    # YOLO 검출
    'yolo_conf_thresh': 0.2,
    
    # 프레임 내 그룹핑 (같은 문 판단)
    'fusion_eps': 0.7,
    
    # DoorMap
    'door_conf_thresh': 0.25,
    'match_dist_tentative': 0.45,
    'match_dist_confirmed': 0.60,
    'min_observations': 5,
    'ttl_tentative': 5.0,
    'ttl_confirmed': 60.0,
    'cluster_merge_dist': 0.5,
    
    # Handle 귀속
    'handle_min_containment': 0.3,
    
    # 프레임 동기화
    'sync_threshold': 0.05,
    
    # 월드 좌표 (카메라 높이)
    'ground_height': 1.7,
}
# ==================================


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


def fuse_dets_by_world_pos(dets, eps=None):
    """프레임 내 det들끼리 월드 (x,z) 기준으로 그룹핑."""
    if eps is None:
        eps = CONFIG['fusion_eps']
    if not dets:
        return [], []
    
    used = [False] * len(dets)
    representatives = []
    groups = []
    
    for i in range(len(dets)):
        if used[i]:
            continue
        base = dets[i]
        group = [base]
        used[i] = True
        pi = base['pos'][[0, 2]]
        
        for j in range(i + 1, len(dets)):
            if used[j]:
                continue
            pj = dets[j]['pos'][[0, 2]]
            if np.linalg.norm(pi - pj) < eps:
                used[j] = True
                group.append(dets[j])
        
        rep = max(group, key=lambda d: d['conf'])
        rep_pos = np.mean([g['pos'] for g in group], axis=0)
        rep['pos'] = rep_pos
        rep['fused_count'] = len(group)
        
        representatives.append(rep)
        groups.append(group)
    
    return representatives, groups


def propagate_gid_to_groups(representatives, groups):
    """DoorMap에서 할당받은 gid를 그룹 전체 det에 전파."""
    all_dets = []
    for rep, group in zip(representatives, groups):
        gid = rep.get('gid', -1)
        for det in group:
            det['gid'] = gid
            all_dets.append(det)
    return all_dets


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
            scale = 1.0
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


class DoorMap:
    def __init__(self):
        self.clusters = {}
        self.next_gid = 0
        self.match_dist_tentative = CONFIG['match_dist_tentative']
        self.match_dist_confirmed = CONFIG['match_dist_confirmed']
        self.door_conf_thresh = CONFIG['door_conf_thresh']
        self.min_observations = CONFIG['min_observations']
        self.ttl_tentative = CONFIG['ttl_tentative']
        self.ttl_confirmed = CONFIG['ttl_confirmed']
        
    def update(self, door_dets):
        now = rospy.Time.now()
        
        valid_dets = [d for d in door_dets if d['conf'] >= self.door_conf_thresh]
        low_conf_dets = [d for d in door_dets if d['conf'] < self.door_conf_thresh]
        for d in low_conf_dets:
            d['gid'] = -1
        
        if not valid_dets:
            self._cleanup(now)
            return door_dets
        
        cluster_ids = list(self.clusters.keys())
        
        if not cluster_ids:
            for det in valid_dets:
                self._create_cluster(det, now)
            return door_dets
        
        n_dets = len(valid_dets)
        n_clusters = len(cluster_ids)
        cost = np.full((n_dets, n_clusters), 1e6)
        
        for i, det in enumerate(valid_dets):
            det_pos = det['pos'][[0, 2]]
            for j, gid in enumerate(cluster_ids):
                cluster = self.clusters[gid]
                cluster_pos = cluster['pos'][[0, 2]]
                dist = np.linalg.norm(det_pos - cluster_pos)
                thresh = self.match_dist_confirmed if cluster['confirmed'] else self.match_dist_tentative
                if dist < thresh:
                    cost[i, j] = dist
        
        row_idx, col_idx = linear_sum_assignment(cost)
        
        matched_dets = set()
        for r, c in zip(row_idx, col_idx):
            cluster = self.clusters[cluster_ids[c]]
            thresh = self.match_dist_confirmed if cluster['confirmed'] else self.match_dist_tentative
            if cost[r, c] < thresh:
                det = valid_dets[r]
                gid = cluster_ids[c]
                self._update_cluster(gid, det, now)
                det['gid'] = gid
                matched_dets.add(r)
        
        for i, det in enumerate(valid_dets):
            if i not in matched_dets:
                self._create_cluster(det, now)
        
        self._cleanup(now)
        return door_dets
    
    def _create_cluster(self, det, now):
        gid = self.next_gid
        self.next_gid += 1
        self.clusters[gid] = {
            'pos': det['pos'].copy(),
            'last_seen': now,
            'observations': 1,
            'confirmed': False
        }
        det['gid'] = gid
    
    def _update_cluster(self, gid, det, now):
        cluster = self.clusters[gid]
        alpha = 0.2
        cluster['pos'] = (1 - alpha) * cluster['pos'] + alpha * det['pos']
        cluster['last_seen'] = now
        cluster['observations'] += 1
        
        if not cluster['confirmed'] and cluster['observations'] >= self.min_observations:
            cluster['confirmed'] = True
            rospy.loginfo(f"[DoorMap] Cluster {gid} CONFIRMED (obs={cluster['observations']})")
    
    def _cleanup(self, now):
        self._merge_close_clusters()
        
        expired = []
        for gid, c in self.clusters.items():
            ttl = self.ttl_confirmed if c['confirmed'] else self.ttl_tentative
            if (now - c['last_seen']).to_sec() > ttl:
                expired.append(gid)
        for gid in expired:
            del self.clusters[gid]
    
    def _merge_close_clusters(self):
        merge_dist = CONFIG['cluster_merge_dist']
        confirmed_ids = [gid for gid, c in self.clusters.items() if c['confirmed']]
        if len(confirmed_ids) < 2:
            return
        
        to_remove = set()
        for i, gid1 in enumerate(confirmed_ids):
            if gid1 in to_remove:
                continue
            c1 = self.clusters[gid1]
            for gid2 in confirmed_ids[i+1:]:
                if gid2 in to_remove:
                    continue
                c2 = self.clusters[gid2]
                dist = np.linalg.norm(c1['pos'][[0,2]] - c2['pos'][[0,2]])
                if dist < merge_dist:
                    if c1['observations'] >= c2['observations']:
                        c1['observations'] += c2['observations']
                        to_remove.add(gid2)
                        rospy.loginfo(f"[DoorMap] Merged cluster {gid2} into {gid1}")
                    else:
                        c2['observations'] += c1['observations']
                        to_remove.add(gid1)
                        rospy.loginfo(f"[DoorMap] Merged cluster {gid1} into {gid2}")
                        break
        
        for gid in to_remove:
            del self.clusters[gid]
    
    def is_confirmed(self, gid):
        return gid in self.clusters and self.clusters[gid]['confirmed']


class SnapSpaceNode:
    def __init__(self):
        rospy.init_node('snapspace_tracker', anonymous=False)
        r = rospkg.RosPack()
        calib_path = r.get_path('ces') + "/MOT/calibration file/2024-09-13-20-37-00-camchain.yaml"
        self.cameras = self._load_cameras(calib_path)
        
        self.door_map = DoorMap()
        
        self.model = YOLO(r.get_path('ces') + "/MOT/pretrained/3_27_witd_model.pt")
        self.bridge = CvBridge(); self.lock = threading.Lock()
        self.imgs = {k: None for k in ['cam0', 'cam1', 'cam2', 'cam3']}
        self.last_timestamps = {k: None for k in ['cam0', 'cam1', 'cam2', 'cam3']}
        
        self.sync_threshold = CONFIG['sync_threshold']
        
        for c in ['front','rear','left','right']:
            cid = {'front':'cam0','rear':'cam1','left':'cam2','right':'cam3'}[c]
            rospy.Subscriber(f"/camera/image_raw_{c}", Image, self._img_cb, callback_args=cid)

    def _load_cameras(self, path):
        with open(path) as f: data = yaml.safe_load(f)
        ground_h = CONFIG['ground_height']
        cams = {}
        
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"[Camera Chain] ground_height = {ground_h}m")
        
        for i in range(4):
            cid = f'cam{i}'
            T = np.eye(4)
            if i > 0:
                prev_T = cams[f'cam{i-1}'].T
                T_cn_cnm1 = np.array(data[cid].get('T_cn_cnm1', np.eye(4)))
                T_rel = np.linalg.inv(T_cn_cnm1)
                T = prev_T @ T_rel
            
            cams[cid] = Camera(cid, data[cid], T, ground_h)
            pos = T[:3, 3]
            rospy.loginfo(f"[{cid}] Global pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        
        rospy.loginfo("=" * 60)
        return cams

    def _img_cb(self, msg, cid):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.cameras[cid].map_x is None: self.cameras[cid].init_maps(img.shape[1], img.shape[0])
            img_u = self.cameras[cid].undistort(img)
            with self.lock: 
                self.imgs[cid] = (img_u, msg.header.stamp)
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"[{cid}] img_cb error: {e}")

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
            
            stamps = [current_stamps[cid].to_sec() for cid in ['cam0','cam1','cam2','cam3']]
            if max(stamps) - min(stamps) > self.sync_threshold:
                rate.sleep()
                continue
            
            is_new_frame = False
            for cid, stamp in current_stamps.items():
                if self.last_timestamps[cid] is None or stamp > self.last_timestamps[cid]:
                    is_new_frame = True
                    break
            
            if not is_new_frame:
                rate.sleep()
                continue

            self.last_timestamps = current_stamps.copy()
            
            processed_results = []
            for i, frame in enumerate(frames):
                result = self.model(frame, verbose=False)[0]
                processed_results.append(result)

            detections = self._process_yolo(processed_results, active_cams, frames)
            
            door_dets = [d for d in detections if d['cls'] == 1]
            handle_dets = [d for d in detections if d['cls'] == 2]
            
            representatives, groups = fuse_dets_by_world_pos(door_dets)
            representatives = self.door_map.update(representatives)
            door_dets = propagate_gid_to_groups(representatives, groups)
            
            self._attribute_handles_to_doors(handle_dets, door_dets)
            
            all_dets = door_dets + handle_dets
            self._vis(frames, active_cams, all_dets)
            
            rate.sleep()

    def _attribute_handles_to_doors(self, handle_dets, door_dets):
        min_score = CONFIG['handle_min_containment']
        for handle in handle_dets:
            handle['gid'] = -1
            handle['parent_door'] = None
            
            h_cam = handle['cam']
            h_box = handle['box']
            h_mask = handle.get('mask')
            
            best_door, best_score = None, min_score
            
            for door in door_dets:
                if door['cam'] != h_cam:
                    continue
                
                d_mask = door.get('mask')
                d_box = door['box']
                
                if h_mask is not None and d_mask is not None:
                    h_area = (h_mask > 0.5).sum()
                    if h_area == 0:
                        continue
                    intersection = np.logical_and(h_mask > 0.5, d_mask > 0.5).sum()
                    score = intersection / h_area
                else:
                    score = compute_iou(h_box, d_box)
                
                if score > best_score:
                    best_score = score
                    best_door = door
            
            if best_door is not None:
                handle['gid'] = best_door['gid']
                handle['parent_door'] = best_door['gid']

    def _get_mask_bottom_uv(self, mask, box, img_shape):
        v_h, v_w = img_shape[:2]
        if mask.shape[:2] != (v_h, v_w):
            mask = cv2.resize(mask, (v_w, v_h), interpolation=cv2.INTER_NEAREST)
        
        ys, xs = np.where(mask > 0.5)
        if len(ys) == 0:
            return [(box[0] + box[2]) / 2, box[3]]
        
        max_y = ys.max()
        bottom_xs = xs[ys == max_y]
        return [(bottom_xs.min() + bottom_xs.max()) / 2, max_y]

    def _process_yolo(self, results, active_cams, frames):
        all_dets_by_cam = {cid: [] for cid in active_cams}
        yolo_conf = CONFIG['yolo_conf_thresh']
        
        for i, res in enumerate(results):
            cid = active_cams[i]
            if not res.boxes: continue
            
            frame_h, frame_w = frames[i].shape[:2]
            
            for j, box in enumerate(res.boxes):
                raw_cls, conf = int(box.cls), float(box.conf)
                if conf < yolo_conf or raw_cls not in {1,2,3,4}: continue
                
                tracker_cls = 1 if raw_cls in {1, 3, 4} else 2
                
                det_data = {
                    'cam': cid,
                    'box': box.xyxy.cpu().numpy()[0],
                    'conf': conf,
                    'cls': tracker_cls,
                    'raw_cls': raw_cls,
                    'frame_shape': (frame_h, frame_w)
                }
                if res.masks is not None and j < len(res.masks.data):
                    mask_raw = res.masks.data[j].cpu().numpy()
                    if mask_raw.shape[:2] != (frame_h, frame_w):
                        mask_raw = cv2.resize(mask_raw, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
                    det_data['mask'] = mask_raw
                all_dets_by_cam[cid].append(det_data)
        
        final_detections = []
        for cid, dets in all_dets_by_cam.items():
            if not dets: continue
            boxes = np.array([d['box'] for d in dets])
            scores = np.array([d['conf'] for d in dets])
            indices = non_max_suppression(boxes, scores, iou_threshold=0.6)
            for idx in indices:
                det = dets[idx]
                xyxy = det['box']
                frame_shape = det.get('frame_shape', (480, 640))
                
                if det['cls'] == 1:
                    if 'mask' in det:
                        uv = self._get_mask_bottom_uv(det['mask'], xyxy, frame_shape)
                    else:
                        uv = [(xyxy[0]+xyxy[2])/2, xyxy[3]]
                    
                    pos = self.cameras[cid].pixel_to_world(uv[0], uv[1])
                    if pos is None: continue
                    det['pos'] = pos
                    det['uv'] = uv
                else:
                    det['pos'] = np.array([0, 0, 0])
                    det['uv'] = [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
                
                det['gid'] = -1
                final_detections.append(det)
        return final_detections

    def _vis(self, imgs, cams, dets):
        vis = {c: i.copy() for c, i in zip(cams, imgs)}
        for d in dets:
            cid, gid = d.get('cam'), d.get('gid', -1)
            if cid not in vis: continue
            
            x1, y1, x2, y2 = map(int, d['box'])
            tracker_cls = d.get('cls', 1)
            is_confirmed = tracker_cls == 1 and self.door_map.is_confirmed(gid)
            
            # 문은 확정된 것만 표시
            if tracker_cls == 1 and not is_confirmed:
                continue
            if tracker_cls == 1:
                color = (0, 255, 0) if is_confirmed else (0, 255, 255)
            else:
                color = (255, 255, 0) if d.get('parent_door') else (128, 128, 128)
            
            if 'mask' in d:
                mask = d['mask']
                v_h, v_w = vis[cid].shape[:2]
                if mask.shape[:2] != (v_h, v_w):
                    mask = cv2.resize(mask, (v_w, v_h), interpolation=cv2.INTER_NEAREST)
                overlay = np.zeros_like(vis[cid], dtype=np.uint8)
                overlay[mask > 0.5] = color
                vis[cid] = cv2.addWeighted(vis[cid], 1.0, overlay, 0.4, 0)

            thickness = 3 if is_confirmed else 2
            cv2.rectangle(vis[cid], (x1, y1), (x2, y2), color, thickness)
            
            if tracker_cls == 1:
                status = "[OK]" if is_confirmed else ""
                label = f"D{gid}{status} ({d['conf']:.2f})"
            else:
                parent = d.get('parent_door')
                label = f"H->D{parent}" if parent else "Handle"
            
            cv2.putText(vis[cid], label, (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        h, w = imgs[0].shape[:2]
        def get_img(k): return cv2.resize(vis.get(k, np.zeros((h,w,3),np.uint8)), (w,h))
        top = np.hstack((get_img('cam0'), get_img('cam2')))
        bottom = np.hstack((get_img('cam1'), get_img('cam3')))
        final = np.vstack((top, bottom))
        cv2.imshow("SnapSpace Tracker", final)
        cv2.waitKey(1)


if __name__ == '__main__':
    try: SnapSpaceNode().spin()
    except rospy.ROSInterruptException: pass