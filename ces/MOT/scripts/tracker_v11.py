#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tracker V11 - Fisheye Overlap Stereo (The Real 3D)
==================================================
185도 어안 렌즈의 넓은 시야각(Overlap)을 활용한 정밀 3D 트래킹.
SLAM 없이도 두 카메라(Front+Side)의 교차각을 이용해 깊이(Depth)를 산출.

원리:
1. Front 카메라와 Side(Left/Right) 카메라의 Extrinsic(상대 위치) 활용.
2. 두 카메라에서 동시에 보이는 물체(Door)를 매칭.
3. Ray-to-Ray Triangulation으로 3D 좌표(x,y,z) 계산.
4. 계산된 3D 좌표 기반으로 Global ID 부여.
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
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from ultralytics import YOLO

# ========== 설정 ==========
CONFIG = {
    'door_conf': 0.25,
    'tracker_type': 'bytetrack.yaml',
    
    # 스테레오 매칭 임계값
    'max_ray_dist': 0.5,   # 두 광선 사이의 최단 거리가 이보다 작아야 매칭 인정 (미터)
    'match_dist_3d': 1.0,  # 기존 3D 문과 이만큼 가까우면 같은 ID (미터)
    'sync_slop': 0.1,      # 카메라 간 시간 동기화 허용 오차 (초)
    
    # 시각화
    'font_scale': 0.8,
}

def closest_point_of_two_lines(p1, d1, p2, d2):
    """
    두 광선(P+t*D) 사이의 가장 가까운 점(중점) 계산
    p1, p2: Ray Origin (3,)
    d1, d2: Ray Direction (Normalized, 3,)
    Return: (midpoint_xyz, min_distance)
    """
    w0 = p1 - p2
    a = np.dot(d1, d1); b = np.dot(d1, d2); c = np.dot(d2, d2)
    d = np.dot(d1, w0); e = np.dot(d2, w0)
    denom = a * c - b * b
    
    if abs(denom) < 1e-6: return None, float('inf') # 평행
    
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    
    # 카메라 뒤쪽(음수)은 제외
    # if s < 0 or t < 0: return None, float('inf')
    
    point1 = p1 + s * d1
    point2 = p2 + t * d2
    dist = np.linalg.norm(point1 - point2)
    midpoint = (point1 + point2) / 2.0
    return midpoint, dist

class Camera:
    def __init__(self, name, intr, T_rig_cam):
        self.name = name
        self.fx = intr[2]
        self.cx = intr[4]
        # Extrinsic: Cam -> Rig
        self.T_rig_cam = T_rig_cam
        self.R = T_rig_cam[:3, :3]
        self.t = T_rig_cam[:3, 3] # Ray Origin in Rig Frame
        
    def pixel_to_ray(self, u):
        # 1. Camera Frame Ray (Z=1 plane, only X varies for bearing)
        # Assuming Y is roughly center (v = cy) for bearing check
        # Vector: [(u-cx)/fx, 0, 1] (Simplified)
        x_c = (u - self.cx) / self.fx
        ray_c = np.array([x_c, 0, 1.0])
        ray_c /= np.linalg.norm(ray_c)
        
        # 2. Rig Frame Ray
        ray_rig = self.R @ ray_c
        return self.t, ray_rig

class DoorObj:
    def __init__(self, uid, xyz):
        self.id = uid
        self.xyz = xyz
        self.last_seen = time.time()
        self.hits = 1
    
    def update(self, xyz):
        # Update Position (Average)
        alpha = 0.2
        self.xyz = (1-alpha)*self.xyz + alpha*xyz
        self.last_seen = time.time()
        self.hits += 1

class TrackerV11:
    def __init__(self):
        rospy.init_node("tracker_v11", anonymous=False)
        r = rospkg.RosPack()
        
        calib_path = r.get_path('ces') + "/MOT/calibration file/2024-09-13-20-37-00-camchain.yaml"
        self.cameras = self._load_cameras(calib_path)
        
        model_path = r.get_path('ces') + "/MOT/pretrained/best.pt"
        self.yolo = YOLO(model_path)
        
        self.doors = {} # global_id -> DoorObj
        self.next_id = 0
        
        self.lock = threading.Lock()
        self.data_buf = {} # {cam_name: (time, img)}
        self.bridge = CvBridge()
        
        # Subs - HARDWARE FIX (SWAP LEFT/REAR)
        # Topic 'left' contains Rear image -> Store as 'rear'
        rospy.Subscriber("/camera/undistorted/left", Image, self._make_cb('rear'))
        # Topic 'rear' contains Left image -> Store as 'left'
        rospy.Subscriber("/camera/undistorted/rear", Image, self._make_cb('left'))
        
        # Normal
        rospy.Subscriber("/camera/undistorted/front", Image, self._make_cb('front'))
        rospy.Subscriber("/camera/undistorted/right", Image, self._make_cb('right'))
        
        self.pub_marker = rospy.Publisher("/door_markers", MarkerArray, queue_size=1)
        
        rospy.loginfo("Tracker V11 (Fisheye Stereo) Started - LEFT/REAR SWAPPED")

    def _load_cameras(self, path):
        with open(path) as f: data = yaml.safe_load(f)
        cams = {}
        
        # Hardcoded Extrinsics (Standard ROS: X-forward, Y-left, Z-up)
        # However, Camera frame is usually: Z-forward, X-right, Y-down (Optical)
        # We need T_rig_cam (Cam in Rig Frame).
        
        # Rig Frame: X-forward, Y-left, Z-up
        # Cam Optical Frame: Z-forward, X-right, Y-down
        
        # Base Rotation (Optical to Body Standard):
        # Z(opt) -> X(body), X(opt) -> -Y(body), Y(opt) -> -Z(body)
        # R_opt_body = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        
        # But let's assume we work in a simplified 2D plane for ray casting
        # Let's define T_rig_cam s.t. Ray(0,0,1) points in the viewing direction
        
        # Front Cam: Looks +X in Rig
        # Left Cam: Looks +Y in Rig
        # Right Cam: Looks -Y in Rig
        # Rear Cam: Looks -X in Rig
        
        # We construct Rotation matrix that rotates Vector(0,0,1) to desired direction
        # Front: Identity (assuming input rays are already in a frame where Z is forward) -> No, wait.
        # The `pixel_to_ray` function generates a vector [x, 0, 1]. This is Z-forward.
        # So we need R that rotates Z-forward to X-forward (for Front).
        
        # R_front: Z -> X
        R_front = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        
        # R_left: Z -> Y
        # Rotate R_front by +90 around Z_rig
        R_z90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        R_left = R_z90 @ R_front
        
        # R_right: Z -> -Y
        R_zn90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        R_right = R_zn90 @ R_front
        
        # R_rear: Z -> -X
        R_z180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        R_rear = R_z180 @ R_front
        
        # Translations (approx 0.2m)
        t_front = np.array([0.2, 0, 0])
        t_left  = np.array([0, 0.2, 0])
        t_right = np.array([0, -0.2, 0])
        t_rear  = np.array([-0.2, 0, 0])
        
        poses = {
            'front': np.eye(4), 'left': np.eye(4), 'right': np.eye(4), 'rear': np.eye(4)
        }
        poses['front'][:3,:3] = R_front; poses['front'][:3,3] = t_front
        poses['left'][:3,:3]  = R_left;  poses['left'][:3,3]  = t_left
        poses['right'][:3,:3] = R_right; poses['right'][:3,3] = t_right
        poses['rear'][:3,:3]  = R_rear;  poses['rear'][:3,3]  = t_rear
        
        # Mapping (Left/Rear Swapped logic maintained)
        # front -> cam0
        # left topic (actual rear) -> rear var -> cam1 calib
        # rear topic (actual left) -> left var -> cam2 calib
        # right -> cam3
        
        # Note: 'poses' are now hardcoded by logical name, so we just need intrinsics
        intr_map = {'front':'cam0', 'left':'cam2', 'rear':'cam1', 'right':'cam3'}
        
        for name, cid in intr_map.items():
            intr = data[cid]['intrinsics']
            cams[name] = Camera(name, intr, poses[name])
            
        return cams

    def _make_cb(self, name):
        def cb(msg):
            with self.lock:
                try:
                    self.data_buf[name] = (msg.header.stamp.to_sec(), self.bridge.imgmsg_to_cv2(msg, "bgr8"))
                except: pass
        return cb

    def detect(self, img):
        # Class 0:Door
        res = self.yolo.predict(img, conf=CONFIG['door_conf'], classes=[0], verbose=False)[0]
        dets = []
        if res.boxes.id is None and res.boxes.xyxy is not None:
             for box in res.boxes.xyxy.cpu().numpy():
                 dets.append(box)
        return dets

    def match_and_triangulate(self, camA, detsA, camB, detsB):
        # Match detections between two cameras
        # Naive approach: Find pair with closest Ray intersection
        
        matches = [] # (boxA_idx, boxB_idx, point3d)
        
        for i, boxA in enumerate(detsA):
            uA = (boxA[0] + boxA[2]) / 2.0
            origA, dirA = camA.pixel_to_ray(uA)
            
            best_dist = float('inf')
            best_j = -1
            best_pt = None
            
            for j, boxB in enumerate(detsB):
                uB = (boxB[0] + boxB[2]) / 2.0
                origB, dirB = camB.pixel_to_ray(uB)
                
                pt, dist = closest_point_of_two_lines(origA, dirA, origB, dirB)
                
                if pt is not None:
                    # Debug print
                    # print(f"DEBUG: Ray Dist: {dist:.3f} | PT: {pt}")
                    pass

                if pt is not None and dist < CONFIG['max_ray_dist']:
                    # Distance check (Valid triangulation?)
                    # Must be somewhat in front of Rig (X > -1)
                    if pt[0] > -1.0: 
                        if dist < best_dist:
                            best_dist = dist
                            best_j = j
                            best_pt = pt
            
            if best_j != -1:
                rospy.loginfo_throttle(1.0, f"Triangulated! Dist: {best_dist:.3f}m at {best_pt}")
                matches.append((i, best_j, best_pt))
                
        return matches

    def update_map(self, point3d):
        # Find closest existing door
        best_id = -1
        min_dist = float('inf')
        
        for uid, door in self.doors.items():
            dist = np.linalg.norm(door.xyz - point3d)
            if dist < CONFIG['match_dist_3d']:
                if dist < min_dist:
                    min_dist = dist
                    best_id = uid
        
        if best_id != -1:
            self.doors[best_id].update(point3d)
            return best_id
        else:
            new_id = self.next_id
            self.doors[new_id] = DoorObj(new_id, point3d)
            self.next_id += 1
            return new_id

    def draw_viz(self, imgs_dict, det_dict, id_map_dict):
        # 2x2 Grid with Directions
        # Front(0,0), Left(0,1) -> No, user wants Direction mapping
        # Let's do:
        #  [LEFT] [FRONT] [RIGHT]
        #      [REAR]
        # But for 2x2 window:
        # [FRONT] [RIGHT]
        # [LEFT]  [REAR]
        # With big arrows
        
        canvases = {}
        h, w = 480, 640
        
        for cam in ['front', 'left', 'rear', 'right']:
            if cam in imgs_dict:
                img = cv2.resize(imgs_dict[cam], (w, h))
            else:
                img = np.zeros((h, w, 3), np.uint8)
            
            # Draw Detections
            if cam in det_dict:
                for idx, box in enumerate(det_dict[cam]):
                    # Check if this box has an ID
                    gid = id_map_dict.get((cam, idx))
                    
                    x1, y1, x2, y2 = map(int, box)
                    if gid is not None:
                        color = (0, 255, 0)
                        label = f"ID:{gid}"
                        # 3D coord display
                        if gid in self.doors:
                            d = self.doors[gid]
                            label += f"({d.xyz[0]:.1f},{d.xyz[1]:.1f})"
                    else:
                        color = (0, 165, 255)
                        label = "Scan.."
                        
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw Direction Label
            text = f"[{cam.upper()}]"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            
            # Direction Arrow (Simple)
            cx, cy = w//2, h-50
            if cam == 'front':
                cv2.arrowedLine(img, (cx, cy+20), (cx, cy-20), (0,0,255), 2) # Up
            elif cam == 'rear':
                cv2.arrowedLine(img, (cx, cy-20), (cx, cy+20), (0,0,255), 2) # Down
            elif cam == 'left':
                cv2.arrowedLine(img, (cx+20, cy), (cx-20, cy), (0,0,255), 2) # Left
            elif cam == 'right':
                cv2.arrowedLine(img, (cx-20, cy), (cx+20, cy), (0,0,255), 2) # Right
                
            canvases[cam] = img

        top = np.hstack((canvases['left'], canvases['front'])) 
        bot = np.hstack((canvases['rear'], canvases['right']))
        
        # Rearrange for intuitive view:
        # Left | Front
        # Rear | Right
        # (User can match this to physical setup)
        
        cv2.imshow("Tracker V11 (Overlap Stereo)", np.vstack((top, bot)))
        cv2.waitKey(1)

    def loop(self):
        rate = rospy.Rate(15)
        while not rospy.is_shutdown():
            with self.lock:
                # Copy current buffer
                current_data = {k: v for k, v in self.data_buf.items()}
            
            if 'front' not in current_data:
                rate.sleep(); continue
            
            t_front = current_data['front'][0]
            
            # Detect on all
            dets = {}
            for cam, (t, img) in current_data.items():
                if abs(t - t_front) < CONFIG['sync_slop']:
                    dets[cam] = self.detect(img)
            
            id_map = {} # (cam, box_idx) -> global_id
            
            # Stereo Pairs: Front-Left, Front-Right
            pairs = [('front', 'left'), ('front', 'right')]
            
            for cA, cB in pairs:
                if cA in dets and cB in dets:
                    matches = self.match_and_triangulate(
                        self.cameras[cA], dets[cA],
                        self.cameras[cB], dets[cB]
                    )
                    
                    for idxA, idxB, pt3d in matches:
                        gid = self.update_map(pt3d)
                        id_map[(cA, idxA)] = gid
                        id_map[(cB, idxB)] = gid
                        
            # Visualize
            self.draw_viz({k:v[1] for k,v in current_data.items()}, dets, id_map)
            
            # Publish Markers
            ma = MarkerArray()
            for gid, d in self.doors.items():
                m = Marker(); m.header.frame_id = "base_link" # Rig frame
                m.header.stamp = rospy.Time.now()
                m.ns = "doors"; m.id = gid; m.type = Marker.CUBE
                m.action = Marker.ADD
                m.pose.position.x = d.xyz[0]; m.pose.position.y = d.xyz[1]; m.pose.position.z = d.xyz[2]
                m.scale.x = 0.5; m.scale.y = 0.5; m.scale.z = 2.0
                m.color.a = 0.8; m.color.g = 1.0
                ma.markers.append(m)
            self.pub_marker.publish(ma)
            
            rate.sleep()

if __name__ == "__main__":
    TrackerV11().loop()
