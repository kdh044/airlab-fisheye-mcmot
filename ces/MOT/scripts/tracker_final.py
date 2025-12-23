#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tracker Final (V12) - Relay + Stereo Verification
=================================================
"이어달리기(Relay)"로 ID를 유지하되, "스테레오(Stereo)"로 위치를 보정하는 하이브리드 방식.
"""

import rospy
import rospkg
import cv2
import yaml
import numpy as np
import threading
import time

from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from ultralytics import YOLO

# ========== 설정 ==========
CONFIG = {
    'door_conf': 0.2,       # 노이즈 제거 위해 높임
    'tracker_type': 'bytetrack.yaml',
    'border_margin': 100,   # Hand-over 영역 (픽셀)
    'handover_time': 2.0,   # 몇 초 안에 넘겨받아야 하는지
    
    # Stereo Filter
    'min_valid_dist': 0.5,  # 50cm보다 가까우면 노이즈 취급 (무시)
    'max_ray_dist': 0.5,    # 두 광선이 이 거리 이내로 만나야 매칭 인정
}

def closest_point(p1, d1, p2, d2):
    """두 직선(Ray) 사이의 최단 거리 및 중점 계산"""
    w0 = p1 - p2
    a = np.dot(d1, d1); b = np.dot(d1, d2); c = np.dot(d2, d2)
    d = np.dot(d1, w0); e = np.dot(d2, w0)
    denom = a * c - b * b
    if abs(denom) < 1e-6: return None, float('inf')
    
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    
    # 카메라 뒤쪽 제외
    # if s < 0 or t < 0: return None, float('inf')
    
    pt1 = p1 + s * d1
    pt2 = p2 + t * d2
    mid = (pt1 + pt2) / 2.0
    dist = np.linalg.norm(pt1 - pt2)
    return mid, dist

class Camera:
    def __init__(self, R, t, fx, cx):
        self.R = R; self.t = t; self.fx = fx; self.cx = cx
    
    def pix2ray(self, u):
        # Camera Frame: Z-forward (Simplified) -> Rig Frame
        # Simple Ray: [(u-cx)/fx, 0, 1]
        ray_c = np.array([(u - self.cx)/self.fx, 0, 1.0])
        ray_c /= np.linalg.norm(ray_c)
        
        # Transform to Rig Frame: R * ray_c + t (Origin is t)
        return self.t, self.R @ ray_c

class TrackerFinal:
    def __init__(self):
        rospy.init_node("tracker_final", anonymous=False)
        r = rospkg.RosPack()
        
        # Load Config
        calib_path = r.get_path('ces') + "/MOT/calibration file/2024-09-13-20-37-00-camchain.yaml"
        self.cams = self._load_cams(calib_path)
        
        model_path = r.get_path('ces') + "/MOT/pretrained/best.pt"
        self.trackers = {k: YOLO(model_path) for k in ['front','left','right','rear']}
        
        # State
        self.handoff_left = []  # Front -> Left Queue
        self.handoff_right = [] # Front -> Right Queue
        self.doors = {} # gid -> xyz (3D location)
        
        self.lock = threading.Lock()
        self.images = {}
        self.bridge = CvBridge()
        
        # Subs (Hardware Fix: Left/Rear Swapped)
        rospy.Subscriber("/camera/undistorted/front", Image, self._cb('front'))
        rospy.Subscriber("/camera/undistorted/right", Image, self._cb('right'))
        rospy.Subscriber("/camera/undistorted/left", Image, self._cb('rear')) # Rear topic -> left var
        rospy.Subscriber("/camera/undistorted/rear", Image, self._cb('left')) # Left topic -> rear var
        
        self.pub_marker = rospy.Publisher("/door_markers", MarkerArray, queue_size=1)
        rospy.loginfo("Tracker Final (Relay + Stereo) Started")

    def _load_cams(self, path):
        with open(path) as f: data = yaml.safe_load(f)
        cams = {}
        # Hardcoded Extrinsics (Rig Frame: X-fwd, Y-left, Z-up)
        # Assuming Cameras are Z-fwd
        
        # R_front: Z->X
        R0 = np.array([[0,0,1],[-1,0,0],[0,-1,0]])
        
        # R_left: Z->Y (+90 rot of R0)
        R_L = np.array([[0,-1,0],[1,0,0],[0,0,1]]) @ R0
        
        # R_right: Z->-Y (-90 rot of R0)
        R_R = np.array([[0,1,0],[-1,0,0],[0,0,1]]) @ R0
        
        # R_rear: Z->-X (180 rot of R0)
        R_B = np.array([[-1,0,0],[0,-1,0],[0,0,1]]) @ R0
        
        poses = {
            'front': (R0, np.array([0.2,0,0])),
            'left':  (R_L, np.array([0,0.2,0])),
            'right': (R_R, np.array([0,-0.2,0])),
            'rear':  (R_B, np.array([-0.2,0,0]))
        }
        
        # Mapping (Left/Rear Swapped Logic Applied)
        # front->cam0, left->cam2, rear->cam1, right->cam3
        imap = {'front':'cam0', 'left':'cam2', 'rear':'cam1', 'right':'cam3'}
        
        for n, cid in imap.items():
            intr = data[cid]['intrinsics']
            cams[n] = Camera(poses[n][0], poses[n][1], intr[2], intr[4])
        return cams

    def _cb(self, name):
        def callback(msg):
            with self.lock: self.images[name] = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        return callback

    def process(self):
        with self.lock: imgs = {k:v.copy() for k,v in self.images.items()}
        if 'front' not in imgs: return
        
        # 1. Front Tracking (Main)
        f_res = self.trackers['front'].track(imgs['front'], persist=True, tracker=CONFIG['tracker_type'],
                                            conf=CONFIG['door_conf'], classes=[0,1], verbose=False)[0]
        
        f_boxes = [] # (lid, box, u)
        if f_res.boxes.id is not None:
            for box, lid, cls in zip(f_res.boxes.xyxy.cpu().numpy(), f_res.boxes.id.cpu().numpy(), f_res.boxes.cls.cpu().numpy()):
                if cls == 1: continue # Handle skip
                u = (box[0]+box[2])/2
                f_boxes.append({'lid':int(lid), 'box':box, 'u':u})
                
                # Hand-over Logic
                h, w = imgs['front'].shape[:2]
                if u < CONFIG['border_margin']: 
                    self.add_handoff(self.handoff_left, int(lid))
                elif u > w - CONFIG['border_margin']:
                    self.add_handoff(self.handoff_right, int(lid))

        # 2. Side Tracking & Stereo Check
        sides = {'left':(self.handoff_left, 'left'), 
                 'right':(self.handoff_right, 'right')}
        
        viz_data = {'front': (imgs['front'], f_boxes)}
        
        for sname, (queue, side_cam) in sides.items():
            if sname not in imgs: continue
            
            s_res = self.trackers[sname].track(imgs[sname], persist=True, tracker=CONFIG['tracker_type'],
                                              conf=CONFIG['door_conf'], classes=[0], verbose=False)[0]
            s_boxes = []
            if s_res.boxes.id is not None:
                for box, lid in zip(s_res.boxes.xyxy.cpu().numpy(), s_res.boxes.id.cpu().numpy()):
                    u = (box[0]+box[2])/2
                    h, w = imgs[sname].shape[:2]
                    
                    # A. Relay ID Assignment
                    gid = f"{sname[0].upper()}{int(lid)}" # Default: L1, R1
                    
                    # Check overlap (Entering side cam)
                    is_entering = (sname=='left' and u > w - CONFIG['border_margin']*2) or \
                                  (sname=='right' and u < CONFIG['border_margin']*2)
                    
                    if is_entering:
                        matched = self.check_queue(queue)
                        if matched is not None:
                            gid = matched # Take Front ID
                    
                    # B. Stereo Check (Verification)
                    is_locked = False
                    for fb in f_boxes:
                        orig1, dir1 = self.cams['front'].pix2ray(fb['u'])
                        orig2, dir2 = self.cams[sname].pix2ray(u)
                        pt, dist = closest_point(orig1, dir1, orig2, dir2)
                        
                        if pt is not None and dist < CONFIG['max_ray_dist']:
                            d_cam = np.linalg.norm(pt)
                            if d_cam > CONFIG['min_valid_dist']: # Valid 3D point
                                # Update Map
                                if isinstance(gid, int): # If already matched
                                    self.doors[gid] = pt
                                    is_locked = True
                                elif isinstance(fb['lid'], int): # Use Front ID if valid
                                    self.doors[fb['lid']] = pt
                                    gid = fb['lid']
                                    is_locked = True
                    
                    s_boxes.append({'gid':gid, 'box':box, 'locked':is_locked})
            
            viz_data[sname] = (imgs[sname], s_boxes)
            
        # Rear (Dummy)
        if 'rear' in imgs: viz_data['rear'] = (imgs['rear'], [])

        self.viz(viz_data)
        self.pub_markers()

    def add_handoff(self, queue, gid):
        now = time.time()
        for item in queue:
            if item['id'] == gid: item['time'] = now; return
        queue.append({'id': gid, 'time': now})

    def check_queue(self, queue):
        now = time.time()
        queue[:] = [x for x in queue if now - x['time'] < CONFIG['handover_time']]
        return queue[-1]['id'] if queue else None

    def viz(self, data):
        canvases = {}
        for cam in ['front', 'left', 'right', 'rear']:
            img = data[cam][0].copy() if cam in data else np.zeros((480,640,3), np.uint8)
            if cam in data:
                for item in data[cam][1]:
                    lid = item.get('lid', item.get('gid'))
                    box = item['box']
                    locked = item.get('locked', False) or (isinstance(lid, int) and lid in self.doors)
                    
                    color = (0, 0, 255) if locked else (0, 255, 0)
                    label = f"ID:{lid}"
                    if locked and isinstance(lid, int) and lid in self.doors:
                        pt = self.doors[lid]
                        #label += f"({pt[0]:.1f},{pt[1]:.1f})"
                    
                    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                    cv2.putText(img, str(label), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.putText(img, f"[{cam.upper()}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            canvases[cam] = img
            
        top = np.hstack((canvases['left'], canvases['front']))
        bot = np.hstack((canvases['rear'], canvases['right']))
        cv2.imshow("Tracker Final", np.vstack((top, bot)))
        cv2.waitKey(1)

    def pub_markers(self):
        ma = MarkerArray()
        for gid, xyz in self.doors.items():
            m = Marker(); m.header.frame_id = "base_link"
            m.header.stamp = rospy.Time.now()
            m.ns = "doors"; m.id = int(gid)
            m.type = Marker.CUBE; m.action = Marker.ADD
            m.pose.position.x = xyz[0]; m.pose.position.y = xyz[1]; m.pose.position.z = xyz[2]
            m.scale.x = 0.2; m.scale.y = 0.5; m.scale.z = 2.0
            m.color.a = 0.8; m.color.r = 1.0; m.color.g = 0.0
            ma.markers.append(m)
        self.pub_marker.publish(ma)

    def loop(self):
        while not rospy.is_shutdown():
            self.process()

if __name__ == "__main__":
    TrackerFinal().loop()
