#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tracker V9 - Pure IoU (Linear Corridor Optimized)
=================================================
복잡한 알고리즘 제거. 직진 주행 시 가장 안정적인 방식.

1. SLAM / Pose 사용 안 함 (영상만 있으면 됨)
2. ByteTrack 제거 -> 순수 YOLO Detection + IoU 매칭
3. 로직:
   - 이전 프레임 박스와 현재 박스의 겹침(IoU) 계산
   - 겹치면 ID 유지, 안 겹치면 새 ID
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

# ========== 설정 ==========
CONFIG = {
    'door_conf_thresh': 0.25,
    'iou_thresh': 0.3,       # 겹침 임계값 (이보다 크면 같은 문)
    'max_lost_frames': 5,    # 몇 프레임 놓치면 잊을지
}

def calculate_iou(box1, box2):
    """(x1, y1, x2, y2) 형식의 두 박스 IoU 계산"""
    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])

    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    inter = w * h

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

class Track:
    def __init__(self, tid, box, cls_id):
        self.id = tid
        self.box = box
        self.cls_id = cls_id # 0:Door, 1:Handle
        self.lost_frames = 0
        self.updated = True

class TrackerV9:
    def __init__(self):
        rospy.init_node("tracker_v9", anonymous=False)
        r = rospkg.RosPack()
        
        model_path = r.get_path('ces') + "/MOT/pretrained/best.pt"
        self.yolo = YOLO(model_path)
        
        self.next_id = 0
        self.tracks = [] # Active Tracks
        
        self.lock = threading.Lock()
        self.image = None
        self.bridge = CvBridge()
        
        # Front Camera Only (Side는 어차피 지나가는 거라 ID 유지 의미 적음)
        rospy.Subscriber("/camera/undistorted/front", Image, self.cb_img)
        
        rospy.loginfo("Tracker V9 (Pure IoU) Started")

    def cb_img(self, msg):
        with self.lock:
            try:
                self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except: pass

    def update(self, img):
        # 1. Detection (No Tracking)
        res = self.yolo.predict(img, conf=CONFIG['door_conf_thresh'], classes=[0,1], verbose=False)[0]
        
        detections = []
        if res.boxes is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            
            for box, cls_id, conf in zip(boxes, clss, confs):
                detections.append({'box': box, 'cls': cls_id, 'conf': conf})
        
        # 2. IoU Matching
        # Reset updated flag
        for t in self.tracks: t.updated = False
        
        # Greedy Matching
        used_dets = [False] * len(detections)
        
        # 기존 트랙들에 대해 가장 잘 겹치는 detection 찾기
        for t in self.tracks:
            best_iou = 0
            best_idx = -1
            
            for i, det in enumerate(detections):
                if used_dets[i]: continue
                if t.cls_id != det['cls']: continue # 클래스 다르면 매칭 X
                
                iou = calculate_iou(t.box, det['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_iou > CONFIG['iou_thresh']:
                # 매칭 성공 -> 업데이트
                t.box = detections[best_idx]['box']
                t.lost_frames = 0
                t.updated = True
                used_dets[best_idx] = True
            else:
                # 매칭 실패 -> Lost 증가
                t.lost_frames += 1
        
        # 3. Create New Tracks
        for i, det in enumerate(detections):
            if not used_dets[i]:
                # 새 트랙 생성
                self.tracks.append(Track(self.next_id, det['box'], det['cls']))
                self.next_id += 1
        
        # 4. Remove Dead Tracks
        self.tracks = [t for t in self.tracks if t.lost_frames < CONFIG['max_lost_frames']]
        
        return self.tracks

    def viz(self, img, tracks):
        viz_img = img.copy()
        for t in tracks:
            # 오래된 유령 트랙은 그리지 않음 (옵션)
            if t.lost_frames > 1: continue
            
            x1, y1, x2, y2 = map(int, t.box)
            if t.cls_id == 0:
                label = f"Door {t.id}"
                color = (0, 255, 0)
            else:
                label = "Handle"
                color = (0, 0, 255)
            
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(viz_img, label, (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        cv2.imshow("Tracker V9 (Pure IoU)", viz_img)
        cv2.waitKey(1)

    def loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            img = None
            with self.lock:
                if self.image is not None:
                    img = self.image.copy()
            
            if img is not None:
                tracks = self.update(img)
                self.viz(img, tracks)
            
            rate.sleep()

if __name__ == "__main__":
    TrackerV9().loop()
