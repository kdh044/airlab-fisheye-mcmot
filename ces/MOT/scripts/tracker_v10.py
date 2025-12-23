#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tracker V10 - "The Relay" (Hand-over Logic)
===========================================
SLAM의 부정확한 스케일 때문에 ID가 꼬이는 문제를 해결.
오직 "화면상의 위치와 타이밍"만으로 카메라 간 ID를 넘겨줍니다.

로직:
1. [Front] 중앙 트래킹 (ByteTrack/IoU)
   - 문이 왼쪽 끝으로 사라지면? -> "Left 카메라 너 받아!" (Queue에 저장)
   - 문이 오른쪽 끝으로 사라지면? -> "Right 카메라 너 받아!"
2. [Side] 사이드 트래킹
   - Side 카메라의 "진입 영역(Front쪽 가장자리)"에 새 문이 나타나면?
   - Queue를 확인해서 "아까 넘겨준 그 ID"를 낚아챔.
"""

import rospy
import rospkg
import cv2
import numpy as np
import threading
import time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

# ========== 설정 ==========
CONFIG = {
    'door_conf': 0.25,
    'tracker_type': 'bytetrack.yaml',
    
    # Hand-over 설정
    'border_margin': 50,     # 화면 가장자리 몇 픽셀을 "사라짐/나타남" 영역으로 볼 건지
    'handover_time': 2.0,    # 사라진 후 몇 초까지 기억할지 (너무 늦게 나타나면 다른 문)
}

class Track:
    def __init__(self, tid, box, cls):
        self.id = tid
        self.box = box
        self.cls = cls
        self.last_seen = time.time()
        self.state = "ACTIVE" # ACTIVE, HANDING_OVER

class TrackerV10:
    def __init__(self):
        rospy.init_node("tracker_v10", anonymous=False)
        r = rospkg.RosPack()
        model_path = r.get_path('ces') + "/MOT/pretrained/best.pt"
        
        # 각 카메라별 독립 트래커
        self.trackers = {
            'front': YOLO(model_path),
            'left':  YOLO(model_path),
            'right': YOLO(model_path),
            'rear':  YOLO(model_path)
        }
        
        # Hand-over Queues (전달 대기열)
        # 구조: {'id': global_id, 'time': timestamp}
        self.handoff_left = []  # Front -> Left
        self.handoff_right = [] # Front -> Right
        
        # ID 관리
        self.global_tracks = {} # global_id -> Track Info
        
        self.images = {}
        self.lock = threading.Lock()
        self.bridge = CvBridge()
        
        # Subs
        # HARDWARE FIX: SWAP LEFT/REAR
        rospy.Subscriber("/camera/undistorted/front", Image, self._make_cb('front'))
        rospy.Subscriber("/camera/undistorted/right", Image, self._make_cb('right'))
        
        # Topic 'left' is actually Rear -> Store as 'rear'
        rospy.Subscriber("/camera/undistorted/left", Image, self._make_cb('rear'))
        # Topic 'rear' is actually Left -> Store as 'left'
        rospy.Subscriber("/camera/undistorted/rear", Image, self._make_cb('left'))
            
        rospy.loginfo("Tracker V10 (Relay System) Started - LEFT/REAR SWAPPED")

    def _make_cb(self, name):
        def cb(msg):
            with self.lock:
                try:
                    self.images[name] = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                except: pass
        return cb

    def run_yolo(self, cam, img):
        # 1. Track (Local ID 생성)
        res = self.trackers[cam].track(img, persist=True, tracker=CONFIG['tracker_type'],
                                       conf=CONFIG['door_conf'], classes=[0,1], verbose=False)[0]
        
        tracks = []
        if res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int)
            clss = res.boxes.cls.cpu().numpy().astype(int)
            
            for box, lid, cls in zip(boxes, ids, clss):
                if cls == 1: continue # Handle skip
                tracks.append({'lid': lid, 'box': box})
                
        return tracks

    def process_front(self, img):
        local_tracks = self.run_yolo('front', img)
        h, w = img.shape[:2]
        
        display_tracks = []
        
        for trk in local_tracks:
            lid = trk['lid']
            box = trk['box']
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            
            # Front는 Local ID = Global ID로 씀 (기준점)
            gid = lid 
            
            # [Hand-over Check]
            # 왼쪽으로 사라짐 (x < margin)
            if cx < CONFIG['border_margin']:
                self.add_handoff(self.handoff_left, gid)
                
            # 오른쪽으로 사라짐 (x > w - margin)
            elif cx > w - CONFIG['border_margin']:
                self.add_handoff(self.handoff_right, gid)
            
            display_tracks.append((box, gid))
            
        return display_tracks

    def add_handoff(self, queue, gid):
        now = time.time()
        # 이미 큐에 있으면 시간만 갱신
        for item in queue:
            if item['id'] == gid:
                item['time'] = now
                return
        # 없으면 추가
        queue.append({'id': gid, 'time': now})
        # print(f"DEBUG: Handing over ID {gid}")

    def process_side(self, cam, img, handoff_queue):
        local_tracks = self.run_yolo(cam, img)
        h, w = img.shape[:2]
        display_tracks = []
        
        # Side 카메라는 Front에서 넘어온 놈인지 확인해야 함
        # Front와 Side의 관계:
        # Front의 Left Edge -> Left Cam의 Right Edge로 진입
        # Front의 Right Edge -> Right Cam의 Left Edge로 진입
        
        for trk in local_tracks:
            lid = trk['lid'] # Side만의 Local ID
            box = trk['box']
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            
            gid = f"{cam[0].upper()}{lid}" # 기본: L1, R1...
            
            # [Hand-over Receive]
            # 진입 영역에 있는지 확인
            is_entering = False
            if cam == 'left' and cx > w - CONFIG['border_margin'] * 2: # Left캠의 오른쪽에서 등장
                is_entering = True
            elif cam == 'right' and cx < CONFIG['border_margin'] * 2: # Right캠의 왼쪽에서 등장
                is_entering = True
                
            if is_entering:
                # 큐에서 가장 최근에 넘겨준 ID 가져오기
                matched_gid = self.check_queue(handoff_queue)
                if matched_gid is not None:
                    # 매칭 성공! (Front ID 승계)
                    # 주의: 한 번 매칭되면 계속 기억해야 하는데, 
                    # V10 단순화를 위해 여기서는 "진입 순간"에만 매칭함. 
                    # (완벽하려면 Local->Global Map 필요)
                    gid = matched_gid 
            
            display_tracks.append((box, gid))
            
        return display_tracks

    def check_queue(self, queue):
        now = time.time()
        # 시간 지난 거 청소
        queue[:] = [x for x in queue if now - x['time'] < CONFIG['handover_time']]
        
        if len(queue) > 0:
            # 가장 최근 것 리턴 (LIFO 유사)
            return queue[-1]['id']
        return None

    def viz(self, imgs, tracks_map):
        # 2x2 Grid
        grid_h, grid_w = 480, 640
        canvases = {}
        
        for cam in ['front', 'left', 'rear', 'right']:
            if imgs.get(cam) is not None:
                img = cv2.resize(imgs[cam], (grid_w, grid_h))
                # Draw
                if cam in tracks_map:
                    for box, gid in tracks_map[cam]:
                        x1, y1, x2, y2 = map(int, box)
                        
                        # ID Color Logic
                        if isinstance(gid, int): # Front ID
                            color = (0, 255, 0)
                            label = f"ID: {gid}"
                        else: # Side Local ID
                            color = (0, 165, 255)
                            label = f"{gid}"
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, label, (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Label
                cv2.putText(img, cam.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                canvases[cam] = img
            else:
                canvases[cam] = np.zeros((grid_h, grid_w, 3), np.uint8)

        top = np.hstack((canvases['front'], canvases['left'])) # Front, Left(실제위치 배치)
        bot = np.hstack((canvases['right'], canvases['rear'])) # Right, Rear
        
        cv2.imshow("Tracker V10 (Relay)", np.vstack((top, bot)))
        cv2.waitKey(1)

    def loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            with self.lock:
                imgs = {k: v.copy() for k, v in self.images.items()}
            
            tracks_map = {}
            
            if 'front' in imgs:
                tracks_map['front'] = self.process_front(imgs['front'])
            
            if 'left' in imgs:
                tracks_map['left'] = self.process_side('left', imgs['left'], self.handoff_left)
                
            if 'right' in imgs:
                tracks_map['right'] = self.process_side('right', imgs['right'], self.handoff_right)
                
            if 'rear' in imgs:
                # Rear는 그냥 보여주기만 (Handover 로직 생략)
                tracks_map['rear'] = self.process_side('rear', imgs['rear'], [])

            self.viz(imgs, tracks_map)
            rate.sleep()

if __name__ == "__main__":
    TrackerV10().loop()
