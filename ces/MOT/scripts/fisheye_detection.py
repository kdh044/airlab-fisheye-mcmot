#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import threading
import time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO


class YOLOSegMultiCam2x2:
    def __init__(self):
        rospy.init_node("yolo_seg_multi_cam_2x2", anonymous=False)

        # ===============================
        # Model (고정 경로 - 도커 내부 절대 경로)
        # ===============================
        self.model_path = "/root/catkin_ws/src/ces/MOT/pretrained/best.pt"
        self.model = YOLO(self.model_path)
        # GPU 사용 시
        # self.model.to("cuda")

        # ===============================
        # Class Info
        # ===============================
        self.class_names = {
            0: "door",
            1: "handle"
        }


        # 클래스별 색상
        self.class_colors = {
            0: (0, 255, 255),   # door
            1: (0, 0, 255),     # handle
        }
        
        self.ignore_class_ids = set()

        # ===============================
        # ROS / CV
        # ===============================
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.images = {
            "front": None,
            "left": None,
            "rear": None,
            "right": None
        }

        # ===============================
        # Subscribers
        # ===============================
        rospy.Subscriber("/camera/image_raw_front", Image, self.cb_front, queue_size=1)
        rospy.Subscriber("/camera/image_raw_left", Image, self.cb_left, queue_size=1)
        rospy.Subscriber("/camera/image_raw_rear", Image, self.cb_rear, queue_size=1)
        rospy.Subscriber("/camera/image_raw_right", Image, self.cb_right, queue_size=1)

        rospy.loginfo("YOLOv11 Segmentation (building ignored) node started")

    # ======================================================
    # Callbacks
    # ======================================================
    def cb_front(self, msg):
        self._update_image("front", msg)

    def cb_left(self, msg):
        self._update_image("left", msg)

    def cb_rear(self, msg):
        self._update_image("rear", msg)

    def cb_right(self, msg):
        self._update_image("right", msg)

    def _update_image(self, key, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.images[key] = img
        except Exception as e:
            rospy.logerr(e)

    # ======================================================
    # Segmentation + Visualization (building 제외)
    # ======================================================
    def run_segmentation(self, img):
        start = time.perf_counter()
        result = self.model(img, verbose=False)[0]
        end = time.perf_counter()

        infer_time_ms = (end - start) * 1000.0

        if result.masks is None:
            return img, infer_time_ms

        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes

        for i in range(len(masks)):
            cls_id = int(boxes.cls[i])

            # ❌ building 완전 배제
            if cls_id in self.ignore_class_ids:
                continue

            if cls_id not in self.class_colors:
                continue

            conf = float(boxes.conf[i])
            mask = masks[i]
            color = self.class_colors[cls_id]

            # ---------- Mask Overlay ----------
            overlay = np.zeros_like(img, dtype=np.uint8)
            overlay[mask > 0.5] = color
            img = cv2.addWeighted(img, 1.0, overlay, 0.4, 0)

            # ---------- Bounding Box ----------
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            label = f"{self.class_names[cls_id]} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                label,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return img, infer_time_ms

    # ======================================================
    # 2x2 Grid
    # ======================================================
    def make_2x2_grid(self, imgs):
        h, w, _ = imgs[0].shape
        resized = [cv2.resize(img, (w, h)) for img in imgs]
        top = np.hstack((resized[0], resized[1]))
        bottom = np.hstack((resized[2], resized[3]))
        return np.vstack((top, bottom))

    # ======================================================
    # Main Loop
    # ======================================================
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

            front, t_front = self.run_segmentation(front)
            left, t_left = self.run_segmentation(left)
            rear, t_rear = self.run_segmentation(rear)
            right, t_right = self.run_segmentation(right)

            rospy.loginfo_throttle(
                1.0,
                f"[Segmentation Time] "
                f"front: {t_front:.1f} ms | "
                f"left: {t_left:.1f} ms | "
                f"rear: {t_rear:.1f} ms | "
                f"right: {t_right:.1f} ms"
            )

            grid = self.make_2x2_grid([front, left, rear, right])
            cv2.imshow("YOLOv11 Segmentation (Doors & Handle Only)", grid)
            cv2.waitKey(1)

            rate.sleep()


if __name__ == "__main__":
    node = YOLOSegMultiCam2x2()
    node.spin()
