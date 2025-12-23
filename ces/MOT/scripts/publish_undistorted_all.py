#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publishes Undistorted Images for All 4 Cameras
Input:  /camera/image_raw_{front, left, rear, right} (Fisheye / Double Sphere)
Output: /camera/undistorted/{front, left, rear, right} (Pinhole)
"""

import rospy
import rospkg
import cv2
import numpy as np
import yaml
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class Camera:
    def __init__(self, cid, data):
        self.cid = cid
        intr = data['intrinsics']
        # Double Sphere Parameters: [xi, alpha, fx, fy, cx, cy]
        if len(intr) >= 6:
            self.xi, self.alpha, self.fx, self.fy, self.cx, self.cy = intr[:6]
        else:
            raise ValueError(f"Invalid intrinsics for {cid}")
        self.map_x, self.map_y = None, None

    def init_maps(self, w, h):
        if self.map_x is not None: return
        rospy.loginfo(f"[{self.cid}] Initializing Undistort maps ({w}x{h})...")
        
        ncx, ncy = w / 2.0, h / 2.0
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        
        # Pinhole -> Double Sphere Mapping
        uy = (grid_x - ncx) / self.fx
        vy = (grid_y - ncy) / self.fy
        
        r2 = uy**2 + vy**2
        norm = np.sqrt(r2 + 1)
        xs, ys, zs = uy / norm, vy / norm, 1.0 / norm
        
        zs_prime = zs + self.xi
        d2 = np.sqrt(xs**2 + ys**2 + zs_prime**2)
        denom = self.alpha * d2 + (1 - self.alpha) * zs_prime
        
        u_ds = xs / denom
        v_ds = ys / denom
        
        self.map_x = (self.fx * u_ds + self.cx).astype(np.float32)
        self.map_y = (self.fy * v_ds + self.cy).astype(np.float32)

    def undistort(self, img):
        if self.map_x is None:
            self.init_maps(img.shape[1], img.shape[0])
        return cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR)

class UndistortAllNode:
    def __init__(self):
        rospy.init_node('publish_undistorted_all', anonymous=False)
        self.bridge = CvBridge()
        
        # Load Calibration
        r = rospkg.RosPack()
        calib_path = r.get_path('ces') + "/MOT/calibration file/2024-09-13-20-37-00-camchain.yaml"
        self.cameras = self._load_cameras(calib_path)
        
        # Topic Mapping
        # Note: Check save_undistorted_images.py for correct mapping
        # cam0: front, cam1: rear, cam2: left, cam3: right
        self.mappings = [
            ('front', 'cam0'),
            ('rear',  'cam1'),
            ('left',  'cam2'),
            ('right', 'cam3')
        ]
        
        self.pubs = {}
        self.subs = []
        
        for name, cid in self.mappings:
            # Publisher
            pub_topic = f"/camera/undistorted/{name}"
            self.pubs[name] = rospy.Publisher(pub_topic, Image, queue_size=1)
            
            # Subscriber
            sub_topic = f"/camera/image_raw_{name}"
            self.subs.append(
                rospy.Subscriber(sub_topic, Image, self._make_callback(name, cid), queue_size=1)
            )
            rospy.loginfo(f"Mapping {sub_topic} -> {pub_topic} using {cid}")

    def _load_cameras(self, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        cams = {}
        for i in range(4):
            cid = f'cam{i}'
            cams[cid] = Camera(cid, data[cid])
        return cams

    def _make_callback(self, name, cid):
        # Closure for callback arguments
        def callback(msg):
            try:
                img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                camera = self.cameras[cid]
                
                img_u = camera.undistort(img)
                
                msg_u = self.bridge.cv2_to_imgmsg(img_u, "bgr8")
                msg_u.header = msg.header
                
                self.pubs[name].publish(msg_u)
            except Exception as e:
                rospy.logerr_throttle(1, f"Error processing {name}: {e}")
        return callback

if __name__ == '__main__':
    try:
        UndistortAllNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
