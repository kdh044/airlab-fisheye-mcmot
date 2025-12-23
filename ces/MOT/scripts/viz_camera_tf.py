#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import rospkg
import yaml
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R

def get_transform_msg(parent, child, T_mat):
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = parent
    t.child_frame_id = child
    
    t.transform.translation.x = T_mat[0, 3]
    t.transform.translation.y = T_mat[1, 3]
    t.transform.translation.z = T_mat[2, 3]
    
    rot_mat = T_mat[:3, :3]
    r = R.from_matrix(rot_mat)
    q = r.as_quat()
    
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    
    return t

def main():
    rospy.init_node('viz_camera_tf_broadcaster')
    r = rospkg.RosPack()
    calib_path = r.get_path('ces') + "/MOT/calibration file/2024-09-13-20-37-00-camchain.yaml"
    
    with open(calib_path) as f:
        data = yaml.safe_load(f)
        
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    transforms = []
    
    # 올바른 매핑 적용 (cam1=Left, cam2=Rear)
    # T_chain 순서는 cam_ids 순서대로 쌓임
    cam_ids = ['cam0', 'cam1', 'cam2', 'cam3'] 
    # 주의: 여기서 cam_ids 순서대로 chain을 계산하되,
    # TF 이름표(child frame)를 붙일 때 올바른 이름을 매칭해야 함
    
    # cam0 -> cam1 -> cam2 -> cam3 체인 구조라고 가정 (YAML 구조상)
    T_prev = np.eye(4)
    
    # ID별 역할 매핑
    id_to_name = {
        'cam0': 'front',
        'cam1': 'left',   # 반시계 방향
        'cam2': 'rear',
        'cam3': 'right'
    }
    
    # 1. Front (cam0)
    transforms.append(get_transform_msg("rig_link", "front_link", np.eye(4)))
    
    # 2. 나머지 (cam1, cam2, cam3 순서로 체인 연결)
    for cid in ['cam1', 'cam2', 'cam3']:
        T_data = np.array(data[cid].get('T_cn_cnm1', np.eye(4)))
        T_rel = np.linalg.inv(T_data) # 이전 -> 현재
        T_curr = T_prev @ T_rel
        T_prev = T_curr
        
        name = id_to_name[cid]
        tf_msg = get_transform_msg("rig_link", f"{name}_link", T_curr)
        transforms.append(tf_msg)
        
        pos = T_curr[:3, 3]
        rospy.loginfo(f"[{name.upper()} ({cid})] Pos: {pos}")

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        for t in transforms:
            t.header.stamp = rospy.Time.now()
        broadcaster.sendTransform(transforms)
        rate.sleep()

if __name__ == '__main__':
    main()
