# Fisheye Camera based MOT - Entrance (CES 2026)

**개발자:** 김동훈

이 프로젝트는 CES 2026 출품작인 스냅스페이스(SnapSpace)의 기능 모듈입니다. 4대의 어안(Fisheye) 카메라를 활용하여 실내 출입문(Entrance)을 실시간으로 검출하고, 여러 카메라에서 중복 검출되는 문을 전역 좌표계(Global Coordinate) 상에서 유일한 ID로 통합하여 추적(Tracking)하는 것을 목표로 합니다.

## 개발 기능

### 1. Multi-view Fisheye Detection (검출부)
- **소스 코드:** [fisheye_detection.py](./ces/MOT/scripts/fisheye_detection.py)
- 4채널 어안 카메라 영상에서 **YOLOv11 Segmentation**을 이용해 문(Glass/Metal/Wood Door)과 손잡이(Handle)를 실시간으로 검출합니다.
- 건물(Building) 등 불필요한 클래스는 제외하고, 필요한 객체의 2D Bounding Box와 Segmentation Mask 정보를 추출합니다.

### 2. Multi-camera Global Tracking (추적부)
- **소스 코드:** [multi_cam_tracker.py](./ces/scripts/multi_cam_tracker.py)
- **핵심 목표:** 여러 카메라 시야각이 겹치는 영역에서 동일한 문이 중복 검출될 경우, 이를 하나로 통합하고 전역 좌표상에서 유일한 ID를 부여합니다.

#### 작동 원리 (Mechanism)
1.  **배치 추론 (Batch Inference):** 4개 카메라(Front, Left, Rear, Right)의 영상을 동시에 받아 YOLOv11로 한 번에 처리합니다.
2.  **3D 투영 (3D Projection):**
    *   Depth 정보가 없는 환경이므로 **바닥 평면 가정(Ground Plane Assumption)**을 사용합니다.
    *   `camchain.yaml`의 캘리브레이션 정보(Intrinsics, Extrinsics)를 로드합니다.
    *   검출된 문 박스의 밑변 중앙점(발 부분)을 3D Ray로 쏘아 바닥(Z=0)과 만나는 지점을 계산합니다.
3.  **좌표계 통일 (Unified Coordinate System):** 모든 카메라의 3D 좌표를 `cam0` (Front Camera) 기준 좌표계로 변환합니다.
4.  **ID 매칭 (ID Association):**
    *   각 카메라에서 검출된 3D 점들 간의 **유클리드 거리(Euclidean Distance)**를 계산합니다.
    *   거리가 가까운(예: 1.0m 이내) 객체들은 "같은 문"으로 간주하여 동일한 Global ID를 부여합니다.

---

# 사용법 (How to Run)

## 1. Docker 환경 실행
```bash
cd ~/3dmot/ces
./run_docker.sh
```

## 2. 데이터 재생 (터미널 #1)
도커 내부에서 Bag 파일을 재생하여 가상의 카메라 데이터를 송출합니다.
```bash
docker exec -it ces /bin/bash  # 도커 접속
roscore &                      # (백그라운드 실행)
cd /root/catkin_ws/src/ces/MOT/bag
rosbag play *.bag -l           # 무한 재생
```

## 3. 트래커 실행 (터미널 #2)
MCMT 트래킹 노드를 실행합니다. (Rviz 마커 발행)
```bash
docker exec -it ces /bin/bash  # 새 터미널
devel                          # 환경변수 로드 (alias)
rosrun ces multi_cam_tracker.py
```

## 4. 시각화 (Host)
호스트 PC에서 Rviz를 실행하여 결과를 확인합니다.
```bash
rviz
# Fixed Frame: camera_link_front
# Add -> MarkerArray -> Topic: /snapspace/markers
```

---

# 진행 상황

## 12월 19일 (완료)
- **개발 방향 확정:** ADA-Track(무거움) 대신 **YOLOv11 + 기하학적 3D 매칭** 방식 채택.
- **프론트엔드 완료:** Docker + ROS Noetic + YOLOv11 환경 구축 및 실시간 검출 성공.
- **백엔드 구현:** `multi_cam_tracker.py` 작성 완료.
    - 캘리브레이션 파일 파싱 및 Extrinsics 적용.
    - 바닥 평면 가정을 통한 2D -> 3D 좌표 변환 구현.
    - 거리 기반 ID 통합 로직 구현.

## 🚀 To-Do (Next Steps)
- [ ] **Rviz 시각화 검증:** 실제 Bag 데이터로 3D 마커가 올바른 위치에 찍히는지 확인.
- [ ] **정확도 튜닝:** `ROBOT_HEIGHT` (카메라 높이) 및 캘리브레이션 파라미터 미세 조정.
- [ ] **SLAM 연동:** `tf`를 구독하여 로봇이 이동해도 문 좌표가 지도(Map)에 고정되도록 업그레이드.