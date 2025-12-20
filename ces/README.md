# Fisheye Camera based MOT - Entrance (CES 2026)

**개발자:** 김동훈

이 프로젝트는 CES 2026 출품작인 스냅스페이스(SnapSpace)의 기능 모듈입니다. 4대의 어안(Fisheye) 카메라를 활용하여 실내 출입문(Entrance)을 실시간으로 검출하고, 여러 카메라에서 중복 검출되는 문을 전역 좌표계(Global Coordinate) 상에서 유일한 ID로 통합하여 추적(Tracking)하는 것을 목표로 합니다.

##  기능

### 1. Multi-view Fisheye Detection (검출부)
- **소스 코드:** [fisheye_detection.py](./CES/MOT/fisheye_detection.py)
- 4채널 어안 카메라 영상에서 **YOLOv11 Segmentation**을 이용해 문(Glass/Metal/Wood Door)과 손잡이(Handle)를 실시간으로 검출합니다.
- 건물(Building) 등 불필요한 클래스는 제외하고, 필요한 객체의 2D Bounding Box와 Segmentation Mask 정보를 추출합니다.

### 2. Multi-camera Global Tracking (추적부)
- **핵심 목표:** 검출부에서 획득한 2D 정보(박스, 클래스, 마스크 등)만을 활용하여 트래킹을 구현합니다.
- 여러 카메라 시야각이 겹치는 영역에서 동일한 문이 중복 검출될 경우, 이를 하나로 통합합니다.
- 전역 좌표(Global Map) 상에서 각 문에 유일한 고유 번호(ID)를 부여하고 유지합니다.

![ces_door_detection.png](ces_door_detection.png)
*(여러 카메라에서 출입문이 검출되는 상황)*

![multicam_entrance_detection.png](multicam_entrance_detection.png)
*(멀티 카메라 검출 예시)*

---

# 진행 상황

## 12월 18일
- 3D MOT 기술 조사 (ADA-Track, EarlyBird 등 관련 논문 및 오픈소스 검토).
- 데이터셋(Wildtrack, nuScenes) 구조 파악 및 개발 환경 설정.

## 12월 19일
- **프론트엔드 코드 분석:** `fisheye_detection.py` (ROS 기반 4채널 YOLOv11) 코드 구조 파악 및 테스트 준비.