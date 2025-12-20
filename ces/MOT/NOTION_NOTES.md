# MCMOT 작업 메모 (Notion용)

## 배경/문제 정의
- 멀티 카메라에서 같은 문이 보일 때, 전역 ID가 일관되게 유지되어야 함.
- Re-ID는 무겁고 문 외형 구분이 어려움 → 기하 기반 매칭이 현실적.
- 기존 ID 분리의 핵심 원인은 핀홀 수식을 어안(DS 모델) 영상에 그대로 적용한 좌표 오차.

## 현재 파이프라인 상세
1) Input
- /camera/image_raw_front|left|rear|right 구독
- 4장 배치로 YOLO 추론

2) Detection 필터
- 문 클래스(1,3,4)만 사용
- conf threshold 적용(`_conf_thresh`)

3) Ground-plane Projection (DS 모델)
- camchain.yaml의 DS intrinsics: [xi, alpha, fx, fy, cx, cy]
- 픽셀 → ray 복원(DS 모델) → 바닥 평면과 교차
- 영상 undistort는 하지 않음 (좌표 계산만 보정)

4) 프레임 내 클러스터링
- 같은 프레임에서 서로 가까운 detection을 클러스터로 묶음
- 오버랩 카메라에서 같은 문이 보일 때 ID 분리 방지

5) Hungarian 매칭
- 클러스터 중심 ↔ 기존 트랙 거리 cost
- `_match_max_dist` 이내 매칭만 허용

6) Kalman filter (CV 2D)
- state: [x, z, vx, vz]
- 매 프레임 predict → 매칭 시 update
- TTL 이후 트랙 삭제

7) Visualization
- 2x2 그리드(Front/Left/Rear/Right)
- 마스크/박스/ID 표시
- RViz MarkerArray 발행

## 개선/튜닝 포인트
- 바닥점: bbox 하단(y2) → mask 최하단 y로 교체
- cam_overlaps 기반 게이팅으로 불필요 매칭 제거
- DS 기반 LUT로 픽셀→지면 좌표 가속화 + 정밀도 개선
- match/cluster dist 튜닝

## 파라미터
- `_conf_thresh` (0.5)
- `_match_max_dist` (1.0)
- `_cluster_max_dist` (match_max_dist)
- `_track_ttl` (5.0)
- `_use_kalman` (true)
- `_kalman_q_pos`, `_kalman_q_vel`, `_kalman_r`, `_kalman_init_var`
- `_ground_height` (1.0)

## 내일 할 일
- conf threshold 낮춰서 성능/ID 안정성 비교
- 손잡이 클래스(2) 포함 검출
- 스크립트 경량화
- 멀티뷰 ID 통합 안정화(오버랩 개선)
