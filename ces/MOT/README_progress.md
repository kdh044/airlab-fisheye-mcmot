# Multi-Camera MOT Progress Summary

## 현재 로직 요약
- 4개 카메라(Front/Left/Rear/Right) 토픽을 배치로 받아 YOLOv11 추론.
- 클래스 필터링(문: 1/3/4) + conf threshold 적용.
- DS(Double Sphere) 카메라 모델 파라미터(xi, alpha, fx, fy, cx, cy)로 픽셀을 레이로 복원 후 바닥 평면과 교차 계산.
- 같은 프레임 내 detection은 클러스터링으로 묶고, 클러스터 중심을 트랙과 Hungarian 매칭.
- 트랙 상태는 2D CV Kalman filter로 예측/보정, TTL 지나면 삭제.
- 2x2 시각화 + 글로벌 ID 표시, RViz MarkerArray 발행.

## 최근 변경 사항
- conf threshold(기본 0.5) 추가.
- 클러스터링 + Hungarian 매칭으로 멀티뷰 오버랩 ID 분리 완화.
- DS 모델 역투영 적용(핀홀 수식 제거)으로 ground 좌표 오차 감소.
- Kalman filter(CV 2D) 추가로 ID 흔들림 완화.

## 주요 파라미터
- `_conf_thresh` (default: 0.5)
- `_match_max_dist` (default: 1.0)
- `_cluster_max_dist` (default: match_max_dist)
- `_track_ttl` (default: 5.0)
- `_use_kalman` (default: true)
- `_kalman_q_pos`, `_kalman_q_vel`, `_kalman_r`, `_kalman_init_var`
- `_ground_height` (default: 1.0)

## 주의사항
- 영상 자체는 undistort하지 않고, 좌표 계산만 DS 모델로 보정함.
- 바닥점은 현재 bbox 하단(y2)을 사용 중(마스크 최하단 y로 개선 가능).

## 다음 작업(실험)
- conf 낮춰서 재실험.
- 손잡이(클래스 2)까지 추적 포함.
- 경량화(불필요 연산 제거, LUT/캐시 고려).
- 멀티뷰 ID 통합 안정화(오버랩 게이팅/투영 정확도 개선).
