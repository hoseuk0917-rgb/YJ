# DACON 모기 미래 좌표 예측 회고 — 표준 고급 루트 체크리스트

작성일: 2026-06-14

## 1. 결론 요약

이번 대회에서 0.683~0.684 근처에서 막힌 핵심 이유는 문제를 너무 오래 **후처리·수식 후보 선택·public score 미세조정 문제**로 본 것이다.

1등 공유 코드를 확인한 뒤의 결론은 다르다. 상위권 접근은 특이한 비밀 트릭이라기보다, 시계열 회귀 대회에서 자주 쓰이는 **표준 고급 루트**였다.

핵심 루트는 다음과 같다.

```text
CV baseline
→ CV residual target
→ 마지막 속도 방향 기준 local/yaw frame 정렬
→ train 내부 window 전이 증강
→ metric-aware soft R-Hit loss
→ GRU / ODE / gray-box physics 등 복수 모델
→ seed ensemble + TTA
```

이번 프로젝트에서는 이 루트를 초반 우선순위로 두지 못했고, 후처리·게이트·후보 조합을 지나치게 오래 탐색했다.

---

## 2. 대회 문제 구조에서 바로 읽었어야 하는 신호

대회 데이터는 다음 구조였다.

- 각 샘플은 40ms 간격의 11개 3D 좌표
- 입력 시점: -400ms, -360ms, ..., -40ms, 0ms
- 예측 타깃: 마지막 관측 기준 +80ms 좌표
- 좌표계: sensor-local 3D 좌표계
- 평가: 예측 좌표와 정답 좌표의 3D 거리 `<= 0.01m` 여부의 평균 hit rate

여기서 초반에 바로 잡았어야 할 관찰은 다음이다.

```text
라벨은 마지막 시점 +80ms 하나뿐이지만,
train 입력 11개 안에도 동일한 +80ms 전이 구조가 여러 개 숨어 있다.
```

즉 1개 sample에서 단순히 최종 label 하나만 쓰지 말고, 내부 시점 `e`에서 `e+2`를 타깃으로 삼아 추가 학습 예시를 만들 수 있었다.

예시:

```text
e = 5, 6, 7, 8
입력: 과거 window up to e
타깃: X[e+2]
```

이 접근은 test를 학습에 쓰는 것이 아니라 train 내부 관측값을 auxiliary transition target으로 쓰는 것이므로, 대회 규칙의 “test는 학습에 사용 금지”와 충돌하지 않는다.

---

## 3. 표준 고급 루트 1 — Baseline residualization

강한 baseline이 있을 때 절대 좌표를 직접 학습하는 것보다 보통 더 안정적인 방식은 다음이다.

```text
base_pred = constant_velocity(X)
residual = true_position - base_pred
model learns residual
final_pred = base_pred + model_residual
```

이번 문제에서 CV는 이미 매우 강한 prior였다. 따라서 모델은 전체 위치를 다시 맞히기보다, CV가 틀리는 잔차만 배워야 했다.

우리가 놓친 점:

```text
CV가 강하다 → 모델이 필요 없다
```

가 아니라,

```text
CV가 강하다 → 모델은 CV residual만 배워야 한다
```

였다는 점이다.

---

## 4. 표준 고급 루트 2 — Local frame / yaw alignment

sensor-local 좌표라고 해도, 샘플마다 마지막 이동 방향이 다르다. 절대 x/y/z 좌표계에서 residual을 학습하면 같은 운동 패턴도 방향에 따라 전혀 다른 패턴처럼 보인다.

따라서 마지막 속도 방향을 기준으로 좌표계를 회전시켜야 한다.

```text
v_last = X[-1] - X[-2]
yaw = atan2(v_last_y, v_last_x)
rotate input sequence into last-velocity-aligned frame
rotate residual target into the same local frame
```

효과:

- 모델이 “앞/옆/위” 기준의 공통 패턴을 학습할 수 있음
- 샘플별 방향 차이가 줄어듦
- residual 분포가 더 단순해짐

이번 프로젝트에서는 local frame류 분석을 후처리 관점으로 일부 했지만, 모델 입력·타깃 전체를 local frame으로 바꾸는 supervised route를 초반 핵심 루트로 잡지 못했다.

---

## 5. 표준 고급 루트 3 — Internal transition pretraining / augmentation

11개 관측점이 있으면, train label 하나 외에도 내부 전이 예제가 생긴다.

기본 구조:

```text
원래 문제:
X[-11:0] → y(+80ms)

내부 전이:
X[:e] → X[e+2]
```

장점:

- 10,000개 샘플을 사실상 수만 개 전이 샘플로 확장
- 모기 운동의 짧은 시간 동역학을 더 많이 학습
- 최종 label만 쓰는 supervised 학습보다 일반화가 좋아질 가능성 큼

주의:

- 내부 전이와 최종 label의 분포 차이를 확인해야 함
- 내부 transition만 과하게 학습하면 최종 +80ms label 예측과 어긋날 수 있음
- pretraining 후 final target fine-tuning 또는 loss weighting이 필요

---

## 6. 표준 고급 루트 4 — Metric-aware loss

평가는 평균 L2 distance가 아니라 hit boundary다.

```text
distance <= 0.01m 이면 hit
아니면 miss
```

따라서 MSE/Huber만 쓰면 평균적으로 좋아져도 R-Hit가 안 오를 수 있다. 상위권식 접근은 보통 지표를 부드럽게 근사한 soft loss를 추가한다.

예시 개념:

```text
soft_hit = sigmoid((0.01 - distance) / tau)
loss = regression_loss + lambda * (1 - soft_hit)
```

의미:

- 1cm boundary 근처 sample에 더 민감해짐
- 평균거리 개선보다 hit 전환에 직접 최적화
- 후처리로 boundary row를 고치려는 것보다 학습 단계에서 반영하는 쪽이 강함

이번 프로젝트에서는 hit boundary를 주로 postprocessing/gating 대상으로 봤고, loss 설계에 초반부터 넣지 못했다.

---

## 7. 표준 고급 루트 5 — Model family ensemble

상위권 접근은 단일 모델 하나보다 서로 다른 inductive bias를 가진 모델을 섞는 경우가 많다.

이 문제에서 적합한 family 예시:

```text
GRU / LSTM / TCN:
  짧은 시계열 패턴 학습

Neural ODE:
  연속 시간 동역학 근사

Gray-box physics / HyperPhysics:
  CV, acceleration, 회전, damping 등 물리 prior를 neural gate로 조절

MLP / Tabular model:
  수작업 feature 기반 residual correction
```

중요한 점은 ODE나 physics model 하나가 단독으로 Public에서 약해도, ensemble diversity로는 가치가 있을 수 있다는 것이다.

이번 프로젝트에서는 ODE를 단독 후보 또는 보정 후보로 보고 빨리 닫았다. 하지만 상위권 루트에서는 ODE를 “단독 정답”이 아니라 30-model blend의 다양성 구성원으로 사용했다.

---

## 8. 표준 고급 루트 6 — TTA와 symmetry augmentation

좌표계가 sensor-local이고 y축이 left라면, 좌우 대칭성 또는 y-flip test-time augmentation을 검토할 수 있다.

개념:

```text
원본 입력 예측
+ y축 flip 입력 예측 후 inverse flip
→ 평균
```

주의:

- 모든 축에 대해 대칭이 성립하는 것은 아님
- sensor-local 좌표와 환경 차이 때문에 실제 효과는 OOF로 검증 필요
- 하지만 짧은 궤적 문제에서는 방향/좌우 편향 완화에 도움이 될 수 있음

---

## 9. 이번 프로젝트에서 실제로 너무 오래 판 축

다음 축들은 의미가 없었던 것은 아니지만, 0.70대 돌파의 핵심 루트는 아니었다.

```text
- 수식 후보 스캔
- HSR/q96/q97/r14 계열 미세 조정
- public feedback 기반 row surgery
- candidate bank selector
- KNN residual rescue
- overlap / leakage / exact match probe
- horizon surface / local formula patch
- constant turn hand-crafted physics
- endpoint denoise 단독 수식
```

이들은 CV baseline 근처를 몇 점 개선하거나 public에 맞춘 candidate를 찾는 데는 쓸 수 있지만, 0.683 → 0.703 같은 점프를 만들기 어렵다.

---

## 10. 다음 유사 대회 초반 체크리스트

시계열 회귀·좌표 예측 대회에서 첫 1~2일 안에 반드시 확인할 것.

### A. 문제 변환

```text
[ ] 강한 baseline이 있는가?
[ ] target을 absolute coordinate가 아니라 residual로 바꿀 수 있는가?
[ ] 좌표계를 local frame으로 정렬할 수 있는가?
[ ] 예측 horizon이 고정되어 있는가?
```

### B. 데이터 증강

```text
[ ] train 내부 window에서 동일 horizon pseudo target을 만들 수 있는가?
[ ] flip / rotation / scale 등 symmetry augmentation이 가능한가?
[ ] noise augmentation이 실제 sensor noise 가정과 맞는가?
```

### C. Loss 설계

```text
[ ] 평가지표가 MSE/MAE와 다른가?
[ ] hit boundary, rank metric, threshold metric이면 soft approximation을 loss에 넣었는가?
[ ] MSE/Huber + metric-aware loss를 같이 썼는가?
```

### D. 모델 family

```text
[ ] GRU/LSTM/TCN 계열 sequence model을 돌렸는가?
[ ] baseline residual MLP/tabular model을 돌렸는가?
[ ] physics prior를 neural gate로 조절하는 gray-box model을 고려했는가?
[ ] Neural ODE는 단독 제출이 아니라 ensemble diversity로 평가했는가?
```

### E. Ensemble

```text
[ ] seed ensemble을 했는가?
[ ] architecture ensemble을 했는가?
[ ] TTA를 했는가?
[ ] OOF에서 ensemble diversity를 봤는가?
```

### F. 후처리 순서

```text
[ ] supervised residual model 이전에 후처리에 빠지지 않았는가?
[ ] postprocessing은 최종 5~10% 튜닝으로 제한했는가?
[ ] public score 기반 row surgery는 마지막 수단으로만 썼는가?
```

---

## 11. 의사결정 규칙

앞으로 유사 대회에서 아래 순서로 강제한다.

```text
1일차:
  baseline 재현
  residual target 정의
  local frame 변환
  내부 window 증강 가능성 확인

2일차:
  GRU/TCN/MLP residual model OOF
  soft metric loss prototype
  seed 3개 ensemble

3일차:
  physics/ODE/gray-box model 추가
  architecture ensemble

4일차 이후:
  candidate selector / postprocessing / public feedback 분석
```

금지 규칙:

```text
강한 supervised residual model을 만들기 전에는
후처리 grid search를 대량으로 돌리지 않는다.
```

---

## 12. 이번 회고의 핵심 문장

이번 대회의 실패는 “세부 후보를 덜 뒤져서”가 아니라, **상위권 표준 고급 루트를 초반에 우선순위로 놓지 못한 것**이다.

다음에는 강한 baseline이 보이면 곧바로 다음 질문을 던진다.

```text
이 baseline을 대체할 것인가?
아니면 이 baseline의 residual을 local frame에서 학습할 것인가?
```

대부분의 경우 정답은 두 번째다.
