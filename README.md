# Chest X-ray Pneumonia Classification

## 프로젝트 개요
흉부 X-ray 이미지를 활용해 정상(NORMAL)과 폐렴(PNEUMONIA)을 분류하는 CNN 기반 딥러닝 모델 개발

## 데이터셋
- 출처: Kaggle - Chest X-Ray Images (Pneumonia) by Paul Mooney
- 규모: 총 5,863장 (NORMAL / PNEUMONIA 이진 분류)
- 재분할: 원본 val(16장) → train에서 stratified 15% 분리 (train 4,433 / val 783 / test 624)

## 주요 발견 (EDA)
- 클래스 불균형: PNEUMONIA가 train의 약 74% 차지 → class weight 적용
- 이미지 해상도 편차 큼 (384~2720px) → 224×224 리사이즈 통일
- PNEUMONIA 이미지에 의료장비(튜브, 패치) 동반 → 모델 편향 위험 존재

## 모델
| 모델 | 특징 |
|------|------|
| Simple CNN | 직접 설계한 베이스라인 (Conv 4블록) |
| EfficientNetB0 | ImageNet 사전학습 기반 Transfer Learning |

## 기술 스택
- Python 3.14
- PyTorch (MPS 가속 - Apple M2)
- torchvision, scikit-learn, matplotlib, seaborn

## 파일 구조
```
chest-x-ray-pneumonia/
├── eda_modeling.ipynb   # EDA + 모델 학습 전체 코드
├── data/                # 데이터셋 (git 제외)
└── README.md
```
## 학습 결과

| 모델 | Test Accuracy | NORMAL Recall | PNEUMONIA Recall | NORMAL F1 | PNEUMONIA F1 |
|------|---------------|----------------|-------------------|-----------|---------------|
| Simple CNN | 75% | 37% | 97% | 0.52 | 0.83 |
| EfficientNetB0 | 83% | 56% | 99% | 0.71 | 0.88 |

- EfficientNetB0가 모든 지표에서 Simple CNN을 상회
- 두 모델 모두 PNEUMONIA recall은 높지만(과소진단 위험 낮음), NORMAL recall이 낮아 정상을 폐렴으로 오진하는 경향 존재 → 향후 개선 필요 지점

## Grad-CAM 해석
- PNEUMONIA 예측 시 폐 하단부(하엽)에 집중 → 임상적으로 타당한 근거
- 일부 NORMAL 예측에서 심장 부근에 집중되는 경향 → 폐 영역 외 단서에 의존할 가능성

## 한계점
- Train 데이터 클래스 불균형 (PNEUMONIA가 NORMAL의 약 2.9배)
- PNEUMONIA 이미지에 의료장비 아티팩트 동반 → 모델 편향 가능성
- 원본 데이터셋이 1~5세 소아 환자 한정 → 성인 데이터 일반화 어려움
- Grad-CAM 해석이 항상 임상적으로 일관되지는 않음

## 향후 개선 방향
- NORMAL 클래스 데이터 증강 또는 오버샘플링으로 recall 개선
- 의료장비가 없는 이미지만 따로 학습/평가해 아티팩트 영향 검증
- 성인 X-ray 데이터셋 추가로 일반화 성능 확인
