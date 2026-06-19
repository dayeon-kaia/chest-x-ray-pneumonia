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
