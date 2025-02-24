# 로지스틱 회귀를 이용한 의료보험 사기탐지 🔍

## 1. 목표 및 프로젝트 기간 

### 1.1 목표 👍🏻
- 보험청구 과정에서 발생하는 사기 행위 근절 
- 보험사의 재정손실 예방 
- 정직한 가입자의 보험료 상승 방지 
- 사기 탐지의 자동화 
- 보험 시스템 신뢰도 상승

### 1.2 프로젝트 기간 ⏳
#### Dec 2024 - Jan 2025 

## 2. 개발 환경 및 개발 방법 👩🏻‍💻
### 2.1 개발 환경
- Jupyter Notebook 
  - colab

### 2.2 데이터셋 
- https://www.kaggle.com/datasets/nyashachizampeni/medical-insurance-claim-fraud
- 14개의 독립변수와 1개의 종속변수(사기여부)
- 결측치 없음

### 2.3 데이터 전처리
- One-hot encoding 활용
  - 성별, 사고 원인과 같은 범주형 변수를 수치형 변수로 변환
- 환자의 생년월일 데이터를 age 데이터로 변환 (나이 데이터를 새로 만들어 추후에 연령대별 사기 비율을 알아야 할 상황에 대비함)
- 표준화와 정규화
  - 데이터 분포 조정을 통해 특정 독립변수가 다른 독립변수의 비해 지나치게 큰 값을 가지는 문제를 해결 
- 차원축소 (주성분 분석, 선형 판별 분석)
  - 주성분 분석 : 데이터 분산이 가장 큰 축을 찾아 그 방향으로 차원을 축소
  - 선형 판별 분석 : 클래스 간 분리를 최대화하면서 데이터 차원을 감소

### 2.4 모델 생성 



## 3. 성과 📝
### 71차 한국컴퓨터정보학회 동계학술대회 2024 우수논문 수상 및 발표 
![우수논문수상연락사진](pictures/컴퓨터정보학회우수논문.png)

