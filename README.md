## # 📈 분류 분석 프로젝트  

## 심부전 질환 환자 예측 분류 분석 프로젝트
<img src="https://newsimg-hams.hankookilbo.com/2022/11/15/74f4c2d2-141e-45c7-99c2-60b420954018.jpg">  

  ### 📌 데이터 세트 주제 
  - 환자들의 각종 질병들을 활용하여 심부전 질환 환자들을  분류 분석하여 예측합니다.
  #### 📌 Feature별 설명
  - Age : 나이
  - Sex: 성별  
  - ChestPainType : 흉통 타입
  - RestingBP : 휴식시 혈압
  - ChestPainType : 흉통 타입
  - RestingBP : 휴식 시 혈압
  - Cholesterol : 콜레스테롤
  - FastingBS : 빠른 혈압 환자
  - RestingECG : 휴식 시 혈압 심전도 타입
  - MaxHR : 최대 심박수
  - ExerciseAngina : 협심증# ## # ### ## # 
  - Oldpeak : 운동 시 심전도 기울기
  - ST_Slope : 운동으로 인한 심박수 증가율 기울기
  - HeartDisease : 심부전 질환 유무
    
  ### ✏️ 심부전 질환 예측 분류 프로젝트 진행 방향성
  - [데이터 전처리 (결측치, 중복된 데이터, 이상치 등 제거 및 일반화 작업)](#전처리-작업)
  - [독립변수와 종속변수들의 상관관계 확인](#correlation-종속변수와의-상관관계-분석)
  - [분류 분석 실시 (1 ~ 7 Cycle)](#📌-전처리-완료)
  - [1 Cycle - Logistic Regression (로지스틱회귀분석)으로 작업](#1-Cycle)
  - [2 Cycle - lda 차원축소 후 로지스틱 회귀분석 ](#2-Cycle)
  - [3 Cycle - 오차행렬 정리 및 그에 따른 임계치 조정 ](#3-Cycle)
  - [4 Cycle - Feature 중요도 확인](#4-Cycle)
  - [5 Cycle - 중요한 Feature들만 있는 데이터세트로 과적합 여부 확인](#5-Cycle)
  - [6 Cycle - feature 제거 후 데이터 vs 차원축소 데이터) 분석](#6-Cycle)
  - [7 Cycle - 나머지 분류모델들과 로지스틱회귀 모델의 성능 비교](#7-Cycle)
  - [최종 결론](#Total-Result)

## 데이터세트(csv파일 PNG) <USA House Price Predict>
<img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/ee6ab461-bab9-44c7-8cc3-a889fd50c21e' width="600px">

## 전처리 작업
- ✏️ 해당 데이터세트 에서는 결측치 및 중복된 데이터가 없었으며 범주형 데이터들은 모두 등급으로 형태가 잡혀있는 구조여서 LabelEncoding 작업을 하여서 수치형으로 바꿔주었습니다.  
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/6d53f1b2-6cf8-4a3c-beee-243690ccb313" width="600px">

## correlation 종속변수와의 상관관계 분석
<img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/54b8c5c4-f8aa-42bf-ae4f-bd5e5ab6da63' width="600px">  

위 이미지와 같이 ST_Slope (ST 분절)가장 상관관계가 높았으며, 가장 상관관계가 낮은 Feature는 RestingECG (안정을 취했을 때, 심전도) 였습니다.   
후에 성능이 좋게 나온 회귀모델에서 Feature들의 permution Importance를 산출해서 작업하겠습니다.   

[4 Cycle 로 이동](#4-Cycle)

### 📌 전처리 완료  

## 1 Cycle  
> ### pytorch로 Logistic Regression (로지스틱회귀분석) 으로 작업

- #### Validatation Data & TEST Data의 loss 값을 비교하여 과적합 분석 시각화 
  <img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/41204183-6357-400a-94d8-09582051d408' width="800px" style="margin-bottom:10px">  
   

> Pytorch로 훈련데이터와 검증데이터와의 loss 값을 비교해보았을 때
둘 다 큰 차이가 발생하지 않았으므로, 해당 데이터는 과적합이 발생하지 않았음을 알 수 있습니다.  

Train Data loss: 0.4127 , Test Data loss: 0.4088

    
- #### pytorch-LogisticRegression 훈련 결과 
  <img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/7a9d4d51-85cc-49c8-b3e5-7be7faefc4eb' width="800px" style="margin-bottom:10px">  
  
> 위 이미지와 같이 정확도가 0.8696, F1 Score가 0.8537로 높게 나타난 것을 알 수 있었으며, pytorch에서 epoch와 learning_rate를 적절하게 준 뒤 과적합도 방지 할 수 있었습니다.  
  
- #### Sklearn을 활용해서 로지스틱회귀분석
<img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/4c645f36-927c-4d6c-a32d-618645ee46bb' width="800px" style="margin-bottom:10px">  

- 성능 점수 산출
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/88613556-8ccb-449d-a353-9211e854db26" width="800px">


### 📃 1 Cycle Result
>Pytorch에서 시그모이드 함수를 사용한 결과와 sklearn에서 Logistic Regression을 사용한 결과가 비슷한 수치를 보이지만

> pytorch로 로지스틱회귀분석을 하였을 때가 더 성능이 높은 것을 알 수 있었습니다.  

> 결론적으로 로지스틱회귀분석을 사용했을 때, 모든 성능점수가 0.8를 상회하는 것을 볼 수 있으므로, 해당 데이터에서는 로지스틱 회귀기법이 좋은 성능을 보입니다.

## 2 Cycle
> ### 차원 축소를 한 다음, 로지스틱 회귀 진행
> lda 차원축소 후 로지스틱 회귀분석 진행 (**후에 로지스틱회귀에서 나온 permutation importance로 중요 Feature 선별해서 차원축소도 진행해보겠습니다.**)
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/ea11c331-20ee-4367-a64f-da8dc9e2595d" width="800px">    

- lda 차원축소 후 로지스틱 회귀 결과 
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/0f43825f-a49f-440f-b2d6-68ebca09a09c" width="800px">  

- 기존 pytorch-Logistic Regression과 lda 차원축소-Logistic Regression 비교 그래프 
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/7aee5c78-db70-4dd5-98d3-c57cb27ec7f3" width="800px">


### 📃 2 Cycle Result
> #### 차원축소를 하지 않고, 로지스틱회귀분석을 사용하였을 경우와 LDA를 통해 1차원으로 차원축소를 하였을 때와 큰 차이가 없음을 알 수 있습니다.  
> #### 로지스틱 회귀 분석을 통해 차원 축소된 데이터에 대한 예측을 수행하고, 이에 따른 임계치를 조정하여 오차 행렬을 분석하겠습니다. 


## 3 Cycle
> ### lda 차원축소  Logistic Regression 으로 분석한 데이터의 임계치(Threshold) 조절 시행
>  오차행렬 정리 

<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/da912431-a391-493c-96ce-1fdb7da3f35a" width="800px" style="margin-bottom: 30px">  

- #### 🏆임계치를 낮춰서 재현율을 높여주는 것이 적합하다. (실제 심부전 환자를 정상으로 잘못분류하지 않게끔 하기 위해)  

## 원본데이터 차원축소  

👉 [Feature 제거 후 임계치 조정 표 보러가기](#8-Cycle)

> **Threshold를 0.4로 낮추면 재현율(Recall)이 기존 0.8660에서 0.9072으로 증가하는 것을 볼 수 있습니다.**

<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/e08ef1ae-ee08-4a22-8123-4add8e8b3a6b" width="800px" style="margin-bottom: 10px">    

> **임계치에 따른 정밀도, 재현율 변화 추이 그래프**  
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/9f6d47a5-8d2e-42aa-b7e4-df6ecb8075a1" width="800px">

> **ROC Curve** 이미지  

<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/b3f6a059-c766-4fa5-92d4-27f80363ef1f" width="800px">


### 📃 3 Cycle Result  

<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/1bc3354d-476f-4c06-be1b-0a865270377a" width="800px">  


> #### 로지스틱 회귀기법을 사용해서 차원축소를 한 결과 임계치를 0.4로 낮출 경우 정확도 점수와 F1 Score는 아주 낮은 폭으로 낮아졌지만, 재현율을 0.86에서 0.9로 높힘으로써 얻고자 하는 결과를 얻을 수 있었습니다.

최종적으로 임계치를 0.4로 낮출 때 최적의 결과를 얻을 수 있었습니다.

## 4 Cycle
> ### Feature Importance - 
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/fb6bd4c1-6844-42f9-a50c-35e458def704" width="600px">   

- 로지스틱 회귀를 사용하였을 때, RestingECG,  FastingBS, Resting BP 등의 Feature의 중요도가 매우 낮은 것을 확인 할 수 있습니다.
  
- 이는 해당 세 개의 Feature가 심부전 질환 예측 하는 것에 대해 다른 독립변수들에 비해 영향력이 적다는 것을 의미합니다.
- 다음으로 해당 세개의 Feature 제거 후 분석을 실시 하겠습니다.  

> FastingBP, RestingBP 와 RestingECG 같은 경우 타겟데이터에 대한 correlation에서 낮은 영향을 보였기 때문에 해당 세개의 Feature를 제거하도록 하겠습니다.   

<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/baa2858b-484e-4053-b041-7674e181cb3f" width="600px">  
 
[독립변수와 종속변수들의 상관관계 확인](#correlation-종속변수와의-상관관계-분석)  

> 다음 사이클에서는 Feature 제거 후 다시 로지스틱 회귀를 진행하도록 하겠습니다.

## 5 Cycle  
> ### Feature 제거 후 로지스틱 회귀 분석 및 과적합 확인

#### Pytorch로 과적합 분석 시각화 (Train Data loss vs Validation Data loss)
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/937666cc-7a7e-453c-b18c-c26b8d50415c" width="700px">  

#### Feature 제거 전 과적합 분석 시각화 
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/966036c7-bbdd-4ed9-9921-b641ee5d7214" width="700px">   

> 📌 로지스틱 회귀에서 중요도가 없었던 Feature들을 제거 후 pytorch로 과적합을 분석하였습니다.  

> 그 결과 이전 Feature 제거 전 보다 train data와 validation data 와의 손실값 차이가 더 줄었으며, 과적합을 더 해소 할 수 있었습니다.

#### Feature 제거 후 로지스틱 회귀 분석 결과
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/1089a0f3-e6a2-4b8d-8455-d29274762252" width="700px">  

> Feature 2개 삭제 후 로지스틱 회귀 분석을 한 결과 마찬가지로 정확도가 0.8587, F1 Score가 0.8762로 높게 나온 것을 알 수 있습니다.   

#### 💡 원본데이터, 차원축소데이터, 다중공선성 해소 데이터의 로지스틱 회귀 성능 비교 시각화    

<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/a5a44006-2be7-4733-9c33-1696d1b36864" width="800px">



### 📃 5 Cycle Result  
> 해당 데이터세트에서 Feature 중요도 확인 후 Feature 3개를 제거하여도, 과적합 문제는 발생하지 않습니다.  
  
> 정확도 측면에서, 차원축소를 했을 때와 차원축소를 하지 않고 Feature를 제거 한 데이터와 점수가 동일하였고, F1 Score에서는 Feature들을 제거한 데이터가 가장 점수가 높았씁니다.

> 다음으로는 Feature를 제거 한 데이터를 기반으로, 다른 분류모델들과 성능을 비교해보겠습니다. 

## 6 Cycle
> ### 로지스틱회귀 성능과  다른 분류모델들의 성능 비교    

- #### Accuracy Score & F1 Score 시각화

<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/b4b582f0-99ee-4eee-b6b0-376a3b492eff" width="800px">

### 📃 6 Cycle Result
> 📌 Feature 제거 후 데이터에서 다른 분류모델들의 성능을 분석하여 시각화 한 결과 RandomForest, Lgbm 같은 분류모델들의 성능이 높았지만, 
  로지스틱 회귀분석 역시 성능이 높게 나왔으므로, 임계치를 낮춘 뒤 재현율을 높혀주어서 성능을 업그레이드 시키도록 하겠습니다.


## 7 Cycle 
> ### Feature를 제거한 데이터의 임계치 조정 후 성능 및 시각화 

- Threshold(임계치)에 따른 성능 점수 시각화 
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/442d5a67-42ae-4843-9d49-9ff4aa03d0ce" width="800px">
  
- Threshold를 기존 0.5에서 0.4로 조정
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/146e0fed-c21a-4c79-af56-f19ccd19c586" width="800px">    

- Threshold 0.4 결과   
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/85726ae3-6579-48ff-954f-3366511ff479" width="800px">  

- Threshold에 따른 정밀도 , 재현율 추이 시각화  
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/5a376361-70f1-45f9-8d5e-2355998a9bef" width="800px">    

- ROC curve 시각화 
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/95a262ae-a477-4571-b620-e02c537f2a8d" width="800px">   

## 8 Cycle 
> ### Feature 제거 후 차원축소 재 시행     

- 아래 이미지는 로지스틱회귀를 사용하였을 때, 중요하지 않은 Feature 제거 한 데이터에서 lda 차원축소를 하였습니다.  
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/8425015f-c73c-4f8a-802f-ab96449388e8" width="800px">  

- 훈련 결과  
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/811455c5-d465-4f56-aec2-2c45f9900700" width="800px">  



- Feature 제거 전 lda 차원축소 후 로지스틱 회귀 결과 
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/0f43825f-a49f-440f-b2d6-68ebca09a09c" width="800px">    

- 분석 결과 (Feature 제거 후 차원축소 vs Feature 제거 전 차원 축소)  
> Feature 3개 제거 후 차원축소를 한 결과, 기존 제거 하기 전 차원축소를 하였을 때와 성능이 똑같았습니다.   

- 임계치 조정 성능 지표  
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/4c459877-fe34-4c0d-855e-da1d30b78727" width="600px">   

> Feature 제거 후 lda 차원축소 데이터의 임계치 분포를 보았을 때 임계치가 0.3일때 최적의 성능을 구할 수 있었습니다.  
 
👉 [Feature 제거 하지 않고 lda 차원축소 임계치 표 보러가기](#원본데이터-차원축소)  

> 임계치 0.3 조정 이미지  (정밀도 , 재현율 분포 추이)
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/46bbdcfe-be8d-4cbe-ac43-9c90c58e02d7" width="600px">  

> ROC-Curve   
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/5c085704-6dd5-465c-b54f-f92430736b82" width="600px">

> 결론적으로 Feature 3개를 제거 후 lda로 차원축소를 진행하였을 때 더 좋은 성능을 보여줍니다.     

<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/d9846724-8f6d-4f4c-9ad8-bf7e6d7dbbee" width="800px">   

- 최종적으로 임계치를 0.3으로 설정한 Feature 제거 후 lda 차원축소 결과가 더 성능이 뛰어나므로, 해당 데이터에서 로지스틱회귀를 사용할 때에는,    


- FastingBS, RestingBP, RestingECG 이 세개의 Feature 제거 후 lda 차원 축소를 해야 합니다.

## Total Result
- ### ✨ 최종결론
- 로지스틱 회귀로 순열 중요도 Feature를 제거한 데이터셋과 기존 데이터셋과, 차원축소한 데이터셋의 로지스틱회귀분석 성능을 비교한 결과 큰 변화폭은 없었으며, 모두 성능이 좋았던 것을 알 수 있었습니다.   
- 과적합의 문제도 하이퍼파라미터 튜닝을 통해 방지 할 수 있었습니다.  

[맨위로 이동](#심부전-질환-환자-예측-분류-분석-프로젝트)
