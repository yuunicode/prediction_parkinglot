# 오산시 주차예측모델
이 모델은 실제데이터셋이 항상 업데이트됐을 것이라는 가정하에 만들어졌습니다.  
하지만, 이 프로젝트는 파일을 제공받은 것이기 때문에 2023년 6월 24일이 현재 시점이라는 가정 하에 진행했습니다. 그러기 때문에 6월 24일~6월 30일까지는 정답(Y_true)가 있지만, 모른다는 가정 하에 했습니다.

하지만 코드는 현재 시점까지의 데이터가 준비되어있다면 예측값이 나오도록 설계했기 때문에 프로젝트를 더 진행하게 된다면 유지보수를 할 수 있습니다.  

간단한 파일과 폴더설명입니다.  

## 폴더
- confusion : 혼잡도를 설정하기 위해 예측값이 프린트된 csv파일을 저장합니다. test파일이 예측을 위한 파일이며, 주차장별로 묶어 시각화합니다.  
- figure: 모델의 적합을 시각화하기 위한 폴더입니다.
    -기간별: total/전체기간, half/절반기간, quarter/사분의일기간, one_eight/팔분의일기간  
    -검증/시험용: val(6월17일~6월23일 실제값과 검증값 확인), test(6월17일~6월23일 실제값에 6월24일~6월30일 예측값 연결), answer(6월24일~6월30일의 예측값을 실제값과 겹쳐 실제에 얼마나 근접한지 확인하기)  
    -즉, ohsaek2_total_test는 오색시장2의 전체기간의 테스트셋으로 6월17일~6월23일 실제값에 6월24일~6월30일 예측값을 연결한 그림입니다.)  
- visualization: 이는 공영주차장 이용자에게 직관적으로 혼잡도를 시각화하기위해 필요한 파일이 저장되어있습니다.  
    -bad, good, very_vad png: 혼잡도를 표시하는 이미지파일  
    -주차장명_현재시점_finalHighMeanPrediction: 모델들 중 값이 가장 높은 두 예측값의 평균으로 최종적인 주차장별 주차대수 예측값을 나타냅니다. **최종예측값** 입니다.  
    -주차장명_현재시점_finalPrediction: 모델 4개의 값과 평균, 중앙값 등 최종예측값의 후보들의 데이터프레임입니다. **참고용**입니다.  
    
- preprocessedData: Holiday API를 받아오다가 오류가 생기는 경우가 빈번하여, 미리 사용자가 현재 날짜(이 프로젝트 기준 6월 24일 현재)를 터미널에 입력하면 그에 따른 **훈련셋과 테스트셋**을 만듭니다. (저희 팀원이 멋지게 자동화했습니다.) (이 또한 깃허브에는 혹시모를 보안문제로 업데이트하지 못했습니다.)
- rawData: 시설관리공단측에서 그대로 받아오는 데이터로, RDBMS 그대로 SQL로 들고온 것으로 알고 있습니다. 매일 업데이트된다면 이 모델 또한 매일 업데이트가 가능합니다. (깃허브에는 혹시모를 보안문제로 업데이트하지 못했습니다.)
  
## 소스파일  
- main.py: 메인이 되는 함수.
- getDataset.py: (부수적인 함수) 데이터셋 정제하기 -> 해당 시점 기준으로 훈련데이터세트가 processedData폴더에 저장됩니다.

- train.py: 모델의 하이퍼파라미터를 옵튜나+샘플러를 통해 구합니다. 하이퍼파라미터는 옵튜나를 통해 조절할 수 있습니다. 16개의 모델이 20분내로 생성됩니다.  
- test.py: 모델의 파라미터에 따른 플롯과 예측값이 csv로 저장되도록 합니다. train에서 알아서 넘어갑니다.  
- main_vis_for_engineer.py: main.py에서 자동으로 호출되는 함수로, 이 프로젝트를 유지보수하는사람이 각 기간별 모델링이 어떻게 됐는지 확인이 가능합니다. high mean과 같은 시각화 자료들이 나옴. 
- main_vis_for_citizen.py: main.py에서 자동으로 호출되는 함수로, 웃음아이콘과 같은 시각화 자료들이 나옵니다. 오산시 시민분들이나 주차장이 필요한 분들에게 직접적으로 나타나는 플롯입니다.  

## 0. 가상환경 venv 생성 후 requirement 설치해주세요.  

    python -m venv myenv
    source myenv/Scripts/activate #for window
    pip install -r requirements.txt

이후 코드를 돌릴때 no module found가 나온다면 pip install 해당모듈로 설치하고 돌리시면 됩니다. ex) optuna not found -> pip install optuna

## 1. 데이터셋 받아오기 (이미 폴더에는 저장되어있지만 확인위해 다시하기 가능합니다.)
main.py를 실행하기 전, getDataset.py를 이용하여 preprocessedData 폴더에 트레이닝을 위한 파일들을 생성합니다.  
실행코드는 다음과 같습니다.  

    python getDataset.py 2023-06-24 #날짜는 현재시점 기준으로 
    
## 2. 데이터 트레이닝  
preprocessedData에 학습에 필요한, 그리고 테스트 시 필요한 파일이 import되었습니다. train.py 를 이용하여 데이터를 training하면 optuna가 training을 하게됩니다.  
범위를 좁혀가며 30번의 Trial로도 빠르게 튜닝되도록 최대한 가볍게 만들었고, 현실에서는 예측모델의 업데이트가 빨라야 할 것 같아 시간이 많이 소요되지 않도록 했습니다.   
+ 컴퓨터 환경에 따라 시청, 오색1의 예측값이 과하게 높게 추정되는 경우가 있습니다. 기간 4개 중 2개는 적정하게 예측하는 모습을 보이는 것이 정상인데, 과하게 높게 추정된다면 파라미터 튜닝을 따로 하셔야합니다.  
+ 과하게 높게 추정되거나, 과하게 낮게 추정되는 것을 파악할 수 있기위해 test plot을 개형했습니다. 잘 확인하면서 튜닝하길 바랍니다.  

    python main.py sichung 2023-06-24
    python main.py unam 2023-06-24
    python main.py ohsaek1 2023-06-24
    python main.py ohsaek2 2023-06-24 #4개를 다 입력해야 4개 주차장의 예보를 할 수 있습니다. 


## 3. 데이터 시각화   

데이터 트레이닝을 마치면 주차장명_현재날짜_finalHighMeanPrediction.csv가 예측의 최종 결과파일로 생성됩니다. (finalPrediction은 각 모델들의 결과입니다.)  
엔지니어의 모델별 예측값 확인 커맨드는 다음과 같습니다.  

    python main_vis_for_engineer.py 2023-06-24  

## 가장중요 - 주차장 이용자에게 보여지는 시각화  

커맨드를 발동하는 시각 기준으로 15분뒤, 30분뒤, 45분뒤, 1시간뒤, 2시간뒤의 주차장 혼잡도가 아이콘으로 팝업됩니다. 이용자가 보기 쉽게 큰 글씨로 하였습니다.

    python main_vis_for_citizen.py 2023-06-24    

입력시간과 맞는 시각화 자료가 pop-up됩니다. (가장 최근 데이터가 없어 6월24일 기준으로 한 것이 아쉽네요).
![KakaoTalk_20230925_162634200](https://github.com/yuunicode/osan_parkinglot/assets/56739105/b00dccb5-462b-4ee7-99d7-b6bf1a9652f6)


아쉽게 2등으로 마무리지은 프로젝트지만, 오류가 없는 입출차나 주차기록이 존재한다면 예측모델이 충분히 역할을 다할 것 같습니다. 이번 프로젝트로 MLFlow를 통해 16개의 모델을 관리하며 팀원들과 협업을 할 수 있어 뜻깊은 프로젝트였습니다. 

이상입니다. 문의: soro24@cau.ac.kr  
    

