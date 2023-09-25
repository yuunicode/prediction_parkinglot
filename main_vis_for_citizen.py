from datetime import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd
from matplotlib import font_manager, rc
import sys

def main(start_day):

    font_path = "C:/Windows/Fonts/NanumGothic.ttf"
    font = font_manager.FontProperties(fname = font_path).get_name()
    rc('font', family = font)
    global predict_images

    df_sichung = readData(start_day, 'sichung')
    df_unam = readData(start_day, 'unam')
    df_ohsaek1 = readData(start_day, 'ohsaek1')
    df_ohsaek2 = readData(start_day, 'ohsaek2')
    
    predict_images = ['./Visualization/very_bad.png', 
                      './Visualization/bad.png', 
                      './Visualization/good.png']
   
    mine(df_sichung, 50, 672, 550, 0.8, 0.6)
    mine(df_unam, 50, 672, 593, 0.8, 0.6)
    mine(df_ohsaek1, 0, 672, 105, 0.8, 0.6)
    mine(df_ohsaek2, 0, 672, 180, 0.8, 0.6) 

    df_sichung_DF = predict(df_sichung, 672, 550, 0.8, 0.6)
    df_unam_DF = predict(df_unam, 672, 593, 0.8, 0.6)
    df_ohsaek1_DF = predict(df_ohsaek1, 672, 105, 0.8, 0.6)
    df_ohsaek2_DF = predict(df_ohsaek2, 672, 180, 0.8, 0.6)

    # 아래는 현재 시각, 15분 뒤, 30분 뒤, 45분 뒤, 1시간 뒤, 2시간 뒤 시각을 출력하는 코드
    current = dt.now()
    current_15M = dt.now() + timedelta(0,15*60)
    current_30M = dt.now() + timedelta(0,30*60)
    current_45M = dt.now() + timedelta(0,45*60)
    current_1H = dt.now() + timedelta(0,60*60)
    current_2H = dt.now() + timedelta(0,2*60*60)

    # 국제표준시 기준이면 아래를 사용하세요.
    # current = dt.now() + timedelta(0,9*60*60)
    # current_15M = dt.now() + timedelta(0,9*60*60+15*60)
    # current_30M = dt.now() + timedelta(0,9*60*60+30*60)
    # current_45M = dt.now() + timedelta(0,9*60*60+45*60)
    # current_1H = dt.now() + timedelta(0,10*60*60)
    # current_2H = dt.now() + timedelta(0,11*60*60)

    # 아래는 현재 시각이 96개의 timing 중 몇 번째인지 찾는 코드.
    current_timing = (current.hour*60 + current.minute) // 15
    data = [['시간', current.strftime("%H시 %M분"), current_15M.strftime("%H시 %M분"),current_30M.strftime("%H시 %M분"),current_45M.strftime("%H시 %M분"),current_1H.strftime("%H시 %M분"),current_2H.strftime("%H시 %M분")],
            ['예보', "", "","","", "", ""]]

    # 테이블 생성+기본 구조 세팅

    plt.rcParams['font.family'] ='Malgun Gothic'
    figure, axs = plt.subplots(figsize=(15,10), nrows=4, ncols=1, sharey = True)

    for i in range(4):
        table = axs[i].table(cellText=data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.scale(1, 3.5)
        table.set_fontsize(25)  # 원하는 크기로 설정
        axs[i].axis('off')

    axs[0].set_title('주차장1(오산시청)', fontsize=20)
    axs[1].set_title('주차장2(운암공영)', fontsize=20)
    axs[2].set_title('주차장3(오색시장1)', fontsize=20)
    axs[3].set_title('주차장4(오색시장2)', fontsize=20)

    plt.suptitle('주차장 현황', fontsize=30)


    # 아래는 이미지를 읽어오고, 크기를 조절하는 함수이다.
    def getImage(path, zoom=0.5):
        return OffsetImage(plt.imread(path), zoom=zoom)


    # _predict에 이미지가 출력을 원하는 timing의 이미지가 순서대로 들어오는 과정.
    sichung_predict=[]
    unam_predict=[]
    ohsaek1_predict=[]
    ohsaek2_predict=[]
    for k in range(0,6):
        a=df_sichung_DF.iloc[current_timing+k,0]
        sichung_predict.append(a)

    for k in range(0,6):
        a=df_unam_DF.iloc[current_timing+k,0]
        unam_predict.append(a)

    for k in range(0,6):
        a=df_ohsaek1_DF.iloc[current_timing+k,0]
        ohsaek1_predict.append(a)

    for k in range(0,6):
        a=df_ohsaek2_DF.iloc[current_timing+k,0]
        ohsaek2_predict.append(a)

    # 아래 리스트는 각 subplot안에 그림이 들어갈 위치이다.
    y=[0.33,0.33,0.33,0.33,0.33,0.33]
    x=[0.21,0.36,0.5,0.64,0.78,0.93]

    # 아래 반복문은 각 subplot안에 그림을 넣는 과정이다.
    for x0, y0, path in zip(x, y, sichung_predict ):
        ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
        axs[0].add_artist(ab)

    for x0, y0, path in zip(x, y, unam_predict ):
        ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
        axs[1].add_artist(ab)

    for x0, y0, path in zip(x, y, ohsaek1_predict ):
        ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
        axs[2].add_artist(ab)

    for x0, y0, path in zip(x, y, ohsaek2_predict ):
        ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
        axs[3].add_artist(ab)


    footer_text = current.strftime('%m월 %d일')
    plt.figtext(0.9, 0.9, '현재 날짜 : '+footer_text, horizontalalignment='right', size=25, weight='light')
    plt.show()



def readData(start_day, lot):
    df = pd.read_csv(f'./Visualization/{lot}_{start_day}_finalHighMeanPrediction(Test).csv')
    df.columns = ['time', 'status']
    return df

def mine(df, init, length, max_parking, very_bad, bad):
  df['status']=df['status']+init


  # '누적_cat'은 주차장의 매우 혼잡/혼잡/보통를 문자열로 나타낸 열
  # '누적_cat.1'은 위의 값을 0,1,2로
  df['누적_cat']=0
  df['누적_cat.1']=0

  for i in range(length):
    if df.iloc[i,1]>=max_parking*very_bad:
      df.iloc[i,2]='매우 혼잡'

    elif (df.iloc[i,1]<max_parking*very_bad)&(df.iloc[i,1]>=max_parking*bad):
      df.iloc[i,2]='혼잡'

    elif (df.iloc[i,1]<max_parking*bad):
      df.iloc[i,2]='보통'

  for i in range(length):
    if df.iloc[i,1]>=(max_parking*very_bad):
      df.iloc[i,3]='2'

    elif (df.iloc[i,1]<max_parking*very_bad)&(df.iloc[i,1]>=max_parking*bad):
      df.iloc[i,3]='1'

    elif (df.iloc[i,1]<max_parking*bad):
      df.iloc[i,3]='0'

def predict(df, length , max_parking, very_bad, bad):

  df1 = df.copy()

  df1 = df1[['time','status']]
  df1['예보']=0

  for i in range(length):
    if df1.iloc[i,1]>=(max_parking*very_bad):
      df1.iloc[i,2]=predict_images[0]

    elif (df1.iloc[i,1]<max_parking*very_bad)&(df1.iloc[i,1]>=max_parking*bad):
      df1.iloc[i,2]=predict_images[1]

    elif (df1.iloc[i,1]<max_parking*bad):
      df1.iloc[i,2]=predict_images[2]

  df1.set_index(df1['time'],inplace=True)
  df1.drop(columns={'status','time'}, inplace=True)

  return df1



if __name__ == '__main__':
    start_day = str(sys.argv[1])
    main(start_day)