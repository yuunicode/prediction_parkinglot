import pandas as pd
import sys
import warnings

def main():
    from datetime import datetime
    from datetime import timedelta
    warnings.filterwarnings("ignore")
    start_day = str(sys.argv[1])
    data, weather = rawData()
    initials = {'unam':100, 'sichung':200, 'ohsaek1':5, 'ohsaek2':20}
    tdy = datetime.strptime(start_day, '%Y-%m-%d') #들어온 스트링을 데이트타입으로
    end_day = tdy + timedelta(days = 7)
    end_day = datetime.strftime(end_day, '%Y-%m-%d') #데이트타입을 다시 문자로

    #해당 날짜 00시에 주차장에 남아있는 주차장 수 입력
    data_dict = converting(data, weather, start_day, end_day, initials)

    lot_set = ['sichung', 'unam', 'ohsaek1', 'ohsaek2']
    for lots in lot_set:
        data_dict[f'{lots}_total'].to_csv(f'./preprocessedData/{lots}_total.csv', encoding = 'utf-8-sig')
        data_dict[f'{lots}_quarter'].to_csv(f'./preprocessedData/{lots}_quarter.csv', encoding = 'utf-8-sig')
        data_dict[f'{lots}_half'].to_csv(f'./preprocessedData/{lots}_half.csv', encoding = 'utf-8-sig')
        data_dict[f'{lots}_pizza'].to_csv(f'./preprocessedData/{lots}_one_eight.csv', encoding = 'utf-8-sig')
        data_dict[f'target_{lots}_total'].to_csv(f'./preprocessedData/target_{lots}_total.csv', encoding = 'utf-8-sig')
        data_dict[f'target_{lots}_half'].to_csv(f'./preprocessedData/target_{lots}_half.csv', encoding = 'utf-8-sig')
        data_dict[f'target_{lots}_quarter'].to_csv(f'./preprocessedData/target_{lots}_quarter.csv', encoding = 'utf-8-sig')
        data_dict[f'target_{lots}_pizza'].to_csv(f'./preprocessedData/target_{lots}_one_eight.csv', encoding = 'utf-8-sig')       

### 데이터셋 들고오기, 이 데이터는 꾸준히 업데이트된다고 가정한 후 모델링
def rawData():
    import logging
    logging.basicConfig(level = logging.WARN)
    logger = logging.getLogger(__name__)
    try:
        print('Read Raw Dataset ...')
        data = pd.read_csv('./rawData/raw.csv')
        weather = pd.read_csv('./rawData/weather.csv', parse_dates = ['일시'])
        print('Dataset Imported')
        return data, weather
    
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.exception('Unable to download raw/weather data. Error: %s', e)

'''
converting: 원데이터를 모델링에 필요한 데이터프레임으로 변형시킨다.
아래는 converting 함수 + converting 함수를 위한 보조함수들로 이루어져있다.
'''
### 집계용 함수
def Q1(series):
  return series.quantile(.25)
def Q3(series):
  return series.quantile(.75)

### 누적값
def melting(df:pd.DataFrame, holiday:pd.DataFrame, initial:int):
  # df: 기본데이터, holiday: 공휴일종합, initial: 주차장별 초깃값
  # df: 데이터프레임, holiday: 데이터프레임, initial: 정수

  import pandas as pd
  import numpy as np
  from datetime import timedelta
  from datetime import datetime as dt
  import warnings
  warnings.filterwarnings('ignore')

  df['In'] = pd.to_datetime(df['In'])
  df['Out'] = pd.to_datetime(df['Out'])
  df['duration'] = df['Out'] - df['In']
  # 결측, 장기주차 제외
  df = drop_till_yesterday(df, 'In')
  df = df[df['duration'] < timedelta(1)]

  # 입출차 수
  df['minus'] = -1
  df['plus'] = 1

  # 입차와 출차 melting
  df_cum = pd.concat([df[['In','plus']].rename(columns={'In':'time'}), df[['Out','minus']].rename(columns={'Out':'time'})], axis=0)

  # 입출차 수 계산
  df_melted = pd.DataFrame()
  df_melted['입차'] = df_cum.set_index('time').resample('15T')['plus'].sum()
  df_melted['출차'] = abs(df_cum.set_index('time').resample('15T')['minus'].sum())

  # 증감, 누적 계산
  df_melted['증감'] = df_melted['입차'] - df_melted['출차']
  df_melted['누적'] = df_melted['증감'].cumsum()
  df_melted['누적'] = df_melted['누적'] + initial

  # 날씨변수
  df_melted = pd.concat([df_melted, holiday],axis = 1, join='inner')
  df_melted = df_melted.reset_index(drop=False).rename(columns={"index": "time"})

  # 요인들
  df_melted['요일'] = df_melted['time'].dt.weekday
  df_melted['주말'] = df_melted['time'].dt.weekday // 5
  df_melted['장날'] = (df_melted['time'].dt.day % 5).map({3:1, 0:0, 1:0, 2:0, 4:0})

  df_melted['월'] = df_melted['time'].dt.month
  df_melted['일'] = df_melted['time'].dt.day
  df_melted['시간대'] = df_melted['time'].dt.hour
  df_melted['timing'] = (df_melted['time'].dt.hour*60 + df_melted['time'].dt.minute)//15
  df_melted['timing_cos'] = 2*np.cos(2*np.pi*df_melted['timing']/96)
  df_melted['timing_sin'] = 2*np.sin(2*np.pi*df_melted['timing']/96)

  # 운암요일
  df_melted['운암요일'] = df_melted['요일'].apply(lambda x: 0 if (x == 6 or x == 0) else 1 if (x == 1 or x == 2 or x == 3) else 2)

  return df_melted

### 공휴일 API 따오기
def get_holiday(years: list):
    # years:[원하는 연도 목록]

    import requests
    from urllib import parse
    import json
    from requests.api import get
    import pandas as pd
    from datetime import datetime as dt
    from datetime import timedelta
    import warnings
    warnings.filterwarnings('ignore')

    geturl = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService"
    api_key_utf8 = "G0JDvgFvJhTpOhYxUIFEgE6w6ihghwLabtTjKdsCmmh3wTggZttGPyQFEljW9YNv8oILkvyxs5HfUkG+JBJUcQ==" # 개인 API키
    api_key_decode = parse.unquote(api_key_utf8)

    total_dic = []  # 최종 반환 사전
    url_holiday = geturl + '/getRestDeInfo'

    for year in years:
        params = {"ServiceKey": api_key_decode, 'solYear': year, 'numOfRows': 100, '_type': 'json'}
        response = requests.get(url_holiday, params=params)
        dic = json.loads(response.text)

        item = dic['response']['body']['items']['item']
        total_dic.extend(item)

    holiday = pd.DataFrame(total_dic)
    holiday['isHoliday'] = 1

    holiday['locdate'] = pd.to_datetime(holiday['locdate'].astype(str))
    holiday = holiday.set_index('locdate').resample(rule='1D').first()
    holiday = holiday.asfreq(freq='15T', method='ffill')

    # 공휴일
    holiday.loc[holiday.dateName.isna(), 'isHoliday'] = 0
    # 주말
    holiday['weekday'] = holiday.index.weekday

    # 3일연휴
    holiday['isSeq'] = 0
    # 월이 공휴일일 때
    idx_mon = holiday[(holiday.weekday == 0) & (holiday.isHoliday == 1)].index
    for i in list(idx_mon):
        holiday.loc[i - timedelta(2), 'isSeq'] = 1
        holiday.loc[i - timedelta(1), 'isSeq'] = 1
        holiday.loc[i, 'isSeq'] = 1
        # 금이 공휴일일 때
    idx_fri = holiday[(holiday.weekday == 4) & (holiday.isHoliday == 1)].index
    for i in list(idx_fri):
        holiday.loc[i, 'isSeq'] = 1
        holiday.loc[i + timedelta(1), 'isSeq'] = 1
        holiday.loc[i + timedelta(2), 'isSeq'] = 1

    return holiday[['isHoliday', 'isSeq']]

### 주차시간 집계함수
def aggregate(df:pd.DataFrame, target_date, term:str, ohsaek=None):
    import pandas as pd
    from datetime import timedelta
    from datetime import datetime as dt
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')

   # 기간별 셋팅
    target_date = pd.to_datetime(target_date)
    terms = {'total':0, 'half':-180, 'quarter':-90, 'pizza':-45}
    start = terms[term]

    # 데이터 셋 준비
    df = df.dropna()
    df['duration'] = df['Out'] - df['In']

    df = df.loc[df.duration < timedelta(1)]
    df['mins_taken'] = df.duration.dt.days * 24 * 60 + df.duration.dt.seconds / 60
    df['log_mins_taken'] = np.log(df['mins_taken'] + 1)

    df['시간대'] = df['In'].dt.hour
    df['timing'] = (df.In.dt.hour * 60 + df.In.dt.minute) // 15

    # 명절 제외
    idx_chuseok = list(df[(df.In >= dt(2022, 9, 7, 0, 0, 0)) & (df.In < dt(2022, 9, 13, 0, 0, 0))].index)
    idx_seol = list(df[(df.In >= dt(2023, 1, 19, 0, 0, 0)) & (df.In < dt(2023, 1, 25, 0, 0, 0))].index)
    df = df.drop(idx_chuseok + idx_seol, axis=0)

    # 야맥축제 제외
    if ohsaek:
        df = df[(df['In'] < dt(2023,6,9,0,0,0))|(df['In'] > dt(2023,6,12,0,0,0))]

    # 기간 조정
    df['passed'] = (df['In'] - df['In'].min()).dt.days
    df = df[df['In'] < target_date] # 예측대상 이전까지만 자르기
    start_date = int(df.passed.unique()[start])
    df = df[df.passed > start_date]

    # 시간대별 집계
    agg_df = df.groupby('시간대')[['mins_taken', 'log_mins_taken']].agg({'mean', 'std', Q1, Q3}).reset_index(
        drop=False)

    # multi index -> single index
    cols_list = []
    for col in agg_df.columns:
        cols_list.append(col[0] + '_' + col[1])
    agg_df.columns = cols_list

    return agg_df

### 누적값, 날씨, 주차시간 합치는 함수
def merging(df_melted:pd.DataFrame, df_agg:pd.DataFrame, weather:pd.DataFrame, target_date, term:str, ohsaek=None):
    # [누적값, 주차시간, 날씨]를 (타겟날짜)와 (기간)을 고려하여 merge하는 함수

    import pandas as pd
    from datetime import timedelta
    from datetime import datetime as dt
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')

    # 기간별 셋팅
    target_date = pd.to_datetime(target_date, errors='ignore')
    terms = {'total':0, 'half':-180, 'quarter':-90, 'pizza':-45}
    start = terms[term]

    # 날씨 동일시점, 이전시점
    weather_past = weather.copy()
    weather_past['일시'] = weather_past['일시'] + timedelta(0, 30 * 60)
    weather_past = weather_past.rename(
        columns={'온도': '30분전_온도', '강수량': '30분전_강수량', '날씨': '30분전_날씨', '날씨.1': '30분전_날씨.1'})

    # merging
    # melted 기간조정
    # 명절 제외
    idx_chuseok = list(df_melted[(df_melted['time'] >= dt(2022, 9, 7, 0, 0, 0)) & (df_melted['time'] < dt(2022, 9, 13, 0, 0, 0))].index)
    idx_seol = list(df_melted[(df_melted['time'] >= dt(2023, 1, 19, 0, 0, 0)) & (df_melted['time'] < dt(2023, 1, 25, 0, 0, 0))].index)
    df_melted = df_melted.drop(idx_chuseok + idx_seol, axis=0)

    # 야맥축제 제외
    if ohsaek:
        df_melted = df_melted[(df_melted['time'] < dt(2023,6,9,0,0,0))|(df_melted['time'] >= dt(2023,6,12,0,0,0))]

    #### 7일전주차시간추가
    df_melted['7daysbefore'] = df_melted['누적'].shift(7 * 24 * 4)
    df_melted['7daysbefore'].fillna(method = 'bfill', inplace = True)
    ####

    # 기간 조정
    df_melted['passed'] = (df_melted['time'] - dt(2022,7,1)).dt.days
    df_melted = df_melted[df_melted['time'] < target_date] # 예측대상 이전까지 자르기
    start_date = int(df_melted.passed.unique()[start])
    df_melted = df_melted[df_melted.passed > start_date]
    df_melted = df_melted[df_melted['time'] >= dt(2022, 7, 1) + timedelta(start_date)] # 최근 n일로 자르기

    # 주차시간
    df_melted = df_melted.merge(df_agg, left_on='시간대', right_on='시간대_')

    # 날씨 변수
    df_melted = df_melted.merge(weather, left_on='time', right_on='일시')  # 현재날씨
    df_melted = df_melted.merge(weather_past, left_on='time', right_on='일시')  # 30분전 날씨

    df_melted = df_melted.sort_values('time')
    return df_melted

### 타겟 데이터프레임 만들기
def making_target(target_start, target_end, weather:pd.DataFrame, holiday:pd.DataFrame, df_agg:pd.DataFrame):
    import pandas as pd
    import numpy as np
    from datetime import datetime as dt
    from datetime import timedelta
    import warnings
    warnings.filterwarnings('ignore')

    num_cols = ['시간대_', 'mins_taken_mean', 'mins_taken_std', 'mins_taken_Q1', 'mins_taken_Q3',
                 'log_mins_taken_mean', 'log_mins_taken_std', 'log_mins_taken_Q1', 'log_mins_taken_Q3']
    df_agg = df_agg[num_cols]

    ## 날짜기준 설정
    idx = pd.date_range(start=target_start, end= target_end - timedelta(0, 15 * 60), freq='15T')
    df_target = pd.DataFrame(index=idx)

    # 파생 요인들
    df_target['요일'] = df_target.index.weekday
    df_target['주말'] = df_target.index.weekday // 5
    df_target['장날'] = (df_target.index.day % 5).map({3: 1, 0: 0, 1: 0, 2: 0, 4: 0})
    df_target['월'] = df_target.index.month
    df_target['일'] = df_target.index.day
    df_target['시간대'] = df_target.index.hour
    df_target['timing'] = (df_target.index.hour * 60 + df_target.index.minute) // 15
    df_target['timing_cos'] = 2*np.cos(2*np.pi*df_target['timing']/96)
    df_target['timing_sin'] = 2*np.sin(2*np.pi*df_target['timing']/96)

    # 공휴일
    df_target = pd.concat([df_target, holiday], axis=1, join='inner')
    df_target = df_target.reset_index(drop=False).rename(columns={"index": "time"})

    # 운암요일
    df_target['운암요일'] = df_target['요일'].apply(lambda x: 0 if (x == 6 or x == 0) else 1 if (x == 1 or x == 2 or x == 3) else 2)

    # 날씨
    # 날씨 동일시점, 이전시점
    weather_past = weather.copy()
    weather_past['일시'] = weather_past['일시'] + timedelta(0, 30 * 60)
    weather_past = weather_past.rename(
        columns={'온도': '30분전_온도', '강수량': '30분전_강수량', '날씨': '30분전_날씨', '날씨.1': '30분전_날씨.1'})

    ## merging (target 날짜설정을 기준으로)
    # 날씨 변수
    df_target = df_target.merge(weather, left_on='time', right_on='일시')  # 현재날씨
    df_target = df_target.merge(weather_past, left_on='time', right_on='일시')  # 30분전 날씨

    # 주차시간
    df_target = df_target.merge(df_agg, left_on='시간대', right_on='시간대_')

    df_target = df_target.sort_values('time')
    return df_target

### 전날 빼고 결측제외시키기
def drop_till_yesterday(df:pd.DataFrame, col:str):
    # df: 대상 데이터프레임, col: 마지막날을 판단하는 기준컬럼

    from datetime import datetime as dt
    from datetime import timedelta
    import warnings
    import pandas as pd
    warnings.filterwarnings('ignore')

    # 마지막날 00시 00분 00초
    last_day = dt(max(df[col]).year, max(df[col]).month, max(df[col]).day,0,0,0)

    # 마지막 날인 애들
    df_last_day = df[(df[col] - last_day)<timedelta(1)]

    # 일반적인 날들
    df_before = df[df[col]-last_day >= timedelta(1)]
    df_before = df_before.dropna()

    # 둘이 합치기
    df = pd.concat([df_before, df_last_day], axis=0)

    return df

### 메인함수
def converting(df, weather, target_start, target_end, initials):
    # df: 원데이터, weather:날씨데이터, target_start:예측시작, target_end:예측종료, intials:주차장별 초깃값들

    # 위의 모든 과정을 종합하는 함수
    # 주차장 분리 melt -> (주차시간, 날씨)merge -> make_target의 순서로 진행된다

    from datetime import datetime as dt
    from datetime import timedelta
    import pandas as pd
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')

    df.columns = ['CarNo','In','Out','lot_name']

    target_start = pd.to_datetime(target_start)
    target_end = pd.to_datetime(target_end)
    df['In'] = pd.to_datetime(df['In'])
    df['Out'] = pd.to_datetime(df['Out'])
    weather['일시'] = pd.to_datetime(weather['일시'])

    ## 날짜정보 확인
    if target_end < target_start:
        print('예측기간을 확인해주십시오! 데이터 프레임 출력에 실패했습니다!')
        return None

    ## 데이터 준비: 결측, 장기주차 제외
    # df = drop_till_yesterday(df, 'In')
    df['duration'] = df['Out'] - df['In']
    df = df.loc[df.duration < timedelta(1)]

    ## 주차장별 분리
    unam = df[df['lot_name'] == '운암공영주차장']
    sichung = df[df['lot_name'] == '오산시청부설주차장']
    ohsaek1 = df[df['lot_name'] == '오색시장1공영주차장']
    ohsaek2 = df[df['lot_name'] == '오색시장2공영주차장']

    ## 공휴일 API 정보
    holiday = get_holiday(list(range(df['In'].dt.year.min(), df['Out'].dt.year.max()+1, 1)))

    ## 주차장별 melting
    unam_melted = melting(df=unam, holiday = holiday, initial = initials['unam'])
    sichung_melted = melting(df=sichung, holiday = holiday, initial = initials['sichung'])
    ohsaek1_melted = melting(df=ohsaek1, holiday = holiday, initial = initials['ohsaek1'])
    ohsaek2_melted = melting(df=ohsaek2, holiday = holiday, initial = initials['ohsaek2'])


    ## 주차장별 기간별 주차시간
    # 운암
    unam_agg_total = aggregate(df=unam, target_date=target_start, term='total', ohsaek=None)
    unam_agg_half = aggregate(df=unam, target_date=target_start, term='half', ohsaek=None)
    unam_agg_quarter = aggregate(df=unam, target_date=target_start, term='quarter', ohsaek=None)
    unam_agg_pizza = aggregate(df=unam, target_date=target_start, term='pizza', ohsaek=None)

    # 시청
    sichung_agg_total = aggregate(df=sichung, target_date=target_start, term='total', ohsaek=None)
    sichung_agg_half = aggregate(df=sichung, target_date=target_start, term='half', ohsaek=None)
    sichung_agg_quarter = aggregate(df=sichung, target_date=target_start, term='quarter', ohsaek=None)
    sichung_agg_pizza = aggregate(df=sichung, target_date=target_start, term='pizza', ohsaek=None)

    # 오색시장1
    ohsaek1_agg_total = aggregate(df=ohsaek1, target_date=target_start, term='total', ohsaek=True)
    ohsaek1_agg_half = aggregate(df=ohsaek1, target_date=target_start, term='half', ohsaek=True)
    ohsaek1_agg_quarter = aggregate(df=ohsaek1, target_date=target_start, term='quarter', ohsaek=True)
    ohsaek1_agg_pizza = aggregate(df=ohsaek1, target_date=target_start, term='pizza', ohsaek=True)

    # 오색시장2
    ohsaek2_agg_total = aggregate(df=ohsaek2, target_date=target_start, term='total', ohsaek=True)
    ohsaek2_agg_half = aggregate(df=ohsaek2, target_date=target_start, term='half', ohsaek=True)
    ohsaek2_agg_quarter = aggregate(df=ohsaek2, target_date=target_start, term='quarter', ohsaek=True)
    ohsaek2_agg_pizza = aggregate(df=ohsaek2, target_date=target_start, term='pizza', ohsaek=True)


    ## 기간별 주차시간과 날씨정보를 merge
    # 운암 기간별
    unam_total = merging(df_melted=unam_melted, df_agg=unam_agg_total, weather=weather, target_date=target_start, term='total', ohsaek=False)
    unam_half = merging(unam_melted, unam_agg_half, weather, target_start, 'half', None)
    unam_quarter = merging(unam_melted, unam_agg_quarter, weather, target_start, 'quarter', None)
    unam_pizza = merging(unam_melted, unam_agg_pizza, weather, target_start, 'pizza', None)

    # 시청 기간별
    sichung_total = merging(sichung_melted, sichung_agg_total, weather, target_start, 'total', None)
    sichung_half = merging(sichung_melted, sichung_agg_half,weather, target_start, 'half', None)
    sichung_quarter = merging(sichung_melted, sichung_agg_quarter, weather, target_start, 'quarter',None)
    sichung_pizza = merging(sichung_melted, sichung_agg_pizza, weather, target_start, 'pizza', None)

    # 오색시장1 기간별
    ohsaek1_total = merging(ohsaek1_melted, ohsaek1_agg_total, weather, target_start, 'total', True)
    ohsaek1_half = merging(ohsaek1_melted, ohsaek1_agg_half, weather, target_start, 'half', True)
    ohsaek1_quarter = merging(ohsaek1_melted, ohsaek1_agg_quarter, weather, target_start, 'quarter', True)
    ohsaek1_pizza = merging(ohsaek1_melted, ohsaek1_agg_pizza, weather, target_start, 'pizza', True)

    # 오색시장2 기간별
    ohsaek2_total = merging(ohsaek2_melted, ohsaek2_agg_total, weather, target_start, 'total', True)
    ohsaek2_half = merging(ohsaek2_melted, ohsaek2_agg_total, weather, target_start, 'half', True)
    ohsaek2_quarter = merging(ohsaek2_melted, ohsaek2_agg_total, weather, target_start, 'quarter', True)
    ohsaek2_pizza = merging(ohsaek2_melted, ohsaek2_agg_total, weather, target_start, 'pizza', True)


    ## 타겟변수 생성
    target_unam_total = making_target(target_start=target_start, target_end=target_end, weather=weather, holiday=holiday, df_agg=unam_agg_total)
    target_unam_half = making_target(target_start, target_end, weather, holiday, unam_agg_half)
    target_unam_quarter = making_target(target_start, target_end, weather, holiday, unam_agg_quarter)
    target_unam_pizza = making_target(target_start, target_end, weather, holiday, unam_agg_pizza)

    target_sichung_total = making_target(target_start, target_end, weather, holiday, sichung_agg_total)
    target_sichung_half = making_target(target_start, target_end, weather, holiday, sichung_agg_half)
    target_sichung_quarter = making_target(target_start, target_end, weather, holiday, sichung_agg_quarter)
    target_sichung_pizza = making_target(target_start, target_end, weather, holiday, sichung_agg_pizza)

    target_ohsaek1_total = making_target(target_start, target_end, weather, holiday, ohsaek1_agg_total)
    target_ohsaek1_half = making_target(target_start, target_end, weather, holiday, ohsaek1_agg_half)
    target_ohsaek1_quarter = making_target(target_start, target_end, weather, holiday, ohsaek1_agg_quarter)
    target_ohsaek1_pizza = making_target(target_start, target_end, weather, holiday, ohsaek1_agg_pizza)

    target_ohsaek2_total = making_target(target_start, target_end, weather, holiday, ohsaek2_agg_total)
    target_ohsaek2_half = making_target(target_start, target_end, weather, holiday, ohsaek2_agg_half)
    target_ohsaek2_quarter = making_target(target_start, target_end, weather, holiday, ohsaek2_agg_quarter)
    target_ohsaek2_pizza = making_target(target_start, target_end, weather, holiday, ohsaek2_agg_pizza)

    ## 출력
    data_dict = {'unam_total':unam_total, 'unam_half':unam_half, 'unam_quarter':unam_quarter, 'unam_pizza':unam_pizza,
                'sichung_total':sichung_total, 'sichung_half':sichung_half, 'sichung_quarter':sichung_quarter, 'sichung_pizza':sichung_pizza,
                'ohsaek1_total':ohsaek1_total, 'ohsaek1_half':ohsaek1_half, 'ohsaek1_quarter':ohsaek1_quarter, 'ohsaek1_pizza':ohsaek1_pizza,
                'ohsaek2_total': ohsaek2_total, 'ohsaek2_half': ohsaek2_half, 'ohsaek2_quarter': ohsaek2_quarter, 'ohsaek2_pizza': ohsaek2_pizza,

                'target_unam_total': target_unam_total, 'target_unam_half': target_unam_half, 'target_unam_quarter': target_unam_quarter, 'target_unam_pizza': target_unam_pizza,
                'target_sichung_total': target_sichung_total, 'target_sichung_half': target_sichung_half, 'target_sichung_quarter': target_sichung_quarter,'target_sichung_pizza': target_sichung_pizza,
                'target_ohsaek1_total': target_ohsaek1_total, 'target_ohsaek1_half': target_ohsaek1_half, 'target_ohsaek1_quarter': target_ohsaek1_quarter,'target_ohsaek1_pizza': target_ohsaek1_pizza,
                'target_ohsaek2_total': target_ohsaek2_total, 'target_ohsaek2_half': target_ohsaek2_half, 'target_ohsaek2_quarter': target_ohsaek2_quarter,'target_ohsaek2_pizza': target_ohsaek2_pizza,
                }

    return data_dict

if __name__ == '__main__':
    main()