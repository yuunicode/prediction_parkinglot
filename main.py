import sys
import pandas as pd
import train_sichung, train_unam, train_ohsaek1, train_ohsaek2

def main():
    lot = str(sys.argv[1])  
    tdy = str(sys.argv[2])
    
    data1, data2, data3, data4, data1_testX, data2_testX, data3_testX, data4_testX = readData(lot)

    ###트레인/테스트 (주차장별로 나눈 이유는 옵튜나 범위, 목적함수와 평가지표가 다르기 때문이었습니다...)
    if lot == 'sichung':
        data1_val, data1_test = train_sichung.train(data1, data1_testX, lot, key = 'total')
        data2_val, data2_test = train_sichung.train(data2, data2_testX, lot, key = 'half')
        data3_val, data3_test = train_sichung.train(data3, data3_testX, lot, key = 'quarter')    
        data4_val, data4_test = train_sichung.train(data4, data4_testX, lot, key = 'one_eight')   

    elif lot == 'unam':
        data1_val, data1_test = train_unam.train(data1, data1_testX, lot, key = 'total')
        data2_val, data2_test = train_unam.train(data2, data2_testX, lot, key = 'half')
        data3_val, data3_test = train_unam.train(data3, data3_testX, lot, key = 'quarter')      
        data4_val, data4_test = train_unam.train(data4, data4_testX, lot, key = 'one_eight')      

    elif lot == 'ohsaek1':
        data1_val, data1_test = train_ohsaek1.train(data1, data1_testX, lot, key = 'total')
        data2_val, data2_test = train_ohsaek1.train(data2, data2_testX, lot, key = 'half')
        data3_val, data3_test = train_ohsaek1.train(data3, data3_testX, lot, key = 'quarter')    
        data4_val, data4_test = train_ohsaek1.train(data4, data4_testX, lot, key = 'one_eight')    
    
    else:
        data1_val, data1_test = train_ohsaek2.train(data1, data1_testX, lot, key = 'total')
        data2_val, data2_test = train_ohsaek2.train(data2, data2_testX, lot, key = 'half')
        data3_val, data3_test = train_ohsaek2.train(data3, data3_testX, lot, key = 'quarter')      
        data4_val, data4_test = train_ohsaek2.train(data4, data4_testX,lot, key = 'one_eight')    

    #최종산출물 -> Visualization에 들어갈 finalHighMeanPrediction(Test).csv
    createFinalPrediction(data1_test, data2_test, data3_test, data4_test, lot, tdy)

    #마지막으로 visualization하면됩니다.



#가장 높은 예측값 두 개의 평균으로 최종 예측값 산정하기. (Final_prediction은 4개의 모델 전부, Final_HighMean_prediction은 최종예측값만)
def createFinalPrediction(data1_test, data2_test, data3_test, data4_test, lot, tdy):
    final_prediction = pd.DataFrame({'전체기간': data1_test, '절반기간': data2_test, '쿼터기간': data3_test, '피자기간': data4_test}, index = createTimeIdx(tdy))
    final_prediction['평균'] = final_prediction[['전체기간','절반기간','쿼터기간','피자기간']].mean(axis=1)
    final_prediction['중앙값'] = final_prediction[['전체기간','절반기간','쿼터기간','피자기간']].median(axis=1)

    for i in final_prediction.index:
        temp = final_prediction.loc[i,'전체기간':'피자기간']
        final_prediction.loc[i,'high_mean'] = temp.sort_values()[-2:].mean()
        final_prediction.loc[i,'max'] = temp.max()

    final_prediction.to_csv(f'./Visualization/{lot}_{tdy}_finalPrediction(Test).csv', encoding = 'utf-8-sig')
    high_mean = pd.DataFrame({'high_mean': final_prediction.high_mean.values.astype(int)}, index = createTimeIdx(tdy))
    high_mean.to_csv(f'./Visualization/{lot}_{tdy}_finalHighMeanPrediction(Test).csv', encoding = 'utf-8-sig')

def createTimeIdx(tdy): #입력된 시간 기준으로 7일 뒤까지의 15분간격 시간인덱스를 뽑습니다.
    from datetime import datetime
    from datetime import timedelta
    tdy_dt = datetime.strptime(tdy, '%Y-%m-%d')
    t = pd.date_range(tdy, periods = 24 * 4 * 7, freq = '15T')    
    time_index = t.map(lambda x: x.strftime('%Y-%m-%d-%H:%M'))            

    return time_index

def readData(lot): #getDataset을 통해 얻은 데이터를 바탕으로 훈련/테스트데이터를 읽어옵니다.
    data1 = pd.read_csv(f'./preprocessedData/{lot}_total.csv')
    data2 = pd.read_csv(f'./preprocessedData/{lot}_half.csv')
    data3 = pd.read_csv(f'./preprocessedData/{lot}_quarter.csv')
    data4 = pd.read_csv(f'./preprocessedData/{lot}_one_eight.csv')
    data1_testX = pd.read_csv(f'./preprocessedData/target_{lot}_total.csv')
    data2_testX = pd.read_csv(f'./preprocessedData/target_{lot}_half.csv')
    data3_testX = pd.read_csv(f'./preprocessedData/target_{lot}_quarter.csv')
    data4_testX = pd.read_csv(f'./preprocessedData/target_{lot}_one_eight.csv')
    return data1, data2, data3, data4, data1_testX, data2_testX, data3_testX, data4_testX 

if __name__ == '__main__':
    main()