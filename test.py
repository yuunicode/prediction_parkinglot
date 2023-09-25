import optuna
from optuna.samplers import TPESampler
import warnings
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import csv
import metric
import pandas as pd
import os



def test(data, target, lot, key, best_param):    
    global dtrain, dtest, y
    warnings.filterwarnings("ignore")
    X, y = select_col_and_split(data, lot)
    Xtest = select_col(target, lot)
    ytest = pd.read_csv(f'./답지파일(그저확인용-안봐도됨)/{lot}_답지_24~30.csv')['누적'].values #-> 원래는 없어야하지만 체크용도로 만들었음. 

    dtrain = xgb.DMatrix(data = X, label = y)
    dtest = xgb.DMatrix(data = Xtest)
  
    if lot == 'sichung':
        model = xgb.train(best_param,
                        dtrain,
                        obj = metric.weighted_mse_loss_sichung,
                        custom_metric = metric.weighted_mae_metric_sichung,
                        evals = [(dtrain, 'train')],
                        verbose_eval = True)           
    
    elif lot == 'unam':
        model = xgb.train(best_param,
                        dtrain,
                        obj = metric.weighted_mse_loss_unam,
                        custom_metric = metric.weighted_mae_metric_unam,
                        evals = [(dtrain, 'train')],
                        verbose_eval = True)           
    
    elif lot == 'ohsaek1':
        model = xgb.train(best_param,
                        dtrain,
                        obj = metric.weighted_mse_loss_ohsaek1,
                        custom_metric = metric.weighted_mae_metric_ohsaek1,
                        evals = [(dtrain, 'train')],
                        verbose_eval = True)     
        
    else: 
         model = xgb.train(best_param,
                        dtrain,
                        obj = metric.weighted_mse_loss_ohsaek2,
                        custom_metric = metric.weighted_mae_metric_ohsaek2,
                        evals = [(dtrain, 'train')],
                        verbose_eval = True)   



    pred = model.predict(dtest)
    pred = pred.astype(int)
    finalPred = pd.DataFrame(data = {'prediction':pred})
    finalPred.to_csv(f'./confusion/{lot}_{key}_prediction(test).csv', encoding = 'utf-8-sig')
    plot_data(y, pred, lot, key)
    answer_plot(ytest, pred, lot, key) #->원래 없어야하지만 체크용도로 만들었음. figure에 answer로 끝나는 파일들이 해당파일
    print('test is Done :D')
    
    return pred

def answer_plot(ytest, pred, lot, key): #->원래 없어야하지만 체크용도로 만들었음. figure에 answer로 끝나는 파일들이 해당파일
    pred = pred.astype(int)
    pred = pd.Series(pred)
    plt.figure(figsize = (25,5))
    plt.title(f'{lot}_Answer_{key}')
    plt.plot(ytest, label = 'true')
    plt.plot(pred, label = 'prediction')
    plt.legend()
    plt.savefig(f'./figure/{lot}_{key}_answer.png')
    
def plot_data(y, pred, lot, key):
    y = y.iloc[-672:] #최근 7일간의 데이터만
    pred = pd.Series(pred)
    pred.index = np.arange(y.index[-1]+1, y.index[-1]+1 + len(pred))

    plt.figure(figsize = (25,5))
    plt.title(f'{lot}_Prediction_{key}')
    plt.plot(y, label = 'true')
    plt.plot(pred, label = 'prediction')
    plt.legend()
    plt.savefig(f'./figure/{lot}_{key}_test.png')


def select_col_and_split(data, lot):#훈련+검증셋의 프리프로세싱
    col_dict = {'sichung' : ['timing_cos','timing_sin','요일','주말','일','isHoliday','isSeq', 'mins_taken_mean','mins_taken_std', 'mins_taken_Q1','mins_taken_Q3','온도','강수량','날씨.1','30분전_온도','30분전_강수량','30분전_날씨.1', '7daysbefore'],
                'unam' : ['timing_cos','timing_sin','요일','운암요일','주말','일','isHoliday','isSeq', 'mins_taken_mean','mins_taken_std', 'mins_taken_Q1','mins_taken_Q3','온도','강수량','날씨.1','30분전_온도','30분전_강수량','30분전_날씨.1'],
                'ohsaek1' : ['timing_cos','timing_sin','요일','주말','일','isHoliday','isSeq','장날','mins_taken_mean','mins_taken_std', 'mins_taken_Q1','mins_taken_Q3','온도','강수량','날씨.1','30분전_온도','30분전_강수량','30분전_날씨.1', '7daysbefore'],
                'ohsaek2' :  ['timing_cos','timing_sin','요일','주말','일','isHoliday','isSeq','장날','mins_taken_mean','mins_taken_std', 'mins_taken_Q1','mins_taken_Q3','온도','강수량','날씨.1','30분전_온도','30분전_강수량','30분전_날씨.1']}
    
    if col_dict.get(lot):
          if '누적' in data:
            X = data.loc[:, col_dict[lot]]
            y = data['누적']
            print('Output: X,y : Cols are selected before training/test')
            return X, y
          
def select_col(target, lot):#테스트셋의 프리프로세싱
    col_dict = {'sichung' : ['timing_cos','timing_sin','요일','주말','일','isHoliday','isSeq', 'mins_taken_mean','mins_taken_std', 'mins_taken_Q1','mins_taken_Q3','온도','강수량','날씨.1','30분전_온도','30분전_강수량','30분전_날씨.1'],
                'unam' : ['timing_cos','timing_sin','요일','운암요일','주말','일','isHoliday','isSeq', 'mins_taken_mean','mins_taken_std', 'mins_taken_Q1','mins_taken_Q3','온도','강수량','날씨.1','30분전_온도','30분전_강수량','30분전_날씨.1'],
                'ohsaek1' : ['timing_cos','timing_sin','요일','주말','일','isHoliday','isSeq','장날','mins_taken_mean','mins_taken_std', 'mins_taken_Q1','mins_taken_Q3','온도','강수량','날씨.1','30분전_온도','30분전_강수량','30분전_날씨.1'],
                'ohsaek2' :  ['timing_cos','timing_sin','요일','주말','일','isHoliday','isSeq','장날','mins_taken_mean','mins_taken_std', 'mins_taken_Q1','mins_taken_Q3','온도','강수량','날씨.1','30분전_온도','30분전_강수량','30분전_날씨.1']}

    testX = target.loc[:, col_dict[lot]]
    if (lot == 'sichung') or (lot == 'ohsaek1'):
        testX['7daysbefore'] = y[-24*4*7:]

    print('Output: X : Cols are selected before training/test')
    return testX        
    