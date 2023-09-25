import optuna
from optuna.samplers import TPESampler
import warnings
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import metric
import param
import test
import pandas as pd

def train(data, testX, lot, key):    
    global dtrain, dval, y, y_train, y_val
    warnings.filterwarnings("ignore")
    X, y = select_col_and_split(data, lot)
    X_train, X_val, y_train, y_val = split_data(X, y)

    dtrain = xgb.DMatrix(data = X_train, label = y_train)
    dval = xgb.DMatrix(data = X_val, label = y_val)    

    '''옵튜나 사용 시 주석처리'''
    best_param = param.param(lot, key)
    '''옵튜나 사용 시 주석제외'''
    # sampler = TPESampler(**TPESampler.hyperopt_parameters(), seed = 10)
    # study = optuna.create_study(study_name = f'{lot}_{key}_optuna', direction = 'minimize', sampler = sampler)
    # study.optimize(objective, n_trials = 30)
    # best_param = study.best_params
    ''''''''''''''''''''''''

    # fig = optuna.visualization.plot_param_importances(study) -> 베스트파라미터 확인용
    # print(f'Best parameters for {key} in {lot}: ', study.best_params)
    # fig.show()

    model = xgb.train(best_param,
                      dtrain,
                      obj = metric.weighted_mse_loss_ohsaek1,
                      custom_metric = metric.weighted_mae_metric_ohsaek1,
                      evals = [(dtrain, 'train'), (dval, 'validation')],
                      verbose_eval = False)           
    
    pred = model.predict(dval)
    pred_labels = np.rint(pred)
    finalPred = pd.DataFrame(data = {'prediction': pred_labels.astype(int), 'true': y_val})
    finalPred.to_csv(f'./confusion/{lot}_{key}_prediction(val).csv', encoding = 'utf-8-sig')
    plot_data(y, pred, lot, key)

    print('training is Done :D')
    print('start prediction!')

    data_test = test.test(data, testX, lot, key, best_param)

    return pred, data_test

def plot_data(y, pred, lot, key):
    pred = pd.Series(pred)
    pred.index = np.arange(y_val.index[0], y_val.index[-1]+1)
    plt.figure(figsize = (25,5))
    plt.plot(y_val, label = 'true')
    plt.plot(pred, label = 'prediction')
    plt.legend()
    plt.savefig(f'./figure/{lot}_{key}_val.png')


def objective(trial):
    params = {
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.6, 0.8),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 8),
            'gamma': trial.suggest_float('gamma', 0.5, 1.0),
            'subsample': trial.suggest_float('subsample', 0.3, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.3, 0.6),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.3, 0.6),
            'random_state': trial.suggest_int('random_state', 1, 30),
        }

    model = xgb.train(params,
                      dtrain,
                      obj = metric.weighted_mse_loss_ohsaek1,
                      custom_metric = metric.weighted_mae_metric_ohsaek1,
                      evals = [(dtrain, 'train'), (dval, 'validation')],
                      verbose_eval = False)           
    pred = model.predict(dval)
    wmae = metric.weighted_mae_metric_ohsaek1(pred, dval)
    return wmae[1]


def select_col_and_split(data, lot):#훈련+검증셋의 프리프로세싱
    col_dict = {'sichung' : ['timing_cos','timing_sin', '요일','주말','일','isHoliday','isSeq', 'mins_taken_mean','mins_taken_std', 'mins_taken_Q1','mins_taken_Q3','온도','강수량','날씨.1','30분전_온도','30분전_강수량','30분전_날씨.1', '7daysbefore'],
                'unam' : ['timing_cos','timing_sin', '요일','운암요일','주말','일','isHoliday','isSeq', 'mins_taken_mean','mins_taken_std', 'mins_taken_Q1','mins_taken_Q3','온도','강수량','날씨.1','30분전_온도','30분전_강수량','30분전_날씨.1'],
                'ohsaek1' : ['timing_cos','timing_sin', '요일','주말','일','isHoliday','isSeq','장날','mins_taken_mean','mins_taken_std', 'mins_taken_Q1','mins_taken_Q3','온도','강수량','날씨.1','30분전_온도','30분전_강수량','30분전_날씨.1','7daysbefore'],
                'ohsaek2' :  ['timing_cos','timing_sin','요일','주말','일','isHoliday','isSeq','장날','mins_taken_mean','mins_taken_std', 'mins_taken_Q1','mins_taken_Q3','온도','강수량','날씨.1','30분전_온도','30분전_강수량','30분전_날씨.1']}
    
    if col_dict.get(lot):
          if '누적' in data:
            X = data.loc[:, col_dict[lot]]
            y = data['누적']
            print('Output: X,y : Cols are selected before training/test')
            return X, y

          
def split_data(X, y):#데이터셋 X,y로 분리
    from sklearn.model_selection import train_test_split
    #General Training Set
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                          shuffle = False,
                                                          test_size = 24 * 7 * 4)
    return X_train, X_test, y_train, y_test   


if __name__ == '__main__':
    import pandas as pd
    lot = 'ohsaek1'
    period = 'total'
    data4 = pd.read_csv(f'./preprocessedData/{lot}_{period}.csv')
    data4_testX = pd.read_csv(f'./preprocessedData/target_{lot}_{period}.csv')
    pred, testset = train(data4, data4_testX, lot, period)

    period = 'half'
    data4 = pd.read_csv(f'./preprocessedData/{lot}_{period}.csv')
    data4_testX = pd.read_csv(f'./preprocessedData/target_{lot}_{period}.csv')
    pred, testset = train(data4, data4_testX, lot, period)

    period = 'quarter'
    data4 = pd.read_csv(f'./preprocessedData/{lot}_{period}.csv')
    data4_testX = pd.read_csv(f'./preprocessedData/target_{lot}_{period}.csv')
    pred, testset = train(data4, data4_testX, lot, period)

    period = 'one_eight'
    data4 = pd.read_csv(f'./preprocessedData/{lot}_{period}.csv')
    data4_testX = pd.read_csv(f'./preprocessedData/target_{lot}_{period}.csv')
    pred, testset = train(data4, data4_testX, lot, period)