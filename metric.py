import numpy as np

### custom loss
def weighted_mse_loss_ohsaek2(y_pred, dmat, alpha = 9):
    # y_val: 실제값, y_pred: 예측값, alpha: 가중치
    y_val = dmat.get_label()
    d = (y_pred-y_val)
    grad = np.where((d)<0, 2*alpha*d, 2*d)
    hess = np.where((d)<0, 2*alpha, 2.0)
    return grad, hess

def weighted_mse_loss_ohsaek1(y_pred, dmat, alpha = 4):
    # y_val: 실제값, y_pred: 예측값, alpha: 가중치
    y_val = dmat.get_label()
    d = (y_pred-y_val)
    grad = np.where((d)<0, 2*alpha*d, 2*d)
    hess = np.where((d)<0, 2*alpha, 2.0)
    return grad, hess

def weighted_mse_loss_unam(y_pred, dmat, alpha = 100):
    # y_val: 실제값, y_pred: 예측값, alpha: 가중치
    y_val = dmat.get_label()
    d = (y_pred-y_val)
    grad = np.where((d)<0, 2*alpha*d, 2*d)
    hess = np.where((d)<0, 2*alpha, 2.0)
    return grad, hess

def weighted_mse_loss_sichung(y_pred, dmat, alpha = 100):
    # y_val: 실제값, y_pred: 예측값, alpha: 가중치
    y_val = dmat.get_label()
    d = (y_pred-y_val)
    grad = np.where((d)<0, 2*alpha*d, 2*d)
    hess = np.where((d)<0, 2*alpha, 2.0)
    return grad, hess

### custom metric
def weighted_mae_metric_ohsaek2(y_pred, dmat, alpha = 3):
    # y_val: 실제값, y_pred: 예측값, alpha: 가중치
    y_val = dmat.get_label()

    resi = np.zeros(shape = len(y_pred), dtype = np.float64)
    loss = np.zeros(shape = len(y_pred), dtype = np.float64)

    for x in range(len(y_pred)):
        resi[x] = y_pred[x] - y_val[x]
        if resi[x] > 0:
            loss[x] = alpha * abs(resi[x])
        else:
            loss[x] = abs(resi[x])

    return 'wmae', loss.mean()

def weighted_mae_metric_ohsaek1(y_pred, dmat, alpha = 2):
    # y_val: 실제값, y_pred: 예측값, alpha: 가중치
    y_val = dmat.get_label()

    resi = np.zeros(shape = len(y_pred), dtype = np.float64)
    loss = np.zeros(shape = len(y_pred), dtype = np.float64)

    for x in range(len(y_pred)):
        resi[x] = y_pred[x] - y_val[x]
        if resi[x] > 0:
            loss[x] = alpha * abs(resi[x])
        else:
            loss[x] = abs(resi[x])

    return 'wmae', loss.mean()

def weighted_mae_metric_unam(y_pred, dmat, alpha = 10):
    # y_val: 실제값, y_pred: 예측값, alpha: 가중치
    y_val = dmat.get_label()

    resi = np.zeros(shape = len(y_pred), dtype = np.float64)
    loss = np.zeros(shape = len(y_pred), dtype = np.float64)

    for x in range(len(y_pred)):
        resi[x] = y_pred[x] - y_val[x]
        if resi[x] > 0:
            loss[x] = alpha * abs(resi[x])
        else:
            loss[x] = abs(resi[x])

    return 'wmae', loss.mean()

def weighted_mae_metric_sichung(y_pred, dmat, alpha = 10):
    # y_val: 실제값, y_pred: 예측값, alpha: 가중치
    y_val = dmat.get_label()

    resi = np.zeros(shape = len(y_pred), dtype = np.float64)
    loss = np.zeros(shape = len(y_pred), dtype = np.float64)

    for x in range(len(y_pred)):
        resi[x] =   y_pred[x] - y_val[x]
        if resi[x] < 0:
            loss[x] = alpha * abs(resi[x])
        else:
            loss[x] = abs(resi[x])

    return 'wmae', loss.mean()
