import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from math import sqrt
import io
import base64

def custom_mape(A,F):
    return np.mean(np.abs((A - F)/A))

def make_lags(ts, lags):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, int(lags) + 1)
        },
        axis=1)

def xgboost_func(features, numlags, option):
    marina = pd.read_csv("./MarinaDataset.csv", parse_dates=["datetime"])
    marina.drop(columns=['datetime'], inplace=True)

    if(option=='manual'):
        marina = marina[features]
    else:
        numlags = 4

    #Split dataset
    size = math.ceil(len(marina) * 0.75)
    predictions_length = len(marina) - size
    train, test = marina[0:size], marina[size:len(marina)]

    X_train, y_train = train.drop(columns='arrivals'), train.arrivals
    X_test, y_test = test.drop(columns='arrivals'), test.arrivals

    if(numlags):
        num_lags = numlags
        lags = make_lags(marina.arrivals, num_lags)
        lags = lags.fillna(0.0)
        train_lags, test_lags = lags[0:size], lags[size:len(marina)]

        for i in range(1, num_lags + 1):
            X_train[f'y_lag_{i}'] = train_lags[f'y_lag_{i}']

        for i in range(1, num_lags + 1):
            X_test[f'y_lag_{i}'] = test_lags[f'y_lag_{i}']

    reg = xgb.XGBRegressor(n_estimators=200)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=500,
        	verbose=False)

    predictions = reg.predict(X_test)
    predictions = np.reshape(predictions, (predictions_length, 1))

    testPredictPlot = np.ndarray(shape=(len(marina),1))
    testPredictPlot[:, :] = np.nan
    testPredictPlot[size:, :] = predictions

    img1 = io.BytesIO()
    img2 = io.BytesIO()
    plt.figure(figsize=(15, 10), dpi=55)

    plot_importance(reg, height=0.9)
    plt.savefig(img1, format='png')
    plt.close()
    img1.seek(0)

    plt.plot(marina.arrivals)
    plt.plot(testPredictPlot)
    plt.savefig(img2, format='png')
    plt.close()
    img2.seek(0)

    plot_png1 = base64.b64encode(img1.getvalue()).decode('utf8')
    plot_png2 = base64.b64encode(img2.getvalue()).decode('utf8')

    # calculate RMSE
    #rmse = custom_mape(test.arrivals, predictions)

    return plot_png1, plot_png2, '67'#, rmse