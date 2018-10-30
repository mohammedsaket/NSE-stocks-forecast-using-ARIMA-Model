from nsepy import get_history
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

data = get_history(symbol="SBIN", start=date(2015, 1, 1), end=date(2015, 12, 31))

'''
SMA_close = pd.Series((data['Close']).rolling(window=10).mean(),name = 'SMA_Close')
data = data.join(SMA_close)

SMA_open = pd.Series((data['Open']).rolling(window=10).mean(),name = 'SMA_Open') 
data = data.join(SMA_open)

SMA_low = pd.Series((data['Low']).rolling(window=10).mean(),name = 'SMA_Low') 
data = data.join(SMA_low)

SMA_high = pd.Series((data['High']).rolling(window=10).mean(),name = 'SMA_High') 
data = data.join(SMA_high)

fig, axes = plt.subplots(nrows=2, ncols=2)
data[['Open']].plot(ax=axes[0,0])
data[['Close']].plot(ax=axes[0,1])
data[['Low']].plot(ax=axes[1,0])
data[['High']].plot(ax=axes[1,1])

data[['SMA_Open']].plot(ax=axes[0,0],color='red')
data[['SMA_Close']].plot(ax=axes[0,1],color='red')
data[['SMA_Low']].plot(ax=axes[1,0],color='red')
data[['SMA_High']].plot(ax=axes[1,1],color='red')


'''
def test_stationarity(timeseries):

    
    rolmean = timeseries.rolling(10).mean()
    rolstd = timeseries.rolling(10).std()

    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
'''
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
'''

def ARIMAfunction(datatype):
    ts_log = np.log(datatype)
    moving_avg = ts_log.rolling(10).mean()
    
    #plt.plot(ts_log_open)
    #plt.plot(moving_avg_open, color='red')
    
    ts_log_moving_avg_diff = ts_log - moving_avg
    ts_log_moving_avg_diff.dropna(inplace=True)

    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.dropna(inplace=True)
    
    model = ARIMA(ts_log, order=(1, 0, 0))
    results_ARIMA = model.fit(disp=-1)
    #print(results_ARIMA.fittedvalues)
    # plt.plot(ts_log_diff)
    # plt.plot(results_ARIMA.fittedvalues, color='red')
    
    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    # print (predictions_ARIMA_diff.head())
    
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    # print (predictions_ARIMA_diff_cumsum.head())
    
    predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff, fill_value=0)
    # print(predictions_ARIMA_log.head())
    
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    # print(predictions_ARIMA.head())
    
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
        
    plt.plot(datatype)
    plt.plot(predictions_ARIMA)


    results_ARIMA.plot_predict(1,348)
    #print(results_ARIMA.forecast(steps=100))
    plt.show()
    
ARIMAfunction(data[['Open']])
ARIMAfunction(data[['Close']])
ARIMAfunction(data[['Low']])
ARIMAfunction(data[['High']])



