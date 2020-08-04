#Import statements

import math
import matplotlib.pyplot as plt 
import numpy as np
from pandas_datareader import data 
import pandas as pd 

def getData(stockSym:str, startDate:str, endDate:str):
    stock = data.DataReader(stockSym, 'yahoo', start=startDate, end=endDate)
    return stock

stock = getData(stockSym='AAPL',startDate='6/1/2020',endDate='7/1/2020')['Close']

def predict(ntrials:int = 1000, predictionWindow:int = 21):
    timeElapsed = (stock.index[-1] - stock.index[0]).days
    # mean = np.mean(stock, axis=0)[3]
    # std = np.std(stock,axis=0)[3]
    stockFrame = pd.DataFrame(stock).pct_change()
    mean = stockFrame.mean()
    std = stockFrame.std()
    mostRecentPrice = stock[-1]

    
    
    t = np.linspace(start=0, stop=predictionWindow,num=predictionWindow, dtype=int)
    W = np.random.standard_normal(size = predictionWindow) 
    W = np.cumsum(W)

    #Drift is the general movement of a stock over time
    drift = (mean - (std**2) * 0.5 )

    #Shock represents the random volatility of a stock's price
    shock = std

    predictions = np.zeros((ntrials, predictionWindow))
    #Set first element of array as most recent price of the given period
    for i in range(ntrials):
       predictions[i][0] = mostRecentPrice
 

    for i in range(ntrials):
        for j in range(predictionWindow):
            predictions[i][j] = mostRecentPrice * np.exp(drift * t[j] + shock * W[j] )
    return predictions
 

S = predict(ntrials=5, predictionWindow= 21)
x = np.linspace(start=0, stop=21,num=21, dtype=int)
#for i in range(len(S)):
plt.plot(S)

plt.savefig('stackoverflow')

