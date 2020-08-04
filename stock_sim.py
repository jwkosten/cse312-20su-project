#Import statements

import math
import matplotlib.pyplot as plt 
import numpy as np
from pandas_datareader import data 
import pandas as pd 

def getData(stockSym:str, startDate:str, endDate:str):
    stock = data.DataReader(stockSym, 'yahoo', start=startDate, end=endDate)
    return stock



def predict(predictionWindow:int = 21, stockSym:str = 'AAPL'):
    stock = getData(stockSym,startDate='6/1/2020',endDate='7/1/2020')['Close']
    stockFrame = pd.DataFrame(stock).pct_change()
    mean = stockFrame.mean()
    std = stockFrame.std()
    mostRecentPrice = stock[-1]

    
    
    t = np.arange(predictionWindow)
    W = np.random.standard_normal(size = predictionWindow) 
    W = np.cumsum(W)

    #Drift is the general movement of a stock over time
    drift = (mean - (std**2) * 0.5 )

    #Shock represents the random volatility of a stock's price
    shock = std

    #initialize empty list for predictions
    predictions = []

    #Set first element of array as most recent price of the given period
    predictions.append(mostRecentPrice)
 

    for j in range(predictionWindow):
        predictions.append(mostRecentPrice * np.exp(drift * t[j] + shock * W[j] ))
    return predictions
 

def plot_data(ntrials:int = 1000, predictionWindow:int = 21, stockSym:str= 'AAPL'):
    for i in range(ntrials):
        S = predict(predictionWindow)
        plt.plot(np.arange(len(S)), S)
    plt.title('Simulation of ' + stockSym + ' over ' + str(predictionWindow) + ' days', fontsize=20)
    plt.xlabel('Days', fontsize=18)
    plt.ylabel('Price in dollars', fontsize=16)
    plt.savefig('beautifulStockInformation')

plot_data(100, 10)
