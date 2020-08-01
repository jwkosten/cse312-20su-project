#Import statements

import math
import matplotlib.pyplot as plt 
import numpy as np
from pandas_datareader import data 

def getData(stockSym:str, startDate:str, endDate:str):
    stock = data.DataReader(stockSym, 'yahoo', start=startDate, end=endDate)
    return stock

stock = getData(stockSym='AAPL',startDate='1/10/2020',endDate='7/1/2020')

def predict(ntrials:int = 1000, predictionWindow:int = 21):
    timeElapsed = (stock.index[-1] - stock.index[0]).days
    mean = np.mean(stock, axis=0)[3]
    print(mean)
    std = np.std(stock,axis=0)[3]
    mostRecentPrice = stock['Close'][-1]
    print(std)
    #Drift is the general movment of a stock over time
    drift = (mean - (std**2)/2)
    #Schock represents the random volatility of a stock's price
    shock = std * np.random.rand()
    
    predictions = np.zeros((ntrials,predictionWindow))
    #Set first element of array as most recent price of the given period
    for i in range(ntrials):
        predictions[i][0] = mostRecentPrice
 


    for i in range(10):
        for j in range(predictionWindow - 1):
            predictions[i][j + 1] = (predictions[i][j] * np.exp(drift + shock))
            print(predictions)
    return predictions

# predictions = predict()
# plt.plot(predictions)
# plt.savefig('predictions2')
# plt.show()
