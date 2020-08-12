#Import statements

import math
import matplotlib.pyplot as plt 
import numpy as np
from pandas_datareader import data 
import pandas as pd 




def predict(ntrials:int, predictionWindow:int, startDate:str, endDate:str, stockSym:str):
    #============================================================================================================#
    #Grab stock data for predictions using pandas data reader/data frame
    predictedStock = data.DataReader(stockSym, 'yahoo', start=startDate, end=endDate)['Close']
    stockFrame = pd.DataFrame(predictedStock).pct_change()
    mean = stockFrame.mean()
    std = stockFrame.std()
    mostRecentPrice = predictedStock[-1]
    


    #============================================================================================================#
    #Variables for GBM. Need to explain more
    t = np.arange(predictionWindow)

    #============================================================================================================#
    #These go in the forloop as they are the random elements for the price and need to be updated each iteration
    #W = np.random.standard_normal(size = predictionWindow) 
    #W = np.cumsum(W)

    #Drift is the general movement of a stock over time
    drift = (mean - (std**2) * 0.5 )

    #Shock represents the random volatility of a stock's price
    shock = std


    predictions = np.zeros((ntrials, predictionWindow))

    #============================================================================================================#
    #set the first element of each array to the most recent price
    for i in range(ntrials):
        predictions[i][0] = mostRecentPrice
    

    #============================================================================================================#
    for i in range(ntrials):
        #Each time we make a new prediction, we want to get a new random sample
        W = np.random.standard_normal(size = predictionWindow) 
        W = np.cumsum(W)
        for j in range(1, predictionWindow):
            predictions[i][j] = mostRecentPrice * np.exp(drift * t[j - 1] + shock * W[j - 1])
    return predictions


def plot_stocks(predictions, startDate:str, endDate:str, stockSym:str):
    #============================================================================================================#
    # Prediction graph
    #============================================================================================================#
    for i in range(len(predictions)):
        plt.plot(np.arange(len(predictions[i])), predictions[i])
    plt.title('Simulation of ' + stockSym + ' over ' + str(len(predictions[i])) + ' days', fontsize=20)
    plt.xlabel('Days', fontsize=18)
    plt.ylabel('Price in dollars', fontsize=16)
    plt.savefig("predicted" + stockSym + 'over' + str(len(predictions[i]))  + 'days')
    plt.close()


    #============================================================================================================#
    # True prices graph
    #============================================================================================================#
    stock = data.DataReader(stockSym, 'yahoo', start=startDate, end=endDate)['Close']

    plt.plot(np.arange(len(stock)), stock)
    plt.title('True ' + stockSym + ' Prices from ' + startDate + ' to ' + endDate)
    plt.xlabel('Days', fontsize=18)
    plt.ylabel('Price in dollars', fontsize=16)
    plt.savefig('true' + stockSym + 'over' + str(len(stock)) +'days')
    plt.close()

    #============================================================================================================#
    # Calculate true average and standard deviation vs predicted mean and standard deviation
    #============================================================================================================#

    stockPredictedAvg = np.mean(predictions, axis=0)
    stockPredictedAvg = np.mean(stockPredictedAvg)
    stockPredictedStd = np.std(predictions, axis=0)
    stockPredictedStd = np.mean(stockPredictedStd)

    stockTrueAvg = stock.mean()
    stockTrueStd = stock.std()
    
    print('#============================================================================================================#')
    print('Predicted mean for ' + stockSym + ' over ' + str(len(stock)) + ' days: ' + str(stockPredictedAvg))
    print('Predicted standard deviation for ' + stockSym + ' over ' + str(len(stock)) + ' days: ' + str(stockPredictedStd))
    print('True mean for ' + stockSym + ' over ' + str(len(stock)) + ' days: ' + str(stockTrueAvg))
    print('True standard deviation for ' + stockSym + ' over ' + str(len(stock)) + ' days: ' + str(stockTrueStd))
    print('#============================================================================================================#')



    #============================================================================================================#
    # We will do 2 simulations each for 3 different stocks: AAPL, GOOG, MSFT
    #
    # First we will use the first 3 quarters of data from 2019 to predict the final quarter of the year
    # So the date range for this is startDate='1/1/2019' endDate='9/30/2019' and predictionWindow = 64
    # The data we compare this to is the final quarter of these stocks so startDate='10/1/2019' endDate='12/31/2019'
    #
    # Next we use the first 3 weeks of a month (September 2019) to predict that final week of that month
    # So the date range for this is startDate='9/1/2019' endDate='9/21/2019' and predictionWindow = 7
    # The data we compare this to is the final week of September 2019 so startDate='9/22/2019' endDate='9/29/2019'
    #
    # At the end of each simulation, we will compare the mean and standard deviations of the stock prices to make 
    # Sure that they fall within a reasonable range of eachother
    #============================================================================================================#

#============================================================================================================#
# Predictions for 3 months
#============================================================================================================#
appleThreeMonthPrediction = predict(ntrials = 50, predictionWindow = 64, startDate = '1/1/2019', endDate = '9/30/2019', stockSym= 'AAPL')

plot_stocks(predictions=appleThreeMonthPrediction, startDate = '10/1/2019', endDate = '12/31/2019', stockSym= 'AAPL')

googleThreeMonthPrediction = predict(ntrials = 50, predictionWindow = 64, startDate = '1/1/2019', endDate = '9/30/2019', stockSym= 'GOOG')

plot_stocks(predictions=googleThreeMonthPrediction, startDate = '10/1/2019', endDate = '12/31/2019', stockSym= 'GOOG')

msftThreeMonthPrediction = predict(ntrials = 50, predictionWindow = 64, startDate = '1/1/2019', endDate = '9/30/2019', stockSym= 'MSFT')

plot_stocks(predictions=msftThreeMonthPrediction, startDate = '10/1/2019', endDate = '12/31/2019', stockSym= 'MSFT')

#============================================================================================================#
# Predictions for 1 week
#============================================================================================================#

appleOneWeekPrediction = predict(ntrials = 50, predictionWindow = 5, startDate = '9/1/2019', endDate = '9/21/2019', stockSym= 'AAPL')

plot_stocks(predictions=appleOneWeekPrediction, startDate = '9/22/2019', endDate = '9/29/2019', stockSym= 'AAPL')

googleOneWeekPrediction = predict(ntrials = 50, predictionWindow = 5, startDate = '9/1/2019', endDate = '9/21/2019', stockSym= 'GOOG')

plot_stocks(predictions=googleOneWeekPrediction, startDate = '9/22/2019', endDate = '9/29/2019', stockSym= 'GOOG')

msftThreeMonthPrediction = predict(ntrials = 50, predictionWindow = 5, startDate = '9/1/2019', endDate = '9/21/2019', stockSym= 'MSFT')

plot_stocks(predictions=msftThreeMonthPrediction, startDate = '9/22/2019', endDate = '9/29/2019', stockSym= 'MSFT')



