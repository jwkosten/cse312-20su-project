
#============================================================================================================#
# Written by An Keopradith & Joseph Kosten 
# This project uses a Monte Carlo simulation technique to model possible movements of stock prices
#============================================================================================================#


#============================================================================================================#
#Import statements for all used libraries
import math
import numpy as np
import matplotlib.pyplot as plt 
from pandas_datareader import data 
import pandas as pd 
import datetime as dt

#============================================================================================================#
# Function Purpose: simulating stock price movements 
# using the historical data of the given time period as the parameters 
# to simulate the possible paths for the interested time interval

    # param ntrials:                number of paths to simulate
    # param stockSym:               stock ticker/symbol of the interested company
    # param startDate_collectData:  starting date to collect historical data
    # param endDate_collectData:    end date of collecting historical data
    # param sim_startDate:          simulation start date
    # param sim_endDate:            simulation end date

    # return predictions:           A 2D array of generated ntrials stock price paths for the given simulated time period
    # return Predicted_avg_daily:   A 1D array the simulated avarage daily value
    # return predictedAvg:          The overall average value for the given simulated time period
    # return predictedStd:          The overall average standard deviation value for the given simulated time period
#============================================================================================================#
def predict(ntrials , stockSym  , startDate_collectData, endDate_collectData , sim_startDate , sim_endDate):
    #============================================================================================================#
    #Grab stock data for predictions using pandas data reader/data frame
    predictedStock = data.DataReader(stockSym, 'yahoo', start=startDate_collectData, end=endDate_collectData)['Close']
    stockFrame = pd.DataFrame(predictedStock).pct_change() 
    mean = stockFrame.mean()
    std = stockFrame.std()
    mostRecentPrice = predictedStock[-1]
    predictionWindow = np.busday_count(sim_startDate,sim_endDate)#counting the number of trading days
    #============================================================================================================#
    #Variables for GBM. 
    t = np.arange(predictionWindow)
    #Drift is the general movement of a stock over time
    drift = (mean - (std**2) * 0.5 )
    #Shock represents the random volatility of a stock's price
    shock = std
    #Initializing the empty array
    predictions = np.zeros((ntrials, predictionWindow))
    #============================================================================================================#
        #Simulation with ntrials    
    for i in range(ntrials):
        #set the first element of each array to the most recent price
        predictions[i][0] = mostRecentPrice 
        # Each time we make a new prediction, we want to get a array of random samples from the standard normal of length predictionWindow
        W = np.random.standard_normal(size = predictionWindow) 
        # Running sum of random samples from W. Determines the shock of each prediction at time t=j-1
        # Because we use stationary value predictions[i][0] (most recent price) as our starting point, each iteration uses the sum of standard normal samples
        # From 0, ... current as the sample to determine the shock of the current prediction. This in effect takes the sum of the shocks from 
        # predictions 0,... current - 1 and applies it to the most recent price along with its current shock as well
        W = np.cumsum(W) 
        for j in range(1, predictionWindow):
            predictions[i][j] = predictions[i][0] * np.exp(drift * t[j - 1] + shock * W[j - 1])
    #============================================================================================================#
        #Calculating the simulation results
    Predicted_avg_daily = np.mean(predictions, axis=0)
    predictedAvg = np.mean(Predicted_avg_daily)
    Predicted_Std_daily = np.std(predictions, axis=0)
    predictedStd = np.mean(Predicted_Std_daily)
    return predictions,Predicted_avg_daily, predictedAvg, predictedStd

#============================================================================================================#
# Function Purpose: Getting the actual stock prices for a given time period

    # param stockSym:               stock ticker/symbol of interested company
    # param startDate:              start date of the time period
    # param endDate:                end date of the time period

    # return stock_actual_prices:   A 1D array of actual stock prices of the given time period
    # return stockTrueAvg:          The overall average value  of the given time period
    # return stockTrueStd:          The standard deviation value  of the given time period
#============================================================================================================#
def actual_data( stock_sym,startDate, endDate):
    stock_actual_prices = data.DataReader(stock_sym, 'yahoo', start=startDate, end=endDate)['Close']
    stockTrueAvg = stock_actual_prices.mean()
    stockTrueStd = stock_actual_prices.std()
    return stock_actual_prices, stockTrueAvg, stockTrueStd

#============================================================================================================#
#Function Purpose: plot result
    # param stock_sym:              Stock ticker/symbol of interested company
    # param predictions:            A 2D array of generated ntrials stock price paths for a given simulated time period
    # param simulate_daily_mean:    A 1D array the simulated avarage daily value
    # param simulate_mean:          The overall average value for the given simulated time period
    # param true_daily_price:       A 1D array of actual stock prices of the given time period
    # param true_mean:              The overall average value  of the given time period
    # param startDate:              start date of the time period
    # param endDate:                end date of the time period
#============================================================================================================#
def plot(stock_sym,predictions,simulate_daily_mean, simulate_mean, true_daily_price,true_mean, startDate,endDate):
    plt.style.use('seaborn')    # choosing the seaborn style for plotting
    
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,sharex=True)  #choosing to graph both plot in 1 figure 

    #plotting the simulated prices for the given time period
    for i in range(len(predictions)):                           
        ax1.plot(np.arange(len(predictions[i])), predictions[i], alpha=0.2)
    ax1.plot(np.arange(len(simulate_daily_mean)), simulate_daily_mean, linewidth=3, color='c',label='Mean')
    #plotting the simulated avarage daily value of the given time period
    ax1.legend()
    ax1.set_title('Simulation (trials = ' + str(ntrials) +')', fontsize=12)
    ax1.set_xlabel('Days', fontsize=12)
    ax1.set_ylabel('Price (USD)', fontsize=12)

    #plotting the simulated vs actual avarage daily value for the given time period
    ax2.plot(np.arange(len(simulate_daily_mean)), simulate_daily_mean, label='Simulation',color='c')
    ax2.plot(np.arange(len(true_daily_price)), true_daily_price,label='Actual',color='g')
    #plotting the overall simulated avarage value of the given time period
    ax2.axhline(simulate_mean, label='Simulated mean',color='c', linestyle='--')
    ax2.text(len(simulate_daily_mean), simulate_mean,"{0:.2f}".format(simulate_mean), horizontalalignment='left', size='small', color='c')
    #plotting the overall true avarage value of the given time period
    ax2.axhline(true_mean, label='True mean',color='g', linestyle='--')
    ax2.text(len(true_daily_price), true_mean,"{0:.2f}".format(true_mean), horizontalalignment='left', size='small', color='g')
    ax2.legend()
    ax2.set_title('Simulation VS Actual', fontsize=12)
    ax2.set_xlabel('Days', fontsize=12)
    fig.suptitle(str(stock_sym) + ' stock from ' + str(startDate) + ' to ' + str(endDate), fontsize=16)
    fig.savefig(stock_sym + 'from' + str(startDate) + 'to' + str(endDate) )



#============================================================================================================#
# output/plot results
    # param ntrials:                number of paths to simulate
    # param stock_sym:              stock ticker/symbol of the interested company
    # param startDate_collectData:  starting date to collect data
    # param endDate_collectData:    end date of collecting data
    # param sim_startDate:          simulation start date
    # param sim_endDate:            simulation end date
#============================================================================================================# 
def test_result(ntrials,stock_sym,startDate_collectData, endDate_collectData,sim_startDate,sim_endDate):
    simulations,sim_daily,sim_mean,sim_std = predict(ntrials,stock_sym,startDate_collectData,endDate_collectData,sim_startDate,sim_endDate)
    true_price,true_mean,true_std = actual_data(stock_sym,sim_startDate,sim_endDate)
    plot(stock_sym,simulations,sim_daily,sim_mean,true_price,true_mean,sim_startDate,sim_endDate)
    print('number of trials: ' + str(ntrials))
    print(stock_sym + ' average simulation price from: '+ str(sim_startDate) + ' to ' +  str(sim_endDate) +' : ' + str(sim_mean))
    print(stock_sym + ' average simulation STD from: '+ str(sim_startDate) + ' to ' +  str(sim_endDate) +' : ' + str(sim_std))
    print(stock_sym + ' average actual price from: '+ str(sim_startDate) + ' to ' +  str(sim_endDate) +' : ' + str(true_mean))
    print(stock_sym + ' average actual STD from: '+ str(sim_startDate) + ' to ' +  str(sim_endDate) +' : ' + str(true_std))


#============================================================================================================# 
    # Test application 
#============================================================================================================# 
stocks = ['AMZN','MSFT','FB','GOOGL','AAPL','TSLA'] # List of interested stock
ntrials = 100 
for stock in stocks: # Simulating all the stocks in the list
    #================================================
    # Collect the first 3 quarter of 2019 data to do the simulation
    # Simulate the last quarter of 2019
    test_result(ntrials,stock,dt.date(2019,1,1),dt.date(2019,9,30),dt.date(2019,10,1),dt.date(2019,12,31))
    #================================================
    # Collect data of 3 months from April to July in order to do the simulation
    # Simulate the last quarter of 2019
    test_result(ntrials,stock,dt.date(2020,4,1),dt.date(2020,7,31),dt.date(2020,8,1),dt.date(2020,8,14))
    #================================================
    #more testing
    #test_result(ntrials,stock,dt.date(2019,1,1),dt.date(2019,12,31),dt.date(2020,1,1),dt.date(2020,8,14))
