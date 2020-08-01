#Import statements

import math
import matplotlib.pyplot as plt 
import numpy as np
from pandas_datareader import data 

apple = data.DataReader('AAPL', 'yahoo', start='1/1/2009')

apple.head()

time_elapsed = (apple.index[-1] - apple.index[0]).days

