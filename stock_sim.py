#Import statements

import math
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd



goog = GOOGLEFINANCE("NASDAQ:GOOG", "price", DATE(2014,1,1), DATE(2014,12,31), "DAILY")
print(goog)
