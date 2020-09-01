import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
#from pandas.plotting import register_matplotlib_converters
#register_matplotlib_converters()

from os.path import dirname, abspath
parentDirectory = dirname(dirname(abspath(__file__)))

prices = pd.read_csv(parentDirectory + '/Dados/IBOV.csv', parse_dates=['Date'], index_col='Date')
print(prices[prices.columns[:10]].tail())