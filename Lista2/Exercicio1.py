import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
#from pandas.plotting import register_matplotlib_converters
#register_matplotlib_converters()

from os.path import dirname, abspath
parentDirectory = dirname(dirname(abspath(__file__)))

def checkPrintStationarity(pricesInterval, asset):
        prices = pricesInterval[asset]
        posPValue = 1
        pValuePrices = adfuller(prices)[posPValue]
        print('P-value for ' + asset + f" not being stationary: {pValuePrices:.5f}")
        diffPrices = np.diff(pricesInterval[asset])
        pValueI1Prices = adfuller(diffPrices)[posPValue]
        print('P-value for ' + asset + f" I(1) not being stationary: {pValueI1Prices:.5f}")

        return prices, diffPrices, pValuePrices, pValueI1Prices

def plotResiduesGraph(residues, title):
#        font = {'family' : 'normal',
#                'weight' : 'bold',
#                'size'   : 16}

#        plt.rc('font', **font)
        ax = residues.plot(title=title)
        ax.axhline(residues.mean(), color='r')

#set monthly locator
        stepBetweenMonths = 2
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=stepBetweenMonths))

        #Forcing inclusion of first month tick
        xTickValue = ax.get_xticks()[0] - 59
        ax.set_xticks(np.append(ax.get_xticks(), xTickValue))
        #Forcing inclusion of last month tick
        xTickValue = ax.get_xticks()[-2] + 60
        ax.set_xticks(np.append(ax.get_xticks(), xTickValue))

        ax.set_xlabel("Datas")
        ax.set_ylabel("Resíduos")

#set formatter
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))

        plt.tight_layout()
        plt.show()

prices = pd.read_csv(parentDirectory + '/Dados/IBOV.csv', parse_dates=['Date'], index_col='Date')

pricesInterval = prices['2018':'2019'].dropna()

#CHECKING IF PETR3 IS STATIONARY
pricesPETR3, diffPricesPETR3, pValuePETR3, pValueI1Prices = checkPrintStationarity(pricesInterval, 'PETR3')

#SELECTING PAIRS
pValueThreshold = 0.05
selectedPairs = []
for asset in pricesInterval.drop(columns=['PETR3']):
        pricesAsset, diffPricesAsset, pValuePrices, pValueI1Asset = checkPrintStationarity(pricesInterval, asset)

        if pValueI1Asset < pValueThreshold:
#LINEAR REGRESSION
                pricesPETR3Reshaped = pricesPETR3.values.reshape(-1,1)
                linReg = LinearRegression().fit(pricesPETR3Reshaped, pricesAsset)
                posBeta = 0
                residues = linReg.intercept_ + linReg.coef_[posBeta]*pricesPETR3 - pricesAsset
                posPValue = 1
                pValueResidues = adfuller(residues)[posPValue]
                print(f'P-value for residues: {pValueResidues:.5f} Alpha: {linReg.intercept_:.5f} Beta: {linReg.coef_[posBeta]:.5f}')
                
                if pValueResidues < pValueThreshold:
                        print('Asset ' + asset + ' selected as cointegrated with PETR3.')
                        selectedPairs.append(asset)
                        plotResiduesGraph(residues, 'PETR3 x ' + asset)
                        print('==============================================================')
                        if len(selectedPairs) >= 3:
                                break
