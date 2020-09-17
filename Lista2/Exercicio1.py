import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
#from pandas.plotting import register_matplotlib_converters
#register_matplotlib_converters()

from os.path import dirname, abspath
parentDirectory = dirname(dirname(abspath(__file__)))

prices = pd.read_csv(parentDirectory + '/Dados/IBOV.csv', parse_dates=['Date'], index_col='Date')

pricesInterval = prices['2018':'2019'].dropna()

#CHECKING IF PETR3 IS STATIONARY
pricesPETR3 = pricesInterval['PETR3']
posPValue = 1
print(f"P-value for PETR3 not being stationary: {adfuller(pricesPETR3)[posPValue]:.3f}")
diffPricesPETR3 = np.diff(pricesInterval['PETR3'])
print(f"P-value for PETR3 I(1) not being stationary: {adfuller(diffPricesPETR3)[posPValue]:.3f}")

#SELECTING PAIRS
pValueThreshold = 0.05
selectedPairs = []
for asset in pricesInterval.drop(columns=['PETR3']):
        pricesAsset = pricesInterval[asset]
        diffPricesAsset = np.diff(pricesAsset)
        pValueAsset = adfuller(diffPricesAsset)[posPValue]
        if pValueAsset < pValueThreshold:
                selectedPairs.append(asset)
                print('Asset: ' + asset + f' p-value: {pValueAsset:.3f}')

#LINEAR REGRESSION
                linReg = LinearRegression().fit([pricesPETR3], pricesAsset)
                posAlpha = 0
                residues = linReg.intercept_ + linReg.coef_[posAlpha]*pricesPETR3 - pricesAsset
                pValueResidues = adfuller(residues)[posPValue]
                if pValueResidues < pValueThreshold:
                        print(f'P-value: {pValueAsset:.3f} Alpha: {reg.coef_[posAlpha]:.3f} Beta: {reg.intercept_:.3f}')
                        if len(selectedPairs) >= 3:
                                break

corrPrecos2019 = pd.df([])
corrRetornos2019 = prices['2019':'2020-01-02'].dropna().pct_change().corr()['IBOV'].drop('IBOV')

print("Média das correlações dos preços: " + str(corrPrecos2019.mean()))
print("Média das correlações dos retornos: " + str(corrRetornos2019.mean()) + "\n")
print("Menor correlação dos preços: " + str(corrPrecos2019.min()) + " aconteceu com a ação: " + corrPrecos2019.idxmin())
print("Menor correlação dos retornos: " + str(corrRetornos2019.min()) + " aconteceu com a ação: " + corrPrecos2019.idxmin() + "\n")
print("Maior correlação dos preços: " + str(corrPrecos2019.max()) + " aconteceu com a ação: " + corrPrecos2019.idxmax())
print("Maior correlação dos retornos: " + str(corrRetornos2019.max()) + " aconteceu com a ação: " + corrPrecos2019.idxmax())

#Correlações de 2020
corrPrecos2020 = prices['2020'].corr()['IBOV'].drop('IBOV')
corrRetornos2020 = prices['2020'].pct_change().corr()['IBOV'].drop('IBOV')

print("Média das correlações dos preços: " + str(corrPrecos2020.mean()))
print("Média das correlações dos retornos: " + str(corrRetornos2020.mean()) + "\n")
print("Menor correlação dos preços: " + str(corrPrecos2020.min()) + " aconteceu com a ação: " + corrPrecos2019.idxmin())
print("Menor correlação dos retornos: " + str(corrRetornos2020.min()) + " aconteceu com a ação: " + corrPrecos2019.idxmin() + "\n")
print("Maior correlação dos preços: " + str(corrPrecos2020.max()) + " aconteceu com a ação: " + corrPrecos2019.idxmax())
print("Maior correlação dos retornos: " + str(corrRetornos2020.max()) + " aconteceu com a ação: " + corrPrecos2019.idxmax())

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)

corrPrecos2019.plot.hist(bins=50, xlim=(-1,1), ylim=(0,12))
plt.ylabel("Frequência")
plt.xlabel("Correlação")
plt.tight_layout()
plt.show()

corrPrecos2020.plot.hist(bins=50, xlim=(-1,1), ylim=(0,12))
plt.axes().axes.get_yaxis().get_label().set_visible(False)
plt.xlabel("Correlação")
plt.tight_layout()
plt.show()