import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
#from pandas.plotting import register_matplotlib_converters
#register_matplotlib_converters()

from os.path import dirname, abspath
parentDirectory = dirname(dirname(abspath(__file__)))

prices = pd.read_csv(parentDirectory + '/Dados/IBOV.csv', parse_dates=['Date'], index_col='Date')
#print(prices[prices.columns[:10]].tail())

#Correlações de 2019
corrPrecos2019 = prices['2019':'2020-01-02'].dropna().corr()['IBOV'].drop('IBOV')
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