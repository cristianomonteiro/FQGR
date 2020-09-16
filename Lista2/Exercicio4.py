import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
#from pandas.plotting import register_matplotlib_converters
#register_matplotlib_converters()

from os.path import dirname, abspath
parentDirectory = dirname(dirname(abspath(__file__)))

def printTable(retornos, asset):
    texto = str(retornos[asset].mean())
    texto += " " + str(retornos[asset].std())
    texto += " " + str(retornos[asset].skew())
    texto += " " + str(retornos[asset].kurtosis())

    retornosOrdenados = retornos[asset].sort_values()
    posicao = round(250*0.05)
    texto += " " + str(retornosOrdenados[posicao])

    print(texto)

prices = pd.read_csv(parentDirectory + '/Dados/IBOV.csv', parse_dates=['Date'], index_col='Date')
#print(prices[prices.columns[:10]].tail())

#Últimos 250 retornos
retornos = prices.dropna().tail(251).pct_change()
printTable(retornos, 'BOVA11')
printTable(retornos, 'PETR4')
printTable(retornos, 'ABEV3')
printTable(retornos, 'SUZB3')
printTable(retornos, 'ITSA4')

print("portfolio")
portfolio = retornos['BOVA11']*0.2 + retornos['PETR4']*0.2 + retornos['ABEV3']*0.2 + retornos['SUZB3']*0.2 + retornos['ITSA4']*0.2
texto = str(portfolio.mean()) + " " + str(portfolio.std()) + " " + str(portfolio.skew()) + " " + str(portfolio.kurtosis())

retornosOrdenados = portfolio.sort_values()
posicao = round(250*0.05)
texto += " " + str(retornosOrdenados[posicao])
print(texto)

ativos = ['BOVA11', 'PETR4', 'ABEV3', 'SUZB3', 'ITSA4']
primeiroSomatorio = 0
for i in range(0, 5):
    primeiroSomatorio += (0.2**2) * retornos[ativos[i]].std()**2

segundoSomatorio = 0
for i in range(0, 4):
    for j in range(i + 1, 5):
        segundoSomatorio += 2*0.2*0.2*retornos[ativos[i]].std()*retornos[ativos[j]].std()*retornos.corr()[ativos[i]][ativos[j]]

varianciaPortfolio = primeiroSomatorio + segundoSomatorio
print(varianciaPortfolio)
print(varianciaPortfolio**(1/2))