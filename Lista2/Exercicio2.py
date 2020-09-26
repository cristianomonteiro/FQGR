import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import math

from os.path import dirname, abspath
parentDirectory = dirname(dirname(abspath(__file__)))

def pricesReturnsMean(prices, year):
    pricesYear = prices[year].dropna()
    returns = pricesYear.pct_change()
    meanReturns = returns.mean()
    sumE = returns.cov()

    return prices, returns, meanReturns, sumE

def varPortfolio(model, sumE):
    w = np.reshape([model.w[asset].value for asset in sumE], (-1,1))
    var = np.matmul(w.transpose(), np.matmul(sumE.to_numpy(), w))

    return var[0]

def createDefaultModel(sumE):
    model = pyo.ConcreteModel()
    model.w = pyo.Var(sumE.columns, domain=pyo.Reals)

    model.objective = pyo.Objective(expr = 1/2*sum(sumE[i][j] * model.w[i] * model.w[j] for i in sumE for j in sumE))
    model.CstrSumW1 = pyo.Constraint(expr = sum(model.w[asset] for asset in sumE) == 1)

    return model

def resultsSimulation(letter, model, meanRet, sumE):
    threshold = 0.0001
    wSelected = pd.DataFrame([[asset, model.w[asset].value] for asset in sumE if abs(model.w[asset].value) >= threshold],
                            columns=['Asset', 'w'])#.set_index('Asset')
    wSelected.plot.bar(x='Asset', y='w')
    plt.ylabel("Peso")
    plt.xlabel("Ativo")
    plt.tight_layout()    
    plt.show()

    meanPortfolio = sum([model.w[asset].value * meanRet[asset] for asset in meanRet.index.values])
    stdPortfolio = math.sqrt(varPortfolio(model, sumE))
    print("##############################################################")
    print('Letra ' + letter + f')\n\tMédia do portfólio: {meanPortfolio:.5f}\n\tDesvio padrão: {stdPortfolio:.5f}')

    return meanPortfolio, stdPortfolio

def letterA(meanRet, sumE):
    model = createDefaultModel(sumE)
    results = pyo.SolverFactory('ipopt').solve(model)
    model.solutions.store_to(results)
    #print(results)

    meanPortfolio, stdPortfolio = resultsSimulation('a', model, meanRet, sumE)

    return meanPortfolio, stdPortfolio

prices = pd.read_csv(parentDirectory + '/Dados/IBOV.csv', parse_dates=['Date'], index_col='Date')

pricesInSample, returnsInSample, meanRetInSample, sumEInSample = pricesReturnsMean(prices, '2019')
pricesOutOfSample, returnsOutOfSample, meanRetOutOfSample, sumEOutOfSample = pricesReturnsMean(prices, '2020')

letterA(meanRetInSample, sumEInSample)
