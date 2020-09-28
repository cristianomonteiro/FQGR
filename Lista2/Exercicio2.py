import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import math

from os.path import dirname, abspath
parentDirectory = dirname(dirname(abspath(__file__)))

def pricesReturnsMean(prices, year):
    pricesYear = prices[year].drop(columns=['IBOV']).dropna()
    returns = pricesYear.pct_change().dropna()
    meanReturns = returns.mean()
    sumE = returns.cov()

    return pricesYear, returns, meanReturns, sumE

def varPortfolio(model, sumE):
    w = np.reshape([model.w[asset].value for asset in sumE], (-1,1))
    var = np.matmul(w.transpose(), np.matmul(sumE.to_numpy(), w))

    return var[0]

def createDefaultModel(sumE, noShorting=True):
    model = pyo.ConcreteModel()
    model.w = pyo.Var(sumE.columns, domain=pyo.Reals)

    model.objective = pyo.Objective(expr = 1/2*sum(sumE[i][j] * model.w[i] * model.w[j] for i in sumE for j in sumE))
    model.CstrSumW1 = pyo.Constraint(expr = sum(model.w[asset] for asset in sumE) == 1)

    if noShorting == True:
        model.noShortingCstr = pyo.ConstraintList()
        for asset in sumE:
            model.noShortingCstr.add(model.w[asset] >= 0)

    return model

def resultsSimulation(letter, model, meanRet, sumE, shouldPrintPlot=True):
    meanPortfolio = sum([model.w[asset].value * meanRet[asset] for asset in meanRet.index.values])
    stdPortfolio = math.sqrt(varPortfolio(model, sumE))

    if shouldPrintPlot == True:
        print("##############################################################")
        print('Letra ' + letter + f')\n\tMédia do portfólio: {meanPortfolio:.5f}\n\tDesvio padrão: {stdPortfolio:.5f}')
        print(f'\tSoma dos pesos: {sum(model.w[asset].value for asset in sumE):.5f}')
        threshold = 0.0001
        wSelected = pd.DataFrame([[asset, model.w[asset].value] for asset in sumE if abs(model.w[asset].value) >= threshold],
                                columns=['Asset', 'w'])#.set_index('Asset')
        wSelected.sort_values(['w']).plot.bar(x='Asset', y='w')
        plt.ylabel("Peso")
        plt.xlabel("Ativo")
        plt.tight_layout()    
        plt.show()

    return meanPortfolio, stdPortfolio

def letterA(meanRet, sumE, shouldPrintPlot=True):
    model = createDefaultModel(sumE, noShorting=False)

    results = pyo.SolverFactory('ipopt').solve(model)
    model.solutions.store_to(results)
    #print(results)

    meanPortfolio, stdPortfolio = resultsSimulation('a', model, meanRet, sumE, shouldPrintPlot)

    return meanPortfolio, stdPortfolio, model

def letterB(meanRet, sumE, shouldPrintPlot=True):
    model = createDefaultModel(sumE)

    results = pyo.SolverFactory('ipopt').solve(model)
    model.solutions.store_to(results)
    #print(results)

    meanPortfolio, stdPortfolio = resultsSimulation('b', model, meanRet, sumE, shouldPrintPlot)

    return meanPortfolio, stdPortfolio, model

def letterC(meanRet, sumE, shouldPrintPlot=True):
    model = createDefaultModel(sumE)
    minReturn = 0.003
    model.cstrMinRet = pyo.Constraint(expr = sum(model.w[asset] * meanRet[asset] for asset in sumE) >= minReturn)

    results = pyo.SolverFactory('ipopt').solve(model)
    model.solutions.store_to(results)
    #print(results)

    meanPortfolio, stdPortfolio = resultsSimulation('c', model, meanRet, sumE, shouldPrintPlot)

    return meanPortfolio, stdPortfolio, model

def letterD(prices, returns, meanRet, sumE):
    meanA, stdA, modelA = letterA(meanRet, sumE, shouldPrintPlot=False)
    meanB, stdB, modelB = letterB(meanRet, sumE, shouldPrintPlot=False)
    meanC, stdC, modelC = letterC(meanRet, sumE, shouldPrintPlot=False)

    seriesIBOV = prices['IBOV']/prices['IBOV'][0]
    seriesA = modelA.w['IBOV'].value * (1 + returns['IBOV'])
    seriesB = modelB.w['IBOV'].value * (1 + returns['IBOV'])
    seriesC = modelC.w['IBOV'].value * (1 + returns['IBOV'])
    for asset in returns.drop(columns=['IBOV']):
        seriesA += modelA.w[asset].value * (1 + returns[asset])
        seriesB += modelB.w[asset].value * (1 + returns[asset])
        seriesC += modelC.w[asset].value * (1 + returns[asset])
    seriesA = np.cumprod(seriesA)
    seriesB = np.cumprod(seriesB)
    seriesC = np.cumprod(seriesC)

    print(seriesIBOV)
    print(seriesA)
    print(seriesB)
    print(seriesC)

    ax = seriesIBOV.plot(label='IBOV', legend=True)
    seriesA.plot(ax=ax, label='P1', legend=True)
    seriesB.plot(ax=ax, label='P2', legend=True)
    seriesC.plot(ax=ax, label='P3', legend=True)

    stepBetweenMonths = 1
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=stepBetweenMonths))

    #Forcing inclusion of first month tick
    xTickValue = ax.get_xticks()[0] - 30
    ax.set_xticks(np.append(ax.get_xticks(), xTickValue))
    #Forcing inclusion of last month tick
    #xTickValue = ax.get_xticks()[-2] + 30
    #ax.set_xticks(np.append(ax.get_xticks(), xTickValue))

    ax.set_xlabel("Datas")
    ax.set_ylabel("Retorno (R$)")

#set formatter
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    plt.tight_layout()
    plt.show()

def letterE(meanRet, sumE):
    meanA, stdA, modelA = letterA(meanRet, sumE, shouldPrintPlot=False)
    meanB, stdB, modelB = letterB(meanRet, sumE, shouldPrintPlot=False)
    meanC, stdC, modelC = letterC(meanRet, sumE, shouldPrintPlot=False)

    print('##########################################################')
    print('Letra e) Out of Sample:')
    print(f'a)\tMédia do portfólio: {meanA:.5f}\n\tDesvio padrão: {stdA:.5f}')
    print(f'b)\tMédia do portfólio: {meanB:.5f}\n\tDesvio padrão: {stdB:.5f}')
    print(f'c)\tMédia do portfólio: {meanC:.5f}\n\tDesvio padrão: {stdC:.5f}')

prices = pd.read_csv(parentDirectory + '/Dados/IBOV.csv', parse_dates=['Date'], index_col='Date')

pricesInSample, returnsInSample, meanRetInSample, sumEInSample = pricesReturnsMean(prices, '2019')
pricesOutOfSample, returnsOutOfSample, meanRetOutOfSample, sumEOutOfSample = pricesReturnsMean(prices, '2020')

letterA(meanRetInSample, sumEInSample)
letterB(meanRetInSample, sumEInSample)
letterC(meanRetInSample, sumEInSample)
letterD(pricesOutOfSample, returnsOutOfSample, meanRetInSample, sumEInSample)
letterE(meanRetOutOfSample, sumEOutOfSample)