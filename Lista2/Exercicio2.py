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

    pricesIBOV = prices[year]['IBOV'].dropna()

    return pricesYear, returns, meanReturns, sumE, pricesIBOV

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
        ax = wSelected.sort_values(['w']).plot.bar(x='Asset', y='w')
        ax.legend(loc=2)
        plt.ylabel("Peso")
        plt.xlabel("Ativo")
        plt.grid(axis='y')
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

def calcPricesSeriesBuyAndHold(model, budget, prices):
    series = (model.w['BOVA11'].value * budget/prices['BOVA11'][0]) * prices['BOVA11']

    for asset in prices.drop(columns=['BOVA11']):
        series += (model.w[asset].value * budget/prices[asset][0]) * prices[asset]
    
    return series

def calcPricesSeries(model, budget, prices):
    series = calcPricesSeriesBuyAndHold(model, budget, prices)

    earlierX = {}
    earlierBudget = series[0]
    newBudget = 0
    for i in range(1, len(series)):
        for asset in prices:
            earlierX[asset] = model.w[asset].value * earlierBudget/prices[asset][i-1]
            newBudget += earlierX[asset] * prices[asset][i]

        series[i] = newBudget
        earlierBudget = newBudget
        newBudget = 0

    return series

def letterD(prices, pricesIBOV, meanRet, sumE, functionToCall):
    meanA, stdA, modelA = letterA(meanRet, sumE, shouldPrintPlot=False)
    meanB, stdB, modelB = letterB(meanRet, sumE, shouldPrintPlot=False)
    meanC, stdC, modelC = letterC(meanRet, sumE, shouldPrintPlot=False)

    seriesIBOV = pricesIBOV/pricesIBOV[0]
    budget = 1.0

    seriesA = functionToCall(modelA, budget, prices)
    seriesB = functionToCall(modelB, budget, prices)
    seriesC = functionToCall(modelC, budget, prices)

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
    plt.grid()
    plt.tight_layout()
    plt.show()

    return seriesIBOV, seriesA, seriesB, seriesC

def letterE(seriesIBOV, seriesA, seriesB, seriesC):
    print('##########################################################')
    print('Letra e) Out of Sample:')
    print(f'a)\tMédia dos retornos: {seriesA.pct_change().dropna().mean():.5f}\n\tDesvio padrão: {seriesA.pct_change().dropna().std():.5f}')
    print(f'b)\tMédia dos retornos: {seriesB.pct_change().dropna().mean():.5f}\n\tDesvio padrão: {seriesB.pct_change().dropna().std():.5f}')
    print(f'c)\tMédia dos retornos: {seriesC.pct_change().dropna().mean():.5f}\n\tDesvio padrão: {seriesC.pct_change().dropna().std():.5f}')
    print(f'IBOV)\tMédia dos retornos: {seriesIBOV.pct_change().dropna().mean():.5f}\n\tDesvio padrão: {seriesIBOV.pct_change().dropna().std():.5f}')

prices = pd.read_csv(parentDirectory + '/Dados/IBOV.csv', parse_dates=['Date'], index_col='Date')

pricesInSample, returnsInSample, meanRetInSample, sumEInSample, pricesIBOVInSample = pricesReturnsMean(prices, '2019')
pricesOutOfSample, returnsOutOfSample, meanRetOutOfSample, sumEOutOfSample, pricesIBOVOutOfSample = pricesReturnsMean(prices, '2020')

letterA(meanRetInSample, sumEInSample)
letterB(meanRetInSample, sumEInSample)
letterC(meanRetInSample, sumEInSample)
seriesIBOV, seriesAOutOfSample, seriesBOutOfSample, seriesCOutOfSample = letterD(pricesOutOfSample, pricesIBOVOutOfSample, meanRetInSample, sumEInSample, calcPricesSeriesBuyAndHold)
letterE(seriesIBOV, seriesAOutOfSample, seriesBOutOfSample, seriesCOutOfSample)
seriesIBOV, seriesAOutOfSample, seriesBOutOfSample, seriesCOutOfSample = letterD(pricesOutOfSample, pricesIBOVOutOfSample, meanRetInSample, sumEInSample, calcPricesSeries)
letterE(seriesIBOV, seriesAOutOfSample, seriesBOutOfSample, seriesCOutOfSample)