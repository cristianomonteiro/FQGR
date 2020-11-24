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
    datesSlice = slice(year, '2020-01-02') if year == '2019' else year
    pricesYear = prices[datesSlice].drop(columns=['IBOV']).dropna()
    pricesIBOV = prices[datesSlice]['IBOV'].dropna()

    returns = pricesYear.pct_change().dropna()
    sumReturns = returns.sum()
    meanReturns = returns.mean()
    sumE = returns.cov()

    return pricesYear, returns, meanReturns, sumE, pricesIBOV, sumReturns

def varPortfolio(model, sumE):
    w = np.reshape([model.w[asset].value for asset in sumE], (-1,1))
    var = np.matmul(w.transpose(), np.matmul(sumE.to_numpy(), w))

    return var[0]

def createDefaultModel(sumE, meanRet, noShorting=True):
    model = pyo.ConcreteModel()
    model.w = pyo.Var(sumE.columns, domain=pyo.Reals)

    model.CstrSumW1 = pyo.Constraint(expr = sum(model.w[asset] for asset in sumE) == 1)

    minReturn = 0.0001
    model.cstrMinRet = pyo.Constraint(expr = sum(model.w[asset] * meanRet[asset] for asset in sumE) >= minReturn)
    
    maxW = 0.15
    model.maxWCstr = pyo.ConstraintList()
    for asset in sumE:
        model.maxWCstr.add(model.w[asset] <= maxW)

    if noShorting == True:
        model.noShortingCstr = pyo.ConstraintList()
        for asset in sumE:
            model.noShortingCstr.add(model.w[asset] >= 0)

    return model

def resultsSimulation(portfolio, model, meanRet, sumE):
    meanPortfolio = sum([model.w[asset].value * meanRet[asset] for asset in meanRet.index.values])
    stdPortfolio = math.sqrt(varPortfolio(model, sumE))

    print("##############################################################")
    print('Portfolio ' + portfolio + f')\n\tMédia do portfólio: {meanPortfolio:.5f}\n\tDesvio padrão: {stdPortfolio:.5f}')
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

def MarkowitzModel(meanRet, sumE):
    model = createDefaultModel(sumE, meanRet)
    model.objective = pyo.Objective(expr = 1/2*sum(sumE[i][j] * model.w[i] * model.w[j] for i in sumE for j in sumE))

    results = pyo.SolverFactory('ipopt').solve(model)
    model.solutions.store_to(results)

    return model

def CVaRModel(meanRet, sumE, sumReturns, alpha=0.05):
    model = createDefaultModel(sumE, meanRet)
    model.V = pyo.Var(domain=pyo.Reals)
    model.rpt = pyo.Var(domain=pyo.Reals)
    numT = len(sumE.index)
    pt = 1/numT
    model.dt = pyo.Var(range(numT), domain=pyo.Reals)
    model.objective = pyo.Objective(sense = pyo.maximize, expr = model.V - (1/alpha) * pt * sum(model.dt[i] for i in range(numT)))

    model.cstrT = pyo.ConstraintList()
    for t in range(numT):
        model.cstrT.add(model.dt[t] >= model.V - model.rpt)
        model.cstrT.add(model.dt[t] >= 0)
    
    model.rptCstr = pyo.Constraint(expr = model.rpt == sum(model.w[asset] * sumReturns[asset] for asset in sumE))

    results = pyo.SolverFactory('ipopt').solve(model)
    model.solutions.store_to(results)

    return model

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

def letterA(prices, pricesIBOV, sumReturns, meanRet, sumE, functionToCall):
    modelMarkowitz = MarkowitzModel(meanRet, sumE)
    modelCVaR = CVaRModel(meanRet, sumE, sumReturns)

    seriesIBOV = pricesIBOV/pricesIBOV[0]
    budget = 1.0

    seriesMarkowitz = functionToCall(modelMarkowitz, budget, prices)
    seriesCVaR = functionToCall(modelCVaR, budget, prices)

    ax = seriesIBOV.plot(label='IBOV', legend=True)
    seriesMarkowitz.plot(ax=ax, label='Markowitz', legend=True)
    seriesCVaR.plot(ax=ax, label='CVaR', legend=True)

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

    #resultsSimulation('Markowitz', modelMarkowitz, meanRetInSample, sumEInSample)

    return seriesIBOV, seriesMarkowitz, seriesCVaR

def CVaR(returns, alpha=0.05):
    scenarioOdds = 1/len(returns)
    numScenariosAtRisk = alpha/scenarioOdds
    sortedReturns = np.sort(returns)
    floorNumScenarios = int(np.floor(numScenariosAtRisk))

    atRiskSorted = sortedReturns[0:floorNumScenarios]
    CVaRValue = (1/alpha) * scenarioOdds * np.sum(atRiskSorted)
    remaining = numScenariosAtRisk - floorNumScenarios
    if remaining != 0:
        CVaRValue += (1/alpha) * remaining * scenarioOdds * sortedReturns[floorNumScenarios]
    
    return CVaRValue

def maxDrawdown(prices):
    maxValue = prices[0]
    maxDrawdownValue = 0
    currentDrawdown = 0

    for value in prices[1:]:
        if value > maxValue:
            maxValue = value
            currentDrawdown = 0
        
        else:
            currentDrawdown = (100 * (maxValue - value)) / maxValue

        if currentDrawdown > maxDrawdownValue:
            maxDrawdownValue = currentDrawdown
    
    return maxDrawdownValue * -1

def letterB(pricesIBOV, pricesMarkw, pricesCVaR):
    dictRet = {}
    dictRet['IBOV'] = [pricesIBOV.pct_change().dropna(), pricesIBOV]
    dictRet['Markowitz'] = [pricesMarkw.pct_change().dropna(), pricesMarkw]
    dictRet['CVaR'] = [pricesCVaR.pct_change().dropna(), pricesCVaR]

    print('##########################################################')
    print('Letra b) Out of Sample:')
    print('\tMédia dos retornos\tDesvio padrão\tCVaR 5%\t\tSharpe Ratio\tSTARR Ratio 5%\tDrawdown Máximo')
    for key in dictRet.keys():
        tab = '\t' if key == 'Markowitz' else '\t\t'
        returns = dictRet[key][0]
        prices = dictRet[key][1]
        retMean = returns.mean()
        retStd = returns.std()
        retCVaR = CVaR(returns)
        maxDrawPrices = maxDrawdown(prices)
        print(key + tab + f'{retMean:.5f}\t{retStd:.5f}\t\t{retCVaR:.5f}\t{retMean/retStd:.5f}\t{retMean/(-1*retCVaR):.5f}\t\t{maxDrawPrices:.5f}')

prices = pd.read_csv(parentDirectory + '/Dados/IBOV.csv', parse_dates=['Date'], index_col='Date')

pricesInSample, returnsInSample, meanRetInSample, sumEInSample, pricesIBOVInSample, sumReturnsInSample = pricesReturnsMean(prices, '2019')
pricesOutOfSample, returnsOutOfSample, meanRetOutOfSample, sumEOutOfSample, pricesIBOVOutOfSample, sumReturnsOutOfSample = pricesReturnsMean(prices, '2020')

seriesIBOV, seriesMarkowitzOutOfSample, seriesCVaROutOfSample = letterA(pricesOutOfSample, pricesIBOVOutOfSample, sumReturnsInSample, meanRetInSample, sumEInSample, calcPricesSeries)
letterB(seriesIBOV, seriesMarkowitzOutOfSample, seriesCVaROutOfSample)