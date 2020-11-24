script.dir <- dirname(sys.frame(1)$ofile);
setwd(script.dir);
source("../../Scripts/optimisation/model.r");
source("../../../../Comum/graficos.r");
source("funcoesAuxiliares.r");
source("funcoesAuxiliaresAdicionais.r");
options(scipen=999);

saveGraphics = 1;

#######################################################
#### LOAD DATA
loadPackage("RCurl");
data <- getURL("http://www.dcc.ufmg.br/~arbex/portfolios/IBOV.csv", ssl.verifypeer=0L, followlocation=1L);
prices = read.csv(text = data, header = TRUE, sep = ",", stringsAsFactors = FALSE);

begin  = which(prices[,1] > 20190000)[1];
end    = which(prices[,1] > 20200000)[1];
os.prices = prices[end:nrow(prices), ];
is.prices = prices[begin:end, ];

#prices = prices[, colSums(prices == 0) == 0];

is.assets = is.prices[,3:ncol(is.prices)];
os.assets = os.prices[,3:ncol(os.prices)];

# Calcula matriz de retornos (cada coluna contém os retornos de um ativo)
is.returns = apply(is.assets, 2, function(x){diff(x)/x[-length(x)]} ); 
os.returns = apply(os.assets, 2, function(x){diff(x)/x[-length(x)]} ); 

# Calcula vetor de retornos esperados e matriz de covariancias
N = ncol(is.returns);
mu = colMeans(is.returns);
numAssets = length(mu);
names = colnames(is.returns);
Sigma = cov(is.returns);
#######################################################


#######################################################
#### CREATE MARKOWITZ MODEL
createMarkowitzModel <- function(mu, Sigma, minExpectedReturn, maxWeight = 1, shorting = 0) {
    
    N = length(mu);
    minWeight = 0;
    if (shorting) minWeight = -Inf;

    model = initialiseModel();
    #model$setModelFilename("markowitz.lp");
    for (i in 1 : N) model$addVariable(paste0("w", i), 0, minWeight, maxWeight); 
    model$addQuadraticMatrix(Sigma);

    # Restricao de retorno esperado minimo
    elements = paste0("w", seq(1, N));
    model$addConstraint(">=", minExpectedReturn, elements, mu, "retornoEsperadoMinimo");

    # Soma dos pesos = 1
    values   = rep(1, N);
    model$addConstraint("=", 1, elements, values, "somaDosPesos");

    return(model);
}
#######################################################

#######################################################
#### CREATE CVaR MODEL
createCVaRModel <- function(scenarios, alpha, minExpectedReturn, maxWeight = 1, shorting = 0) {
    
    T = nrow(scenarios);
    N = ncol(scenarios);
    prob = 1/T;
    minWeight = 0;
    if (shorting) minWeight = -Inf;

    model = initialiseModel();
    model$setModelFilename("cvar.lp");
    model$setSolverDebug(0);
    model$setDirection(1);
    for (i in 1 : N) model$addVariable(paste0("w", i), 0, minWeight, maxWeight); 
    for (t in 1 : T) model$addVariable(paste0("d", t), -1/(T*alpha), 0, Inf); 
    model$addVariable("V", 1, -Inf, Inf);

    # Restricoes CVaR
    for (t in 1 : T) {
        elements = paste0("w", seq(1, N));
        values   = round(scenarios[t,], 10);
        elements = c(elements, paste0("d", t));
        values   = c(values, 1);
        elements = c(elements, "V");
        values   = c(values, -1);
        model$addConstraint(">=", 0, elements, values, paste0("cvarDifference", t));
    }

    # Restricao de retorno esperado minimo
    elements = paste0("w", seq(1, N));
    model$addConstraint(">=", minExpectedReturn, elements, colMeans(scenarios), "retornoEsperadoMinimo");

    # Soma dos pesos = 1
    values   = rep(1, N);
    model$addConstraint("=", 1, elements, values, "somaDosPesos");

    return(model);
}


###################
minReturn = 0.001;
maxWeight = 0.15;


#######################################################
#######################################################
###### MARKOWITZ
model = createMarkowitzModel(mu, Sigma, minReturn, maxWeight = maxWeight);
model$solve();
printf("-------------------------\n");
printf("Solving with minimum return 0.3%%\n");
printf("Solver took %.2fs\n", model$solverTime);
printf("Solution status: %s\n", model$status);
ret.vars     = character(0);
ret.weights  = numeric(0);
ret.solution = numeric(0);
printf("Solution objective value: %.5f\n", model$objValue);
for (i in 1 : model$numVariables) {
    val = model$getSolutionValue(model$variables[i]);
    ret.solution = c(ret.solution, val);
    if (val >= 0.0001 || val <= -0.0001) {
        ret.vars    = c(ret.vars   , colnames(is.assets)[i]);
        ret.weights = c(ret.weights, val                );
    }
}
ret.return = mu %*% ret.solution;
ret.var    = t(ret.solution) %*% Sigma %*% ret.solution;
ret.sd     = sqrt(ret.var)
ret.os     = os.returns %*% ret.solution;
ret.port   = rep(1, length(ret.os)+1);
for (i in 2 : length(ret.port)) {
    ret.port[i] = ret.port[i-1]*(1 + ret.os[i-1]);
}
printf("Portfolio return  : %.5f\n", ret.return);
printf("Portfolio variance: %.5f\n", ret.var);
printf("Portfolio stdev   : %.5f\n", ret.sd);
printf("-------------------------\n");
#######################################################

#######################################################
#######################################################
###### CVAR
model = createCVaRModel(is.returns, 0.05, minReturn, maxWeight = maxWeight);
model$solve();
printf("-------------------------\n");
printf("Solving with minimum return 0.3%%\n");
printf("Solver took %.2fs\n", model$solverTime);
printf("Solution status: %s\n", model$status);
cvr.vars     = character(0);
cvr.weights  = numeric(0);
cvr.solution = numeric(0);
printf("Solution objective value: %.5f\n", model$objValue);
for (i in 1 : numAssets) {
    val = model$getSolutionValue(model$variables[i]);
    cvr.solution = c(cvr.solution, val);
    if (val >= 0.0001 || val <= -0.0001) {
        cvr.vars    = c(cvr.vars   , colnames(is.assets)[i]);
        cvr.weights = c(cvr.weights, val                );
    }
}
cvr.return = mu %*% cvr.solution;
cvr.var    = t(cvr.solution) %*% Sigma %*% cvr.solution;
cvr.sd     = sqrt(cvr.var)
cvr.os     = os.returns %*% cvr.solution;
cvr.port   = rep(1, length(cvr.os)+1);
for (i in 2 : length(cvr.port)) {
    cvr.port[i] = cvr.port[i-1]*(1 + cvr.os[i-1]);
}
printf("Portfolio return  : %.5f\n", cvr.return);
printf("Portfolio variance: %.5f\n", cvr.var);
printf("Portfolio stdev   : %.5f\n", cvr.sd);
printf("-------------------------\n");
#######################################################




#######################################################
## BAR CHARTS
ggOptions = getGGOptions();
ggOptions$saveGraphics = saveGraphics;
ggOptions$height    = 6;
ggOptions$width     = 10;
ggOptions$axisTitleSize = 16;
ggOptions$axisXLabelSize = 8;
ggOptions$axisYLabelSize = 14;
ggOptions$xTitle = "Ativo";
ggOptions$yTitle = "Peso";
ggOptions$percentageInYAxis = 1;
ggOptions$removeXAxisSpace  = 0;
ggOptions$barChart = 1;
ggOptions$axisXLabelAngle = 90;

ggOptions$imageName = "L3_2a.pdf";
ggOptions$title  = "Markowitz";
plotGraph(ret.weights, xValues = ret.vars, ggOptions = ggOptions);

ggOptions$imageName = "L3_2b.pdf";
ggOptions$title  = "CVaR";
plotGraph(cvr.weights, xValues = cvr.vars, ggOptions = ggOptions);
#######################################################


ibov = os.prices[,2];
ibov = ibov / ibov[1];
dates = os.prices[,1];
ibv.os = os.returns[,2];

ggOptions$axisXLabelAngle = 0;
ggOptions$percentageInYAxis = 0;
ggOptions$barChart  = 0;
ggOptions$lineChart = 1;
ggOptions$xTitle = "Tempo";
ggOptions$yTitle = "Valor";
ggOptions$formatXAxisAsDate = 1;
ggOptions$removeYAxisSpace  = 1;
ggOptions$removeXAxisSpace  = 1;
ggOptions$axisXLabelSize = 14;
ggOptions$addLegend = 1;
#ggOptions$legend = c("iBOV", "Min. variance", "Min. return 0.3%");
ggOptions$legend = c("iBOV", "Markowitz", "CVaR");
ggOptions$imageName = "L3_2c.pdf";
ggOptions$title  = "Portfolios, comparação";

plotGraph(ibov, ret.port, cvr.port, xValues = dates, ggOptions = ggOptions);
#plotGraph(ibov, min.port, ret.port, xValues = dates, ggOptions = ggOptions);


printf("Retornos out-of-sample:\n");
printf("IBOV       : R = %8.5f, SD = %8.5f, CVAR = %8.5f, SHARPE = %8.5f, STARR = %8.5f, DD = %5.2f%%\n", stats.mean(ibv.os), stats.sd(ibv.os), stats.CVaR(ibv.os), stats.sharpeRatio(ibv.os), 
        stats.STARRRatio(ibv.os), stats.maxDrawdown(ibov)); 
printf("Markowitz  : R = %8.5f, SD = %8.5f, CVAR = %8.5f, SHARPE = %8.5f, STARR = %8.5f, DD = %5.2f%%\n", stats.mean(ret.os), stats.sd(ret.os), stats.CVaR(ret.os), stats.sharpeRatio(ret.os), 
        stats.STARRRatio(ret.os), stats.maxDrawdown(ret.port)); 
printf("CVaR       : R = %8.5f, SD = %8.5f, CVAR = %8.5f, SHARPE = %8.5f, STARR = %8.5f, DD = %5.2f%%\n", stats.mean(cvr.os), stats.sd(cvr.os), stats.CVaR(cvr.os), stats.sharpeRatio(cvr.os), 
        stats.STARRRatio(cvr.os), stats.maxDrawdown(cvr.port));




