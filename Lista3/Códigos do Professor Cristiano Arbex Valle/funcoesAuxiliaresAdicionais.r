stats.VaR <- function(x, probability = 0.05) {
    
    # Package PerformanceAnalytics:
    #
    # VaR(xts(b$portfolio, order.by = as.Date(as.character(b$dates), "%Y%m%d")), method = "historical", p = 0.95)
    # 
    # It uses quantile:
    #
    # quantile(b$portfolio, probs = 0.05)
    #
    # The method below is equivalent to quantile type 7
    # quantile(b$portfolio, probs = 0.05, type = 7)
    
    x = sort(x);
    pos = stats.posQuantile(length(x), probability);
    x[pos];
}



stats.CVaR <- function(x, probability = 0.05) {
    x = sort(x);
    pos = probability * length(x);
    posInt = floor(pos);
    cvar = sum(x[1:pos]/(length(x)*probability));

    probSoFar = (1/length(x))*posInt;
    cvar + (probability - probSoFar)*x[pos+1]/probability;
}


stats.sharpeRatio <- function(x, rf = 0) {
    stats.mean(x - rf)/stats.sd(x, sample = 0);
}

stats.VaRRatio <- function(x, rf = 0, probability = 0.05) {
    if (stats.VaR(x, probability) > 0) return (Inf);
    stats.mean(x - rf)/(-stats.VaR(x, probability));
}

stats.STARRRatio <- function(x, rf = 0, probability = 0.05) {
    if (stats.CVaR(x, probability) > 0) return (Inf);
    stats.mean(x - rf)/(-stats.CVaR(x, probability));
}

stats.downsideDeviation <- function(x, rf = 0) {
    r = subset(x, x < rf);
    sqrt(sum((r - rf)^2)/length(x));
}


stats.sortinoRatio <- function(x, rf = 0) {
    stats.mean(x - rf)/stats.downsideDeviation(x, rf);
}

stats.maxDrawdown <- function(x) {
    max1 = 0;
    maxdraw1 = 0;
  
    draw1 = 0;
    for (i in 1 : length(x)) {
        if (x[i] > max1) {
            max1 = x[i];
        } else {
            perc = (max1 - x[i])/max1;
            if (maxdraw1 < perc) maxdraw1 = perc; 
        }
    }
  
    maxdraw1*100;

}

