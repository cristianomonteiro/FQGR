import numpy as np

#Calculando a esperança
#15/30 = 1/2 chances de perder o valor total
#14/30 = 7/15 chances de perder metade do valor
#1/30 chances de aumentar o total 50 vezes
#E(X) = (1/2) * (1 - 1) + (7/15) * (1 - 0,5) + (1/30) * (1 + 50) - 1
#E(X) = (1/2) * 0 + (7/15) * 0,5 + (1/30) * 51 - 1
#E(X) = 0,933333333

#Calculando a média temporal
def proporcaoM(w):
    return ((1 - w)**(15/30)) * ((1 - w/2)**(14/30)) * ((1 + 50*w)**(1/30)) - 1

def proporcaoMSimplificado(w):
    return ((1 - w)**(1/2)) * ((1 - w/2)**(7/15)) * ((1 + 50*w)**(1/30)) - 1

print("RESULTADOS")

print('1% ' + str(proporcaoM(0.01)))
print('1% ' + str(proporcaoMSimplificado(0.01)) + '\n')
print('2% ' + str(proporcaoM(0.02)))
print('2% ' + str(proporcaoMSimplificado(0.02)) + '\n')
print('2.5% ' + str(proporcaoM(0.025)))
print('2.5% ' + str(proporcaoMSimplificado(0.025)) + '\n')
print('3% ' + str(proporcaoM(0.03)))
print('3% ' + str(proporcaoMSimplificado(0.03)) + '\n')
print('4% ' + str(proporcaoM(0.04)))
print('4% ' + str(proporcaoMSimplificado(0.04)) + '\n')
print('5% ' + str(proporcaoM(0.05)))
print('5% ' + str(proporcaoMSimplificado(0.05)) + '\n')