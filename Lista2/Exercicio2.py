import numpy as np
import matplotlib.pyplot as plt

#Calculando a esperança
#15/30 = 1/2 chances de perder o valor total
#14/30 = 7/15 chances de perder metade do valor
#1/30 chances de aumentar o total 50 vezes
#E(X) = (1/2) * (1 - 1) + (7/15) * (1 - 0,5) + (1/30) * (1 + 50) - 1
#E(X) = (1/2) * 0 + (7/15) * 0,5 + (1/30) * 51 - 1
#E(X) = 0,933333333

#Calculando a média temporal
def mediaAritmetica(w):
    return (1 - w)*(15/30) + (1 - w/2)*(14/30) + (1 + 50*w)*(1/30) - 1

def mediaAritmeticaSimplificada(w):
    return (1 - w)*(1/2) + (1 - w/2)*(7/15) + (1 + 50*w)*(1/30) - 1

def mediaGeometrica(w):
    return ((1 - w)**(15/30)) * ((1 - w/2)**(14/30)) * ((1 + 50*w)**(1/30)) - 1

def mediaGeometricaSimplificada(w):
    return ((1 - w)**(1/2)) * ((1 - w/2)**(7/15)) * ((1 + 50*w)**(1/30)) - 1

print("RESULTADOS")

wValues = [w for w in np.arange(0, 1, 0.001)]
esperanca = [mediaAritmetica(w) for w in wValues]
mediaTemporal = [mediaGeometrica(w) for w in wValues]

print("Maior esperança: " + str(mediaAritmetica(1)) + " com 100%% do patrimônio.")

maiorMediaTemporal = np.where(mediaTemporal == np.amax(mediaTemporal))[0][0]
print("Maior média memporal: " + str(mediaTemporal[maiorMediaTemporal]*100) + " com " + str(wValues[maiorMediaTemporal]*100) + "% do patrimônio.")

wValues100 = [100*w for w in wValues]
esperanca100 = [100*x for x in esperanca]
mediaTemporal100 = [100*x for x in mediaTemporal]

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)

plt.plot(wValues100, esperanca100)
plt.plot(wValues100, mediaTemporal100)
plt.axhline(color='r')
plt.xlabel("Posição")
plt.ylabel("Retorno")
plt.xlim(0, 100)
plt.ylim(-100, 100)
plt.tight_layout()
plt.show()