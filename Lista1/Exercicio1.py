import numpy as np

preco1Acao = 100
precoOpcao = 1.6
pExercicio = 102

#Cenário R$105
preco2Acao = 105
patrimonio105 = preco1Acao - preco2Acao - precoOpcao + (preco2Acao - pExercicio)
print(patrimonio105)

#Cenário R$95
preco2Acao = 95
patrimonio95 = preco1Acao - preco2Acao - precoOpcao
print(patrimonio95)

#Calculando o Delta
#0 + PrecoOpcao + 100*Delta - 105*Delta + (105 - 102) = 0 + PrecoOpcao + 100*Delta - 95*Delta
#PrecoOpcao - 5*Delta + 3 = 5*Delta + PrecoOpcao
#-10*Delta = -3
#Delta = 0.3

#Calculando o preço justo para a opção
#0 + PrecoOpcao = 100*Delta - 105*Delta + (105 - 102)
#0 + PrecoOpcao = 100*0,3 - 105*0,3 + 3
#PrecoOpcao = 30 - 31,5 + 3 = 1,5

#Calculando o Delta considerando taxa livre de risco de 1%
#0*1,01 + PrecoOpcao + 100*Delta*1,01 - 105*Delta + (105 - 102) = 0*1,01 + PrecoOpcao + 100*Delta*1,01 - 95*Delta
#PrecoOpcao + 101*Delta - 105*Delta + 3 = PrecoOpcao + 101*Delta - 95*Delta
#PrecoOpcao - 4*Delta + 3 = PrecoOpcao + 6*Delta
#-10*Delta = -3
#Delta = 0,3

#Calculando o preço justo para a opção
#0 + PrecoOpcao = 100*Delta - 105*Delta + (105 - 102)
#0 + PrecoOpcao = 101*0,3 - 105*0,3 + 3
#PrecoOpcao = 30,3 - 31,5 + 3 = 1,8