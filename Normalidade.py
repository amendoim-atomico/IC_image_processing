#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 23:34:21 2021

@author: caio
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from phantominator import shepp_logan
import mochila_cp as bag

fig=plt.figure(figsize=(40,20))
f1=fig.add_subplot(2,3,1)
f2=fig.add_subplot(2,3,2)
f3=fig.add_subplot(2,3,3)
f4=fig.add_subplot(2,3,4)
f5=fig.add_subplot(2,3,5)
f6=fig.add_subplot(2,3,6)

I0 = 0*shepp_logan(256) #Cria o phantom simulado
sigma=10
I1 = bag.ruido(I0,sigma) #Adiciona ruido awgn

print(sigma)

dentro1 = bag.segmentação(I1) #Separa regiões de sinal e ruído
#A função retorna um array com True na região de dentro e False na região de fora
fora1 = ~dentro1 #Para chamar com maior facilidade, vamos definir fora como o contrário de dentro

Sinal1 = I1[dentro1]
Ruido1 = I1[fora1]

A1=Sinal1.mean() #Média do sinal
sigma_med1 = np.std(Ruido1) #Desvio padrão do ruido


SNR1 = A1/sigma_med1 #Relação sinal ruído
f1.imshow(I1,cmap='gray')


#Aplicando shapiro ao ruído
stat_i, p_i = shapiro(Ruido1)
print('stat1=',stat_i,'p1=',p_i)

#Histograma do ruído
hist_i, bins_i = np.histogram(Ruido1, 1000)
f2.plot(bins_i[:-1],hist_i)

#QQplot do ruído
linha_i=[]
for k in range(256):
    linha_i=np.concatenate((linha_i,Ruido1[k]),axis=None)
qqplot(linha_i,line='s',ax=f3)










I = bag.ruido_vdd(I0,sigma) #adiciona ruído realístico
#Mesmo processo de antes
dentro = bag.segmentação(I)
fora = ~dentro
Sinal = I[dentro]
Ruido = I[fora]
A2=Sinal.mean()
sigma_med2 = np.std(Ruido)
SNR2 = A1/sigma_med2
f4.imshow(I,cmap='gray')
#Aplicando shapiro ao ruído
stat_i, p_i = shapiro(Ruido)
print('stat2=',stat_i,'p2=',p_i)
#Histograma do ruído
hist_i, bins_i = np.histogram(Ruido, 1000)
f5.plot(bins_i[:-1],hist_i)
#QQplot do ruído
linha_i=[]
for k in range(256):
    linha_i=np.concatenate((linha_i,Ruido[k]),axis=None)
qqplot(linha_i,line='s',ax=f6)




