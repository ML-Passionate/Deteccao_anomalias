# -*- coding: utf-8 -*-
#
#   Deteção de anomalias em servidores
#
#

import numpy as np
import matplotlib.pyplot as plt
from utils import *


#%%

# Load the dataset
#
# X_train para ajustar uma distribuição Gaussiana
#
# X_val e y_val como um conjunto de validação cruzada para selecionar um limiar e determinar exemplos anômalos versus normais
# o y_val é que estará a variável dependente que acusará se é anomalia 1 ou não 0

X_train, X_val, y_val = load_data()

#%%
# Mostras os primeiros elementos dos dados
print("Os primeiros 5 elementos de X_train são:\n", X_train[:5])  

#%%

# Mostra os primeiros 5 elementos de X_val
print("Os primeiros 5 elementos de X_val são\n", X_val[:5])
#%%
# Mostra os primeiros 5 elementos de y_val
print("Os primeiros 5 elementos de y_val são\n", y_val[:5])  

#%%

print ('O tamanho de X_train é :', X_train.shape)
print ('O tamanho de X_val é :', X_val.shape)
print ('O tamanho de y_val é : ', y_val.shape)

#%%
# Scatter plot dos dados
# 
plt.scatter(X_train[:, 0], X_train[:, 1], marker='.', c='b') 

# Set the title
plt.title("Detecção de anomalias no servidor")
# Set the y-axis label
plt.ylabel('Throughput (mb/s)')
# Set the x-axis label
plt.xlabel('Latência (ms)')
# Set axis range
plt.axis([0, 30, 0, 30])
plt.show()

#%%
# 
# Estimatica gaussiana

def estimate_gaussian(X): 
    """
     Calcula a média e a variância de todas as features
     no conjunto de dados

     Argumentos:

     X (ndarray): Matriz de dados (m, n)

     Retorna:

     mu (ndarray): Média de todas as features (n,)

     var (ndarray): Variância de todas as features (n,)
    """

    m, n = X.shape
    
    mu = 1 / m * np.sum(X, axis = 0) # Your code here to calculate the mean of every feature
    var = 1 / m * np.sum((X - mu) ** 2, axis = 0) # Your code here to calculate the variance of every feature 
    
        
    return mu, var

#%%

# Estima a média e a variança de cada atributo
mu, var = estimate_gaussian(X_train)              

print("Média de cada atributo:", mu)
print("Variância de cada attibuto:", var)
    

#%%
#
# Retorna a densidade da normal multivariada
# em cada ponto de dados (linha) de X_train
#


p = multivariate_gaussian(X_train, mu, var)

#Plota no gráfico

# Set the title
plt.title("Detecção de anomalias no servidor")
# Set the y-axis label
plt.ylabel('Throughput (mb/s)')
# Set the x-axis label
plt.xlabel('Latência (ms)')
visualize_fit(X_train, mu, var)

#%%
#
#  seleciona o limite 
#
# 

def select_threshold(y_val, p_val): 
    """
     Encontra o melhor limiar a ser usado para selecionar outliers
     com base nos resultados de um conjunto de validação (p_val)
     e a verdade de solo (y_val)

     Argumentos:

     y_val (ndarray): Verdade de solo no conjunto de validação

     p_val (ndarray): Resultados no conjunto de validação

     Retorna:

     epsilon (float): Limiar escolhido

     F1 (float): Pontuação F1 ao escolher o epsilon como limiar
    """ 

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
    
        
        predictions = (p_val < epsilon) 
        tp = np.sum((predictions == 1) & (y_val == 1)) 
        fp = sum((predictions == 1) & (y_val == 0)) 
        fn = np.sum((predictions == 0) & (y_val == 1)) 
        prec = tp / (tp + fp) # calculando a precisão
        rec = tp / (tp + fn) # calculando o recall
        F1 = 2 * prec * rec / (prec + rec) # calculando F1
        
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1

#%%
#
p_val = multivariate_gaussian(X_val, mu, var)
epsilon, F1 = select_threshold(y_val, p_val)

print('Melhor epsilon usando validação cruzada: %e' % epsilon)
print('Melhor F1 na validação cruzada: %f' % F1)
    

#%%
# Achando os pontos outliers 
#
# 
outliers = p < epsilon

# Visualizando o fit
visualize_fit(X_train, mu, var)

# Set the title
plt.title("Detecção de anomalias no servidor")
# Set the y-axis label
plt.ylabel('Throughput (mb/s)')
# Set the x-axis label
plt.xlabel('Latência (ms)')

# Desenha um circula ao redo dos outliers
plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro',
         markersize= 10,markerfacecolor='none', markeredgewidth=2)

#%%
# fim