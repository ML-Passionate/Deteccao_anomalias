# Detecção de anomalias em um servidor

![image](https://github.com/user-attachments/assets/ce10d314-e490-4af6-9d65-f21c38d8f7d1)


O código utiliza uma abordagem de distribuição Gaussiana multivariada para detectar anomalias em servidores. A média e a variância dos dados de treinamento (X_train) são usadas para calcular a densidade de probabilidade. Com base nos resultados de validação (X_val e y_val), um limiar (epsilon) é determinado para classificar pontos como normais ou anômalos, utilizando a pontuação F1 para otimizar a escolha do limiar. Ponto abaixo do limiar são considerados anômalos e visualizados no gráfico.











