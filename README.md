# projetos-if702
Este repositório contém os projetos e experimentos para a disciplina de Redes Neurais do CIn-UFPE desenvolvidos por [César Rio](https://github.com/CesarC15), [Gabriel Marques](https://github.com/marquesgbr), [Lucas Carvalho](https://github.com/LucasSilvaa0), [Maria Fernanda](https://github.com/MariaFFA) e [Thomaz Cabral](https://github.com/usuario5).

---

## Estrutura do Repositório

* `mini-projeto-1/`: Classificação de imagens no dataset CIFAR-10 usando MLPs e CNNs.
* `mini-projeto-2/`: Previsão do valor do Bitcoin usando LSTM em séries temporais.

---

## Mini-projeto 1: Classificação com MLP e CNN

Este primeiro mini-projeto foca na classificação de imagens utilizando o dataset **CIFAR-10**:

### Dataset: CIFAR-10
O dataset é composto por 60.000 imagens coloridas divididas em 10 classes de tamanhos iguais:
- Aeronaves, Automóveis, Pássaros, Gatos, Cervos, Cães, Sapos, Cavalos, Navios e Caminhões;
- Divisão: 50.000 imagens para treino e 10.000 para teste.

O trabalho deve ser dividido em duas fases principais:

### 1. Fase MLP
Treinamento de uma rede neural densa para classificar imagens de 32x32 pixels.
- **Hiperparâmetros:** Número de camadas, número de neurônios, taxa de aprendizagem, funções de ativação, regularização, algoritmo de aprendizagem, drop-out, função de erro, entre outros.
- **Métricas de Avaliação:** Acurácia por classe, acurácia total, precisão e recall.

### 2. Fase CNN
Evolução do modelo utilizando camadas convolucionais para extração de características.
- **Hiperparâmetros:** Tamaho da rede, tamanho da janela de convolução (filtro), stride, padding, drop-out, janela de pooling, taxa de aprendizagem, entre outros.
- **Métricas de Avaliação:** Acurácia por classe, acurácia total, precisão e recall.

### Links Importantes
- **Documento de Especificação:** [Aula-Miniprojeto1-MLP-CNN-cifar10-vf.pdf](https://classroom.google.com/c/ODQ2MDkwNTk4MzU2/a/ODUxNTk2MDI2Nzcy/details)
- **Código Base (MLP) com CIFAR-10:** [Google Colab - MLP](https://colab.research.google.com/drive/1ATzhzbAwpX6fvBIql3-xe43IT8z-EXqQ?usp=sharing)
- **Código Base (CNN) com CIFAR-10:** [Google Colab - CNN](https://colab.research.google.com/drive/1WtHX82W0k8wvlgr9cP91wXhd7O1DuDaw?usp=sharing)
- **Material de Apoio:** [Neural Networks and Deep Learning - Chap 6](http://neuralnetworksanddeeplearning.com/chap6.html)

---

## Mini-projeto 2: Previsão de Séries Temporais com LSTM (Bitcoin)

Este segundo mini-projeto foca na **previsão de séries temporais** aplicadas à cotação do **Bitcoin**, utilizando redes recorrentes do tipo **LSTM (Long Short-Term Memory)**.

O objetivo é modelar o comportamento do preço ao longo do tempo, respeitando a natureza temporal dos dados (evitando vazamento de informação/look-ahead bias) e avaliando tanto métricas de erro quanto métricas de **direção de tendência**.

**Atenção (anti look-ahead bias):** para prever o fechamento de um dia, devem ser utilizados **apenas dados disponíveis até o dia anterior/atual**, sem usar variáveis do **dia seguinte** (por exemplo: *o valor de abertura do dia seguinte não deve ser usado para prever o fechamento daquele dia*).

### Dataset: Série temporal do Bitcoin (2017–2023)
O dataset contém dados históricos do Bitcoin (com recorte temporal utilizado no projeto de **2017-01-01 a 2023-12-31**).  
A divisão dos dados é feita de forma **cronológica**, preservando a ordem temporal:
- **80%** para treino
- **20%** para teste

### O trabalho é dividido em duas abordagens principais

### 1. Abordagem Simples (Univariada)
Modelo de previsão utilizando apenas o atributo **close** (preço de fechamento).
- Pré-processamento com **RobustScaler**.
- Construção de janelas (sequências) com tamanho variável (`seq_length`) para alimentar a LSTM.
- Otimização de hiperparâmetros com **Optuna (TPE)**.

Arquivo relacionado:
- `mini-projeto-2/Equipe_2_Miniprojeto_LSTM_bitcoins_simples.ipynb`

### 2. Abordagem Avançada (Multivariada + Feature Engineering)
Modelo de previsão com múltiplas variáveis e indicadores técnicos, incluindo:
- **EMA 9 e EMA 21**
- **RSI**
- **MACD** e **Signal Line**
- **Bollinger Bands** (média, banda superior e inferior)

Além disso, o treinamento inclui:
- Função de perda customizada com penalização direcional (*directional_loss*), combinando **Huber Loss** com um termo que penaliza erros de direção.
- Validação cruzada temporal (**TimeSeriesSplit**) para avaliar estabilidade do modelo.
- Otimização de hiperparâmetros com **Optuna (TPE)**.

Arquivo relacionado:
- `mini-projeto-2/Equipe_2_Miniprojeto_LSTM_bitcoins.ipynb`

### Métricas de Avaliação
- **MSE (Mean Squared Error)**: erro quadrático médio.
- **RMSE (Root Mean Squared Error)**: raiz do erro quadrático médio (erro absoluto na mesma escala do preço).
- **POCID (Prediction of Change in Direction)**: métrica direcional que avalia o percentual de acertos na **mudança de tendência** (se o modelo acerta subida/queda entre instantes).

### Links Importantes
- **Documento de Especificação:** [Aula-Miniprojeto-LSTMs-2025.pdf](https://classroom.google.com/c/ODQ2MDkwNTk4MzU2/a/Nzk3MzIyOTYxOTg2/details)
- **Notebook (modelo simples):** `mini-projeto-2/Equipe_2_Miniprojeto_LSTM_bitcoins_simples.ipynb`
- **Notebook (modelo avançado):** `mini-projeto-2/Equipe_2_Miniprojeto_LSTM_bitcoins.ipynb`
