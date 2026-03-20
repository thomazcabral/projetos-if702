# projetos-if702
Este repositório contém os projetos e experimentos para a disciplina de **Redes Neurais** do CIn-UFPE desenvolvidos por [César Rio](https://github.com/CesarC15), [Gabriel Marques](https://github.com/marquesgbr), [Lucas Carvalho](https://github.com/LucasSilvaa0), [Maria Fernanda](https://github.com/MariaFFA) e [Thomaz Cabral](https://github.com/usuario5).

---

## Estrutura do Repositório

* `mini-projeto-1/`: Classificação de imagens no dataset CIFAR-10 usando MLPs e CNNs.

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
