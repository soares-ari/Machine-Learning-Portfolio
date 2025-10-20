# 📊 Benchmark de Algoritmos de Machine Learning

Este projeto tem como objetivo comparar o desempenho de diferentes algoritmos de aprendizado de máquina em problemas clássicos de regressão e classificação. A proposta é entender os trade-offs entre acurácia, interpretabilidade, tempo de execução e complexidade dos modelos.

---

## 🎯 Objetivos

- Aplicar múltiplos algoritmos em datasets clássicos
- Avaliar desempenho com métricas padronizadas
- Visualizar comparações de forma clara e interativa
- Consolidar aprendizado prático em ML supervisionado

---

## 🧠 Algoritmos Avaliados

### Regressão
- Regressão Linear
- Árvore de Decisão
- Random Forest
- XGBoost
- Redes Neurais (MLP)

### Classificação
- K-Nearest Neighbors (KNN)
- Árvore de Decisão
- Random Forest
- SVM
- Naive Bayes
- XGBoost
- Redes Neurais (MLP)

---

## 📁 Datasets Utilizados

- `cars.csv` — Consumo e eficiência de veículos
- `iris.csv` — Classificação de espécies de flores
- `boston.csv` — Preço de imóveis
- `wine.csv` — Qualidade de vinhos
- `diabetes.csv` — Diagnóstico médico

Todos os datasets são públicos e disponíveis via [scikit-learn](https://scikit-learn.org/stable/datasets/toy_dataset.html) ou [Kaggle](https://www.kaggle.com/datasets).

---

## 📐 Métricas de Avaliação

### Regressão
- MAE (Erro Absoluto Médio)
- RMSE (Raiz do Erro Quadrático Médio)
- R² (Coeficiente de Determinação)

### Classificação
- Acurácia
- Precisão
- Recall
- F1-Score
- Matriz de Confusão

---

## 🛠️ Estrutura do Projeto

benchmark-ml-models/ 
├── data/ # Datasets 
├── notebooks/ # Prototipagem e EDA 
├── src/ # Scripts de modelagem 
├── results/ # Gráficos e comparações 
├── README.md # Documentação 
└── requirements.txt # Dependências


---

## 🚀 Como Executar

1. Clone o repositório:
```bash
git clone https://github.com/soares-ari/Machine-Learning-Portfolio.git 
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```
3. Execute os scripts ou abra os notebooks:
```bash
python src/models.py
```

---
## 📈 Resultados
Os resultados das comparações entre algoritmos são salvos na pasta results/, incluindo gráficos de desempenho e tabelas com métricas.
---
## 👨‍💻 Autor
Ari Soares — Pós-graduando em Inteligência Artificial com ênfase em Machine Learning. Background em UX/UI, Cinema e Agronegócios. Apaixonado por sistemas inteligentes e aprendizado contínuo.
---
## 📬 Contato
- [LinkedIn](https://www.linkedin.com/in/ari-soares/)
- [GitHub](https://github.com/soares-ari)