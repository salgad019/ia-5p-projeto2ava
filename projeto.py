from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Carregar e preparar dados

# DIFERENÇA: Este arquivo usa o dataset Wine do sklearn (built-in)
# Outros arquivos: diabetes.py usa CSV online, soybean.py usa UCI repository
wine = load_wine()
X = wine.data
y = wine.target

# DIFERENÇA: Apenas este arquivo inclui visualização dos dados originais
# Outros arquivos não fazem essa visualização inicial
# Visualização dos dados
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='Set1', s=60)
plt.title('Distribuição dos Dados Originais')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.tight_layout()
plt.show()

# DIFERENÇA: Este arquivo tem pré-processamento mais simples
# Apenas normalização, sem tratamento de valores faltantes ou encoding
# diabetes.py: inclui one-hot encoding de variáveis categóricas
# soybean.py: inclui imputação de valores faltantes + label encoding

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão treino/teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.3,
    random_state=42, ## para reprodutibilidade
    stratify=y
)

print("Formato X:", X.shape)
print("Formato y:", y.shape)
print("Formato X_train:", X_train.shape)
print("Formato X_test:", X_test.shape)
print("Formato y_train:", y_train.shape)
print("Formato y_test:", y_test.shape)


# 2. PCA

pca_90 = PCA(n_components=0.90)
pca_90.fit(X_train)
print("Número de componentes para 90% variância:", pca_90.n_components_)

# PCA 2D
pca_2 = PCA(n_components=2)
X_train_pca2 = pca_2.fit_transform(X_train)
X_test_pca2 = pca_2.transform(X_test)

# PCA 3D
pca_3 = PCA(n_components=3)
X_train_pca3 = pca_3.fit_transform(X_train)
X_test_pca3 = pca_3.transform(X_test)

# Curva de variância acumulada
pca_full = PCA().fit(X_train)
var_acumulada = np.cumsum(pca_full.explained_variance_ratio_)
plt.figure(figsize=(8,5))
plt.plot(range(1, len(var_acumulada)+1), var_acumulada, marker='o')
plt.xlabel('Número de Componentes')
plt.ylabel('Variância Acumulada')
plt.title('Curva de Variância Acumulada (PCA)')
plt.grid(True)
plt.tight_layout()
plt.show()


# 3. Scatter plot PCA 2D

# DIFERENÇA: Este arquivo usa classes numéricas diretas (0, 1, 2) 
# diabetes.py: usa rótulos descritivos ('Sem diabetes', 'Com diabetes')
# soybean.py: usa loop para múltiplas classes com encoder.inverse_transform
df_pca2 = pd.DataFrame({
    'PC1': X_train_pca2[:, 0],
    'PC2': X_train_pca2[:, 1],
    'Classe': y_train
})
print(df_pca2.head())
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_pca2, x='PC1', y='PC2', hue='Classe', palette='Set1', s=60)
plt.title('Projeção PCA (2 PCs)')
plt.tight_layout()
plt.show()

# 4. Função para treinar e medir

def avaliar_modelo(model, Xtr, ytr, Xte, yte):
    t0 = time.time()
    model.fit(Xtr, ytr)
    tempo_treino = time.time() - t0
    t1 = time.time()
    y_pred = model.predict(Xte)
    tempo_pred = time.time() - t1
    acc = accuracy_score(yte, y_pred)
    return acc, tempo_treino, tempo_pred


# 5. Treinamento e resultados

knn = KNeighborsClassifier(n_neighbors=5)
logreg = LogisticRegression(max_iter=1000)

# k-NN
acc_knn_orig, tt_knn_orig, tp_knn_orig = avaliar_modelo(knn, X_train, y_train, X_test, y_test)
acc_knn_pca2, tt_knn_pca2, tp_knn_pca2 = avaliar_modelo(knn, X_train_pca2, y_train, X_test_pca2, y_test)
acc_knn_pca3, tt_knn_pca3, tp_knn_pca3 = avaliar_modelo(knn, X_train_pca3, y_train, X_test_pca3, y_test)

# Logistic Regression
acc_log_orig, tt_log_orig, tp_log_orig = avaliar_modelo(logreg, X_train, y_train, X_test, y_test)
acc_log_pca2, tt_log_pca2, tp_log_pca2 = avaliar_modelo(logreg, X_train_pca2, y_train, X_test_pca2, y_test)
acc_log_pca3, tt_log_pca3, tp_log_pca3 = avaliar_modelo(logreg, X_train_pca3, y_train, X_test_pca3, y_test)

# DataFrame de resultados
resultados = [
    ["Original", "k-NN", acc_knn_orig, tt_knn_orig, tp_knn_orig],
    ["PCA-2D", "k-NN", acc_knn_pca2, tt_knn_pca2, tp_knn_pca2],
    ["PCA-3D", "k-NN", acc_knn_pca3, tt_knn_pca3, tp_knn_pca3],
    ["Original", "LogisticRegression", acc_log_orig, tt_log_orig, tp_log_orig],
    ["PCA-2D", "LogisticRegression", acc_log_pca2, tt_log_pca2, tp_log_pca2],
    ["PCA-3D", "LogisticRegression", acc_log_pca3, tt_log_pca3, tp_log_pca3]
]
df_result = pd.DataFrame(resultados, columns=["Abordagem","Modelo","Acurácia","Tempo Treino (s)","Tempo Inferência (s)"])
print(df_result)


# 6. Gráficos comparativos

plt.figure(figsize=(8,5))
sns.barplot(data=df_result, x="Abordagem", y="Acurácia", hue="Modelo")
plt.title("Comparação de Acurácia")
plt.ylim(0,1.05)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(data=df_result, x="Abordagem", y="Tempo Treino (s)", hue="Modelo")
plt.title("Tempo de Treinamento por Abordagem")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(data=df_result, x="Abordagem", y="Tempo Inferência (s)", hue="Modelo")
plt.title("Tempo de Inferência por Abordagem")
plt.tight_layout()
plt.show()