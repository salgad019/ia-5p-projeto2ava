from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# 1.1 Carregar dataset
wine = load_wine()
X = wine.data        # (178, 13)
y = wine.target      # classes 0,1,2

# 1.2 Normalizar atributos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1.3 Dividir em treino/teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print("Formato X:", X.shape)
print("Formato y:", y.shape)

# 1.2 Normalizar atributos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 1.3 Dividir em treino/teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.3,
    random_state=42,     # garante reprodutibilidade
    stratify=y           # mantém proporção das classes
)

print("Formato X_train:", X_train.shape)
print("Formato X_test:", X_test.shape)
print("Formato y_train:", y_train.shape)
print("Formato y_test:", y_test.shape)


# ----- PCA para reter 90% da variância -----
pca_90 = PCA(n_components=0.90)
pca_90.fit(X_train)

print("Número de componentes necessários para reter 90% da variância:", pca_90.n_components_)

# ----- PCA com 2 componentes -----
pca_2 = PCA(n_components=2)
X_train_pca2 = pca_2.fit_transform(X_train)
X_test_pca2 = pca_2.transform(X_test)

print("Formato X_train_pca2:", X_train_pca2.shape)
print("Formato X_test_pca2:", X_test_pca2.shape)


# ----- PCA com 3 componentes -----
pca_3 = PCA(n_components=3)
X_train_pca3 = pca_3.fit_transform(X_train)
X_test_pca3 = pca_3.transform(X_test)

print("Formato X_train_pca3:", X_train_pca3.shape)
print("Formato X_test_pca3:", X_test_pca3.shape)


# Ajustar PCA completo (todas as componentes)
pca_full = PCA().fit(X_train)
var_acumulada = np.cumsum(pca_full.explained_variance_ratio_)

# Plotar curva
plt.figure(figsize=(8,5))
plt.plot(range(1, len(var_acumulada)+1), var_acumulada, marker='o', color='b')
plt.xlabel('Número de Componentes')
plt.ylabel('Variância Acumulada')
plt.title('Curva de Variância Acumulada (PCA)')
plt.grid(True)
plt.tight_layout()
plt.show()


#Passo 4

