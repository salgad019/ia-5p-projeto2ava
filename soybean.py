from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import seaborn as sns

# ==============================
# 1. Carregar dataset Soybean Large
# ==============================
# DIFERENÇA: Este arquivo usa UCI ML Repository com fetch_ucirepo
# projeto.py: usa dataset Wine do sklearn (built-in)
# diabetes.py: carrega dados de CSV online via URL
soybean_large = fetch_ucirepo(id=90)

X = soybean_large.data.features
y = soybean_large.data.targets

# DIFERENÇA: Apenas este arquivo mostra informações detalhadas do dataset
# Outros arquivos fazem apenas prints básicos de formato
print("Formato X:", X.shape)
print("Formato y:", y.shape)
print("Colunas:", list(X.columns))
print("Classes únicas:", y.iloc[:, 0].unique())

# ==============================
# 2. Pré-processamento
# ==============================
# DIFERENÇA: Este arquivo tem o pré-processamento mais complexo
# Inclui imputação de valores faltantes + label encoding
# projeto.py: apenas normalização
# diabetes.py: one-hot encoding + normalização
# Imputar valores faltantes com a média
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Codificar labels
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y.values.ravel())

# ==============================
# 3. Dividir em treino/teste (70/30)
# ==============================
# DIFERENÇA: Este arquivo não usa stratify na primeira divisão
# projeto.py e diabetes.py: usam stratify=y para manter proporção das classes
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.3, random_state=42
)
print("Formato treino:", X_train.shape)
print("Formato teste:", X_test.shape)

# ==============================
# 4. PCA - variância acumulada
# ==============================
pca_full = PCA()
pca_full.fit(X_scaled)

explained_var_ratio = np.cumsum(pca_full.explained_variance_ratio_)

# Número de componentes para 90% da variância
n_components_90 = np.argmax(explained_var_ratio >= 0.9) + 1
print("Número de componentes para 90% da variância:", n_components_90)

# DIFERENÇA: Este arquivo inclui linhas de referência (90% variância)
# projeto.py e diabetes.py: apenas a curva simples sem linhas de referência

# Plot curva de variância acumulada
plt.figure(figsize=(8,5))
plt.plot(range(1, len(explained_var_ratio)+1), explained_var_ratio, marker='o')
plt.axhline(y=0.9, color='r', linestyle='--')
plt.axvline(x=n_components_90, color='g', linestyle='--')
plt.xlabel('Número de Componentes')
plt.ylabel('Variância Acumulada')
plt.title('Curva de Variância Acumulada - Soybean Large')
plt.grid(True)
plt.show()

# ==============================
# 5. Projeções PCA 2D e 3D
# ==============================
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# DIFERENÇA: Este arquivo usa loop para visualizar múltiplas classes
# projeto.py: usa classes numéricas diretas no seaborn
# diabetes.py: usa rótulos personalizados mapeados

# Scatter plot PCA 2D
plt.figure(figsize=(8,6))
for cls in np.unique(y_enc):
    plt.scatter(X_pca_2d[y_enc == cls, 0], X_pca_2d[y_enc == cls, 1],
                label=encoder.inverse_transform([cls])[0])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Projeção PCA 2D - Soybean Large')
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================
# 6. Avaliação de modelos
# ==============================

# DIFERENÇA: Este arquivo usa nomes de variáveis diferentes para tempo
# projeto.py e diabetes.py: usam t0, t1, tempo_treino, tempo_pred
# Este arquivo: usa start_train, start_pred, train_time, pred_time
def avaliar_modelo(model, Xtr, ytr, Xte, yte):
    start_train = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - start_train

    start_pred = time.time()
    y_pred = model.predict(Xte)
    pred_time = time.time() - start_pred

    acc = accuracy_score(yte, y_pred)
    return acc, train_time, pred_time

knn = KNeighborsClassifier(n_neighbors=5)
logreg = LogisticRegression(max_iter=1000)

resultados = []

# Original
acc_knn_orig, tt_knn_orig, tp_knn_orig = avaliar_modelo(knn, X_train, y_train, X_test, y_test)
resultados.append(["Original", "k-NN", acc_knn_orig, tt_knn_orig, tp_knn_orig])

acc_lr_orig, tt_lr_orig, tp_lr_orig = avaliar_modelo(logreg, X_train, y_train, X_test, y_test)
resultados.append(["Original", "LogisticRegression", acc_lr_orig, tt_lr_orig, tp_lr_orig])

# DIFERENÇA: Este arquivo faz nova divisão treino/teste para PCA
# projeto.py e diabetes.py: reutilizam a divisão original aplicando PCA aos conjuntos já divididos
# Este approach pode introduzir variabilidade adicional nos resultados

# PCA-2D
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_pca_2d, y_enc, test_size=0.3, random_state=42
)
acc_knn_2d, tt_knn_2d, tp_knn_2d = avaliar_modelo(knn, X_train_2d, y_train_2d, X_test_2d, y_test_2d)
resultados.append(["PCA-2D", "k-NN", acc_knn_2d, tt_knn_2d, tp_knn_2d])

acc_lr_2d, tt_lr_2d, tp_lr_2d = avaliar_modelo(logreg, X_train_2d, y_train_2d, X_test_2d, y_test_2d)
resultados.append(["PCA-2D", "LogisticRegression", acc_lr_2d, tt_lr_2d, tp_lr_2d])

# PCA-3D
X_train_3d, X_test_3d, y_train_3d, y_test_3d = train_test_split(
    X_pca_3d, y_enc, test_size=0.3, random_state=42
)
acc_knn_3d, tt_knn_3d, tp_knn_3d = avaliar_modelo(knn, X_train_3d, y_train_3d, X_test_3d, y_test_3d)
resultados.append(["PCA-3D", "k-NN", acc_knn_3d, tt_knn_3d, tp_knn_3d])

acc_lr_3d, tt_lr_3d, tp_lr_3d = avaliar_modelo(logreg, X_train_3d, y_train_3d, X_test_3d, y_test_3d)
resultados.append(["PCA-3D", "LogisticRegression", acc_lr_3d, tt_lr_3d, tp_lr_3d])

# ==============================
# 7. Resultados
# ==============================

# DIFERENÇA: Este arquivo usa 'df_resultados' como nome da variável
# projeto.py e diabetes.py: usam 'df_result'
df_resultados = pd.DataFrame(resultados, columns=["Abordagem","Modelo","Acurácia","Tempo Treino (s)","Tempo Inferência (s)"])
print(df_resultados)

# Comparação de Acurácia
plt.figure(figsize=(8,5))
sns.barplot(data=df_resultados, x="Abordagem", y="Acurácia", hue="Modelo")
plt.title("Comparação de Acurácia")
plt.ylim(0,1.05)
plt.tight_layout()
plt.show()

# Comparação de Tempo de Treinamento
plt.figure(figsize=(8,5))
sns.barplot(data=df_resultados, x="Abordagem", y="Tempo Treino (s)", hue="Modelo")
plt.title("Tempo de Treinamento por Abordagem")
plt.tight_layout()
plt.show()

# Comparação de Tempo de Inferência
plt.figure(figsize=(8,5))
sns.barplot(data=df_resultados, x="Abordagem", y="Tempo Inferência (s)", hue="Modelo")
plt.title("Tempo de Inferência por Abordagem")
plt.tight_layout()
plt.show()