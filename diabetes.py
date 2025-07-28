import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

# --- 1. Carregar e preparar dados ---

# DIFERENÇA: Este arquivo carrega dados de um CSV online via URL
# projeto.py: usa dataset Wine do sklearn (built-in)
# soybean.py: usa UCI repository com fetch_ucirepo
url = "https://www.dropbox.com/scl/fi/4q2lfh5grcsblh2e0q4pq/diabetes_dataset.csv?rlkey=duspqdg8a6v9nrhq7vam6s17f&dl=1"
df = pd.read_csv(url)

# DIFERENÇA: Este arquivo faz one-hot encoding de variáveis categóricas
# projeto.py: não tem variáveis categóricas para tratar
# soybean.py: usa LabelEncoder para as classes target

# Converter colunas categóricas em dummies (one-hot encoding)
df_encoded = pd.get_dummies(df, columns=['gender', 'location', 'smoking_history'])

# Separar atributos e alvo
X = df_encoded.drop('diabetes', axis=1).values
y = df_encoded['diabetes'].values

print("Formato X:", X.shape)
print("Formato y:", y.shape)

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão treino/teste 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")

# --- 2. PCA ---

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

# --- 3. Scatter plot PCA 2D ---

# DIFERENÇA: Este arquivo usa rótulos descritivos personalizados
# projeto.py: usa classes numéricas diretas (0, 1, 2)
# soybean.py: usa loop com encoder.inverse_transform para múltiplas classes

# Mapeando as classes 0 e 1 para rótulos mais descritivos
rotulos = {0: 'Sem diabetes', 1: 'Com diabetes'}
classe_legenda = pd.Series(y_train).map(rotulos)

# Criando o DataFrame com os rótulos legíveis
df_pca2 = pd.DataFrame({
    'PC1': X_train_pca2[:, 0],
    'PC2': X_train_pca2[:, 1],
    'Classe': classe_legenda
})

# Plot
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_pca2, x='PC1', y='PC2', hue='Classe', palette='Set1', s=60)
plt.title('Projeção PCA (2 PCs)')
plt.tight_layout()
plt.show()

# --- 4. Função para treinar e medir ---

def avaliar_modelo(model, Xtr, ytr, Xte, yte):
    t0 = time.time()
    model.fit(Xtr, ytr)
    tempo_treino = time.time() - t0
    t1 = time.time()
    y_pred = model.predict(Xte)
    tempo_pred = time.time() - t1
    acc = accuracy_score(yte, y_pred)
    return acc, tempo_treino, tempo_pred

# --- 5. Treinamento e resultados ---

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

# --- 6. Gráficos comparativos ---

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
