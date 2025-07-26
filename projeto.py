from sklearn.datasets import load_wine, load_digits, fetch_olivetti_faces
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


# Função para processar cada dataset
def processar_dataset(nome_dataset, X, y, target_names=None):
    print(f"\n{'='*50}")
    print(f"PROCESSANDO DATASET: {nome_dataset}")
    print(f"{'='*50}")
    
    # Visualização inicial (apenas para os primeiros 2 features)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='Set1', s=60)
    plt.title(f'Distribuição dos Dados Originais - {nome_dataset}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Divisão treino/teste (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    print(f"Formato X: {X.shape}")
    print(f"Formato y: {y.shape}")
    print(f"Formato X_train: {X_train.shape}")
    print(f"Formato X_test: {X_test.shape}")
    print(f"Número de classes: {len(np.unique(y))}")

    # PCA para 90% da variância
    pca_90 = PCA(n_components=0.90)
    pca_90.fit(X_train)
    print(f"Componentes para 90% variância: {pca_90.n_components_}")

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
    plt.title(f'Curva de Variância Acumulada (PCA) - {nome_dataset}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Scatter plot PCA 2D
    df_pca2 = pd.DataFrame({
        'PC1': X_train_pca2[:, 0],
        'PC2': X_train_pca2[:, 1],
        'Classe': y_train
    })
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df_pca2, x='PC1', y='PC2', hue='Classe', palette='Set1', s=60)
    plt.title(f'Projeção PCA (2 PCs) - {nome_dataset}')
    plt.tight_layout()
    plt.show()

    return X_train, X_test, y_train, y_test, X_train_pca2, X_test_pca2, X_train_pca3, X_test_pca3

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


# Função para executar experimentos completos
def executar_experimentos(nome_dataset, X_train, X_test, y_train, y_test, 
                         X_train_pca2, X_test_pca2, X_train_pca3, X_test_pca3):
    
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
        [nome_dataset, "Original", "k-NN", acc_knn_orig, tt_knn_orig, tp_knn_orig],
        [nome_dataset, "PCA-2D", "k-NN", acc_knn_pca2, tt_knn_pca2, tp_knn_pca2],
        [nome_dataset, "PCA-3D", "k-NN", acc_knn_pca3, tt_knn_pca3, tp_knn_pca3],
        [nome_dataset, "Original", "LogisticRegression", acc_log_orig, tt_log_orig, tp_log_orig],
        [nome_dataset, "PCA-2D", "LogisticRegression", acc_log_pca2, tt_log_pca2, tp_log_pca2],
        [nome_dataset, "PCA-3D", "LogisticRegression", acc_log_pca3, tt_log_pca3, tp_log_pca3]
    ]
    
    return resultados

# PROCESSAMENTO DOS DATASETS

# 1. Wine Dataset (original)
print("Carregando Wine Dataset...")
wine = load_wine()
resultados_wine = processar_dataset("Wine", wine.data, wine.target, wine.target_names)
exp_wine = executar_experimentos("Wine", *resultados_wine)

# 2. Digits Dataset (8x8 = 64 features, 1797 samples, 10 classes)
print("\nCarregando Digits Dataset...")
digits = load_digits()
resultados_digits = processar_dataset("Digits", digits.data, digits.target, digits.target_names)
exp_digits = executar_experimentos("Digits", *resultados_digits)

# 3. Olivetti Faces Dataset (64x64 = 4096 features, 400 samples, 40 classes)
print("\nCarregando Olivetti Faces Dataset...")
faces = fetch_olivetti_faces()
resultados_faces = processar_dataset("Olivetti Faces", faces.data, faces.target)
exp_faces = executar_experimentos("Olivetti Faces", *resultados_faces)

# CONSOLIDAR TODOS OS RESULTADOS
todos_resultados = exp_wine + exp_digits + exp_faces
df_result = pd.DataFrame(todos_resultados, 
                        columns=["Dataset", "Abordagem", "Modelo", 
                                "Acurácia", "Tempo Treino (s)", "Tempo Inferência (s)"])

print("\n" + "="*80)
print("RESULTADOS CONSOLIDADOS DE TODOS OS DATASETS")
print("="*80)
print(df_result)


# 6. Gráficos comparativos

# Comparação de Acurácia por Dataset
plt.figure(figsize=(15, 8))
sns.barplot(data=df_result, x="Dataset", y="Acurácia", hue="Modelo")
plt.title("Comparação de Acurácia entre Datasets")
plt.xticks(rotation=45)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

# Tempo de Treinamento por Dataset
plt.figure(figsize=(15, 8))
sns.barplot(data=df_result, x="Dataset", y="Tempo Treino (s)", hue="Modelo")
plt.title("Tempo de Treinamento por Dataset")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Tempo de Inferência por Dataset
plt.figure(figsize=(15, 8))
sns.barplot(data=df_result, x="Dataset", y="Tempo Inferência (s)", hue="Modelo")
plt.title("Tempo de Inferência por Dataset")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Gráfico de Abordagem vs Acurácia
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_result, x="Abordagem", y="Acurácia", hue="Dataset")
plt.title("Distribuição de Acurácia por Abordagem e Dataset")
plt.tight_layout()
plt.show()

# Tabela resumo por dataset
print("\n" + "="*80)
print("RESUMO ESTATÍSTICO POR DATASET")
print("="*80)
resumo = df_result.groupby('Dataset').agg({
    'Acurácia': ['mean', 'std', 'min', 'max'],
    'Tempo Treino (s)': ['mean', 'std'],
    'Tempo Inferência (s)': ['mean', 'std']
}).round(4)
print(resumo)