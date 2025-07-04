# IA-projeto8

Projeto: Redução de Dimensionalidade com PCA no Dataset Wine (UCI)

1. Carregue e normalize o Wine dataset (13 atributos, 178 amostras) e divida em treino/teste
70/30.
2. Ajuste PCA(n_components=0.90) para reter 90 % da variância e extraia projeções em
2 e 3 componentes.
3. Plote a curva de variância acumulada versus número de componentes.
4. Faça scatter plot das amostras nos 2 primeiros PCs, colorindo pelas três classes.
5. Treine k-NN (k=5) e Regressão Logística nos dados originais e nos dados PCA-2D/3D.
6. Meça acurácia, tempo de treinamento e inferência em cada abordagem.
7. Compare trade-offs entre redução de dimensão e preservação de informação.
