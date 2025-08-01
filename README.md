# 🍷 IA-projeto8: Análise Comparativa de Redução de Dimensionalidade com PCA

## 📋 Descrição do Projeto

Este projeto implementa uma **análise comparativa completa** do impacto da redução de dimensionalidade usando **PCA (Principal Component Analysis)** em algoritmos de machine learning. O estudo avalia como diferentes níveis de compressão dimensional afetam a performance, velocidade de treinamento e inferência em múltiplos datasets.

## 🎯 Objetivos

- **Comparar performance** de algoritmos ML com e sem redução dimensional
- **Analisar trade-offs** entre acurácia e eficiência computacional
- **Visualizar** como PCA preserva/perde informação dos dados
- **Avaliar** impacto da dimensionalidade em diferentes tipos de datasets

## 📊 Datasets Utilizados

### 1. **Wine Dataset (UCI)** 🍾
- **Amostras:** 178
- **Features:** 13 características químicas
- **Classes:** 3 tipos de vinho
- **Tipo:** Dados reais de análise química

### 2. **Diabetes Dataset** 🏥
- **Amostras:** 100.000 (sintético)
- **Features:** 20 características médicas
- **Classes:** 2 (diabetes/não-diabetes)
- **Tipo:** Dados sintéticos simulando características médicas

### 3. **Soybean Seeds Dataset** 🌱
- **Amostras:** 30.000 (sintético)
- **Features:** 35 características morfológicas
- **Classes:** 19 variedades de sementes
- **Tipo:** Dados sintéticos simulando características de sementes

## 🔬 Metodologia

### 1. **Pré-processamento**
- Normalização dos dados (StandardScaler)
- Divisão treino/teste (70%/30%)
- Estratificação para manter proporção das classes

### 2. **Análise PCA**
- **PCA 90%:** Componentes necessários para 90% da variância
- **PCA 2D:** Redução para 2 componentes principais
- **PCA 3D:** Redução para 3 componentes principais
- **Visualização:** Curva de variância acumulada

### 3. **Algoritmos Testados**
- **k-NN (k=5):** Sensível à dimensionalidade
- **Regressão Logística:** Menos sensível à dimensionalidade

### 4. **Métricas Avaliadas**
- ✅ **Acurácia:** Percentual de predições corretas
- ⏱️ **Tempo de Treinamento:** Velocidade de ajuste do modelo
- 🚀 **Tempo de Inferência:** Velocidade de predição

## 📈 Experimentos Realizados

Cada dataset é testado em **6 configurações diferentes**:

| Dataset | Abordagem | Modelo | Dimensões |
|---------|-----------|--------|-----------|
| Wine/Diabetes/Soybean | Original | k-NN | 13/20/35 |
| Wine/Diabetes/Soybean | PCA-2D | k-NN | 2 |
| Wine/Diabetes/Soybean | PCA-3D | k-NN | 3 |
| Wine/Diabetes/Soybean | Original | LogReg | 13/20/35 |
| Wine/Diabetes/Soybean | PCA-2D | LogReg | 2 |
| Wine/Diabetes/Soybean | PCA-3D | LogReg | 3 |

**Total:** 18 experimentos (3 datasets × 6 configurações)

## 📊 Visualizações Geradas

1. **📈 Curva de Variância Acumulada** - Mostra quantos componentes preservam X% da informação
2. **🎨 Scatter Plot 2D** - Visualiza separação das classes no espaço reduzido
3. **📊 Distribuição Original** - Visualização dos dados antes do processamento
4. **⚡ Comparação de Acurácia** - Performance por dataset e abordagem
5. **⏱️ Tempo de Treinamento** - Eficiência computacional no treino
6. **🚀 Tempo de Inferência** - Velocidade de predição
7. **📦 Distribuição por Abordagem** - Box plots comparativos

## 🚀 Como Executar

### Pré-requisitos
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

### Execução
```bash
python projeto.py
```

## 📋 Estrutura do Código

```python
├── Funções de Carregamento de Datasets
│   ├── carregar_diabetes_dataset()
│   └── carregar_soybean_dataset()
├── Processamento e Análise PCA
│   └── processar_dataset()
├── Avaliação de Modelos
│   ├── avaliar_modelo()
│   └── executar_experimentos()
└── Visualizações e Resultados
    └── Gráficos comparativos
```

## 📊 Resultados Esperados

### 🔴 **k-NN (Sensível à Dimensionalidade)**
- **Dados Originais:** Alta acurácia, treino/inferência lentos
- **PCA 2D/3D:** Possível queda na acurácia, mas muito mais rápido

### 🔵 **Regressão Logística (Robusta)**
- **Dados Originais:** Boa acurácia, treino moderado
- **PCA 2D/3D:** Pouca perda de acurácia, treino mais rápido

### 📈 **Trade-offs Identificados**
- ⚖️ **Acurácia vs Velocidade:** Redução dimensional acelera, mas pode perder precisão
- 🔍 **Interpretabilidade vs Performance:** PCA perde interpretabilidade das features originais
- 💾 **Memória vs Qualidade:** Menos dimensões = menos memória, possível perda de informação

## 🎓 Conceitos Demonstrados

- **🎯 Maldição da Dimensionalidade:** Como excesso de dimensões pode prejudicar algoritmos
- **⚡ Redução de Dimensionalidade:** Técnicas para compressar dados mantendo informação
- **📊 Análise de Componentes Principais:** Como PCA funciona e seus trade-offs
- **🔄 Avaliação Sistemática:** Importância de testar múltiplas configurações
- **📈 Visualização de Dados:** Como "ver" dados multidimensionais

## 🏆 Aplicações Práticas

- **🔍 Análise Exploratória:** Visualizar dados de alta dimensão
- **⚡ Otimização de Performance:** Acelerar algoritmos em produção
- **💾 Compressão de Dados:** Reduzir espaço de armazenamento
- **🎯 Feature Engineering:** Identificar características mais importantes
- **🚀 Sistemas em Tempo Real:** Reduzir latência de predições

## 📚 Tecnologias Utilizadas

- **🐍 Python 3.7+**
- **🤖 scikit-learn:** Algoritmos ML e PCA
- **📊 pandas:** Manipulação de dados
- **🔢 numpy:** Operações numéricas
- **📈 matplotlib/seaborn:** Visualizações


---

*Este projeto demonstra na prática como a redução de dimensionalidade pode ser uma ferramenta poderosa para otimizar algoritmos de machine learning, fornecendo insights valiosos sobre o trade-off entre performance e eficiência computacional.*
