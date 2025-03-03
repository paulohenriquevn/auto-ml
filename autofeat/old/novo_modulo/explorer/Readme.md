# Documentação do Módulo Explorer - Versão 1.0

## Visão Geral

O módulo `Explorer` é um componente fundamental do sistema AutoFE, responsável pela seleção e engenharia de features. Ele trabalha após o `PreProcessor` e é projetado para identificar automaticamente as features mais relevantes, criar novas features derivadas e aplicar técnicas de redução de dimensionalidade para otimizar o desempenho de modelos de machine learning.

## Características Principais

- **Análise automática de importância** de features existentes
- **Geração inteligente de novas features**:
  - Features polinomiais
  - Features de interação entre variáveis importantes
  - Features baseadas em clustering
- **Redução de dimensionalidade** com PCA ou SVD
- **Seleção automática** das melhores features
- **Adaptação inteligente** ao tipo de problema (classificação ou regressão)
- **Persistência** para salvar/carregar o estado de exploração
- **Compatibilidade com scikit-learn** através da API fit/transform

## Instalação

### Requisitos

- Python 3.7 ou superior
- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 1.0.0
- joblib >= 1.1.0

### Instalação via pip

```bash
pip install -r requirements.txt
```

## Estrutura do Módulo

O módulo principal é composto pela classe `Explorer` e pela função auxiliar `create_explorer()`.

## Classe Explorer

A classe `Explorer` é a componente principal, implementando a interface para engenharia e seleção de features.

### Inicialização

```python
from explorer import Explorer

# Inicialização com configurações padrão
explorer = Explorer()

# Inicialização com configurações personalizadas
config = {
    'feature_selection_method': 'mutual_info',
    'polynomial_features': True,
    'polynomial_degree': 2,
    'clustering_features': True,
    'feature_reduction_method': 'pca',
    'problem_type': 'classification'
}
explorer = Explorer(config)
```

### Parâmetros de Configuração

| Parâmetro | Tipo | Valores Possíveis | Padrão | Descrição |
|-----------|------|-------------------|--------|-----------|
| `feature_selection_method` | str | 'mutual_info', 'anova', 'chi2', 'recursive' | 'mutual_info' | Método para seleção de features |
| `feature_selection_k` | str/int | 'auto' ou inteiro positivo | 'auto' | Número de features a selecionar |
| `feature_selection_threshold` | float | 0 a 1 | 0.01 | Threshold para filtragem por importância |
| `feature_reduction_method` | str | 'pca', 'svd', 'none' | 'pca' | Método para redução de dimensionalidade |
| `feature_reduction_components` | float/int | 0-1 (variância) ou inteiro positivo | 0.95 | Componentes a manter |
| `polynomial_features` | bool | True, False | True | Se deve gerar features polinomiais |
| `polynomial_degree` | int | inteiro positivo | 2 | Grau máximo para features polinomiais |
| `interaction_features` | bool | True, False | True | Se deve gerar features de interação |
| `clustering_features` | bool | True, False | True | Se deve gerar features baseadas em clustering |
| `n_clusters` | str/int | 'auto' ou inteiro positivo | 'auto' | Número de clusters a formar |
| `agg_functions` | list | lista de strings | ['mean', 'min', 'max', 'std'] | Funções de agregação para features de grupo |
| `max_features_to_try` | int | inteiro positivo | 1000 | Limite máximo de features a explorar |
| `evaluation_model` | objeto | modelo de ML | None | Modelo para avaliar as features geradas |
| `evaluation_metric` | str | 'auto', 'accuracy', 'roc_auc', 'f1', 'r2', 'rmse' | 'auto' | Métrica para avaliação |
| `evaluation_cv` | int | inteiro positivo | 5 | Folds para validação cruzada |
| `problem_type` | str | 'auto', 'classification', 'regression' | 'auto' | Tipo de problema |
| `random_state` | int | qualquer inteiro | 42 | Semente para reprodutibilidade |
| `verbosity` | int | 0, 1, 2 | 1 | Nível de detalhamento dos logs |

## Métodos Principais

### fit(X, y=None)

Ajusta o Explorer aos dados de entrada, identificando as melhores features e gerando transformações.

**Parâmetros:**
- `X` (pandas.DataFrame): DataFrame com as features
- `y` (pandas.Series, opcional): Valores alvo para problemas supervisionados

**Retorno:**
- Instância do próprio Explorer (para encadeamento de métodos)

**Exemplo:**
```python
explorer.fit(X_train, y_train)
```

### transform(X)

Aplica as transformações descobertas durante o fit a um novo conjunto de dados.

**Parâmetros:**
- `X` (pandas.DataFrame): Dados a serem transformados

**Retorno:**
- pandas.DataFrame: Dados com as transformações aplicadas

**Exemplo:**
```python
X_transformed = explorer.transform(X_test)
```

### fit_transform(X, y=None)

Ajusta o Explorer aos dados e retorna o resultado transformado.

**Parâmetros:**
- `X` (pandas.DataFrame): Features de entrada
- `y` (pandas.Series, opcional): Valores alvo para problemas supervisionados

**Retorno:**
- pandas.DataFrame: Dados transformados

**Exemplo:**
```python
X_transformed = explorer.fit_transform(X_train, y_train)
```

### get_feature_importances()

Retorna as importâncias das features calculadas durante o fit.

**Retorno:**
- dict: Dicionário com nomes de features e suas importâncias

**Exemplo:**
```python
importances = explorer.get_feature_importances()
```

### get_best_features()

Retorna a lista das melhores features selecionadas.

**Retorno:**
- list: Lista com nomes das melhores features

**Exemplo:**
```python
best_features = explorer.get_best_features()
```

### save(filepath)

Salva o Explorer em um arquivo para uso posterior.

**Parâmetros:**
- `filepath` (str): Caminho para salvar o modelo

**Exemplo:**
```python
explorer.save("explorer_model.joblib")
```

### load(filepath)

**Método de classe** para carregar um Explorer previamente salvo.

**Parâmetros:**
- `filepath` (str): Caminho para o arquivo do Explorer

**Retorno:**
- Instância carregada do Explorer

**Exemplo:**
```python
loaded_explorer = Explorer.load("explorer_model.joblib")
```

## Função Auxiliar

### create_explorer(config=None)

Função de conveniência para criar uma instância do Explorer com configurações opcionais.

**Parâmetros:**
- `config` (dict, opcional): Configurações personalizadas

**Retorno:**
- Instância configurada do Explorer

**Exemplo:**
```python
from explorer import create_explorer

config = {
    'polynomial_features': True,
    'clustering_features': True,
    'problem_type': 'classification'
}
explorer = create_explorer(config)
```

## Fluxo de Trabalho Típico

```python
import pandas as pd
from preprocessor import create_preprocessor
from explorer import create_explorer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Carregar dados
data = pd.read_csv("dataset.csv")

# Dividir entre features e target
X = data.drop('target', axis=1)
y = data['target']

# Dividir entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Etapa 1: Preprocessamento
preprocessor = create_preprocessor()
X_train_clean = preprocessor.fit_transform(X_train)
X_test_clean = preprocessor.transform(X_test)

# Etapa 2: Exploração e engenharia de features
config = {
    'polynomial_features': True,
    'interaction_features': True,
    'clustering_features': True,
    'feature_reduction_method': 'pca',
    'problem_type': 'classification',
    'evaluation_metric': 'auto'
}
explorer = create_explorer(config)

# Ajustar o explorer nos dados de treino e aplicar as transformações
X_train_engineered = explorer.fit_transform(X_train_clean, y_train)
X_test_engineered = explorer.transform(X_test_clean)

# Verificar as features mais importantes identificadas
importances = explorer.get_feature_importances()
print("Features mais importantes:", sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10])

# Verificar as melhores features selecionadas
best_features = explorer.get_best_features()
print("Melhores features selecionadas:", best_features)

# Treinar um modelo usando as features melhoradas
model = RandomForestClassifier()
model.fit(X_train_engineered, y_train)

# Avaliar o modelo
accuracy = model.score(X_test_engineered, y_test)
print(f"Acurácia do modelo: {accuracy:.4f}")

# Salvar o explorer para uso posterior
explorer.save("explorer_model.joblib")

# Mais tarde, em produção...
from explorer import Explorer

# Carregar o explorer salvo
loaded_explorer = Explorer.load("explorer_model.joblib")

# Preparar novos dados
new_data = pd.read_csv("new_data.csv")
new_data_clean = preprocessor.transform(new_data)
new_data_engineered = loaded_explorer.transform(new_data_clean)
```

## Exemplos Detalhados

### Exemplo 1: Exploração Completa do Espaço de Features

```python
import pandas as pd
import numpy as np
from preprocessor import create_preprocessor
from explorer import create_explorer
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression

# Carregar dataset de exemplo (diabetes)
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target)

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar e aplicar preprocessamento
preprocessor = create_preprocessor({'normalization': True})
X_train_clean = preprocessor.fit_transform(X_train)
X_test_clean = preprocessor.transform(X_test)

# Avaliar modelo baseline (com features originais)
baseline_model = LinearRegression()
baseline_scores = cross_val_score(baseline_model, X_train_clean, y_train, cv=5, scoring='r2')
print(f"Baseline R² score: {np.mean(baseline_scores):.4f}")

# Configurar explorador com várias transformações
explorer_config = {
    'polynomial_features': True,
    'polynomial_degree': 2,
    'interaction_features': True,
    'clustering_features': True,
    'n_clusters': 3,
    'feature_reduction_method': 'pca',
    'feature_reduction_components': 0.95,
    'problem_type': 'regression'
}

# Criar e aplicar o explorador
explorer = create_explorer(explorer_config)
X_train_eng = explorer.fit_transform(X_train_clean, y_train)
X_test_eng = explorer.transform(X_test_clean)

# Avaliar modelo com features engenheiradas
eng_model = LinearRegression()
eng_scores = cross_val_score(eng_model, X_train_eng, y_train, cv=5, scoring='r2')
print(f"Engineered features R² score: {np.mean(eng_scores):.4f}")

# Examinar melhoria
improvement = np.mean(eng_scores) - np.mean(baseline_scores)
print(f"Melhoria absoluta no R²: {improvement:.4f}")
print(f"Melhoria percentual: {improvement/abs(np.mean(baseline_scores))*100:.2f}%")

# Examinar as features mais importantes
importances = explorer.get_feature_importances()
print("\nTop 10 features por importância:")
for feature, score in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {feature}: {score:.4f}")
```

### Exemplo 2: Seleção de Features para Classificação

```python
import pandas as pd
import numpy as np
from preprocessor import create_preprocessor
from explorer import create_explorer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Carregar dataset de exemplo (cancer de mama)
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar preprocessamento básico
preprocessor = create_preprocessor()
X_train_clean = preprocessor.fit_transform(X_train)
X_test_clean = preprocessor.transform(X_test)

# Configurar explorador focado em seleção de features
explorer_config = {
    'feature_selection_method': 'mutual_info',
    'feature_selection_k': 10,  # Selecionar as 10 melhores features
    'polynomial_features': False,
    'interaction_features': False,
    'clustering_features': False,
    'feature_reduction_method': 'none',
    'problem_type': 'classification'
}

# Criar e aplicar o explorador
explorer = create_explorer(explorer_config)
X_train_selected = explorer.fit_transform(X_train_clean, y_train)
X_test_selected = explorer.transform(X_test_clean)

# Verificar quais features foram selecionadas
selected_features = explorer.get_best_features()
print(f"Features selecionadas ({len(selected_features)}):")
for feature in selected_features:
    print(f"  {feature}")

# Treinar um classificador com as features selecionadas
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_selected, y_train)

# Avaliar o classificador
y_pred = model.predict(X_test_selected)
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))
```