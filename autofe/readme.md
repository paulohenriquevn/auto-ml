# AutoFeatureEngineering

Uma biblioteca Python para automação de engenharia de features, baseada em algoritmos avançados que exploram transformações e aprendem quais são as mais eficazes para diferentes tipos de dados.

## Índice

- [Visão Geral](#visão-geral)
- [Principais Recursos](#principais-recursos)
- [Instalação](#instalação)
- [Requisitos](#requisitos)
- [Uso Básico](#uso-básico)
- [Tipos de Datasets Suportados](#tipos-de-datasets-suportados)
- [Arquitetura do Sistema](#arquitetura-do-sistema)
- [Exemplos Detalhados](#exemplos-detalhados)
- [Transformações Disponíveis](#transformações-disponíveis)
- [Customização](#customização)
- [FAQ](#faq)
- [Licença](#licença)

## Visão Geral

AutoFeatureEngineering automatiza o processo de engenharia de features, permitindo a exploração eficiente de diferentes transformações de dados e a seleção daquelas que maximizam a qualidade dos modelos de machine learning. O sistema utiliza dois componentes principais:

1. **Explorer**: Navega no espaço de possíveis transformações, construindo uma árvore hierárquica e utilizando busca heurística para selecionar as melhores.

2. **Learner-Predictor**: Aprende quais transformações são mais eficazes para diferentes tipos de datasets e as recomenda automaticamente, utilizando técnicas como meta-aprendizado e imagificação de features.

## Principais Recursos

- 🔄 **Automação completa** da engenharia de features
- 🌲 **Exploração estruturada** através de árvore de transformações
- 🧠 **Meta-aprendizado** para recomendar transformações eficazes
- 📊 **Suporte para múltiplos tipos de dados**: classificação tabular, regressão, séries temporais e texto
- 🔍 **Avaliação inteligente** de transformações e features
- 📝 **Visualizações detalhadas** do processo de transformação
- 🚀 **Otimização** para maximizar a qualidade dos modelos

## Instalação

```bash
git clone https://github.com/username/AutoFeatureEngineering.git
cd AutoFeatureEngineering
pip install -r requirements.txt
```

## Requisitos

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- networkx
- scipy

## Uso Básico

```python
from autofeature.main import AutoFeatureEngineering

# Inicializar o sistema
auto_feature = AutoFeatureEngineering()

# Aplicar engenharia de features automatizada
transformed_data = auto_feature.fit_transform(
    data=df,
    target='target_column',
    dataset_type='tabular_classification'
)

# Obter importância das features
feature_importance = auto_feature.get_feature_importance()
print(feature_importance.head(10))

# Salvar transformações para uso futuro
auto_feature.save_transformations('transformations.pkl')
```

## Tipos de Datasets Suportados

### Classificação Tabular

Para problemas de classificação com dados tabulares.

```python
transformed_data = auto_feature.fit_transform(
    data=df,
    target='class_column',
    dataset_type='tabular_classification'
)
```

### Regressão Tabular

Para problemas de regressão com dados tabulares.

```python
transformed_data = auto_feature.fit_transform(
    data=df,
    target='value_column',
    dataset_type='tabular_regression'
)
```

### Tabular para Texto

Para problemas onde o alvo é uma variável de texto.

```python
transformed_data = auto_feature.fit_transform(
    data=df,
    target='text_column',
    dataset_type='tabular_to_text'
)
```

### Previsão de Séries Temporais

Para previsão de séries temporais com horizonte personalizado.

```python
transformed_data = auto_feature.fit_transform(
    data=df,
    target='value_column',
    dataset_type='time_series',
    date_column='date_column',
    forecast_horizon=7
)
```

## Arquitetura do Sistema

O sistema é composto por vários módulos principais:

### Explorer

- **TransformationTree**: Constrói e mantém uma estrutura hierárquica de transformações
- **HeuristicSearch**: Executa busca eficiente para encontrar as melhores transformações
- **FeatureRefinement**: Elimina redundâncias e prioriza features interpretáveis

### Learner-Predictor

- **MetaLearner**: Aprende quais transformações são eficazes com base em experiências anteriores
- **FeatureImagification**: Transforma features em representações visuais para facilitar o meta-aprendizado
- **TransformationPredictor**: Recomenda transformações para novos datasets

### Handlers de Datasets

- **TabularClassificationHandler**: Para classificação tabular
- **TabularRegressionHandler**: Para regressão tabular
- **TabularToTextHandler**: Para dados tabulares com alvo de texto
- **TimeSeriesHandler**: Para previsão de séries temporais

### Utilitários

- **Transformations**: Implementa todas as transformações disponíveis
- **Evaluation**: Funções para avaliar a qualidade de features
- **Visualization**: Visualizações para análise de features e transformações

## Exemplos Detalhados

### Classificação de Crédito

```python
import pandas as pd
from autofeature.main import AutoFeatureEngineering

# Carregar dados
credit_data = pd.read_csv('credit_data.csv')

# Inicializar sistema
auto_feature = AutoFeatureEngineering()

# Aplicar transformações automaticamente
transformed_data = auto_feature.fit_transform(
    data=credit_data,
    target='default',
    dataset_type='tabular_classification'
)

# Visualizar features mais importantes
print(auto_feature.get_feature_importance().head(10))

# Salvar transformações
auto_feature.save_transformations('credit_transformations.pkl')
```

### Previsão de Vendas (Série Temporal)

```python
import pandas as pd
from autofeature.main import AutoFeatureEngineering

# Carregar dados
sales_data = pd.read_csv('sales_data.csv', parse_dates=['date'])

# Inicializar sistema
auto_feature = AutoFeatureEngineering()

# Aplicar transformações automáticas para séries temporais
transformed_data = auto_feature.fit_transform(
    data=sales_data,
    target='sales',
    dataset_type='time_series',
    date_column='date',
    forecast_horizon=30  # Horizonte de 30 dias
)

# Visualizar features mais importantes
print(auto_feature.get_feature_importance().head(10))
```

## Transformações Disponíveis

### Variáveis Numéricas

- **Matemáticas**: log, sqrt, square, cube, reciprocal, sin, cos, tan
- **Normalização**: standardize, normalize, min_max_scale, robust_scale
- **Avançadas**: quantile_transform, power_transform, boxcox, winsorize

### Variáveis Categóricas

- **Codificação**: one_hot_encode, label_encode, target_encode, count_encode
- **Estatísticas**: frequency_encode, mean_encode, weight_of_evidence
- **Avançadas**: hash_encode

### Variáveis de Data/Hora

- **Componentes**: extract_year, extract_month, extract_day, extract_hour
- **Ciclos**: extract_dayofweek, extract_quarter, is_weekend
- **Delta**: time_since_reference, time_to_event

### Variáveis de Texto

- **Contagens**: word_count, char_count, stop_word_count, unique_word_count
- **Características**: uppercase_count, lowercase_count, punctuation_count
- **Semântica**: tfidf, word_embeddings, sentiment_score, readability_score

### Séries Temporais

- **Defasagens**: lag
- **Janelas**: rolling_mean, rolling_std, rolling_min, rolling_max, rolling_median
- **Tendências**: exponential_moving_average, differencing, decompose_trend
- **Sazonalidade**: decompose_seasonal, fourier_features, autocorrelation

### Interações

- **Básicas**: sum, difference, product, ratio
- **Avançadas**: polynomial

## Customização

### Configuração do Explorer

Você pode personalizar o comportamento do Explorer modificando o arquivo `config.py`:

```python
EXPLORER_CONFIG = {
    'max_transformation_depth': 3,   # Profundidade máxima de transformações encadeadas
    'max_features_per_level': 20,    # Número máximo de features por nível
    'min_performance_gain': 0.01,    # Ganho mínimo para aceitar uma feature
    'exploration_timeout': 3600,     # Timeout em segundos
    'redundancy_threshold': 0.95,    # Limiar para considerar features redundantes
}
```

### Configuração do Learner-Predictor

```python
LEARNER_PREDICTOR_CONFIG = {
    'history_path': './transformation_history/',  # Caminho para histórico
    'n_similar_datasets': 5,                      # Número de datasets similares a considerar
    'recommendation_threshold': 0.7,              # Limiar de confiança para recomendações
    'imagification_bins': 30,                     # Bins para imagificação
}
```

### Adição de Novas Transformações

Você pode adicionar novas transformações no arquivo `utils/transformations.py`:

```python
@register_transformation('numeric')
def my_custom_transform(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Minha transformação personalizada.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        
    Returns:
        Series com valores transformados
    """
    # Implementação da transformação
    return data[column] ** 3 - data[column]
```

## FAQ

### Q: Quantas features o sistema pode gerar?

**A:** O sistema pode gerar um número muito grande de features devido à natureza combinatória das transformações. Por padrão, limitamos o número de features por nível de profundidade a 20 para evitar explosão combinatória. Este valor pode ser ajustado em `config.py`.

### Q: Como lidar com features categóricas com alta cardinalidade?

**A:** O sistema automaticamente detecta features categóricas com alta cardinalidade e aplica técnicas como target encoding ou hash encoding, que funcionam bem nesses casos.

### Q: Como salvar e carregar transformações?

**A:** Você pode salvar as transformações com:
```python
auto_feature.save_transformations('transformations.pkl')
```

E carregá-las posteriormente:
```python
auto_feature.load_transformations('transformations.pkl')
```

### Q: O sistema suporta GPU?

**A:** O sistema em si não utiliza GPU diretamente, mas pode ser acelerado em sistemas com múltiplos núcleos de CPU. Para integração com GPU, considere usar os modelos subjacentes com suporte a GPU.

## Licença

Este projeto está licenciado sob os termos da licença MIT.
