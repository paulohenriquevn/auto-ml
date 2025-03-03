# Referência Técnica do AutoFE

Esta referência técnica detalha a API do sistema AutoFE, descrevendo as classes, métodos e parâmetros disponíveis para usuários com conhecimento técnico.

## Módulo PreProcessor

O `PreProcessor` é o componente central do AutoFE, responsável por transformar os dados brutos em features de alta qualidade para modelos de machine learning.

### Classe `PreProcessor`

#### Inicialização

```python
PreProcessor(config: Optional[Dict] = None)
```

**Parâmetros:**
- `config` (Dict, opcional): Dicionário com configurações personalizadas.

**Configurações Disponíveis:**

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `missing_values_strategy` | str | 'median' | Estratégia para tratar valores ausentes. Opções: 'mean', 'median', 'most_frequent', 'knn' |
| `outlier_method` | str | 'iqr' | Método para detecção de outliers. Opções: 'zscore', 'iqr', 'isolation_forest' |
| `categorical_strategy` | str | 'onehot' | Estratégia para codificação de variáveis categóricas. Opções: 'onehot', 'ordinal' |
| `scaling` | str | 'standard' | Método de normalização/padronização. Opções: 'standard', 'minmax', 'robust' |
| `dimensionality_reduction` | str/None | None | Método de redução de dimensionalidade. Opções: 'pca', None |
| `feature_selection` | str/None | 'variance' | Método de seleção de features. Opções: 'variance', None |
| `generate_features` | bool | True | Se deve gerar automaticamente novas features |
| `verbosity` | int | 1 | Nível de detalhamento dos logs. Opções: 0 (mínimo), 1 (normal), 2 (detalhado) |
| `min_pca_components` | int | 10 | Número mínimo de componentes PCA a manter |
| `correlation_threshold` | float | 0.95 | Limiar para detecção de alta correlação entre features |

#### Métodos

##### `fit(df: pd.DataFrame, target_col: Optional[str] = None) -> 'PreProcessor'`

Ajusta o preprocessador aos dados, aprendendo os parâmetros necessários para as transformações.

**Parâmetros:**
- `df` (pandas.DataFrame): DataFrame com os dados de treinamento
- `target_col` (str, opcional): Nome da coluna alvo

**Retorno:**
- Instância do próprio `PreProcessor`, permitindo encadear métodos

**Exemplo:**
```python
preprocessor = PreProcessor()
preprocessor.fit(train_data, target_col='target')
```

##### `transform(df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame`

Aplica as transformações aprendidas a um conjunto de dados.

**Parâmetros:**
- `df` (pandas.DataFrame): DataFrame a ser transformado
- `target_col` (str, opcional): Nome da coluna alvo

**Retorno:**
- pandas.DataFrame: Dados transformados

**Exemplo:**
```python
df_transformed = preprocessor.transform(test_data, target_col='target')
```

##### `save(filepath: str) -> None`

Salva o preprocessador em um arquivo para uso futuro.

**Parâmetros:**
- `filepath` (str): Caminho do arquivo onde o preprocessador será salvo

**Exemplo:**
```python
preprocessor.save('/path/to/save/preprocessor.joblib')
```

##### `load(filepath: str) -> 'PreProcessor'` (método de classe)

Carrega um preprocessador previamente salvo.

**Parâmetros:**
- `filepath` (str): Caminho do arquivo onde o preprocessador foi salvo

**Retorno:**
- Instância de `PreProcessor` carregada

**Exemplo:**
```python
preprocessor = PreProcessor.load('/path/to/saved/preprocessor.joblib')
```

### Métodos Internos (para usuários avançados)

##### `_identify_column_types(df: pd.DataFrame) -> Dict`

Identifica automaticamente o tipo de cada coluna do DataFrame.

**Parâmetros:**
- `df` (pandas.DataFrame): DataFrame a ser analisado

**Retorno:**
- Dict: Dicionário com as chaves 'numeric' e 'categorical', cada uma contendo uma lista dos nomes das colunas do respectivo tipo

##### `_remove_outliers(df: pd.DataFrame) -> pd.DataFrame`

Remove outliers do DataFrame usando o método especificado.

**Parâmetros:**
- `df` (pandas.DataFrame): DataFrame do qual remover outliers

**Retorno:**
- pandas.DataFrame: DataFrame sem outliers

##### `_generate_features(df: pd.DataFrame) -> pd.DataFrame`

Gera automaticamente novas features baseadas nas existentes.

**Parâmetros:**
- `df` (pandas.DataFrame): DataFrame original

**Retorno:**
- pandas.DataFrame: DataFrame com as novas features adicionadas

##### `_remove_high_correlation(df: pd.DataFrame) -> pd.DataFrame`

Remove features altamente correlacionadas.

**Parâmetros:**
- `df` (pandas.DataFrame): DataFrame a ser analisado

**Retorno:**
- pandas.DataFrame: DataFrame sem as features altamente correlacionadas

## Módulo Explorer

O `Explorer` é responsável por testar diversas configurações de transformação e identificar as mais eficazes para os dados.

### Classe `Explorer`

#### Inicialização

```python
Explorer(heuristic: Callable[[pd.DataFrame], float] = None, target_col: Optional[str] = None)
```

**Parâmetros:**
- `heuristic` (Callable, opcional): Função heurística para avaliar transformações
- `target_col` (str, opcional): Nome da coluna alvo

#### Métodos

##### `analyze_transformations(df: pd.DataFrame) -> pd.DataFrame`

Testa diferentes configurações de transformação e retorna o DataFrame com as melhores transformações aplicadas.

**Parâmetros:**
- `df` (pandas.DataFrame): DataFrame a ser analisado

**Retorno:**
- pandas.DataFrame: DataFrame transformado com a melhor configuração encontrada

**Exemplo:**
```python
explorer = Explorer(target_col='target')
df_optimized = explorer.analyze_transformations(df)
```

##### `add_transformation(parent: str, name: str, data, score: float = 0.0) -> None`

Adiciona uma transformação à árvore de transformações com uma pontuação atribuída.

**Parâmetros:**
- `parent` (str): Nó pai na árvore
- `name` (str): Nome da transformação
- `data`: Dados transformados
- `score` (float, opcional): Pontuação da transformação

##### `find_best_transformation() -> str`

Retorna a melhor transformação com base na busca heurística.

**Retorno:**
- str: Nome da melhor transformação

**Exemplo:**
```python
best_transform = explorer.find_best_transformation()
```

### Classe `TransformationTree`

A `TransformationTree` é utilizada internamente pelo Explorer para manter um registro hierárquico das transformações testadas.

#### Inicialização

```python
TransformationTree()
```

#### Métodos

##### `add_transformation(parent: str, name: str, data, score: float = 0.0) -> None`

Adiciona uma transformação à árvore.

**Parâmetros:**
- `parent` (str): Nó pai na árvore
- `name` (str): Nome da transformação
- `data`: Dados transformados
- `score` (float, opcional): Pontuação da transformação

##### `get_best_transformations(heuristic: Callable[[Dict], float]) -> List[str]`

Retorna as melhores transformações baseadas em uma função heurística.

**Parâmetros:**
- `heuristic` (Callable): Função para pontuar transformações

**Retorno:**
- List[str]: Lista de nomes das transformações, ordenadas pela pontuação (da melhor para a pior)

### Classe `HeuristicSearch`

A `HeuristicSearch` implementa algoritmos de busca para encontrar a melhor transformação na árvore.

#### Inicialização

```python
HeuristicSearch(heuristic: Callable[[pd.DataFrame], float])
```

**Parâmetros:**
- `heuristic` (Callable): Função para avaliar a qualidade de um DataFrame transformado

#### Métodos

##### `search(tree: TransformationTree) -> str`

Executa uma busca heurística na árvore de transformações.

**Parâmetros:**
- `tree` (TransformationTree): Árvore de transformações a ser pesquisada

**Retorno:**
- str: Nome da melhor transformação encontrada

##### `custom_heuristic(df: pd.DataFrame) -> float` (método estático)

Heurística padrão para avaliar a qualidade de um DataFrame transformado.

**Parâmetros:**
- `df` (pandas.DataFrame): DataFrame a ser avaliado

**Retorno:**
- float: Pontuação de qualidade do DataFrame

## Funções Utilitárias

### `create_preprocessor(config: Optional[Dict] = None) -> PreProcessor`

Função auxiliar para criar uma instância do PreProcessor com configurações opcionais.

**Parâmetros:**
- `config` (Dict, opcional): Dicionário com configurações personalizadas

**Retorno:**
- PreProcessor: Instância configurada do PreProcessor

**Exemplo:**
```python
preprocessor = create_preprocessor({'scaling': 'minmax', 'generate_features': False})
```

## Fluxo de Trabalho Típico

### Pré-processamento Básico

```python
import pandas as pd
from autofe.preprocessor import PreProcessor

# Carregar dados
df = pd.read_csv('dados.csv')

# Criar e ajustar preprocessador
preprocessor = PreProcessor()
preprocessor.fit(df, target_col='target')

# Transformar dados
df_transformed = preprocessor.transform(df, target_col='target')

# Salvar preprocessador
preprocessor.save('preprocessor.joblib')
```

### Exploração Avançada

```python
import pandas as pd
from autofe.preprocessor import Explorer

# Carregar dados
df = pd.read_csv('dados.csv')

# Criar explorador
explorer = Explorer(target_col='target')

# Encontrar melhor transformação
df_optimized = explorer.analyze_transformations(df)
```

### Pipeline Completo

```python
import pandas as pd
from autofe.preprocessor import PreProcessor, Explorer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregar dados
df = pd.read_csv('dados.csv')

# Separar dados de treino e teste
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Explorar melhores transformações
explorer = Explorer(target_col='target')
best_train_df = explorer.analyze_transformations(train_df)

# Extrair configuração ótima e criar preprocessador
best_config = {
    'missing_values_strategy': 'median',
    'outlier_method': 'iqr',
    'categorical_strategy': 'onehot',
    'scaling': 'standard',
    'generate_features': True
}
preprocessor = PreProcessor(best_config)
preprocessor.fit(train_df, target_col='target')

# Transformar dados de treino e teste
train_transformed = preprocessor.transform(train_df, target_col='target')
test_transformed = preprocessor.transform(test_df, target_col='target')

# Treinar modelo
X_train = train_transformed.drop(columns=['target'])
y_train = train_transformed['target']
X_test = test_transformed.drop(columns=['target'])
y_test = test_transformed['target']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliar modelo
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Acurácia: {accuracy:.4f}")

# Salvar preprocessador para uso futuro
preprocessor.save('best_preprocessor.joblib')
```

## Dicas de Implementação Avançada

### Criação de Heurísticas Personalizadas

Você pode criar suas próprias heurísticas para avaliar a qualidade das transformações:

```python
def my_custom_heuristic(df: pd.DataFrame) -> float:
    """
    Heurística personalizada que valoriza:
    - Baixa correlação entre features
    - Distribuição mais próxima da normal
    - Baixa proporção de valores ausentes
    """
    # Penalidade por correlação
    corr_penalty = 0
    if df.shape[1] > 1:
        corr_matrix = df.corr().abs()
        high_corr = (corr_matrix > 0.7).sum().sum() - df.shape[1]
        corr_penalty = high_corr / (df.shape[1] ** 2)
    
    # Recompensa por distribuição mais próxima da normal
    normality_score = 0
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        # Usar skewness como medida de não-normalidade (0 = perfeitamente normal)
        skew = abs(df[col].skew())
        normality_score += 1 / (1 + skew)
    normality_score = normality_score / max(1, len(numeric_cols))
    
    # Penalidade por valores ausentes
    missing_penalty = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
    
    # Score final
    final_score = normality_score - corr_penalty - missing_penalty
    
    return final_score

# Usar heurística personalizada
explorer = Explorer(heuristic=my_custom_heuristic, target_col='target')
df_optimized = explorer.analyze_transformations(df)
```

### Extendendo o PreProcessor

Para adicionar funcionalidades personalizadas ao PreProcessor, você pode extender a classe:

```python
class CustomPreProcessor(PreProcessor):
    def __init__(self, config=None):
        super().__init__(config)
        # Configurações adicionais
        self.custom_settings = {
            'text_vectorization': 'tfidf',
            'image_processing': 'hog'
        }
        if config:
            self.custom_settings.update(config.get('custom_settings', {}))
    
    def _process_text_data(self, df):
        """Processamento especializado para dados de texto"""
        # Implementação do processamento de texto
        return df
    
    def _process_image_data(self, df):
        """Processamento especializado para dados de imagem"""
        # Implementação do processamento de imagem
        return df
    
    def transform(self, df, target_col=None):
        # Primeiro, aplica transformações padrão
        df_transformed = super().transform(df, target_col)
        
        # Depois, aplica transformações personalizadas
        if 'text_features' in self.custom_settings and self.custom_settings['text_features']:
            df_transformed = self._process_text_data(df_transformed)
        
        if 'image_features' in self.custom_settings and self.custom_settings['image_features']:
            df_transformed = self._process_image_data(df_transformed)
        
        return df_transformed
```

## Considerações de Desempenho

### Otimização para Grandes Datasets

Para datasets muito grandes, considere as seguintes estratégias:

1. **Amostragem**: Use uma amostra dos dados para exploração inicial
   ```python
   sample_df = df.sample(n=10000, random_state=42)
   explorer = Explorer(target_col='target')
   best_config = explorer.analyze_transformations(sample_df)
   ```

2. **Redução de Escopo**: Desative geração de features ou use métodos menos intensivos
   ```python
   config = {
       'generate_features': False,
       'outlier_method': 'zscore',  # Mais rápido que isolation_forest
       'dimensionality_reduction': None  # Desativar PCA
   }
   preprocessor = PreProcessor(config)
   ```

3. **Processamento Paralelo**: Em sistemas multicore, algumas operações podem ser paralelizadas
   ```python
   # Exemplo com joblib para paralelização
   from joblib import Parallel, delayed
   
   def process_chunk(chunk, preprocessor, target_col):
       return preprocessor.transform(chunk, target_col=target_col)
   
   # Divide o DataFrame em chunks
   chunks = np.array_split(df, 4)  # Divide em 4 partes
   
   # Processa em paralelo
   results = Parallel(n_jobs=-1)(
       delayed(process_chunk)(chunk, preprocessor, 'target') for chunk in chunks
   )
   
   # Combina os resultados
   df_transformed = pd.concat(results)
   ```

## Referências

- Documentação do scikit-learn: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- Pandas Documentation: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
- Feature Engineering for Machine Learning: [https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)