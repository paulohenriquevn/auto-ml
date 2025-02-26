"""
Implementações das transformações disponíveis para engenharia de features.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
import re

logger = logging.getLogger(__name__)

# Dicionário de funções de transformação por categoria
TRANSFORMATION_FUNCTIONS = {}


def register_transformation(category: str, name: str = None):
    """
    Decorador para registrar uma função de transformação.
    
    Args:
        category: Categoria da transformação
        name: Nome da transformação (se None, usa o nome da função)
    """
    def decorator(func):
        transform_name = name if name else func.__name__
        if category not in TRANSFORMATION_FUNCTIONS:
            TRANSFORMATION_FUNCTIONS[category] = {}
        TRANSFORMATION_FUNCTIONS[category][transform_name] = func
        return func
    return decorator


def apply_transformation(
    data: pd.DataFrame,
    transformation_type: str,
    params: Dict[str, Any]
) -> Optional[pd.Series]:
    """
    Aplica uma transformação especificada aos dados.
    
    Args:
        data: DataFrame com os dados
        transformation_type: Tipo de transformação a aplicar
        params: Parâmetros para a transformação
        
    Returns:
        Series com os dados transformados ou None em caso de erro
    """
    # Buscar a função de transformação em todas as categorias
    transform_func = None
    
    for category in TRANSFORMATION_FUNCTIONS:
        if transformation_type in TRANSFORMATION_FUNCTIONS[category]:
            transform_func = TRANSFORMATION_FUNCTIONS[category][transformation_type]
            break
    
    if transform_func is None:
        logger.warning(f"Transformação {transformation_type} não encontrada")
        return None
    
    try:
        # Aplicar transformação
        return transform_func(data, **params)
    except Exception as e:
        logger.warning(f"Erro ao aplicar transformação {transformation_type}: {str(e)}")
        return None


# ====== Transformações para variáveis numéricas ======

@register_transformation('numeric')
def log(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Aplica transformação logarítmica (base e).
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        
    Returns:
        Series com transformação log
    """
    # Garantir que a coluna existe e é numérica
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Coluna {column} não existe ou não é numérica")
    
    # Adicionar pequeno offset para evitar log(0)
    min_val = data[column].min()
    offset = 1.0 if min_val >= 0 else abs(min_val) + 1.0
    
    return np.log(data[column] + offset)


@register_transformation('numeric')
def sqrt(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Aplica transformação raiz quadrada.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        
    Returns:
        Series com transformação raiz quadrada
    """
    # Garantir que a coluna existe e é numérica
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Coluna {column} não existe ou não é numérica")
    
    # Adicionar pequeno offset para garantir valores não negativos
    min_val = data[column].min()
    offset = 0.0 if min_val >= 0 else abs(min_val) + 0.01
    
    return np.sqrt(data[column] + offset)


@register_transformation('numeric')
def square(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Aplica transformação quadrática.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        
    Returns:
        Series com transformação quadrática
    """
    # Garantir que a coluna existe e é numérica
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Coluna {column} não existe ou não é numérica")
    
    return data[column] ** 2


@register_transformation('numeric')
def cube(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Aplica transformação cúbica.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        
    Returns:
        Series com transformação cúbica
    """
    # Garantir que a coluna existe e é numérica
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Coluna {column} não existe ou não é numérica")
    
    return data[column] ** 3


@register_transformation('numeric')
def reciprocal(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Aplica transformação recíproca (1/x).
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        
    Returns:
        Series com transformação recíproca
    """
    # Garantir que a coluna existe e é numérica
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Coluna {column} não existe ou não é numérica")
    
    # Evitar divisão por zero
    eps = 1e-10
    return 1.0 / (data[column] + eps)


@register_transformation('numeric')
def sin(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Aplica transformação seno.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        
    Returns:
        Series com transformação seno
    """
    # Garantir que a coluna existe e é numérica
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Coluna {column} não existe ou não é numérica")
    
    return np.sin(data[column])


@register_transformation('numeric')
def cos(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Aplica transformação cosseno.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        
    Returns:
        Series com transformação cosseno
    """
    # Garantir que a coluna existe e é numérica
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Coluna {column} não existe ou não é numérica")
    
    return np.cos(data[column])


@register_transformation('numeric')
def tan(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Aplica transformação tangente.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        
    Returns:
        Series com transformação tangente
    """
    # Garantir que a coluna existe e é numérica
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Coluna {column} não existe ou não é numérica")
    
    return np.tan(data[column])


@register_transformation('numeric')
def sigmoid(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Aplica transformação sigmoide (1 / (1 + exp(-x))).
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        
    Returns:
        Series com transformação sigmoide
    """
    # Garantir que a coluna existe e é numérica
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Coluna {column} não existe ou não é numérica")
    
    return 1.0 / (1.0 + np.exp(-data[column]))


@register_transformation('numeric')
def tanh(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Aplica transformação tangente hiperbólica.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        
    Returns:
        Series com transformação tanh
    """
    # Garantir que a coluna existe e é numérica
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Coluna {column} não existe ou não é numérica")
    
    return np.tanh(data[column])


@register_transformation('numeric')
def standardize(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Padroniza a coluna (subtrai média e divide pelo desvio padrão).
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        
    Returns:
        Series com valores padronizados
    """
    # Garantir que a coluna existe e é numérica
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Coluna {column} não existe ou não é numérica")
    
    # Usar StandardScaler do scikit-learn
    scaler = StandardScaler()
    values = data[column].values.reshape(-1, 1)
    return pd.Series(scaler.fit_transform(values).flatten(), index=data.index)


@register_transformation('numeric')
def normalize(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Normaliza a coluna para o intervalo [0, 1].
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        
    Returns:
        Series com valores normalizados
    """
    # Garantir que a coluna existe e é numérica
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Coluna {column} não existe ou não é numérica")
    
    # Usar MinMaxScaler do scikit-learn
    scaler = MinMaxScaler()
    values = data[column].values.reshape(-1, 1)
    return pd.Series(scaler.fit_transform(values).flatten(), index=data.index)


@register_transformation('numeric', 'min_max_scale')
def min_max_scale(data: pd.DataFrame, column: str, min_val: float = 0, max_val: float = 1) -> pd.Series:
    """
    Transforma a coluna para um intervalo específico [min_val, max_val].
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        min_val: Valor mínimo do intervalo de saída
        max_val: Valor máximo do intervalo de saída
        
    Returns:
        Series com valores transformados
    """
    # Garantir que a coluna existe e é numérica
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Coluna {column} não existe ou não é numérica")
    
    # Usar MinMaxScaler do scikit-learn
    scaler = MinMaxScaler(feature_range=(min_val, max_val))
    values = data[column].values.reshape(-1, 1)
    return pd.Series(scaler.fit_transform(values).flatten(), index=data.index)


@register_transformation('numeric')
def quantile_transform(data: pd.DataFrame, column: str, n_quantiles: int = 100) -> pd.Series:
    """
    Aplica transformação de quantis para uma distribuição mais uniforme.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        n_quantiles: Número de quantis a usar
        
    Returns:
        Series com valores transformados
    """
    # Garantir que a coluna existe e é numérica
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Coluna {column} não existe ou não é numérica")
    
    # Transformação de quantis usando scipy
    from scipy.stats import rankdata
    
    # Remover NaNs para o cálculo do rank
    values = data[column].dropna()
    if len(values) == 0:
        return pd.Series(index=data.index)
    
    # Calcular ranks normalizados
    ranks = rankdata(values, method='average')
    normalized_ranks = (ranks - 0.5) / len(ranks)
    
    # Criar Series de resultado
    result = pd.Series(index=data.index)
    result[values.index] = normalized_ranks
    
    return result


@register_transformation('numeric')
def power_transform(data: pd.DataFrame, column: str, method: str = 'yeo-johnson') -> pd.Series:
    """
    Aplica transformação de potência para aproximar normalidade.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        method: Método de transformação ('yeo-johnson' ou 'box-cox')
        
    Returns:
        Series com valores transformados
    """
    # Garantir que a coluna existe e é numérica
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Coluna {column} não existe ou não é numérica")
    
    # Para Box-Cox, todos os valores devem ser positivos
    if method == 'box-cox' and data[column].min() <= 0:
        raise ValueError("Box-Cox requer valores estritamente positivos")
    
    # Usar PowerTransformer do scikit-learn
    transformer = PowerTransformer(method=method)
    values = data[column].fillna(data[column].median()).values.reshape(-1, 1)
    return pd.Series(transformer.fit_transform(values).flatten(), index=data.index)


@register_transformation('numeric')
def boxcox(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Aplica transformação Box-Cox para normalizar dados.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        
    Returns:
        Series com valores transformados
    """
    # Garantir que a coluna existe e é numérica
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Coluna {column} não existe ou não é numérica")
    
    # Box-Cox requer valores estritamente positivos
    min_val = data[column].min()
    
    # Adicionar offset se necessário
    offset = 0.0
    if min_val <= 0:
        offset = abs(min_val) + 1.0
    
    # Aplicar transformação
    from scipy import stats
    transformed, _ = stats.boxcox(data[column] + offset)
    
    return pd.Series(transformed, index=data.index)


@register_transformation('numeric')
def winsorize(data: pd.DataFrame, column: str, limits: Tuple[float, float] = (0.05, 0.05)) -> pd.Series:
    """
    Aplica winsorização para limitar outliers.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        limits: Tuple com limites inferior e superior (ex: (0.05, 0.05) = 5% em cada extremo)
        
    Returns:
        Series com valores transformados
    """
    # Garantir que a coluna existe e é numérica
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Coluna {column} não existe ou não é numérica")
    
    # Aplicar winsorização
    from scipy import stats
    return pd.Series(stats.mstats.winsorize(data[column], limits=limits), index=data.index)


@register_transformation('numeric')
def robust_scale(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Escala robusta usando mediana e IQR.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        
    Returns:
        Series com valores transformados
    """
    # Garantir que a coluna existe e é numérica
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Coluna {column} não existe ou não é numérica")
    
    # Escala robusta
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    values = data[column].values.reshape(-1, 1)
    return pd.Series(scaler.fit_transform(values).flatten(), index=data.index)


# ====== Transformações para variáveis categóricas ======

@register_transformation('categorical')
def one_hot_encode(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Aplica one-hot encoding para variável categórica.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        
    Returns:
        DataFrame com colunas codificadas
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Aplicar one-hot encoding
    dummies = pd.get_dummies(data[column], prefix=column, drop_first=False)
    
    # Retornar a primeira coluna de dummies como exemplo
    # Na prática, todas as colunas dummies devem ser adicionadas ao DataFrame
    if len(dummies.columns) > 0:
        return dummies[dummies.columns[0]]
    else:
        return pd.Series(0, index=data.index)


@register_transformation('categorical')
def label_encode(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Aplica label encoding para variável categórica.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        
    Returns:
        Series com valores codificados
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Aplicar label encoding
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    
    # Tratar valores ausentes
    values = data[column].fillna('missing')
    return pd.Series(encoder.fit_transform(values), index=data.index)


@register_transformation('categorical')
def target_encode(
    data: pd.DataFrame, 
    column: str, 
    target_column: Optional[str] = None,
    target_values: Optional[pd.Series] = None,
    smoothing: float = 10.0
) -> pd.Series:
    """
    Aplica target encoding para variável categórica.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        target_column: Nome da coluna alvo
        target_values: Valores do alvo (opcional, alternativa a target_column)
        smoothing: Fator de suavização
        
    Returns:
        Series com valores codificados
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Obter valores do alvo
    if target_values is not None:
        y = target_values
    elif target_column is not None and target_column in data.columns:
        y = data[target_column]
    else:
        raise ValueError("É necessário fornecer target_column ou target_values")
    
    # Verificar se o alvo é numérico
    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError("O alvo deve ser numérico para target encoding")
    
    # Calcular médias por categoria
    global_mean = y.mean()
    df = pd.DataFrame({'category': data[column], 'target': y})
    
    # Calcular estatísticas por categoria
    stats = df.groupby('category')['target'].agg(['count', 'mean'])
    stats.columns = ['count', 'mean']
    
    # Aplicar suavização
    smoothed_mean = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
    
    # Mapear valores para categorias
    mapping = smoothed_mean.to_dict()
    return data[column].map(mapping).fillna(global_mean)


@register_transformation('categorical')
def count_encode(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Substitui categorias pela contagem de ocorrências.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        
    Returns:
        Series com valores codificados
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Calcular contagens
    counts = data[column].value_counts()
    
    # Mapear valores
    return data[column].map(counts)


@register_transformation('categorical')
def frequency_encode(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Substitui categorias pela frequência relativa.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        
    Returns:
        Series com valores codificados
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Calcular frequências
    n = len(data)
    freqs = data[column].value_counts() / n
    
    # Mapear valores
    return data[column].map(freqs)


@register_transformation('categorical')
def mean_encode(
    data: pd.DataFrame, 
    column: str, 
    numeric_column: str
) -> pd.Series:
    """
    Substitui categorias pela média de outra coluna numérica.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna categórica a transformar
        numeric_column: Nome da coluna numérica para calcular médias
        
    Returns:
        Series com valores codificados
    """
    # Garantir que as colunas existem
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    if numeric_column not in data.columns:
        raise ValueError(f"Coluna numérica {numeric_column} não existe")
    
    # Garantir que a coluna numérica é numérica
    if not pd.api.types.is_numeric_dtype(data[numeric_column]):
        raise ValueError(f"Coluna {numeric_column} não é numérica")
    
    # Calcular médias por categoria
    means = data.groupby(column)[numeric_column].mean()
    
    # Mapear valores
    global_mean = data[numeric_column].mean()
    return data[column].map(means).fillna(global_mean)


@register_transformation('categorical')
def hash_encode(data: pd.DataFrame, column: str, n_features: int = 8) -> pd.Series:
    """
    Aplica hash encoding para variável categórica.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        n_features: Número de features a gerar
        
    Returns:
        Series com valores codificados (primeira característica do hash)
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Função hash simples
    def hash_value(val, n_features):
        import hashlib
        val_str = str(val)
        hash_obj = hashlib.md5(val_str.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        return hash_int % n_features
    
    # Aplicar hash
    return data[column].apply(lambda x: hash_value(x, n_features))


@register_transformation('categorical')
def weight_of_evidence(
    data: pd.DataFrame, 
    column: str, 
    target_column: Optional[str] = None,
    target_values: Optional[pd.Series] = None
) -> pd.Series:
    """
    Aplica WoE (Weight of Evidence) para variável categórica vs alvo binário.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        target_column: Nome da coluna alvo binária
        target_values: Valores do alvo (opcional, alternativa a target_column)
        
    Returns:
        Series com valores WoE
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Obter valores do alvo
    if target_values is not None:
        y = target_values
    elif target_column is not None and target_column in data.columns:
        y = data[target_column]
    else:
        raise ValueError("É necessário fornecer target_column ou target_values")
    
    # Verificar se o alvo é binário
    unique_vals = y.unique()
    if len(unique_vals) != 2:
        raise ValueError("O alvo deve ser binário para WoE")
    
    # Garantir que os valores são 0 e 1
    if not set(unique_vals).issubset({0, 1}):
        y = (y == unique_vals[1]).astype(int)
    
    # Calcular WoE
    df = pd.DataFrame({'category': data[column], 'target': y})
    
    # Contagem de eventos (1s) e não-eventos (0s) por categoria
    grouped = df.groupby('category')['target'].agg(['sum', 'count'])
    grouped['non_event'] = grouped['count'] - grouped['sum']
    
    # Evitar divisão por zero
    eps = 1e-10
    
    # Totais
    total_event = grouped['sum'].sum()
    total_non_event = grouped['non_event'].sum()
    
    # Calcular WoE
    grouped['event_rate'] = (grouped['sum'] / total_event).clip(eps, 1-eps)
    grouped['non_event_rate'] = (grouped['non_event'] / total_non_event).clip(eps, 1-eps)
    grouped['woe'] = np.log(grouped['event_rate'] / grouped['non_event_rate'])
    
    # Mapear valores
    woe_dict = grouped['woe'].to_dict()
    
    # Para categorias não vistas, usar WoE = 0
    return data[column].map(woe_dict).fillna(0)


# ====== Transformações para variáveis de data/hora ======

@register_transformation('datetime')
def extract_year(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Extrai o ano de uma data.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de data
        
    Returns:
        Series com o ano
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Tentar converter para datetime se não for
    if not pd.api.types.is_datetime64_any_dtype(data[column]):
        try:
            date_col = pd.to_datetime(data[column])
        except:
            raise ValueError(f"Coluna {column} não pode ser convertida para data")
    else:
        date_col = data[column]
    
    return date_col.dt.year


@register_transformation('datetime')
def extract_month(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Extrai o mês de uma data.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de data
        
    Returns:
        Series com o mês
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Tentar converter para datetime se não for
    if not pd.api.types.is_datetime64_any_dtype(data[column]):
        try:
            date_col = pd.to_datetime(data[column])
        except:
            raise ValueError(f"Coluna {column} não pode ser convertida para data")
    else:
        date_col = data[column]
    
    return date_col.dt.month


@register_transformation('datetime')
def extract_day(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Extrai o dia do mês de uma data.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de data
        
    Returns:
        Series com o dia
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Tentar converter para datetime se não for
    if not pd.api.types.is_datetime64_any_dtype(data[column]):
        try:
            date_col = pd.to_datetime(data[column])
        except:
            raise ValueError(f"Coluna {column} não pode ser convertida para data")
    else:
        date_col = data[column]
    
    return date_col.dt.day


@register_transformation('datetime')
def extract_hour(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Extrai a hora de uma data/hora.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de data/hora
        
    Returns:
        Series com a hora
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Tentar converter para datetime se não for
    if not pd.api.types.is_datetime64_any_dtype(data[column]):
        try:
            date_col = pd.to_datetime(data[column])
        except:
            raise ValueError(f"Coluna {column} não pode ser convertida para data")
    else:
        date_col = data[column]
    
    return date_col.dt.hour


@register_transformation('datetime')
def extract_minute(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Extrai o minuto de uma data/hora.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de data/hora
        
    Returns:
        Series com o minuto
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Tentar converter para datetime se não for
    if not pd.api.types.is_datetime64_any_dtype(data[column]):
        try:
            date_col = pd.to_datetime(data[column])
        except:
            raise ValueError(f"Coluna {column} não pode ser convertida para data")
    else:
        date_col = data[column]
    
    return date_col.dt.minute


@register_transformation('datetime')
def extract_second(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Extrai o segundo de uma data/hora.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de data/hora
        
    Returns:
        Series com o segundo
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Tentar converter para datetime se não for
    if not pd.api.types.is_datetime64_any_dtype(data[column]):
        try:
            date_col = pd.to_datetime(data[column])
        except:
            raise ValueError(f"Coluna {column} não pode ser convertida para data")
    else:
        date_col = data[column]
    
    return date_col.dt.second


@register_transformation('datetime')
def extract_dayofweek(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Extrai o dia da semana de uma data.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de data
        
    Returns:
        Series com o dia da semana (0=Segunda, 6=Domingo)
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Tentar converter para datetime se não for
    if not pd.api.types.is_datetime64_any_dtype(data[column]):
        try:
            date_col = pd.to_datetime(data[column])
        except:
            raise ValueError(f"Coluna {column} não pode ser convertida para data")
    else:
        date_col = data[column]
    
    return date_col.dt.dayofweek


@register_transformation('datetime')
def extract_quarter(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Extrai o trimestre de uma data.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de data
        
    Returns:
        Series com o trimestre (1-4)
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Tentar converter para datetime se não for
    if not pd.api.types.is_datetime64_any_dtype(data[column]):
        try:
            date_col = pd.to_datetime(data[column])
        except:
            raise ValueError(f"Coluna {column} não pode ser convertida para data")
    else:
        date_col = data[column]
    
    return date_col.dt.quarter


@register_transformation('datetime')
def is_weekend(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Verifica se a data é um fim de semana.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de data
        
    Returns:
        Series booleana (1=Fim de semana, 0=Dia útil)
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Tentar converter para datetime se não for
    if not pd.api.types.is_datetime64_any_dtype(data[column]):
        try:
            date_col = pd.to_datetime(data[column])
        except:
            raise ValueError(f"Coluna {column} não pode ser convertida para data")
    else:
        date_col = data[column]
    
    # Verificar se é sábado (5) ou domingo (6)
    return (date_col.dt.dayofweek >= 5).astype(int)


@register_transformation('datetime')
def time_since_reference(
    data: pd.DataFrame, 
    column: str, 
    reference_date: Optional[str] = None,
    unit: str = 'days'
) -> pd.Series:
    """
    Calcula o tempo desde uma data de referência.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de data
        reference_date: Data de referência (opcional, padrão=mínimo da coluna)
        unit: Unidade de tempo ('days', 'hours', 'minutes', 'seconds')
        
    Returns:
        Series com o tempo decorrido
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Tentar converter para datetime se não for
    if not pd.api.types.is_datetime64_any_dtype(data[column]):
        try:
            date_col = pd.to_datetime(data[column])
        except:
            raise ValueError(f"Coluna {column} não pode ser convertida para data")
    else:
        date_col = data[column]
    
    # Definir data de referência
    if reference_date is None:
        ref_date = date_col.min()
    else:
        ref_date = pd.to_datetime(reference_date)
    
    # Calcular diferença de tempo
    time_diff = date_col - ref_date
    
    # Converter para a unidade especificada
    if unit == 'days':
        return time_diff.dt.total_seconds() / (24 * 3600)
    elif unit == 'hours':
        return time_diff.dt.total_seconds() / 3600
    elif unit == 'minutes':
        return time_diff.dt.total_seconds() / 60
    elif unit == 'seconds':
        return time_diff.dt.total_seconds()
    else:
        raise ValueError(f"Unidade inválida: {unit}")


@register_transformation('datetime')
def time_to_event(
    data: pd.DataFrame, 
    column: str, 
    event_date: str,
    unit: str = 'days'
) -> pd.Series:
    """
    Calcula o tempo até uma data de evento.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de data
        event_date: Data do evento
        unit: Unidade de tempo ('days', 'hours', 'minutes', 'seconds')
        
    Returns:
        Series com o tempo até o evento
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Tentar converter para datetime se não for
    if not pd.api.types.is_datetime64_any_dtype(data[column]):
        try:
            date_col = pd.to_datetime(data[column])
        except:
            raise ValueError(f"Coluna {column} não pode ser convertida para data")
    else:
        date_col = data[column]
    
    # Converter data do evento
    event_date = pd.to_datetime(event_date)
    
    # Calcular diferença de tempo
    time_diff = event_date - date_col
    
    # Converter para a unidade especificada
    if unit == 'days':
        return time_diff.dt.total_seconds() / (24 * 3600)
    elif unit == 'hours':
        return time_diff.dt.total_seconds() / 3600
    elif unit == 'minutes':
        return time_diff.dt.total_seconds() / 60
    elif unit == 'seconds':
        return time_diff.dt.total_seconds()
    else:
        raise ValueError(f"Unidade inválida: {unit}")


# ====== Transformações para variáveis de texto ======

@register_transformation('text')
def word_count(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Conta o número de palavras em um texto.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de texto
        
    Returns:
        Series com a contagem de palavras
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Converter para string e contar palavras
    return data[column].astype(str).apply(lambda x: len(x.split()))


@register_transformation('text')
def char_count(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Conta o número de caracteres em um texto.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de texto
        
    Returns:
        Series com a contagem de caracteres
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Contar caracteres
    return data[column].astype(str).str.len()


@register_transformation('text')
def stop_word_count(data: pd.DataFrame, column: str, language: str = 'english') -> pd.Series:
    """
    Conta o número de stopwords em um texto.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de texto
        language: Idioma das stopwords
        
    Returns:
        Series com a contagem de stopwords
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words(language))
    except:
        # Fallback para uma lista básica de stopwords em inglês
        stop_words = {
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
            'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
            'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'i',
            'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'would', 'could', 'should', 'ought', 'i\'m', 'you\'re', 'he\'s',
            'she\'s', 'it\'s', 'we\'re', 'they\'re', 'i\'ve', 'you\'ve', 'we\'ve', 'they\'ve',
            'i\'d', 'you\'d', 'he\'d', 'she\'d', 'we\'d', 'they\'d', 'i\'ll', 'you\'ll', 'he\'ll',
            'she\'ll', 'we\'ll', 'they\'ll', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'hasn\'t',
            'haven\'t', 'hadn\'t', 'doesn\'t', 'don\'t', 'didn\'t', 'won\'t', 'wouldn\'t',
            'shan\'t', 'shouldn\'t', 'can\'t', 'cannot', 'couldn\'t', 'mustn\'t', 'let\'s', 'that\'s',
            'who\'s', 'what\'s', 'here\'s', 'there\'s', 'when\'s', 'where\'s', 'why\'s', 'how\'s'
        }
    
    # Contar stopwords
    def count_stopwords(text):
        return sum(1 for word in str(text).lower().split() if word in stop_words)
    
    return data[column].apply(count_stopwords)


@register_transformation('text')
def unique_word_count(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Conta o número de palavras únicas em um texto.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de texto
        
    Returns:
        Series com a contagem de palavras únicas
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Contar palavras únicas
    return data[column].astype(str).apply(lambda x: len(set(x.lower().split())))


@register_transformation('text')
def uppercase_count(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Conta o número de caracteres maiúsculos em um texto.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de texto
        
    Returns:
        Series com a contagem de maiúsculas
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Contar maiúsculas
    return data[column].astype(str).apply(lambda x: sum(1 for c in x if c.isupper()))


@register_transformation('text')
def lowercase_count(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Conta o número de caracteres minúsculos em um texto.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de texto
        
    Returns:
        Series com a contagem de minúsculas
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Contar minúsculas
    return data[column].astype(str).apply(lambda x: sum(1 for c in x if c.islower()))


@register_transformation('text')
def punctuation_count(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Conta o número de sinais de pontuação em um texto.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de texto
        
    Returns:
        Series com a contagem de pontuação
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    import string
    
    # Conjunto de pontuação
    punct = set(string.punctuation)
    
    # Contar pontuação
    return data[column].astype(str).apply(lambda x: sum(1 for c in x if c in punct))


@register_transformation('text')
def tfidf(data: pd.DataFrame, column: str, max_features: int = 10) -> pd.Series:
    """
    Aplica TF-IDF para extrair a feature mais importante do texto.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de texto
        max_features: Número máximo de features a extrair
        
    Returns:
        Series com um escalar representativo (primeira componente do TF-IDF)
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Aplicar TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(data[column].astype(str))
    
    # Retornar a primeira feature para cada documento
    return pd.Series(tfidf_matrix[:, 0].toarray().flatten(), index=data.index)


@register_transformation('text')
def word_embeddings(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Aplica word embeddings simples (média de word2vec).
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de texto
        
    Returns:
        Series com um escalar representativo (primeira componente do embedding)
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Usar método simplificado (pseudo-embeddings) se bibliotecas não disponíveis
    # Para uma implementação real, seria necessário carregar modelo word2vec ou similar
    
    # Método simplificado: Hash de palavras
    def simple_embedding(text):
        words = str(text).lower().split()
        if not words:
            return 0
        
        # Calcular hash de cada palavra e média
        word_hashes = [hash(word) % 1000 for word in words]
        return sum(word_hashes) / len(word_hashes) / 1000  # Normalizar para [0, 1]
    
    return data[column].apply(simple_embedding)


@register_transformation('text')
def sentiment_score(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Calcula um score simplificado de sentimento baseado em palavras-chave.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de texto
        
    Returns:
        Series com o score de sentimento
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Palavras positivas e negativas simples
    positive_words = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'terrific',
        'outstanding', 'superb', 'awesome', 'best', 'happy', 'love', 'like', 'positive',
        'beautiful', 'perfect', 'joy', 'grateful', 'impressive', 'easy', 'useful'
    }
    
    negative_words = {
        'bad', 'awful', 'terrible', 'horrible', 'worst', 'poor', 'negative', 'hate',
        'dislike', 'disappointed', 'disappointing', 'difficult', 'useless', 'ugly',
        'hard', 'problem', 'issues', 'complaint', 'fail', 'failure', 'wrong', 'error'
    }
    
    # Calcular sentimento
    def calculate_sentiment(text):
        text = str(text).lower()
        words = re.findall(r'\b\w+\b', text)
        
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        total = pos_count + neg_count
        
        if total == 0:
            return 0.5  # Neutro
        
        return pos_count / total  # 0 (negativo) a 1 (positivo)
    
    return data[column].apply(calculate_sentiment)


@register_transformation('text')
def readability_score(data: pd.DataFrame, column: str) -> pd.Series:
    """
    Calcula um score simplificado de legibilidade.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna de texto
        
    Returns:
        Series com o score de legibilidade
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Calcular legibilidade simplificada (tamanho médio de palavras e sentenças)
    def calculate_readability(text):
        text = str(text)
        
        # Dividir em sentenças (simplificado)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.5  # Valor médio
        
        # Contar palavras
        words = []
        for sentence in sentences:
            words.extend(re.findall(r'\b\w+\b', sentence))
        
        if not words:
            return 0.5  # Valor médio
        
        # Calcular métricas
        avg_word_length = sum(len(word) for word in words) / len(words)
        avg_sentence_length = len(words) / len(sentences)
        
        # Score simplificado de Flesch: normalizado para [0, 1]
        # Quanto maior, mais fácil de ler
        score = 1.0 - min(1.0, ((avg_word_length * 0.1) + (avg_sentence_length * 0.05)))
        
        return score
    
    return data[column].apply(calculate_readability)


# ====== Transformações para séries temporais ======

@register_transformation('time_series')
def lag(data: pd.DataFrame, column: str, periods: int = 1) -> pd.Series:
    """
    Cria um lag da série temporal.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        periods: Número de períodos para o lag
        
    Returns:
        Series com valores defasados
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Aplicar lag
    return data[column].shift(periods)


@register_transformation('time_series')
def rolling_mean(data: pd.DataFrame, column: str, window: int = 3) -> pd.Series:
    """
    Calcula a média móvel da série.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        window: Tamanho da janela
        
    Returns:
        Series com a média móvel
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Aplicar média móvel
    return data[column].rolling(window=window, min_periods=1).mean()


@register_transformation('time_series')
def rolling_std(data: pd.DataFrame, column: str, window: int = 3) -> pd.Series:
    """
    Calcula o desvio padrão móvel da série.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        window: Tamanho da janela
        
    Returns:
        Series com o desvio padrão móvel
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Aplicar desvio padrão móvel
    return data[column].rolling(window=window, min_periods=1).std()


@register_transformation('time_series')
def rolling_min(data: pd.DataFrame, column: str, window: int = 3) -> pd.Series:
    """
    Calcula o mínimo móvel da série.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        window: Tamanho da janela
        
    Returns:
        Series com o mínimo móvel
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Aplicar mínimo móvel
    return data[column].rolling(window=window, min_periods=1).min()


@register_transformation('time_series')
def rolling_max(data: pd.DataFrame, column: str, window: int = 3) -> pd.Series:
    """
    Calcula o máximo móvel da série.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        window: Tamanho da janela
        
    Returns:
        Series com o máximo móvel
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Aplicar máximo móvel
    return data[column].rolling(window=window, min_periods=1).max()


@register_transformation('time_series')
def rolling_median(data: pd.DataFrame, column: str, window: int = 3) -> pd.Series:
    """
    Calcula a mediana móvel da série.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        window: Tamanho da janela
        
    Returns:
        Series com a mediana móvel
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Aplicar mediana móvel
    return data[column].rolling(window=window, min_periods=1).median()


@register_transformation('time_series')
def exponential_moving_average(
    data: pd.DataFrame, 
    column: str, 
    alpha: float = 0.3
) -> pd.Series:
    """
    Calcula a média móvel exponencial da série.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        alpha: Fator de suavização (0 < alpha < 1)
        
    Returns:
        Series com a média móvel exponencial
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Validar alpha
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"Alpha deve estar entre 0 e 1 (exclusivo)")
    
    # Aplicar EMA
    return data[column].ewm(alpha=alpha, adjust=False).mean()


@register_transformation('time_series')
def differencing(data: pd.DataFrame, column: str, periods: int = 1) -> pd.Series:
    """
    Calcula a diferença da série.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        periods: Número de períodos para diferenciação
        
    Returns:
        Series com a diferença
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Aplicar diferenciação
    return data[column].diff(periods=periods)


@register_transformation('time_series')
def decompose_trend(data: pd.DataFrame, column: str, window: int = 7) -> pd.Series:
    """
    Extrai a tendência da série usando média móvel.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        window: Tamanho da janela para média móvel
        
    Returns:
        Series com a tendência
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Extrair tendência usando média móvel centralizada
    return data[column].rolling(window=window, center=True, min_periods=1).mean()


@register_transformation('time_series')
def decompose_seasonal(data: pd.DataFrame, column: str, window: int = 7) -> pd.Series:
    """
    Extrai a componente sazonal da série (valor - tendência).
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        window: Tamanho da janela para média móvel
        
    Returns:
        Series com a componente sazonal
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Extrair tendência usando média móvel centralizada
    trend = data[column].rolling(window=window, center=True, min_periods=1).mean()
    
    # Calcular componente sazonal
    return data[column] - trend


@register_transformation('time_series')
def fourier_features(
    data: pd.DataFrame, 
    column: str, 
    date_column: Optional[str] = None,
    period: int = 365,  # Padrão: anual
    n_harmonics: int = 1
) -> pd.Series:
    """
    Cria features de Fourier para capturar sazonalidade.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna (ou coluna de data se date_column não fornecido)
        date_column: Nome da coluna de data (opcional)
        period: Período da sazonalidade
        n_harmonics: Número de harmônicos
        
    Returns:
        Series com a primeira característica de Fourier (seno)
    """
    # Verificar parâmetros
    if date_column is not None:
        if date_column not in data.columns:
            raise ValueError(f"Coluna de data {date_column} não existe")
        
        # Tentar converter para datetime se não for
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            try:
                date_col = pd.to_datetime(data[date_column])
            except:
                raise ValueError(f"Coluna {date_column} não pode ser convertida para data")
        else:
            date_col = data[date_column]
        
        # Converter para timestamp (número de segundos)
        t = date_col.astype(int) / 10**9 / (24 * 3600)  # Em dias
    else:
        # Usar índice numérico simples
        t = np.arange(len(data))
    
    # Calcular primeira característica de Fourier (seno)
    w = 2 * np.pi / period
    return np.sin(w * t)


@register_transformation('time_series')
def autocorrelation(data: pd.DataFrame, column: str, lag: int = 1) -> pd.Series:
    """
    Calcula a autocorrelação da série.
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna a transformar
        lag: Defasagem para autocorrelação
        
    Returns:
        Series com valores de autocorrelação
    """
    # Garantir que a coluna existe
    if column not in data.columns:
        raise ValueError(f"Coluna {column} não existe")
    
    # Calcular autocorrelação (simplificada - apenas um lag)
    series = data[column]
    
    # Remover NaNs
    series = series.dropna()
    
    if len(series) <= lag:
        return pd.Series(np.nan, index=data.index)
    
    # Calcular autocorrelação para o lag específico
    # Correlação entre série original e série defasada
    lagged = series.shift(lag)
    valid = ~(series.isna() | lagged.isna())
    
    if valid.sum() <= 1:
        return pd.Series(np.nan, index=data.index)
    
    # Calcular coeficiente de correlação
    corr = series[valid].corr(lagged[valid])
    
    # Retornar valor constante para todos os pontos
    return pd.Series(corr, index=data.index)


# ====== Transformações de interação entre features ======

@register_transformation('interaction')
def sum(data: pd.DataFrame, column1: str, column2: str) -> pd.Series:
    """
    Soma duas colunas.
    
    Args:
        data: DataFrame com os dados
        column1: Nome da primeira coluna
        column2: Nome da segunda coluna
        
    Returns:
        Series com a soma
    """
    # Garantir que as colunas existem
    if column1 not in data.columns:
        raise ValueError(f"Coluna {column1} não existe")
    if column2 not in data.columns:
        raise ValueError(f"Coluna {column2} não existe")
    
    # Converter para numérico se necessário
    col1 = pd.to_numeric(data[column1], errors='coerce')
    col2 = pd.to_numeric(data[column2], errors='coerce')
    
    return col1 + col2


@register_transformation('interaction')
def difference(data: pd.DataFrame, column1: str, column2: str) -> pd.Series:
    """
    Calcula a diferença entre duas colunas.
    
    Args:
        data: DataFrame com os dados
        column1: Nome da primeira coluna
        column2: Nome da segunda coluna
        
    Returns:
        Series com a diferença
    """
    # Garantir que as colunas existem
    if column1 not in data.columns:
        raise ValueError(f"Coluna {column1} não existe")
    if column2 not in data.columns:
        raise ValueError(f"Coluna {column2} não existe")
    
    # Converter para numérico se necessário
    col1 = pd.to_numeric(data[column1], errors='coerce')
    col2 = pd.to_numeric(data[column2], errors='coerce')
    
    return col1 - col2


@register_transformation('interaction')
def product(data: pd.DataFrame, column1: str, column2: str) -> pd.Series:
    """
    Multiplica duas colunas.
    
    Args:
        data: DataFrame com os dados
        column1: Nome da primeira coluna
        column2: Nome da segunda coluna
        
    Returns:
        Series com o produto
    """
    # Garantir que as colunas existem
    if column1 not in data.columns:
        raise ValueError(f"Coluna {column1} não existe")
    if column2 not in data.columns:
        raise ValueError(f"Coluna {column2} não existe")
    
    # Converter para numérico se necessário
    col1 = pd.to_numeric(data[column1], errors='coerce')
    col2 = pd.to_numeric(data[column2], errors='coerce')
    
    return col1 * col2


@register_transformation('interaction')
def ratio(data: pd.DataFrame, column1: str, column2: str) -> pd.Series:
    """
    Calcula a razão entre duas colunas.
    
    Args:
        data: DataFrame com os dados
        column1: Nome da primeira coluna (numerador)
        column2: Nome da segunda coluna (denominador)
        
    Returns:
        Series com a razão
    """
    # Garantir que as colunas existem
    if column1 not in data.columns:
        raise ValueError(f"Coluna {column1} não existe")
    if column2 not in data.columns:
        raise ValueError(f"Coluna {column2} não existe")
    
    # Converter para numérico se necessário
    col1 = pd.to_numeric(data[column1], errors='coerce')
    col2 = pd.to_numeric(data[column2], errors='coerce')
    
    # Evitar divisão por zero
    epsilon = 1e-10
    
    return col1 / (col2 + epsilon)


@register_transformation('interaction')
def polynomial(
    data: pd.DataFrame, 
    column1: str, 
    column2: str, 
    degree: int = 2
) -> pd.Series:
    """
    Cria interação polinomial entre duas colunas.
    
    Args:
        data: DataFrame com os dados
        column1: Nome da primeira coluna
        column2: Nome da segunda coluna
        degree: Grau do polinômio
        
    Returns:
        Series com a interação polinomial
    """
    # Garantir que as colunas existem
    if column1 not in data.columns:
        raise ValueError(f"Coluna {column1} não existe")
    if column2 not in data.columns:
        raise ValueError(f"Coluna {column2} não existe")
    
    # Converter para numérico se necessário
    col1 = pd.to_numeric(data[column1], errors='coerce')
    col2 = pd.to_numeric(data[column2], errors='coerce')
    
    # Para grau 2, retornar x^2 + y^2 + xy (exemplo de interação polinomial)
    if degree == 2:
        return col1**2 + col2**2 + col1 * col2
    
    # Para outros graus, retornar (x+y)^grau (expansão binomial)
    return (col1 + col2) ** degree
