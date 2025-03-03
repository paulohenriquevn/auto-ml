# explorer/transformers.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from abc import ABC, abstractmethod
from datetime import datetime
import sys
import os
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, OneHotEncoder, 
    LabelEncoder, PolynomialFeatures, KBinsDiscretizer
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data_types import (
    DataType, ProblemType, DatasetInfo, ColumnInfo, 
    TransformationType, TransformationResult
)

logger = logging.getLogger("AutoFE.Transformers")

class BaseTransformer(ABC):
    """
    Classe base para todos os transformadores de features.
    """
    
    @abstractmethod
    def transform(self, df: pd.DataFrame, column_name: str, 
                 transformation_type: TransformationType, 
                 dataset_info: DatasetInfo) -> Optional[TransformationResult]:
        """
        Aplica uma transformação a uma coluna do DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame a ser transformado.
            column_name (str): Nome da coluna a ser transformada.
            transformation_type (TransformationType): Tipo de transformação a ser aplicada.
            dataset_info (DatasetInfo): Informações sobre o dataset.
            
        Returns:
            TransformationResult or None: Resultado da transformação ou None se falhou.
        """
        pass


class MathematicalTransformer(BaseTransformer):
    """
    Transformador responsável por transformações matemáticas em colunas numéricas.
    """
    
    def transform(self, df: pd.DataFrame, column_name: str, 
                 transformation_type: TransformationType, 
                 dataset_info: DatasetInfo) -> Optional[TransformationResult]:
        """
        Aplica transformações matemáticas a uma coluna numérica.
        
        Args:
            df (pd.DataFrame): DataFrame a ser transformado.
            column_name (str): Nome da coluna a ser transformada.
            transformation_type (TransformationType): Tipo de transformação a ser aplicada.
            dataset_info (DatasetInfo): Informações sobre o dataset.
            
        Returns:
            TransformationResult or None: Resultado da transformação ou None se falhou.
        """
        try:
            # Faz uma cópia do DataFrame para não modificar o original
            df_transformed = df.copy()
            
            # Valida se a coluna existe
            if column_name not in df.columns:
                logger.warning(f"Coluna {column_name} não encontrada no DataFrame")
                return None
            
            # Valida se a coluna é numérica
            if not pd.api.types.is_numeric_dtype(df[column_name]):
                logger.warning(f"Coluna {column_name} não é numérica, não pode aplicar transformação matemática")
                return None
                
            # Prepara o prefixo para o nome da nova coluna
            prefix = f"{column_name}_"
            
            # Aplica a transformação específica
            created_columns = []
            
            if transformation_type == TransformationType.LOG:
                # Transformação logarítmica (log(x+1) para lidar com zeros)
                new_column_name = f"{prefix}log"
                # Garantir que não há valores negativos
                min_val = df[column_name].min()
                offset = max(0, -min_val + 1)  # Adiciona 1 para evitar log(0)
                df_transformed[new_column_name] = np.log1p(df[column_name] + offset)
                created_columns.append(new_column_name)
                
            elif transformation_type == TransformationType.SQUARE_ROOT:
                # Transformação raiz quadrada
                new_column_name = f"{prefix}sqrt"
                # Garantir que não há valores negativos
                min_val = df[column_name].min()
                offset = max(0, -min_val)
                df_transformed[new_column_name] = np.sqrt(df[column_name] + offset)
                created_columns.append(new_column_name)
                
            elif transformation_type == TransformationType.SQUARE:
                # Transformação ao quadrado
                new_column_name = f"{prefix}squared"
                df_transformed[new_column_name] = df[column_name] ** 2
                created_columns.append(new_column_name)
                
            elif transformation_type == TransformationType.CUBE:
                # Transformação ao cubo
                new_column_name = f"{prefix}cubed"
                df_transformed[new_column_name] = df[column_name] ** 3
                created_columns.append(new_column_name)
                
            elif transformation_type == TransformationType.RECIPROCAL:
                # Transformação recíproca (1/x)
                new_column_name = f"{prefix}reciprocal"
                # Evita divisão por zero
                df_transformed[new_column_name] = 1 / (df[column_name] + 1e-10)
                created_columns.append(new_column_name)
                
            elif transformation_type == TransformationType.STANDARDIZE:
                # Padronização (z-score)
                new_column_name = f"{prefix}standardized"
                scaler = StandardScaler()
                df_transformed[new_column_name] = scaler.fit_transform(df[[column_name]])
                created_columns.append(new_column_name)
                
            elif transformation_type == TransformationType.NORMALIZE:
                # Normalização (min-max scaling)
                new_column_name = f"{prefix}normalized"
                scaler = MinMaxScaler()
                df_transformed[new_column_name] = scaler.fit_transform(df[[column_name]])
                created_columns.append(new_column_name)
                
            elif transformation_type == TransformationType.BIN:
                # Discretização em bins
                new_column_name = f"{prefix}binned"
                num_bins = min(10, df[column_name].nunique())
                if num_bins > 1:  # Precisa de pelo menos 2 bins
                    binner = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='quantile')
                    df_transformed[new_column_name] = binner.fit_transform(df[[column_name]])
                    created_columns.append(new_column_name)
                
            elif transformation_type == TransformationType.POLYNOMIAL:
                # Gera termos polinomiais
                poly = PolynomialFeatures(degree=2, include_bias=False)
                poly_features = poly.fit_transform(df[[column_name]])
                
                # Gera um nome para cada termo polinomial (ignorando o primeiro, que é o original)
                for i in range(1, poly_features.shape[1]):
                    new_column_name = f"{prefix}poly_{i}"
                    df_transformed[new_column_name] = poly_features[:, i]
                    created_columns.append(new_column_name)
            
            # Verifica se alguma coluna foi criada
            if not created_columns:
                logger.warning(f"Nenhuma coluna criada com a transformação {transformation_type.name}")
                return None
                
            # Valida se as novas colunas contêm valores válidos
            for col in created_columns:
                if df_transformed[col].isna().all() or np.isinf(df_transformed[col]).any():
                    logger.warning(f"Coluna {col} contém valores inválidos após transformação")
                    # Remove a coluna inválida
                    df_transformed = df_transformed.drop(columns=[col])
                    created_columns.remove(col)
            
            # Se todas as colunas foram removidas, retorna None
            if not created_columns:
                return None
                
            # Cria e retorna o resultado da transformação
            return TransformationResult(
                transformed_data=df_transformed,
                transformation_name=transformation_type.name,
                created_columns=created_columns,
                removed_columns=[]
            )
            
        except Exception as e:
            logger.warning(f"Erro ao aplicar transformação {transformation_type.name} em {column_name}: {str(e)}")
            return None



class TemporalTransformer(BaseTransformer):
    """
    Transformador responsável por transformações temporais em colunas de data/hora.
    """
    
    def transform(self, df: pd.DataFrame, column_name: str, 
                 transformation_type: TransformationType, 
                 dataset_info: DatasetInfo) -> Optional[TransformationResult]:
        """
        Aplica transformações temporais a uma coluna de data/hora.
        
        Args:
            df (pd.DataFrame): DataFrame a ser transformado.
            column_name (str): Nome da coluna a ser transformada.
            transformation_type (TransformationType): Tipo de transformação a ser aplicada.
            dataset_info (DatasetInfo): Informações sobre o dataset.
            
        Returns:
            TransformationResult or None: Resultado da transformação ou None se falhou.
        """
        try:
            # Faz uma cópia do DataFrame para não modificar o original
            df_transformed = df.copy()
            
            # Valida se a coluna existe
            if column_name not in df.columns:
                logger.warning(f"Coluna {column_name} não encontrada no DataFrame")
                return None
            
            # Verifica se a coluna é de data/hora ou tenta converter
            is_datetime = pd.api.types.is_datetime64_dtype(df[column_name])
            
            if not is_datetime:
                try:
                    df_transformed[column_name] = pd.to_datetime(df[column_name])
                    is_datetime = True
                except:
                    # Se não for uma coluna de data/hora e não puder ser convertida,
                    # verifica se é uma série temporal com valores numéricos
                    if (dataset_info.problem_type == ProblemType.TIME_SERIES and 
                        pd.api.types.is_numeric_dtype(df[column_name])):
                        # Continua com transformações para séries temporais numéricas
                        pass
                    else:
                        logger.warning(f"Coluna {column_name} não pode ser convertida para datetime")
                        return None
                
            # Prepara o prefixo para o nome da nova coluna
            prefix = f"{column_name}_"
            
            # Aplica a transformação específica
            created_columns = []
            
            # Transformações para colunas de data/hora
            if is_datetime and transformation_type in [
                TransformationType.EXTRACT_DAY,
                TransformationType.EXTRACT_MONTH,
                TransformationType.EXTRACT_YEAR,
                TransformationType.EXTRACT_WEEKDAY,
                TransformationType.EXTRACT_HOUR
            ]:
                if transformation_type == TransformationType.EXTRACT_DAY:
                    # Extrai o dia do mês
                    new_column_name = f"{prefix}day"
                    df_transformed[new_column_name] = df_transformed[column_name].dt.day
                    created_columns.append(new_column_name)
                    
                elif transformation_type == TransformationType.EXTRACT_MONTH:
                    # Extrai o mês
                    new_column_name = f"{prefix}month"
                    df_transformed[new_column_name] = df_transformed[column_name].dt.month
                    created_columns.append(new_column_name)
                    
                elif transformation_type == TransformationType.EXTRACT_YEAR:
                    # Extrai o ano
                    new_column_name = f"{prefix}year"
                    df_transformed[new_column_name] = df_transformed[column_name].dt.year
                    created_columns.append(new_column_name)
                    
                elif transformation_type == TransformationType.EXTRACT_WEEKDAY:
                    # Extrai o dia da semana (0-6, com 0 sendo segunda-feira)
                    new_column_name = f"{prefix}weekday"
                    df_transformed[new_column_name] = df_transformed[column_name].dt.dayofweek
                    created_columns.append(new_column_name)
                    
                elif transformation_type == TransformationType.EXTRACT_HOUR:
                    # Extrai a hora do dia
                    new_column_name = f"{prefix}hour"
                    df_transformed[new_column_name] = df_transformed[column_name].dt.hour
                    created_columns.append(new_column_name)
            
            # Transformações para colunas temporais (mas não necessariamente datetime)
            elif dataset_info.problem_type == ProblemType.TIME_SERIES:
                if transformation_type == TransformationType.LAG:
                    # Cria features de lag para séries temporais
                    # Define lags diferentes
                    lags = [1, 3, 7, 14, 30]
                    
                    # Filtra para lags que fazem sentido dado o tamanho do dataset
                    max_lag = min(lags[-1], len(df) // 10)
                    valid_lags = [lag for lag in lags if lag <= max_lag]
                    
                    for lag in valid_lags:
                        new_column_name = f"{prefix}lag_{lag}"
                        # Certifica-se de que estamos fazendo shift em uma coluna numérica
                        if pd.api.types.is_numeric_dtype(df[dataset_info.target_column]):
                            df_transformed[new_column_name] = df_transformed[dataset_info.target_column].shift(lag)
                            created_columns.append(new_column_name)
                        
                elif transformation_type == TransformationType.ROLLING_MEAN:
                    # Cria média móvel para séries temporais
                    # Define janelas diferentes
                    windows = [3, 7, 14, 30]
                    
                    # Filtra para janelas que fazem sentido dado o tamanho do dataset
                    max_window = min(windows[-1], len(df) // 5)
                    valid_windows = [window for window in windows if window <= max_window]
                    
                    target_column = dataset_info.target_column
                    
                    # Certifica-se de que estamos aplicando rolling em uma coluna numérica
                    if pd.api.types.is_numeric_dtype(df[target_column]):
                        for window in valid_windows:
                            new_column_name = f"{target_column}_rolling_mean_{window}"
                            df_transformed[new_column_name] = df_transformed[target_column].rolling(window=window).mean()
                            created_columns.append(new_column_name)
                    
                elif transformation_type == TransformationType.ROLLING_STD:
                    # Cria desvio padrão móvel para séries temporais
                    # Define janelas diferentes
                    windows = [3, 7, 14, 30]
                    
                    # Filtra para janelas que fazem sentido dado o tamanho do dataset
                    max_window = min(windows[-1], len(df) // 5)
                    valid_windows = [window for window in windows if window <= max_window]
                    
                    target_column = dataset_info.target_column
                    
                    # Certifica-se de que estamos aplicando rolling em uma coluna numérica
                    if pd.api.types.is_numeric_dtype(df[target_column]):
                        for window in valid_windows:
                            new_column_name = f"{target_column}_rolling_std_{window}"
                            df_transformed[new_column_name] = df_transformed[target_column].rolling(window=window).std()
                            created_columns.append(new_column_name)
            
            # Verifica se alguma coluna foi criada
            if not created_columns:
                logger.warning(f"Nenhuma coluna criada com a transformação {transformation_type.name}")
                return None
                
            # Lida com valores ausentes criados pelas transformações de série temporal
            for col in created_columns:
                # Preenche valores ausentes com a média ou zero
                if df_transformed[col].isna().any():
                    if transformation_type in [TransformationType.LAG, TransformationType.ROLLING_MEAN, TransformationType.ROLLING_STD]:
                        # Preenche com 0 os valores ausentes no início das séries
                        df_transformed[col] = df_transformed[col].fillna(0)
                
            # Cria e retorna o resultado da transformação
            return TransformationResult(
                transformed_data=df_transformed,
                transformation_name=transformation_type.name,
                created_columns=created_columns,
                removed_columns=[]
            )
            
        except Exception as e:
            logger.warning(f"Erro ao aplicar transformação {transformation_type.name} em {column_name}: {str(e)}")
            return None



class CategoricalTransformer(BaseTransformer):
    """
    Transformador responsável por transformações em colunas categóricas.
    """
    
    def transform(self, df: pd.DataFrame, column_name: str, 
                 transformation_type: TransformationType, 
                 dataset_info: DatasetInfo) -> Optional[TransformationResult]:
        """
        Aplica transformações a uma coluna categórica.
        
        Args:
            df (pd.DataFrame): DataFrame a ser transformado.
            column_name (str): Nome da coluna a ser transformada.
            transformation_type (TransformationType): Tipo de transformação a ser aplicada.
            dataset_info (DatasetInfo): Informações sobre o dataset.
            
        Returns:
            TransformationResult or None: Resultado da transformação ou None se falhou.
        """
        try:
            # Faz uma cópia do DataFrame para não modificar o original
            df_transformed = df.copy()
            
            # Valida se a coluna existe
            if column_name not in df.columns:
                logger.warning(f"Coluna {column_name} não encontrada no DataFrame")
                return None
            
            # Verifica se a coluna é categórica ou string
            if not (pd.api.types.is_categorical_dtype(df[column_name]) or 
                   pd.api.types.is_object_dtype(df[column_name]) or
                   pd.api.types.is_string_dtype(df[column_name])):
                logger.warning(f"Coluna {column_name} não é categórica ou string")
                return None
                
            # Prepara o prefixo para o nome da nova coluna
            prefix = f"{column_name}_"
            
            # Aplica a transformação específica
            created_columns = []
            removed_columns = []
            
            if transformation_type == TransformationType.ONE_HOT_ENCODE:
                # Aplica one-hot encoding
                
                # Verifica o número de categorias únicas
                n_unique = df[column_name].nunique()
                if n_unique > 20:  # Limita a 20 categorias para evitar dimensionalidade excessiva
                    logger.warning(f"Coluna {column_name} tem muitas categorias ({n_unique}). One-hot encoding não aplicado.")
                    return None
                
                # Usa o pandas get_dummies para one-hot encoding
                dummies = pd.get_dummies(df[column_name], prefix=prefix)
                
                # Adiciona as colunas dummy ao DataFrame transformado
                for col in dummies.columns:
                    df_transformed[col] = dummies[col]
                    created_columns.append(col)
                
            elif transformation_type == TransformationType.LABEL_ENCODE:
                # Aplica label encoding
                new_column_name = f"{prefix}label_encoded"
                
                # Usa LabelEncoder do sklearn
                encoder = LabelEncoder()
                df_transformed[new_column_name] = encoder.fit_transform(df[column_name].astype(str))
                created_columns.append(new_column_name)
                
            elif transformation_type == TransformationType.FREQUENCY_ENCODE:
                # Aplica frequency encoding (substituir categorias por sua frequência)
                new_column_name = f"{prefix}freq_encoded"
                
                # Calcula a frequência de cada categoria
                freq_map = df[column_name].value_counts(normalize=True).to_dict()
                df_transformed[new_column_name] = df[column_name].map(freq_map)
                created_columns.append(new_column_name)
                
            elif transformation_type == TransformationType.TARGET_ENCODE:
                # Aplica target encoding (substituir categorias pela média do target)
                target_column = dataset_info.target_column
                
                if not target_column:
                    logger.warning("Target encoding requer uma coluna alvo")
                    return None
                
                # Verifica se o target é numérico
                if not pd.api.types.is_numeric_dtype(df[target_column]):
                    logger.warning(f"Target encoding requer um target numérico. {target_column} não é numérico.")
                    return None
                
                new_column_name = f"{prefix}target_encoded"
                
                # Calcula a codificação
                target_means = df.groupby(column_name)[target_column].mean().to_dict()
                df_transformed[new_column_name] = df[column_name].map(target_means)
                
                # Lida com categorias que não aparecem no treinamento
                global_mean = df[target_column].mean()
                df_transformed[new_column_name] = df_transformed[new_column_name].fillna(global_mean)
                
                created_columns.append(new_column_name)
            
            # Verifica se alguma coluna foi criada
            if not created_columns:
                logger.warning(f"Nenhuma coluna criada com a transformação {transformation_type.name}")
                return None
                
            # Cria e retorna o resultado da transformação
            return TransformationResult(
                transformed_data=df_transformed,
                transformation_name=transformation_type.name,
                created_columns=created_columns,
                removed_columns=removed_columns
            )
            
        except Exception as e:
            logger.warning(f"Erro ao aplicar transformação {transformation_type.name} em {column_name}: {str(e)}")
            return None


class TextTransformer(BaseTransformer):
    """
    Transformador responsável por transformações em colunas de texto.
    """
    
    def transform(self, df: pd.DataFrame, column_name: str, 
                 transformation_type: TransformationType, 
                 dataset_info: DatasetInfo) -> Optional[TransformationResult]:
        """
        Aplica transformações a uma coluna de texto.
        
        Args:
            df (pd.DataFrame): DataFrame a ser transformado.
            column_name (str): Nome da coluna a ser transformada.
            transformation_type (TransformationType): Tipo de transformação a ser aplicada.
            dataset_info (DatasetInfo): Informações sobre o dataset.
            
        Returns:
            TransformationResult or None: Resultado da transformação ou None se falhou.
        """
        try:
            # Faz uma cópia do DataFrame para não modificar o original
            df_transformed = df.copy()
            
            # Valida se a coluna existe
            if column_name not in df.columns:
                logger.warning(f"Coluna {column_name} não encontrada no DataFrame")
                return None
            
            # Verifica se a coluna contém texto
            if not pd.api.types.is_string_dtype(df[column_name]) and not pd.api.types.is_object_dtype(df[column_name]):
                logger.warning(f"Coluna {column_name} não é de texto")
                return None
                
            # Converte a coluna para string, se necessário
            df_transformed[column_name] = df[column_name].astype(str)
                
            # Prepara o prefixo para o nome da nova coluna
            prefix = f"{column_name}_"
            
            # Aplica a transformação específica
            created_columns = []
            
            if transformation_type == TransformationType.TF_IDF:
                # Aplica TF-IDF vectorization
                
                # Número máximo de features para evitar dimensionalidade excessiva
                max_features = min(100, len(df) // 10)
                
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    stop_words='english',
                    lowercase=True,
                    ngram_range=(1, 2)
                )
                
                # Aplica a vetorização
                tfidf_matrix = vectorizer.fit_transform(df_transformed[column_name])
                
                # Converte a matriz para DataFrame
                feature_names = vectorizer.get_feature_names_out()
                
                # Limita o número de features para evitar explosão dimensional
                for i, feature_name in enumerate(feature_names[:max_features]):
                    new_column_name = f"{prefix}tfidf_{i}"
                    df_transformed[new_column_name] = tfidf_matrix[:, i].toarray().flatten()
                    created_columns.append(new_column_name)
                
            elif transformation_type == TransformationType.COUNT_VECTORIZE:
                # Aplica Count Vectorization
                
                # Número máximo de features para evitar dimensionalidade excessiva
                max_features = min(100, len(df) // 10)
                
                vectorizer = CountVectorizer(
                    max_features=max_features,
                    stop_words='english',
                    lowercase=True,
                    ngram_range=(1, 1)
                )
                
                # Aplica a vetorização
                count_matrix = vectorizer.fit_transform(df_transformed[column_name])
                
                # Converte a matriz para DataFrame
                feature_names = vectorizer.get_feature_names_out()
                
                # Adiciona as features ao DataFrame
                for i, feature_name in enumerate(feature_names[:max_features]):
                    new_column_name = f"{prefix}count_{i}"
                    df_transformed[new_column_name] = count_matrix[:, i].toarray().flatten()
                    created_columns.append(new_column_name)
                
            elif transformation_type == TransformationType.WORD_EMBEDDING:
                # Para simplificar, aqui apenas extraímos características básicas do texto
                # Em um sistema real, usaríamos embeddings como Word2Vec, GloVe ou BERT
                
                # Comprimento do texto
                new_column_name = f"{prefix}length"
                df_transformed[new_column_name] = df_transformed[column_name].apply(len)
                created_columns.append(new_column_name)
                
                # Número de palavras
                new_column_name = f"{prefix}word_count"
                df_transformed[new_column_name] = df_transformed[column_name].apply(lambda x: len(x.split()))
                created_columns.append(new_column_name)
                
                # Número de caracteres únicos
                new_column_name = f"{prefix}unique_chars"
                df_transformed[new_column_name] = df_transformed[column_name].apply(lambda x: len(set(x)))
                created_columns.append(new_column_name)
                
                # Número de palavras únicas
                new_column_name = f"{prefix}unique_words"
                df_transformed[new_column_name] = df_transformed[column_name].apply(lambda x: len(set(x.lower().split())))
                created_columns.append(new_column_name)
            
            # Verifica se alguma coluna foi criada
            if not created_columns:
                logger.warning(f"Nenhuma coluna criada com a transformação {transformation_type.name}")
                return None
                
            # Cria e retorna o resultado da transformação
            return TransformationResult(
                transformed_data=df_transformed,
                transformation_name=transformation_type.name,
                created_columns=created_columns,
                removed_columns=[]
            )
            
        except Exception as e:
            logger.warning(f"Erro ao aplicar transformação {transformation_type.name} em {column_name}: {str(e)}")
            return None


class InteractionTransformer(BaseTransformer):
    """
    Transformador responsável por gerar interações entre colunas numéricas.
    """
    
    def transform(self, df: pd.DataFrame, column_name: str, 
                 transformation_type: TransformationType, 
                 dataset_info: DatasetInfo) -> Optional[TransformationResult]:
        """
        Gera características de interação entre features numéricas.
        
        Args:
            df (pd.DataFrame): DataFrame a ser transformado.
            column_name (str): Nome da coluna base para interações.
            transformation_type (TransformationType): Tipo de transformação a ser aplicada.
            dataset_info (DatasetInfo): Informações sobre o dataset.
            
        Returns:
            TransformationResult or None: Resultado da transformação ou None se falhou.
        """
        try:
            # Faz uma cópia do DataFrame para não modificar o original
            df_transformed = df.copy()
            
            # Valida se a coluna existe
            if column_name not in df.columns:
                logger.warning(f"Coluna {column_name} não encontrada no DataFrame")
                return None
            
            # Verifica se a coluna é numérica
            if not pd.api.types.is_numeric_dtype(df[column_name]):
                logger.warning(f"Coluna {column_name} não é numérica")
                return None
                
            # Encontra outras colunas numéricas (exceto a coluna alvo)
            target_column = dataset_info.target_column
            numeric_columns = []
            
            for col in df.columns:
                if (col != column_name and 
                    col != target_column and 
                    pd.api.types.is_numeric_dtype(df[col])):
                    numeric_columns.append(col)
            
            if not numeric_columns:
                logger.warning(f"Não foram encontradas outras colunas numéricas para interagir com {column_name}")
                return None
                
            # Limita o número de interações para evitar explosão combinatória
            max_interactions = 5
            if len(numeric_columns) > max_interactions:
                # Prioriza colunas com maior correlação com o target
                if target_column and pd.api.types.is_numeric_dtype(df[target_column]):
                    correlations = []
                    for col in numeric_columns:
                        corr = abs(df[[col, target_column]].corr().iloc[0, 1])
                        if not pd.isna(corr):
                            correlations.append((col, corr))
                    
                    # Ordena por correlação e pega as top features
                    numeric_columns = [col for col, _ in sorted(correlations, key=lambda x: x[1], reverse=True)][:max_interactions]
                else:
                    # Se não há target numérico, seleciona aleatoriamente
                    import random
                    random.seed(42)
                    numeric_columns = random.sample(numeric_columns, max_interactions)
            
            # Aplica a transformação específica
            created_columns = []
            
            for other_col in numeric_columns:
                # Evita interações com features derivadas da mesma coluna original
                if other_col.startswith(f"{column_name}_"):
                    continue
                    
                if transformation_type == TransformationType.MULTIPLY:
                    # Multiplicação
                    new_column_name = f"{column_name}_mult_{other_col}"
                    df_transformed[new_column_name] = df[column_name] * df[other_col]
                    created_columns.append(new_column_name)
                    
                elif transformation_type == TransformationType.DIVIDE:
                    # Divisão (com proteção contra divisão por zero)
                    new_column_name = f"{column_name}_div_{other_col}"
                    df_transformed[new_column_name] = df[column_name] / (df[other_col] + 1e-10)
                    created_columns.append(new_column_name)
                    
                elif transformation_type == TransformationType.ADD:
                    # Adição
                    new_column_name = f"{column_name}_add_{other_col}"
                    df_transformed[new_column_name] = df[column_name] + df[other_col]
                    created_columns.append(new_column_name)
                    
                elif transformation_type == TransformationType.SUBTRACT:
                    # Subtração
                    new_column_name = f"{column_name}_sub_{other_col}"
                    df_transformed[new_column_name] = df[column_name] - df[other_col]
                    created_columns.append(new_column_name)
            
            # Verifica se alguma coluna foi criada
            if not created_columns:
                logger.warning(f"Nenhuma coluna criada com a transformação {transformation_type.name}")
                return None
                
            # Valida se as novas colunas contêm valores válidos
            for col in created_columns:
                if df_transformed[col].isna().all() or np.isinf(df_transformed[col]).any():
                    logger.warning(f"Coluna {col} contém valores inválidos após transformação")
                    # Remove a coluna inválida
                    df_transformed = df_transformed.drop(columns=[col])
                    created_columns.remove(col)
            
            # Se todas as colunas foram removidas, retorna None
            if not created_columns:
                return None
                
            # Cria e retorna o resultado da transformação
            return TransformationResult(
                transformed_data=df_transformed,
                transformation_name=transformation_type.name,
                created_columns=created_columns,
                removed_columns=[]
            )
            
        except Exception as e:
            logger.warning(f"Erro ao aplicar transformação {transformation_type.name} em {column_name}: {str(e)}")
            return None


class GroupingTransformer(BaseTransformer):
    """
    Transformador responsável por gerar features de agrupamento.
    """
    
    def transform(self, df: pd.DataFrame, column_name: str, 
                 transformation_type: TransformationType, 
                 dataset_info: DatasetInfo) -> Optional[TransformationResult]:
        """
        Gera características baseadas em agrupamentos dos dados.
        
        Args:
            df (pd.DataFrame): DataFrame a ser transformado.
            column_name (str): Nome da coluna para agrupar.
            transformation_type (TransformationType): Tipo de transformação a ser aplicada.
            dataset_info (DatasetInfo): Informações sobre o dataset.
            
        Returns:
            TransformationResult or None: Resultado da transformação ou None se falhou.
        """
        try:
            # Faz uma cópia do DataFrame para não modificar o original
            df_transformed = df.copy()
            
            # Valida se a coluna existe
            if column_name not in df.columns:
                logger.warning(f"Coluna {column_name} não encontrada no DataFrame")
                return None
            
            # Verifica se a coluna pode ser usada para agrupamento (categórica ou datetime)
            is_categorical = (pd.api.types.is_categorical_dtype(df[column_name]) or 
                             pd.api.types.is_object_dtype(df[column_name]) or 
                             pd.api.types.is_string_dtype(df[column_name]))
            
            is_datetime = pd.api.types.is_datetime64_dtype(df[column_name])
            
            if not (is_categorical or is_datetime):
                logger.warning(f"Coluna {column_name} não é categórica ou datetime")
                return None
                
            # Encontra colunas numéricas para agregar
            target_column = dataset_info.target_column
            numeric_columns = []
            
            for col in df.columns:
                if (col != column_name and 
                    col != target_column and 
                    pd.api.types.is_numeric_dtype(df[col])):
                    numeric_columns.append(col)
            
            if not numeric_columns:
                logger.warning(f"Não foram encontradas colunas numéricas para agregar")
                return None
                
            # Limita o número de colunas para agregação
            max_cols = 3
            if len(numeric_columns) > max_cols:
                # Prioriza colunas com maior correlação com o target
                if target_column and pd.api.types.is_numeric_dtype(df[target_column]):
                    correlations = []
                    for col in numeric_columns:
                        corr = abs(df[[col, target_column]].corr().iloc[0, 1])
                        if not pd.isna(corr):
                            correlations.append((col, corr))
                    
                    # Ordena por correlação e pega as top features
                    numeric_columns = [col for col, _ in sorted(correlations, key=lambda x: x[1], reverse=True)][:max_cols]
                else:
                    # Se não há target numérico, seleciona aleatoriamente
                    import random
                    random.seed(42)
                    numeric_columns = random.sample(numeric_columns, max_cols)
            
            # Aplica a transformação específica
            created_columns = []
            
            for num_col in numeric_columns:
                if transformation_type == TransformationType.GROUP_MEAN:
                    # Média por grupo
                    new_column_name = f"{num_col}_by_{column_name}_mean"
                    group_means = df.groupby(column_name)[num_col].mean().to_dict()
                    df_transformed[new_column_name] = df[column_name].map(group_means)
                    created_columns.append(new_column_name)
                    
                elif transformation_type == TransformationType.GROUP_MEDIAN:
                    # Mediana por grupo
                    new_column_name = f"{num_col}_by_{column_name}_median"
                    group_medians = df.groupby(column_name)[num_col].median().to_dict()
                    df_transformed[new_column_name] = df[column_name].map(group_medians)
                    created_columns.append(new_column_name)
                    
                elif transformation_type == TransformationType.GROUP_MAX:
                    # Máximo por grupo
                    new_column_name = f"{num_col}_by_{column_name}_max"
                    group_max = df.groupby(column_name)[num_col].max().to_dict()
                    df_transformed[new_column_name] = df[column_name].map(group_max)
                    created_columns.append(new_column_name)
                    
                elif transformation_type == TransformationType.GROUP_MIN:
                    # Mínimo por grupo
                    new_column_name = f"{num_col}_by_{column_name}_min"
                    group_min = df.groupby(column_name)[num_col].min().to_dict()
                    df_transformed[new_column_name] = df[column_name].map(group_min)
                    created_columns.append(new_column_name)
                    
                elif transformation_type == TransformationType.GROUP_COUNT:
                    # Contagem por grupo
                    new_column_name = f"{column_name}_count"
                    group_count = df.groupby(column_name).size().to_dict()
                    df_transformed[new_column_name] = df[column_name].map(group_count)
                    created_columns.append(new_column_name)
                    break  # Só precisamos fazer isso uma vez
            
            # Verifica se alguma coluna foi criada
            if not created_columns:
                logger.warning(f"Nenhuma coluna criada com a transformação {transformation_type.name}")
                return None
                
            # Trata valores ausentes que possam ter sido criados
            for col in created_columns:
                if df_transformed[col].isna().any():
                    # Usa a média da coluna para preencher valores ausentes
                    df_transformed[col] = df_transformed[col].fillna(df_transformed[col].mean())
                
            # Cria e retorna o resultado da transformação
            return TransformationResult(
                transformed_data=df_transformed,
                transformation_name=transformation_type.name,
                created_columns=created_columns,
                removed_columns=[]
            )
            
        except Exception as e:
            logger.warning(f"Erro ao aplicar transformação {transformation_type.name} em {column_name}: {str(e)}")
            return None