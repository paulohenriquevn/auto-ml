# common/data_types.py
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np

class DataType(Enum):
    """Tipos de dados suportados pelo sistema."""
    NUMERIC = auto()
    CATEGORICAL = auto()
    TEXT = auto()
    DATETIME = auto()
    BOOLEAN = auto()
    UNKNOWN = auto()

class ProblemType(Enum):
    """Tipos de problemas suportados pelo sistema."""
    CLASSIFICATION = auto()
    REGRESSION = auto()
    TEXT = auto()
    TIME_SERIES = auto()

@dataclass
class ColumnInfo:
    """Informações sobre uma coluna do DataFrame."""
    name: str
    data_type: DataType
    is_target: bool = False
    is_time_column: bool = False
    num_unique_values: Optional[int] = None
    missing_percentage: float = 0.0
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    is_encoded: bool = False
    original_name: Optional[str] = None
    importance_score: Optional[float] = None
    
    def __hash__(self):
        """
        Implementa método hash para permitir que ColumnInfo seja usado em sets e como chaves de dicionários.
        """
        return hash(self.name)
    
    def __eq__(self, other):
        """
        Define equivalência entre objetos ColumnInfo para uso em operações de conjunto.
        """
        if not isinstance(other, ColumnInfo):
            return False
        return self.name == other.name
    
    @classmethod
    def from_series(cls, series: pd.Series, is_target: bool = False, 
                   is_time_column: bool = False) -> 'ColumnInfo':
        """
        Cria uma instância de ColumnInfo a partir de uma Series do pandas.
        
        Args:
            series (pd.Series): Série do pandas representando uma coluna.
            is_target (bool): Se a coluna é a variável alvo.
            is_time_column (bool): Se a coluna é a variável temporal.
            
        Returns:
            ColumnInfo: Informações sobre a coluna.
        """
        # Determina o tipo de dados
        if pd.api.types.is_numeric_dtype(series):
            data_type = DataType.NUMERIC
            mean = series.mean() if not series.isna().all() else None
            median = series.median() if not series.isna().all() else None
            std = series.std() if not series.isna().all() else None
            min_value = series.min() if not series.isna().all() else None
            max_value = series.max() if not series.isna().all() else None
        elif pd.api.types.is_datetime64_dtype(series):
            data_type = DataType.DATETIME
            mean = None
            median = None
            std = None
            min_value = series.min() if not series.isna().all() else None
            max_value = series.max() if not series.isna().all() else None
        elif pd.api.types.is_categorical_dtype(series) or series.nunique() < len(series) * 0.05:
            data_type = DataType.CATEGORICAL
            mean = None
            median = None
            std = None
            min_value = None
            max_value = None
        elif pd.api.types.is_bool_dtype(series):
            data_type = DataType.BOOLEAN
            mean = series.mean() if not series.isna().all() else None
            median = None
            std = None
            min_value = None
            max_value = None
        elif series.apply(lambda x: isinstance(x, str) and len(x) > 50).any():
            data_type = DataType.TEXT
            mean = None
            median = None
            std = None
            min_value = None
            max_value = None
        else:
            data_type = DataType.UNKNOWN
            mean = None
            median = None
            std = None
            min_value = None
            max_value = None
            
        return cls(
            name=series.name,
            data_type=data_type,
            is_target=is_target,
            is_time_column=is_time_column,
            num_unique_values=series.nunique(),
            missing_percentage=(series.isna().sum() / len(series)) * 100,
            mean=mean,
            median=median,
            std=std,
            min_value=min_value,
            max_value=max_value
        )

@dataclass
class DatasetInfo:
    """Informações sobre o dataset completo."""
    columns: List[ColumnInfo]
    num_rows: int
    num_columns: int
    target_column: Optional[str] = None
    time_column: Optional[str] = None
    problem_type: Optional[ProblemType] = None
    
    def get_column_names(self) -> List[str]:
        """
        Retorna apenas os nomes das colunas.
        
        Returns:
            list: Lista com os nomes das colunas.
        """
        return [col.name for col in self.columns]
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, target_column: Optional[str] = None,
                      time_column: Optional[str] = None, 
                      problem_type: Optional[ProblemType] = None) -> 'DatasetInfo':
        """
        Cria uma instância de DatasetInfo a partir de um DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame a ser analisado.
            target_column (str, optional): Nome da coluna alvo.
            time_column (str, optional): Nome da coluna temporal.
            problem_type (ProblemType, optional): Tipo do problema.
            
        Returns:
            DatasetInfo: Informações sobre o dataset.
        """
        columns = []
        
        for col_name in df.columns:
            is_target = col_name == target_column
            is_time = col_name == time_column
            
            col_info = ColumnInfo.from_series(
                df[col_name], 
                is_target=is_target,
                is_time_column=is_time
            )
            
            columns.append(col_info)
            
        return cls(
            columns=columns,
            num_rows=len(df),
            num_columns=len(df.columns),
            target_column=target_column,
            time_column=time_column,
            problem_type=problem_type
        )

@dataclass
class TransformationResult:
    """Resultado de uma transformação aplicada a um dataset."""
    transformed_data: pd.DataFrame
    transformation_name: str
    created_columns: List[str]
    removed_columns: List[str]
    performance_score: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    
    def get_report(self) -> Dict[str, Any]:
        """
        Gera um relatório sobre a transformação.
        
        Returns:
            dict: Relatório da transformação.
        """
        return {
            "transformation": self.transformation_name,
            "created_columns": self.created_columns,
            "removed_columns": self.removed_columns,
            "performance_score": self.performance_score,
            "feature_importance": self.feature_importance
        }

class TransformationType(Enum):
    """Tipos de transformações suportadas pelo sistema."""
    # Transformações matemáticas
    LOG = auto()
    SQUARE_ROOT = auto()
    SQUARE = auto()
    CUBE = auto()
    RECIPROCAL = auto()
    STANDARDIZE = auto()
    NORMALIZE = auto()
    
    # Transformações temporais
    EXTRACT_DAY = auto()
    EXTRACT_MONTH = auto()
    EXTRACT_YEAR = auto()
    EXTRACT_WEEKDAY = auto()
    EXTRACT_HOUR = auto()
    LAG = auto()
    ROLLING_MEAN = auto()
    ROLLING_STD = auto()
    
    # Transformações categóricas
    ONE_HOT_ENCODE = auto()
    LABEL_ENCODE = auto()
    TARGET_ENCODE = auto()
    FREQUENCY_ENCODE = auto()
    
    # Transformações de texto
    TF_IDF = auto()
    COUNT_VECTORIZE = auto()
    WORD_EMBEDDING = auto()
    
    # Transformações de interação
    MULTIPLY = auto()
    DIVIDE = auto()
    ADD = auto()
    SUBTRACT = auto()
    
    # Transformações de grupo
    GROUP_MEAN = auto()
    GROUP_MEDIAN = auto()
    GROUP_MAX = auto()
    GROUP_MIN = auto()
    GROUP_COUNT = auto()
    
    # Transformações de redução de dimensionalidade
    PCA = auto()
    TSNE = auto()
    
    # Outras
    OUTLIER_INDICATOR = auto()
    MISSING_INDICATOR = auto()
    BIN = auto()
    POLYNOMIAL = auto()

@dataclass
class TransformationInfo:
    """Informações sobre uma transformação."""
    transformation_type: TransformationType
    params: Dict[str, Any]
    input_columns: List[str]
    output_columns: List[str]
    score: Optional[float] = None
    
    def get_report(self) -> Dict[str, Any]:
        """
        Gera um relatório sobre a informação de transformação.
        
        Returns:
            dict: Relatório da informação de transformação.
        """
        return {
            "type": self.transformation_type.name,
            "params": self.params,
            "input_columns": self.input_columns,
            "output_columns": self.output_columns,
            "score": self.score
        }