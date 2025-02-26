"""
Módulo de Handlers de Datasets para a biblioteca AutoFeatureEngineering.

Este módulo contém os handlers específicos para diferentes tipos de datasets,
cada um responsável por:
1. Preparar os dados para o processo de engenharia de features
2. Aplicar transformações específicas para o tipo de dataset
3. Avaliar a qualidade das features geradas
"""

from .tabular_classification import TabularClassificationHandler
from .tabular_regression import TabularRegressionHandler
from .tabular_to_text import TabularToTextHandler
from .time_series import TimeSeriesHandler

__all__ = [
    'TabularClassificationHandler',
    'TabularRegressionHandler',
    'TabularToTextHandler',
    'TimeSeriesHandler'
]
