"""
Módulo de Utilitários para a biblioteca AutoFeatureEngineering.

Este módulo contém funções e classes utilitárias para:
1. Implementar transformações para diferentes tipos de dados
2. Avaliar a qualidade de features e transformações
3. Visualizar dados, features e resultados de transformações
"""

from .transformations import apply_transformation, TRANSFORMATION_FUNCTIONS
from .evaluation import (
    evaluate_classification_features, evaluate_regression_features,
    calculate_feature_importance, calculate_correlation_matrix,
    identify_redundant_features, assess_feature_quality,
    evaluate_transformation_impact, assess_dataset_quality
)
from .visualization import (
    plot_feature_distribution, plot_feature_importance,
    plot_correlation_matrix, plot_transformation_effect,
    plot_transformation_tree, plot_feature_space,
    plot_feature_pair_grid, plot_partial_dependence,
    plot_time_series_features, create_reports_dashboard
)

__all__ = [
    'apply_transformation',
    'TRANSFORMATION_FUNCTIONS',
    'evaluate_classification_features',
    'evaluate_regression_features',
    'calculate_feature_importance',
    'calculate_correlation_matrix',
    'identify_redundant_features',
    'assess_feature_quality',
    'evaluate_transformation_impact',
    'assess_dataset_quality',
    'plot_feature_distribution',
    'plot_feature_importance',
    'plot_correlation_matrix',
    'plot_transformation_effect',
    'plot_transformation_tree',
    'plot_feature_space',
    'plot_feature_pair_grid',
    'plot_partial_dependence',
    'plot_time_series_features',
    'create_reports_dashboard'
]
