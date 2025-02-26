"""
Módulo Learner-Predictor para a biblioteca AutoFeatureEngineering.

Este módulo contém os componentes responsáveis por:
1. Aprender quais transformações são eficazes com base em experiências anteriores
2. Imagificar features para facilitar o meta-aprendizado
3. Recomendar transformações para novos datasets
"""

from .meta_learning import MetaLearner
from .feature_imagification import FeatureImagification
from .transformation_predictor import TransformationPredictor

__all__ = [
    'MetaLearner',
    'FeatureImagification',
    'TransformationPredictor'
]
