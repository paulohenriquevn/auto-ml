"""
Módulo Explorer para a biblioteca AutoFeatureEngineering.

Este módulo contém os componentes responsáveis por:
1. Navegar no espaço de possíveis transformações
2. Construir e manter uma árvore hierárquica de transformações
3. Executar busca heurística para encontrar as melhores transformações
4. Refinar o conjunto de features selecionadas
"""

from .transformation_tree import TransformationTree, TransformationNode
from .heuristic_search import HeuristicSearch
from .refinement import FeatureRefinement

__all__ = [
    'TransformationTree',
    'TransformationNode',
    'HeuristicSearch',
    'FeatureRefinement'
]
