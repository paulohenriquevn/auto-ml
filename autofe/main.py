"""
Ponto de entrada principal para o sistema de automação de engenharia de features.
"""

import os
import argparse
import logging
import pandas as pd
from typing import Dict, Any, Optional, Union, List

# Importações internas
from config import LOGGING_CONFIG
from explorer.transformation_tree import TransformationTree
from explorer.heuristic_search import HeuristicSearch
from explorer.refinement import FeatureRefinement
from predictor.meta_learning import MetaLearner
from predictor.transformation_predictor import TransformationPredictor
from handlers.tabular_classification import TabularClassificationHandler
from handlers.tabular_regression import TabularRegressionHandler
from handlers.tabular_to_text import TabularToTextHandler
from handlers.time_series import TimeSeriesHandler


class AutoFeatureEngineering:
    """
    Classe principal para automação de engenharia de features.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o sistema de automação de engenharia de features.
        
        Args:
            config: Configurações personalizadas para o sistema (opcional)
        """
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Inicializando sistema de automação de engenharia de features")
        
        # Configurações
        self.config = config if config else {}
        
        # Componentes do sistema
        self.transformation_tree = TransformationTree()
        self.heuristic_search = HeuristicSearch()
        self.feature_refinement = FeatureRefinement()
        self.meta_learner = MetaLearner()
        self.transformation_predictor = TransformationPredictor(self.meta_learner)
        
        # Handlers para diferentes tipos de datasets
        self.dataset_handlers = {
            'tabular_classification': TabularClassificationHandler(),
            'tabular_regression': TabularRegressionHandler(),
            'tabular_to_text': TabularToTextHandler(),
            'time_series': TimeSeriesHandler(),
        }
        
        self.logger.info("Sistema inicializado com sucesso")
    
    def _setup_logging(self):
        """Configuração do sistema de logging"""
        os.makedirs(os.path.dirname(LOGGING_CONFIG['log_file']), exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, LOGGING_CONFIG['level']),
            format=LOGGING_CONFIG['format'],
            handlers=[
                logging.FileHandler(LOGGING_CONFIG['log_file']),
                logging.StreamHandler()
            ]
        )
    
    def fit_transform(
        self, 
        data: pd.DataFrame, 
        target: Union[str, pd.Series], 
        dataset_type: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Executa o pipeline completo de engenharia de features automática.
        
        Args:
            data: DataFrame com os dados de entrada
            target: Nome da coluna alvo ou Series com valores alvo
            dataset_type: Tipo de dataset ('tabular_classification', 'tabular_regression', 
                         'tabular_to_text', 'time_series')
            **kwargs: Parâmetros adicionais específicos para o tipo de dataset
            
        Returns:
            DataFrame com as features transformadas
        """
        self.logger.info(f"Iniciando processo de engenharia de features para {dataset_type}")
        
        # 1. Selecionar o handler adequado para o tipo de dataset
        if dataset_type not in self.dataset_handlers:
            raise ValueError(f"Tipo de dataset não suportado: {dataset_type}")
        
        handler = self.dataset_handlers[dataset_type]
        
        # 2. Preparar dados com o handler específico
        data_prepared = handler.prepare_data(data, target, **kwargs)
        
        # 3. Usar o Learner-Predictor para recomendar transformações com base em histórico
        self.logger.info("Obtendo recomendações de transformações do Learner-Predictor")
        recommended_transformations = self.transformation_predictor.predict_transformations(
            data_prepared, dataset_type
        )
        
        # 4. Usar o Explorer para gerar e avaliar features
        self.logger.info("Iniciando exploração de features com o Explorer")
        
        # 4.1 Construir árvore de transformações inicial
        self.transformation_tree.build(data_prepared, recommended_transformations)
        
        # 4.2 Executar busca heurística para encontrar as melhores transformações
        best_features = self.heuristic_search.search(
            self.transformation_tree, 
            data_prepared, 
            target, 
            handler
        )
        
        # 4.3 Refinar features (eliminar redundâncias, priorizar interpretabilidade)
        refined_features = self.feature_refinement.refine(best_features, data_prepared, target, handler)
        
        # 5. Aplicar transformações selecionadas aos dados
        transformed_data = handler.apply_transformations(data, refined_features)
        
        # 6. Atualizar o histórico de transformações para melhorar recomendações futuras
        self.logger.info("Atualizando histórico de transformações no Learner-Predictor")
        self.meta_learner.update_history(
            dataset_type=dataset_type,
            data_properties=handler.extract_data_properties(data),
            successful_transformations=refined_features,
            performance_metrics=handler.evaluate_transformations(transformed_data, target)
        )
        
        self.logger.info("Processo de engenharia de features concluído com sucesso")
        return transformed_data
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Retorna a importância de cada feature gerada.
        
        Returns:
            DataFrame com as features e suas importâncias
        """
        return self.heuristic_search.get_feature_importance()
    
    def get_transformation_tree(self) -> Dict:
        """
        Retorna a árvore de transformações utilizada.
        
        Returns:
            Dicionário representando a árvore de transformações
        """
        return self.transformation_tree.to_dict()
    
    def save_transformations(self, path: str):
        """
        Salva as transformações para uso posterior.
        
        Args:
            path: Caminho para salvar as transformações
        """
        self.logger.info(f"Salvando transformações em {path}")
        self.transformation_tree.save(path)
    
    def load_transformations(self, path: str):
        """
        Carrega transformações previamente salvas.
        
        Args:
            path: Caminho para carregar as transformações
        """
        self.logger.info(f"Carregando transformações de {path}")
        self.transformation_tree.load(path)


def main():
    """Função principal para execução via linha de comando"""
    parser = argparse.ArgumentParser(description='Automação de Engenharia de Features')
    parser.add_argument('--data', required=True, help='Caminho para o arquivo de dados (CSV, parquet, etc.)')
    parser.add_argument('--target', required=True, help='Nome da coluna alvo')
    parser.add_argument('--type', required=True, choices=[
        'tabular_classification', 'tabular_regression', 'tabular_to_text', 'time_series'
    ], help='Tipo de dataset')
    parser.add_argument('--output', default='transformed_data.csv', help='Caminho para salvar os dados transformados')
    parser.add_argument('--save-transformations', default=None, help='Caminho para salvar as transformações')
    
    args = parser.parse_args()
    
    # Determinar formato do arquivo e carregar dados
    file_ext = os.path.splitext(args.data)[1].lower()
    if file_ext == '.csv':
        data = pd.read_csv(args.data)
    elif file_ext == '.parquet':
        data = pd.read_parquet(args.data)
    elif file_ext == '.json':
        data = pd.read_json(args.data)
    else:
        raise ValueError(f"Formato de arquivo não suportado: {file_ext}")
    
    # Inicializar e executar o sistema
    auto_feature = AutoFeatureEngineering()
    transformed_data = auto_feature.fit_transform(data, args.target, args.type)
    
    # Salvar resultados
    transformed_data.to_csv(args.output, index=False)
    print(f"Dados transformados salvos em {args.output}")
    
    # Salvar transformações se solicitado
    if args.save_transformations:
        auto_feature.save_transformations(args.save_transformations)
        print(f"Transformações salvas em {args.save_transformations}")
    
    # Exibir importância das features
    feature_importance = auto_feature.get_feature_importance()
    print("\nImportância das Features:")
    print(feature_importance.head(10))


if __name__ == "__main__":
    main()
