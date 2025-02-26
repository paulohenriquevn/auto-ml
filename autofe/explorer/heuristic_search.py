"""
Implementação da busca heurística para o Explorer.
"""

import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

# Importações internas
from config import EXPLORER_CONFIG
from explorer.transformation_tree import TransformationTree, TransformationNode
from utils.transformations import apply_transformation


class HeuristicSearch:
    """
    Implementa o algoritmo de busca heurística para encontrar as melhores transformações
    no espaço de features.
    """
    
    def __init__(self):
        """
        Inicializa o algoritmo de busca heurística.
        """
        self.logger = logging.getLogger(__name__)
        self.feature_importance = None
        self.base_performance = None
        self.best_features = []
        self.explored_features = set()
        
    def search(
        self,
        tree: TransformationTree,
        data: pd.DataFrame,
        target: Union[str, pd.Series],
        dataset_handler: Any,
        max_iterations: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Executa a busca heurística para encontrar as melhores transformações.
        
        Args:
            tree: Árvore de transformações
            data: Dados para avaliação
            target: Variável alvo
            dataset_handler: Handler específico para o tipo de dataset
            max_iterations: Número máximo de iterações (None = sem limite)
            timeout: Tempo máximo em segundos (None = sem limite)
            
        Returns:
            Lista de dicionários descrevendo as melhores features encontradas
        """
        self.logger.info("Iniciando busca heurística")
        
        # Definir limites de tempo e iterações
        if timeout is None:
            timeout = EXPLORER_CONFIG['exploration_timeout']
        
        start_time = time.time()
        if isinstance(target, str):
            y = data[target]
            X = data.drop(columns=[target])
        else:
            y = target
            X = data
        
        # Estabelecer performance base com as features originais
        self.logger.info("Estabelecendo performance base com features originais")
        self.base_performance = self._evaluate_features(X, y, dataset_handler)
        self.logger.info(f"Performance base: {self.base_performance:.4f}")
        
        # Inicializar conjunto de features exploradas e níveis
        self.explored_features = set(X.columns)
        current_level_nodes = [tree.get_node_by_name(col) for col in X.columns]
        iteration = 0
        
        # Lista para armazenar as melhores features encontradas
        self.best_features = []
        
        # Iniciar busca por níveis
        self.logger.info("Iniciando exploração por níveis na árvore de transformações")
        
        while current_level_nodes and (max_iterations is None or iteration < max_iterations):
            # Verificar timeout
            if time.time() - start_time > timeout:
                self.logger.info(f"Timeout atingido ({timeout}s)")
                break
            
            iteration += 1
            self.logger.info(f"Iteração {iteration}, explorando {len(current_level_nodes)} nós")
            
            # Avaliar nós do nível atual
            evaluated_nodes = self._evaluate_nodes(current_level_nodes, X, y, dataset_handler)
            
            # Selecionar melhores nós para expansão
            next_level_nodes = self._select_nodes_for_expansion(evaluated_nodes, tree)
            
            # Expandir nós selecionados (gerar próximo nível da árvore)
            self._expand_nodes(next_level_nodes, X, tree)
            
            # Atualizar nós do nível atual para a próxima iteração
            current_level_nodes = []
            for node in next_level_nodes:
                current_level_nodes.extend(node.children)
            
            # Se não há mais nós para explorar, encerrar
            if not current_level_nodes:
                self.logger.info("Não há mais nós para explorar, encerrando busca")
                break
            
            # Atualizar as melhores features
            self._update_best_features(evaluated_nodes)
        
        self.logger.info(f"Busca concluída após {iteration} iterações")
        self.logger.info(f"Tempo total: {time.time() - start_time:.2f}s")
        self.logger.info(f"Número de features selecionadas: {len(self.best_features)}")
        
        return self.best_features
    
    def _evaluate_nodes(
        self,
        nodes: List[TransformationNode],
        X: pd.DataFrame,
        y: pd.Series,
        dataset_handler: Any
    ) -> List[TransformationNode]:
        """
        Avalia os nós de transformação, calculando métricas de importância e desempenho.
        
        Args:
            nodes: Lista de nós a serem avaliados
            X: Dados de entrada
            y: Variável alvo
            dataset_handler: Handler específico para o tipo de dataset
            
        Returns:
            Lista de nós avaliados
        """
        # Para cada nó, aplicar a transformação e avaliar
        for node in nodes:
            # Pular se já explorado
            if node.name in self.explored_features:
                continue
            
            # Obter linhagem do nó para aplicar transformações em sequência
            lineage = self._get_transformation_sequence(node)
            
            # Aplicar transformações
            try:
                transformed_feature = self._apply_transformation_sequence(lineage, X)
                
                # Se a transformação resultou em erro ou valores inválidos, pular
                if transformed_feature is None:
                    continue
                
                # Criar DataFrame temporário com as features originais e a nova
                X_temp = X.copy()
                X_temp[node.name] = transformed_feature
                
                # Avaliar performance
                performance = self._evaluate_features(X_temp, y, dataset_handler)
                node.performance_gain = performance - self.base_performance
                
                # Calcular importância da feature
                node.importance = self._calculate_feature_importance(transformed_feature, y, dataset_handler)
                
                # Calcular redundância
                node.redundancy_score = self._calculate_redundancy(transformed_feature, X)
                
                # Marcar como explorado
                self.explored_features.add(node.name)
                
            except Exception as e:
                self.logger.warning(f"Erro ao avaliar nó {node.name}: {str(e)}")
                node.performance_gain = 0
                node.importance = 0
                node.redundancy_score = 1  # Alta redundância indica problemas
        
        return nodes
    
    def _get_transformation_sequence(self, node: TransformationNode) -> List[Dict[str, Any]]:
        """
        Obtém a sequência de transformações que leva a um nó específico.
        
        Args:
            node: Nó para o qual a sequência será gerada
            
        Returns:
            Lista de dicionários descrevendo as transformações
        """
        sequence = []
        current = node
        
        # Trabalhar de trás para frente, do nó atual até a raiz
        while current.parent_id:
            if current.transformation_type:
                sequence.insert(0, {
                    'type': current.transformation_type,
                    'params': current.transformation_params,
                    'output_name': current.name
                })
            
            # Obter nó pai
            parent_id = current.parent_id
            current = self._find_node_by_id(parent_id)
            
            if current is None:
                break
        
        return sequence
    
    def _find_node_by_id(self, node_id: str) -> Optional[TransformationNode]:
        """
        Método auxiliar para encontrar um nó pelo ID.
        
        Args:
            node_id: ID do nó
            
        Returns:
            Nó correspondente ou None se não encontrado
        """
        # Este método normalmente usaria a árvore, mas aqui é simplificado
        # Em uma implementação completa, você usaria tree.get_node_by_id(node_id)
        return None
    
    def _apply_transformation_sequence(
        self,
        transformation_sequence: List[Dict[str, Any]],
        X: pd.DataFrame
    ) -> Optional[pd.Series]:
        """
        Aplica uma sequência de transformações aos dados.
        
        Args:
            transformation_sequence: Sequência de transformações a aplicar
            X: Dados de entrada
            
        Returns:
            Série com os valores transformados ou None em caso de erro
        """
        if not transformation_sequence:
            return None
        
        data = X.copy()
        
        for transform in transformation_sequence:
            try:
                # Aplicar transformação
                result = apply_transformation(
                    data, 
                    transform['type'], 
                    transform['params']
                )
                
                # Verificar se a transformação resultou em valores válidos
                if result is None or result.isnull().all():
                    return None
                
                # Para transformações intermediárias, adicionar ao dataframe
                if transform != transformation_sequence[-1]:
                    data[transform['output_name']] = result
                else:
                    return result
                
            except Exception as e:
                self.logger.warning(f"Erro ao aplicar transformação {transform['type']}: {str(e)}")
                return None
        
        # Se chegou aqui é porque a sequência estava vazia ou algo deu errado
        return None
    
    def _evaluate_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_handler: Any
    ) -> float:
        """
        Avalia a performance de um conjunto de features.
        
        Args:
            X: Dados de entrada
            y: Variável alvo
            dataset_handler: Handler específico para o tipo de dataset
            
        Returns:
            Valor numérico representando a performance
        """
        # Usar o dataset_handler para avaliar a performance
        performance = dataset_handler.evaluate_features(X, y)
        return performance
    
    def _calculate_feature_importance(
        self,
        feature: pd.Series,
        y: pd.Series,
        dataset_handler: Any
    ) -> float:
        """
        Calcula a importância de uma feature individual.
        
        Args:
            feature: Feature a ser avaliada
            y: Variável alvo
            dataset_handler: Handler específico para o tipo de dataset
            
        Returns:
            Valor numérico representando a importância
        """
        # Em uma implementação completa, usaria o dataset_handler
        # Aqui vamos usar uma abordagem simplificada com informação mútua
        try:
            # Verificar se a variável alvo é categórica ou numérica
            is_classification = dataset_handler.is_classification()
            
            # Remover valores NaN
            valid_indices = ~(feature.isnull() | y.isnull())
            if valid_indices.sum() == 0:
                return 0
            
            valid_feature = feature[valid_indices].values.reshape(-1, 1)
            valid_y = y[valid_indices]
            
            # Calcular informação mútua
            if is_classification:
                mi = mutual_info_classif(valid_feature, valid_y, random_state=42)[0]
            else:
                mi = mutual_info_regression(valid_feature, valid_y, random_state=42)[0]
            
            return mi
        except Exception as e:
            self.logger.warning(f"Erro ao calcular importância da feature: {str(e)}")
            return 0
    
    def _calculate_redundancy(self, feature: pd.Series, X: pd.DataFrame) -> float:
        """
        Calcula o nível de redundância de uma feature em relação às features existentes.
        
        Args:
            feature: Nova feature a ser avaliada
            X: DataFrame com features existentes
            
        Returns:
            Valor entre 0 (não redundante) e 1 (totalmente redundante)
        """
        try:
            # Se a feature tem muitos valores ausentes, considerá-la redundante
            if feature.isnull().mean() > 0.5:
                return 1.0
            
            # Para features numéricas, calcular correlação máxima
            if pd.api.types.is_numeric_dtype(feature):
                max_corr = 0
                
                for col in X.columns:
                    if pd.api.types.is_numeric_dtype(X[col]):
                        # Usar correlação de Spearman que é menos sensível a outliers
                        corr = abs(stats.spearmanr(
                            feature.fillna(feature.median()), 
                            X[col].fillna(X[col].median()),
                            nan_policy='omit'
                        )[0])
                        
                        if not np.isnan(corr):
                            max_corr = max(max_corr, corr)
                
                return max_corr
            
            # Para features categóricas, calcular V de Cramer máximo
            if pd.api.types.is_categorical_dtype(feature) or feature.dtype == 'object':
                max_cramer_v = 0
                
                for col in X.columns:
                    if pd.api.types.is_categorical_dtype(X[col]) or X[col].dtype == 'object':
                        cramer_v = self._cramers_v(feature, X[col])
                        max_cramer_v = max(max_cramer_v, cramer_v)
                
                return max_cramer_v
            
            # Caso contrário, retornar redundância média
            return 0.5
            
        except Exception as e:
            self.logger.warning(f"Erro ao calcular redundância: {str(e)}")
            return 1.0
    
    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """
        Calcula o V de Cramer entre duas variáveis categóricas.
        
        Args:
            x: Primeira variável
            y: Segunda variável
            
        Returns:
            Valor do V de Cramer entre 0 e 1
        """
        confusion_matrix = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
    
    def _select_nodes_for_expansion(
        self,
        evaluated_nodes: List[TransformationNode],
        tree: TransformationTree
    ) -> List[TransformationNode]:
        """
        Seleciona os melhores nós para expansão na próxima iteração.
        
        Args:
            evaluated_nodes: Lista de nós avaliados
            tree: Árvore de transformações
            
        Returns:
            Lista com os melhores nós para expansão
        """
        # Filtrar nós com ganho positivo e baixa redundância
        valid_nodes = [
            node for node in evaluated_nodes
            if node.performance_gain > EXPLORER_CONFIG['min_performance_gain'] 
            and node.redundancy_score < EXPLORER_CONFIG['redundancy_threshold']
        ]
        
        # Ordenar por performance_gain * (1 - redundancy_score) * importance
        valid_nodes.sort(
            key=lambda n: n.performance_gain * (1 - n.redundancy_score) * n.importance,
            reverse=True
        )
        
        # Limitar número de nós expandidos
        max_nodes = EXPLORER_CONFIG['max_features_per_level']
        selected_nodes = valid_nodes[:max_nodes]
        
        self.logger.info(f"Selecionados {len(selected_nodes)} nós para expansão (de {len(evaluated_nodes)} avaliados)")
        
        # Marcar nós selecionados como tal na árvore
        for node in selected_nodes:
            tree.mark_node_as_selected(node.id, True)
        
        return selected_nodes
    
    def _expand_nodes(
        self,
        nodes: List[TransformationNode],
        X: pd.DataFrame,
        tree: TransformationTree
    ):
        """
        Expande os nós selecionados, gerando novos nós de transformação.
        
        Args:
            nodes: Lista de nós a expandir
            X: Dados de entrada
            tree: Árvore de transformações
        """
        self.logger.info(f"Expandindo {len(nodes)} nós")
        
        for node in nodes:
            # Determinar tipo de dados da feature
            feature_name = node.name
            
            # Se a feature não existe no X, criar a partir da sequência de transformações
            if feature_name not in X.columns:
                lineage = self._get_transformation_sequence(node)
                feature = self._apply_transformation_sequence(lineage, X)
                
                if feature is None:
                    continue
                
                # Adicionar temporariamente ao X
                X_temp = X.copy()
                X_temp[feature_name] = feature
            else:
                X_temp = X
            
            feature = X_temp[feature_name]
            feature_type = self._detect_feature_type(feature)
            
            # Obter transformações aplicáveis para esse tipo
            applicable_transformations = self._get_applicable_transformations(feature_type)
            
            # Para cada transformação, adicionar novo nó à árvore
            for transformation in applicable_transformations:
                new_name = f"{transformation}({feature_name})"
                
                # Verificar se já existe
                if new_name in self.explored_features:
                    continue
                
                # Adicionar à árvore
                tree.add_transformation_node(
                    parent_id=node.id,
                    name=new_name,
                    transformation_type=transformation,
                    transformation_params={'column': feature_name}
                )
    
    def _detect_feature_type(self, feature: pd.Series) -> str:
        """
        Detecta o tipo de uma feature.
        
        Args:
            feature: Feature a ser analisada
            
        Returns:
            Tipo da feature ('numeric', 'categorical', 'datetime', 'text')
        """
        if pd.api.types.is_numeric_dtype(feature):
            return 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(feature):
            return 'datetime'
        elif feature.dtype == 'object':
            if feature.str.len().mean() > 10:
                return 'text'
            else:
                return 'categorical'
        else:
            return 'categorical'
    
    def _get_applicable_transformations(self, feature_type: str) -> List[str]:
        """
        Retorna as transformações aplicáveis para um tipo de feature.
        
        Args:
            feature_type: Tipo da feature
            
        Returns:
            Lista de transformações aplicáveis
        """
        from config import TRANSFORMATIONS
        return TRANSFORMATIONS.get(feature_type, [])
    
    def _update_best_features(self, evaluated_nodes: List[TransformationNode]):
        """
        Atualiza a lista das melhores features encontradas.
        
        Args:
            evaluated_nodes: Lista de nós avaliados
        """
        for node in evaluated_nodes:
            if node.performance_gain > EXPLORER_CONFIG['min_performance_gain']:
                # Verificar se já temos esta feature
                existing = [f for f in self.best_features if f['name'] == node.name]
                
                if not existing:
                    # Adicionar nova feature
                    self.best_features.append({
                        'name': node.name,
                        'node_id': node.id,
                        'importance': node.importance,
                        'performance_gain': node.performance_gain,
                        'redundancy_score': node.redundancy_score,
                        'transformation_type': node.transformation_type,
                        'transformation_params': node.transformation_params
                    })
                else:
                    # Atualizar feature existente
                    existing[0].update({
                        'importance': node.importance,
                        'performance_gain': node.performance_gain,
                        'redundancy_score': node.redundancy_score
                    })
        
        # Ordenar por performance_gain * importance
        self.best_features.sort(
            key=lambda f: f['performance_gain'] * f['importance'] * (1 - f['redundancy_score']),
            reverse=True
        )
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Retorna um DataFrame com a importância das features.
        
        Returns:
            DataFrame com nome das features e importância
        """
        if not self.best_features:
            return pd.DataFrame(columns=['feature', 'importance', 'performance_gain', 'redundancy'])
        
        return pd.DataFrame([
            {
                'feature': f['name'],
                'importance': f['importance'],
                'performance_gain': f['performance_gain'],
                'redundancy': f['redundancy_score']
            }
            for f in self.best_features
        ])
