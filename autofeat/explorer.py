import pandas as pd
import numpy as np
import networkx as nx
import logging
import itertools
import time
from typing import List, Dict, Callable, Optional, Tuple, Any, Set
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from collections import defaultdict
import concurrent.futures
import joblib
import os
from config import IMBALANCED_CONFIGS
from robust_cross_validation import RobustCrossValidator, create_validator

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutoFE.Explorer")

# Função global para processar transformações
# IMPORTANTE: Esta função precisa estar no escopo global para funcionar com paralelização
def process_transformation(config, df, target_col, evaluator):
    """
    Função global para processar uma transformação (definida no escopo global para permitir paralelização).
    
    Args:
        config: Configuração da transformação
        df: DataFrame a transformar
        target_col: Nome da coluna alvo
        evaluator: Avaliador de transformações
        
    Returns:
        Dicionário com os resultados da transformação
    """
    # Importa o PreProcessor de forma segura
    try:
        # Tenta importar o módulo local primeiro
        from preprocessor import PreProcessor
    except ImportError:
        # Se falhar, busca no caminho do sistema
        import sys
        import os
        # Adiciona o diretório atual ao caminho
        sys.path.append(os.path.abspath('.'))
        try:
            from preprocessor import PreProcessor
        except ImportError:
            raise ImportError("Não foi possível importar o módulo PreProcessor")
    
    import time
    start_time = time.time()
    
    # Cria um nome legível para a transformação
    # Remove valores booleanos do nome para maior clareza
    name = "_".join([f"{key}-{value}" for key, value in config.items() 
                     if str(value) != "True" and str(value) != "False"])
    if not name:
        name = "refinement"
    
    try:
        # Aplica a transformação
        preprocessor = PreProcessor(config)
        preprocessor.fit(df, target_col=target_col)
        transformed_df = preprocessor.transform(df, target_col=target_col)
        
        execution_time = time.time() - start_time
        
        # Avalia a transformação
        metrics = evaluator.evaluate_transformation(transformed_df)
        
        # Adiciona tempo de execução às métricas
        metrics['execution_time'] = execution_time
        
        # Calcula score geral
        score = evaluator.compute_overall_score(metrics)
        
        # Calcula importância das features
        feature_importance = {}
        if transformed_df is not None and target_col in transformed_df.columns:
            feature_importance = evaluator.calculate_feature_importance(transformed_df)
        
        return {
            'name': name,
            'config': config,
            'data': transformed_df,
            'metrics': metrics,
            'score': score,
            'execution_time': metrics.get('execution_time', 0),
            'feature_importance': feature_importance
        }
    except Exception as e:
        # Retorna dados estruturados mesmo em caso de erro
        return {
            'name': name,
            'config': config,
            'data': None,
            'metrics': {'error': str(e)},
            'score': float('-inf'),
            'execution_time': time.time() - start_time,
            'feature_importance': {}
        }

class TransformationNode:
    """
    Representa um nó na árvore de transformações com metadados adicionais.
    """
    def __init__(self, name: str, config: Dict[str, Any], data=None, score: float = 0.0,
                 metrics: Dict[str, float] = None, parent: str = None):
        self.name = name
        self.config = config
        self.data = data
        self.score = score
        self.metrics = metrics or {}
        self.parent = parent
        self.children = []
        self.execution_time = 0
        self.feature_importance = {}
        self.parameters = {}
        self._setup_logging()
        self.logger.info("TransformationNode inicializado com sucesso.")

    def _setup_logging(self):
        self.logger = logging.getLogger("AutoFE.TransformationNode")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def add_child(self, child: 'TransformationNode'):
        """Adiciona um nó filho"""
        self.children.append(child)
    
    def to_dict(self) -> Dict:
        """Converte o nó para um dicionário (sem os dados)"""
        return {
            'name': self.name,
            'config': self.config,
            'score': self.score,
            'metrics': self.metrics,
            'parent': self.parent,
            'execution_time': self.execution_time,
            'feature_importance': self.feature_importance,
            'parameters': self.parameters,
            'data_shape': None if self.data is None else self.data.shape
        }


class TransformationTree:
    """
    Representa uma árvore de transformações usando um grafo direcionado.
    Cada nó contém uma transformação e seus resultados.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        # Nó raiz com configuração vazia
        self.graph.add_node("root", node=TransformationNode("root", {}))
        self.nodes = {"root": TransformationNode("root", {})}
        self._setup_logging()
        self.logger.info("TransformationTree inicializado com sucesso.")

    def _setup_logging(self):
        self.logger = logging.getLogger("AutoFE.TransformationTree")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def add_transformation(self, parent: str, name: str, config: Dict[str, Any], 
                          data=None, score: float = 0.0, metrics: Dict[str, float] = None,
                          execution_time: float = 0, feature_importance: Dict[str, float] = None) -> str:
        """
        Adiciona uma transformação à árvore.
        
        Args:
            parent: Nome do nó pai
            name: Nome da transformação
            config: Configuração da transformação
            data: Dados transformados
            score: Pontuação da transformação
            metrics: Métricas de avaliação adicionais
            execution_time: Tempo de execução
            feature_importance: Importância das features
            
        Returns:
            Nome do nó criado
        """
        # Cria um identificador único para o nó
        node_id = f"{name}_{len(self.nodes)}"
        
        # Cria o objeto TransformationNode
        node = TransformationNode(
            name=name,
            config=config,
            data=data,
            score=score,
            metrics=metrics or {},
            parent=parent
        )
        node.execution_time = execution_time
        node.feature_importance = feature_importance or {}
        
        # Adiciona o nó ao grafo
        self.graph.add_node(node_id, node=node)
        self.graph.add_edge(parent, node_id)
        
        # Adiciona ao dicionário para acesso rápido
        self.nodes[node_id] = node
        
        # Adiciona como filho do nó pai
        if parent in self.nodes:
            self.nodes[parent].add_child(node)
        
        # Log dimensões e métricas
        feature_diff = 0
        if data is not None and parent in self.nodes and self.nodes[parent].data is not None:
            feature_diff = data.shape[1] - self.nodes[parent].data.shape[1]
        
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in (metrics or {}).items()])
        self.logger.info(f"Transformação '{name}' adicionada como '{node_id}' com score {score:.4f}. " +
                   f"Dimensão: {data.shape if data is not None else 'N/A'}. " +
                   f"Alteração nas features: {feature_diff}. " +
                   f"Métricas: {metrics_str}. " +
                   f"Tempo: {execution_time:.2f}s")
        
        return node_id
    
    def get_best_nodes(self, limit: int = 5, metric: str = 'score', min_score: float = None) -> List[str]:
        """
        Retorna os melhores nós com base em uma métrica específica.
        
        Args:
            limit: Número máximo de nós a retornar
            metric: Métrica para ordenação ('score' ou nome de uma métrica específica)
            min_score: Score mínimo para incluir o nó
            
        Returns:
            Lista de IDs dos melhores nós
        """
        scored_nodes = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            node = node_data.get('node')
            if node is None or node_id == "root":
                continue
            
            # Determina o valor para ordenação
            if metric == 'score':
                value = node.score
            elif metric in node.metrics:
                value = node.metrics[metric]
            else:
                value = float('-inf')
            
            # Aplica filtro de score mínimo
            if min_score is not None and value < min_score:
                continue
                
            scored_nodes.append((node_id, value))
        
        # Ordena e retorna os melhores
        best_nodes = [node_id for node_id, _ in sorted(scored_nodes, key=lambda x: x[1], reverse=True)[:limit]]
        
        if best_nodes:
            self.logger.info(f"Melhores {len(best_nodes)} nós por {metric}: {best_nodes}")
        else:
            self.logger.warning(f"Nenhum nó encontrado com métrica {metric} acima de {min_score}")
            
        return best_nodes
    
    def get_transformation_path(self, node_id: str) -> List[Dict]:
        """
        Retorna o caminho de transformações da raiz até o nó especificado.
        
        Args:
            node_id: ID do nó final
            
        Returns:
            Lista de transformações no caminho
        """
        if node_id not in self.graph:
            return []
        
        path = []
        current = node_id
        
        while current != "root":
            # Obter nó atual
            node_data = self.graph.nodes[current].get('node')
            if node_data:
                path.append(node_data.to_dict())
            
            # Mover para o pai
            predecessors = list(self.graph.predecessors(current))
            if not predecessors:
                break
            current = predecessors[0]
        
        # Adiciona o nó raiz
        root_data = self.graph.nodes["root"].get('node')
        if root_data:
            path.append(root_data.to_dict())
        
        # Inverte para ter ordem da raiz até o nó
        return list(reversed(path))
    
    def get_level_nodes(self, level: int) -> List[str]:
        """
        Retorna todos os nós em um determinado nível da árvore.
        
        Args:
            level: Nível da árvore (0 = raiz)
            
        Returns:
            Lista de IDs dos nós no nível
        """
        if level == 0:
            return ["root"]
        
        # BFS para encontrar nós no nível especificado
        nodes = []
        for node_id in self.graph.nodes():
            if node_id == "root":
                continue
            
            # Calcula o comprimento do caminho da raiz até este nó
            try:
                path_length = len(nx.shortest_path(self.graph, "root", node_id)) - 1
                if path_length == level:
                    nodes.append(node_id)
            except nx.NetworkXNoPath:
                continue
        
        return nodes
    
    def prune(self, keep_best: int = 5, metric: str = 'score') -> None:
        """
        Remove nós de baixa qualidade para reduzir o tamanho da árvore.
        
        Args:
            keep_best: Número de melhores nós a manter em cada nível
            metric: Métrica para avaliar a qualidade
        """
        # Determina a profundidade máxima da árvore
        max_depth = 0
        for node_id in self.graph.nodes():
            if node_id == "root":
                continue
            try:
                depth = len(nx.shortest_path(self.graph, "root", node_id)) - 1
                max_depth = max(max_depth, depth)
            except nx.NetworkXNoPath:
                continue
        
        # Para cada nível, manter apenas os melhores nós
        for level in range(1, max_depth + 1):
            level_nodes = self.get_level_nodes(level)
            
            # Avalia cada nó do nível
            scored_nodes = []
            for node_id in level_nodes:
                node = self.graph.nodes[node_id].get('node')
                if node is None:
                    continue
                
                # Determina o valor para ordenação
                if metric == 'score':
                    value = node.score
                elif metric in node.metrics:
                    value = node.metrics[metric]
                else:
                    value = float('-inf')
                    
                scored_nodes.append((node_id, value))
            
            # Ordena e identifica nós a manter
            if len(scored_nodes) > keep_best:
                sorted_nodes = sorted(scored_nodes, key=lambda x: x[1], reverse=True)
                keep_nodes = set(node_id for node_id, _ in sorted_nodes[:keep_best])
                remove_nodes = [node_id for node_id, _ in sorted_nodes[keep_best:]]
                
                # Remove nós de baixa qualidade
                for node_id in remove_nodes:
                    # Verifica se tem filhos
                    has_children = len(list(self.graph.successors(node_id))) > 0
                    
                    if not has_children:  # Remove apenas nós sem filhos
                        self.logger.info(f"Removendo nó de baixo score: {node_id}")
                        self.graph.remove_node(node_id)
                        if node_id in self.nodes:
                            del self.nodes[node_id]
    
    def save(self, filepath: str) -> None:
        """
        Salva a árvore de transformações em um arquivo.
        
        Args:
            filepath: Caminho do arquivo para salvar
        """
        # Remove os dados para economizar espaço
        serializable_nodes = {}
        for node_id, node in self.nodes.items():
            node_copy = TransformationNode(
                name=node.name,
                config=node.config,
                data=None,  # Não salva os dados
                score=node.score,
                metrics=node.metrics,
                parent=node.parent
            )
            node_copy.execution_time = node.execution_time
            node_copy.feature_importance = node.feature_importance
            node_copy.parameters = node.parameters
            serializable_nodes[node_id] = node_copy
        
        # Cria uma cópia do grafo sem dados
        graph_copy = nx.DiGraph()
        for node_id in self.graph.nodes():
            node = self.graph.nodes[node_id].get('node')
            if node:
                node_copy = serializable_nodes[node_id] if node_id in serializable_nodes else TransformationNode(
                    name="unknown",
                    config={},
                    data=None
                )
                graph_copy.add_node(node_id, node=node_copy)
        
        # Copia as arestas
        for edge in self.graph.edges():
            graph_copy.add_edge(*edge)
        
        # Salva a estrutura serializada
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'wb') as f:
            joblib.dump({
                'graph': graph_copy,
                'nodes': serializable_nodes
            }, f)
        
        self.logger.info(f"Árvore de transformações salva em {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TransformationTree':
        """
        Carrega uma árvore de transformações de um arquivo.
        
        Args:
            filepath: Caminho do arquivo para carregar
            
        Returns:
            Árvore de transformações carregada
        """
        with open(filepath, 'rb') as f:
            data = joblib.load(f)
        
        tree = TransformationTree()
        tree.graph = data['graph']
        tree.nodes = data['nodes']
        return tree


class TransformationEvaluator:
    """
    Avalia transformações aplicadas a um conjunto de dados.
    """
    def __init__(self, target_col: Optional[str] = None, problem_type: str = 'auto',
                 cv_folds: int = 5, random_state: int = 42, use_robust_cv: bool = True):
        """
        Inicializa o avaliador de transformações.
        
        Args:
            target_col: Nome da coluna alvo (opcional)
            problem_type: Tipo de problema ('classification', 'regression', ou 'auto')
            cv_folds: Número de folds para validação cruzada
            random_state: Semente aleatória para reprodutibilidade
            use_robust_cv: Se deve usar validação cruzada robusta
        """
        self.target_col = target_col
        self.problem_type = problem_type
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.metrics = {}
        
        # Inicializa o validador robusto se necessário
        if use_robust_cv:
            self.validator = create_validator(
                problem_type=problem_type,
                n_splits=cv_folds,
                random_state=random_state,
                verbosity=1
            )
            
        self._setup_logging()
        self.logger.info("TransformationEvaluator inicializado com sucesso.")

    def _setup_logging(self):
        """Configura o logger para a classe."""
        self.logger = logging.getLogger("AutoFE.TransformationEvaluator")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def detect_problem_type(self, y: pd.Series) -> str:
        """
        Detecta automaticamente o tipo de problema com base na variável alvo.
        
        Args:
            y: Série com a variável alvo
            
        Returns:
            'classification' ou 'regression'
        """
        # Verifica se é categórico/objeto
        if pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y):
            return 'classification'
        
        # Verifica número de valores únicos
        if y.nunique() <= 10:
            return 'classification'
        
        # Caso contrário, assume regressão
        return 'regression'
    
    def evaluate_transformation(self, df: pd.DataFrame, prefix: str = "") -> Dict[str, float]:
        """
        Avalia uma transformação calculando métricas relevantes.
        
        Args:
            df: DataFrame transformado
            prefix: Prefixo para nomes das métricas
            
        Returns:
            Dicionário com métricas calculadas
        """
        metrics = {}
        
        # Se não houver target_col ou não estiver presente no df, apenas métricas básicas
        if not self.target_col or self.target_col not in df.columns:
            # Métricas básicas
            metrics[f'{prefix}dimensionality'] = df.shape[1]
            metrics[f'{prefix}missing_ratio'] = df.isna().mean().mean()
            metrics[f'{prefix}high_cardinality_ratio'] = sum(df.nunique() > 100) / max(1, df.shape[1])
            
            # Verifica correlação entre features
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.shape[1] > 1:
                try:
                    corr_matrix = numeric_df.fillna(numeric_df.median()).corr().abs()
                    np.fill_diagonal(corr_matrix.values, 0)
                    metrics[f'{prefix}avg_correlation'] = corr_matrix.values.mean()
                    metrics[f'{prefix}high_correlation_ratio'] = (corr_matrix > 0.7).sum().sum() / (corr_matrix.shape[0] * corr_matrix.shape[1])
                except Exception as e:
                    self.logger.warning(f"Erro ao calcular correlação: {e}")
            
            return metrics
        
        # Separa features e target
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        # Detecta o tipo de problema se for 'auto'
        problem_type = self.problem_type
        if problem_type == 'auto':
            problem_type = self.detect_problem_type(y)
        
        # Métricas básicas (mesmas de acima)
        metrics[f'{prefix}dimensionality'] = X.shape[1]
        metrics[f'{prefix}missing_ratio'] = X.isna().mean().mean()
        metrics[f'{prefix}high_cardinality_ratio'] = sum(X.nunique() > 100) / max(1, X.shape[1])
        
        # Métricas específicas para o tipo de problema
        try:
            if problem_type == 'classification':
                # Métricas de balanceamento de classes
                class_counts = y.value_counts(normalize=True)
                metrics[f'{prefix}class_imbalance'] = class_counts.max() - class_counts.min()
                
                try:
                    # Entropia da distribuição (maior = mais balanceado)
                    from scipy.stats import entropy
                    metrics[f'{prefix}class_entropy'] = entropy(class_counts) / np.log(len(class_counts))
                except Exception:
                    pass
                
                # Validação cruzada com modelo base
                if X.shape[0] >= 50 and X.shape[1] >= 1:  # Precisa ter dados suficientes
                    # Preenche valores ausentes para poder treinar o modelo
                    X_filled = X.copy()
                    for col in X_filled.select_dtypes(include=['number']).columns:
                        X_filled[col] = X_filled[col].fillna(X_filled[col].median())
                    
                    for col in X_filled.select_dtypes(include=['object', 'category']).columns:
                        X_filled[col] = X_filled[col].fillna(X_filled[col].mode()[0] if len(X_filled[col].mode()) > 0 else 'missing')
                    
                    # Codifica variáveis categóricas
                    X_encoded = pd.get_dummies(X_filled, drop_first=True)
                    
                    # Treina um modelo simples
                    try:
                        model = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
                        
                        # Calcula accuracy, f1 e ROC-AUC quando possível
                        metrics[f'{prefix}cv_accuracy'] = np.mean(cross_val_score(
                            model, X_encoded, y, cv=min(self.cv_folds, 3), scoring='accuracy'
                        ))
                        
                        # F1 multiclasse
                        metrics[f'{prefix}cv_f1'] = np.mean(cross_val_score(
                            model, X_encoded, y, cv=min(self.cv_folds, 3), 
                            scoring='f1_weighted'
                        ))
                        
                        # ROC-AUC (apenas para binário)
                        if y.nunique() == 2:
                            metrics[f'{prefix}cv_roc_auc'] = np.mean(cross_val_score(
                                model, X_encoded, y, cv=min(self.cv_folds, 3), 
                                scoring='roc_auc'
                            ))
                    except Exception as e:
                        self.logger.warning(f"Erro ao calcular métricas de CV para classificação: {e}")
                
            elif problem_type == 'regression':
                # Estatísticas da variável alvo
                metrics[f'{prefix}target_skew'] = abs(stats.skew(y.dropna()))
                metrics[f'{prefix}target_kurtosis'] = stats.kurtosis(y.dropna())
                
                # Validação cruzada com modelo base
                if X.shape[0] >= 50 and X.shape[1] >= 1:
                    # Preenche valores ausentes
                    X_filled = X.copy()
                    for col in X_filled.select_dtypes(include=['number']).columns:
                        X_filled[col] = X_filled[col].fillna(X_filled[col].median())
                    
                    for col in X_filled.select_dtypes(include=['object', 'category']).columns:
                        X_filled[col] = X_filled[col].fillna(X_filled[col].mode()[0] if len(X_filled[col].mode()) > 0 else 'missing')
                    
                    # Codifica variáveis categóricas
                    X_encoded = pd.get_dummies(X_filled, drop_first=True)
                    
                    # Treina um modelo simples
                    try:
                        model = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
                        
                        # RMSE negativo (maior = melhor)
                        neg_mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
                        metrics[f'{prefix}cv_neg_rmse'] = -np.sqrt(-np.mean(cross_val_score(
                            model, X_encoded, y, cv=min(self.cv_folds, 3), scoring=neg_mse_scorer
                        )))
                        
                        # R² 
                        metrics[f'{prefix}cv_r2'] = np.mean(cross_val_score(
                            model, X_encoded, y, cv=min(self.cv_folds, 3), scoring='r2'
                        ))
                    except Exception as e:
                        self.logger.warning(f"Erro ao calcular métricas de CV para regressão: {e}")
        
        except Exception as e:
            self.logger.error(f"Erro ao avaliar transformação: {e}")
        
        return metrics
    
    def calculate_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula a importância das features usando um modelo base.
        
        Args:
            df: DataFrame com as features e target
            
        Returns:
            Dicionário com importância de cada feature
        """
        if not self.target_col or self.target_col not in df.columns:
            return {}
        
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        # Detecta o tipo de problema
        problem_type = self.problem_type
        if problem_type == 'auto':
            problem_type = self.detect_problem_type(y)
        
        feature_importance = {}
        
        try:
            # Preenche valores ausentes
            X_filled = X.copy()
            for col in X_filled.select_dtypes(include=['number']).columns:
                X_filled[col] = X_filled[col].fillna(X_filled[col].median())
            
            for col in X_filled.select_dtypes(include=['object', 'category']).columns:
                X_filled[col] = X_filled[col].fillna(X_filled[col].mode()[0] if len(X_filled[col].mode()) > 0 else 'missing')
            
            # Codifica variáveis categóricas
            X_encoded = pd.get_dummies(X_filled, drop_first=True)
            
            # Seleciona modelo adequado
            if problem_type == 'classification':
                model = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
            
            # Treina o modelo
            model.fit(X_encoded, y)
            
            # Extrai importância das features
            importances = model.feature_importances_
            feature_names = X_encoded.columns
            
            # Cria dicionário de importâncias
            feature_importance = {name: float(imp) for name, imp in zip(feature_names, importances)}
            
        except Exception as e:
            self.logger.warning(f"Erro ao calcular importância das features: {e}")
        
        return feature_importance
    
    def compute_overall_score(self, metrics: Dict[str, float], problem_type: str = None) -> float:
        """
        Calcula uma pontuação geral com base nas métricas.
        
        Args:
            metrics: Dicionário com métricas
            problem_type: Tipo de problema ('classification', 'regression', ou None)
            
        Returns:
            Pontuação geral (maior = melhor)
        """
        # Se não especificado, usa o tipo de problema da classe
        if problem_type is None:
            problem_type = self.problem_type
            
            # Se ainda for 'auto', tenta inferir pelas métricas
            if problem_type == 'auto':
                if 'cv_accuracy' in metrics or 'cv_f1' in metrics:
                    problem_type = 'classification'
                elif 'cv_r2' in metrics or 'cv_neg_rmse' in metrics:
                    problem_type = 'regression'
                else:
                    # Caso não seja possível inferir, usa um cálculo genérico
                    problem_type = 'unknown'
        
        # Inicializa componentes do score
        score_components = []
        
        # Componentes comuns (penalidades)
        if 'missing_ratio' in metrics:
            # Penaliza valores ausentes (0 = melhor)
            score_components.append((1 - metrics['missing_ratio']) * 0.1)
        
        if 'high_correlation_ratio' in metrics:
            # Penaliza alta correlação entre features (0 = melhor)
            score_components.append((1 - metrics['high_correlation_ratio']) * 0.1)
        
        if 'high_cardinality_ratio' in metrics:
            # Penaliza alta cardinalidade (0 = melhor)
            score_components.append((1 - metrics['high_cardinality_ratio']) * 0.05)
        
        # Métricas específicas do problema
        if problem_type == 'classification':
            # Accuracy (maior = melhor)
            if 'cv_accuracy' in metrics:
                score_components.append(metrics['cv_accuracy'] * 0.25)
            
            # F1 (maior = melhor)
            if 'cv_f1' in metrics:
                score_components.append(metrics['cv_f1'] * 0.25)
            
            # ROC-AUC (maior = melhor)
            if 'cv_roc_auc' in metrics:
                score_components.append(metrics['cv_roc_auc'] * 0.2)
            
            # Entropia de classes (maior = melhor)
            if 'class_entropy' in metrics:
                score_components.append(metrics['class_entropy'] * 0.1)
            
        elif problem_type == 'regression':
            # R² (maior = melhor)
            if 'cv_r2' in metrics:
                score_components.append(max(0, metrics['cv_r2']) * 0.3)  # Limita a valores não negativos
            
            # RMSE normalizado (converter para 0-1, maior = melhor)
            if 'cv_neg_rmse' in metrics:
                # O RMSE negativo já está na direção "maior = melhor", mas precisa normalizar
                rmse = metrics['cv_neg_rmse']
                # Converte para uma pontuação entre 0 e 1
                # Limita em +/-10 para evitar valores extremos
                norm_rmse = 1 / (1 + min(10, abs(rmse)))
                score_components.append(norm_rmse * 0.3)
            
            # Penaliza skewness extrema (0 = melhor)
            if 'target_skew' in metrics:
                skew_penalty = 1 - min(1, abs(metrics['target_skew']) / 10)
                score_components.append(skew_penalty * 0.05)
        
        # Para datasets sem target ou tipo desconhecido
        elif problem_type == 'unknown':
            # Usa métricas básicas de estrutura dos dados
            if 'dimensionality' in metrics:
                # Penaliza dimensionalidade muito alta
                dim_score = 1 - min(1, metrics['dimensionality'] / 100)
                score_components.append(dim_score * 0.1)
        
        # Calcula média ponderada dos componentes
        if score_components:
            return sum(score_components)
        else:
            return 0.0  # Nenhuma métrica disponível
            
    def evaluate_with_robust_cv(self, df: pd.DataFrame, model, prefix: str = "") -> Dict[str, float]:
        """
        Avalia usando validação cruzada robusta.
        
        Args:
            df: DataFrame com dados
            model: Modelo a ser avaliado
            prefix: Prefixo para nomes das métricas
            
        Returns:
            Dicionário com métricas calculadas
        """
        if not hasattr(self, 'validator') or not self.target_col or self.target_col not in df.columns:
            # Fallback para avaliação padrão
            return self.evaluate_transformation(df, prefix)
        
        metrics = {}
        
        # Métricas básicas
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        metrics[f'{prefix}dimensionality'] = X.shape[1]
        metrics[f'{prefix}missing_ratio'] = X.isna().mean().mean()
        
        try:
            # Usa o RobustCrossValidator para uma avaliação mais confiável
            cv_results = self.validator.cross_validate(model, X, y, return_train_score=True)
            
            # Adiciona métricas da validação robusta
            for key, value in cv_results.items():
                if key.endswith('_mean'):
                    metrics[f'{prefix}robust_{key}'] = value
            
            # Adiciona score de confiabilidade
            if 'reliability_score' in cv_results:
                metrics[f'{prefix}reliability'] = cv_results['reliability_score']
            
            # Adiciona grau de overfitting se disponível
            if 'overfitting_grade' in cv_results:
                metrics[f'{prefix}overfitting_grade'] = cv_results['overfitting_grade']
                
        except Exception as e:
            self.logger.warning(f"Erro ao usar validação cruzada robusta: {e}")
            # Fallback para avaliação padrão
            return self.evaluate_transformation(df, prefix)
            
        return metrics


class MetaLearner:
    """
    Usa meta-aprendizado para recomendar transformações eficazes com base
    nos dados e em experiências anteriores.
    """
    def __init__(self, experience_db: Optional[str] = None):
        """
        Inicializa o meta-aprendiz.
        
        Args:
            experience_db: Caminho para o banco de dados de experiências anteriores
        """
        self.experience_db = experience_db
        self.dataset_profiles = []
        self.transformation_results = []
        self._setup_logging()
        # Carrega experiências anteriores, se disponíveis
        if experience_db and os.path.exists(experience_db):
            try:
                data = joblib.load(experience_db)
                self.dataset_profiles = data.get('profiles', [])
                self.transformation_results = data.get('results', [])
                self.logger.info(f"Carregadas {len(self.dataset_profiles)} experiências anteriores")
            except Exception as e:
                self.logger.warning(f"Erro ao carregar banco de experiências: {e}")
        
        self.logger.info("MetaLearner inicializado")
    

    def _setup_logging(self):
        self.logger = logging.getLogger("AutoFE.MetaLearner")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def profile_dataset(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict:
        """
        Cria um perfil do dataset para comparação com experiências anteriores.
        
        Args:
            df: DataFrame a ser analisado
            target_col: Nome da coluna alvo
            
        Returns:
            Dicionário com características do dataset
        """
        profile = {
            'n_samples': df.shape[0],
            'n_features': df.shape[1],
            'n_categorical': len(df.select_dtypes(include=['object', 'category']).columns),
            'n_numeric': len(df.select_dtypes(include=['number']).columns),
            'missing_ratio': df.isna().mean().mean(),
            'has_target': target_col is not None and target_col in df.columns
        }
        
        # Características adicionais se tiver target
        if profile['has_target'] and target_col in df.columns:
            y = df[target_col]
            profile['target_type'] = 'categorical' if pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y) or y.nunique() <= 10 else 'numeric'
            profile['n_classes'] = y.nunique() if profile['target_type'] == 'categorical' else 0
            profile['class_imbalance'] = y.value_counts(normalize=True).max() if profile['target_type'] == 'categorical' else 0
            
            if profile['target_type'] == 'numeric':
                profile['target_skew'] = float(stats.skew(y.dropna())) if len(y.dropna()) > 2 else 0
                profile['target_range'] = float(y.max() - y.min()) if len(y) > 0 else 0
        
        # Características das colunas
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            # Correlação média entre features numéricas
            try:
                corr_matrix = df[numeric_cols].fillna(df[numeric_cols].median()).corr().abs()
                np.fill_diagonal(corr_matrix.values, 0)
                profile['avg_correlation'] = float(corr_matrix.values.mean())
            except Exception:
                profile['avg_correlation'] = 0
        
        self.logger.info(f"Perfil do dataset criado: {profile}")
        return profile
    
    def find_similar_datasets(self, profile: Dict, top_k: int = 3) -> List[int]:
        """
        Encontra datasets similares na base de experiências.
        
        Args:
            profile: Perfil do dataset atual
            top_k: Número de datasets similares a retornar
            
        Returns:
            Índices dos datasets mais similares
        """
        if not self.dataset_profiles:
            return []
        
        # Calcula similaridade entre o perfil atual e anteriores
        similarities = []
        for i, prev_profile in enumerate(self.dataset_profiles):
            similarity = self._calculate_profile_similarity(profile, prev_profile)
            similarities.append((i, similarity))
        
        # Ordena por similaridade (maior primeiro)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Retorna os índices dos top_k mais similares
        top_indices = [idx for idx, _ in similarities[:top_k]]
        
        self.logger.info(f"Datasets similares encontrados: {top_indices}")
        return top_indices
    
    def _calculate_profile_similarity(self, profile1: Dict, profile2: Dict) -> float:
        """
        Calcula a similaridade entre dois perfis de dataset.
        
        Args:
            profile1: Primeiro perfil
            profile2: Segundo perfil
            
        Returns:
            Valor de similaridade entre 0 e 1
        """
        # Pesos para diferentes características
        weights = {
            'n_samples': 0.1,
            'n_features': 0.15,
            'n_categorical': 0.1,
            'n_numeric': 0.1,
            'missing_ratio': 0.1,
            'target_type': 0.2,
            'n_classes': 0.1,
            'class_imbalance': 0.05,
            'target_skew': 0.05,
            'avg_correlation': 0.05
        }
        
        # Inicializa similaridade
        similarity = 0.0
        total_weight = 0.0
        
        # Compara características numéricas
        for key in ['n_samples', 'n_features', 'n_categorical', 'n_numeric', 'missing_ratio',
                   'n_classes', 'class_imbalance', 'target_skew', 'avg_correlation']:
            if key in profile1 and key in profile2:
                weight = weights.get(key, 0.05)
                total_weight += weight
                
                # Normaliza valores para comparação
                if key in ['n_samples', 'n_features']:
                    # Escala logarítmica para números grandes
                    val1 = max(1, profile1[key])
                    val2 = max(1, profile2[key])
                    ratio = min(val1, val2) / max(val1, val2)
                    similarity += weight * ratio
                else:
                    # Diferença absoluta normalizada
                    diff = abs(profile1.get(key, 0) - profile2.get(key, 0))
                    max_val = max(1, abs(profile1.get(key, 0)), abs(profile2.get(key, 0)))
                    similarity += weight * (1 - min(1, diff / max_val))
        
        # Compara características categóricas
        if 'target_type' in profile1 and 'target_type' in profile2:
            weight = weights.get('target_type', 0.2)
            total_weight += weight
            similarity += weight * (1 if profile1['target_type'] == profile2['target_type'] else 0)
        
        if 'has_target' in profile1 and 'has_target' in profile2:
            weight = 0.05
            total_weight += weight
            similarity += weight * (1 if profile1['has_target'] == profile2['has_target'] else 0)
        
        # Normaliza similaridade pelo peso total
        return similarity / total_weight if total_weight > 0 else 0
    
    def recommend_transformations(self, df: pd.DataFrame, target_col: Optional[str] = None,
                                 n_recommendations: int = 3) -> List[Dict]:
        """
        Recomenda transformações com base em experiências anteriores.
        
        Args:
            df: DataFrame para analisar
            target_col: Nome da coluna alvo
            n_recommendations: Número de recomendações a retornar
            
        Returns:
            Lista de configurações de transformação recomendadas
        """
        # Cria perfil do dataset atual
        profile = self.profile_dataset(df, target_col)
        
        # Encontra datasets similares
        similar_indices = self.find_similar_datasets(profile, top_k=5)
        
        if not similar_indices:
            # Se não encontrou similares, retorna recomendações padrão
            self.logger.info("Nenhum dataset similar encontrado. Usando recomendações padrão.")
            return self._get_default_recommendations(df, target_col)
        
        # Coleta transformações que funcionaram bem em datasets similares
        effective_transformations = []
        for idx in similar_indices:
            # Recupera as transformações associadas a este dataset
            for result in self.transformation_results:
                if result.get('dataset_idx') == idx:
                    # Adiciona com um peso baseado no score e na similaridade
                    similarity = self._calculate_profile_similarity(profile, self.dataset_profiles[idx])
                    weighted_score = result.get('score', 0) * similarity
                    
                    effective_transformations.append({
                        'config': result.get('config', {}),
                        'score': weighted_score,
                        'original_score': result.get('score', 0),
                        'similarity': similarity
                    })
        
        # Ordena por pontuação ponderada
        effective_transformations.sort(key=lambda x: x['score'], reverse=True)
        
        # Remove duplicatas (configurações idênticas)
        unique_configs = []
        seen_configs = set()
        
        for trans in effective_transformations:
            config_str = str(sorted(trans['config'].items()))
            if config_str not in seen_configs:
                seen_configs.add(config_str)
                unique_configs.append(trans)
        
        # Limita ao número desejado de recomendações
        recommendations = unique_configs[:n_recommendations]
        
        # Se não encontrou recomendações suficientes, complementa com padrões
        if len(recommendations) < n_recommendations:
            default_recs = self._get_default_recommendations(df, target_col, 
                                                          n_recommendations - len(recommendations))
            recommendations.extend(default_recs)
        
        self.logger.info(f"Geradas {len(recommendations)} recomendações de transformações")
        return [r['config'] for r in recommendations]
    
    def _get_default_recommendations(self, df: pd.DataFrame, target_col: Optional[str] = None,
                                   n_recommendations: int = 3) -> List[Dict]:
        """
        Gera recomendações padrão quando não há experiências anteriores.
        
        Args:
            df: DataFrame para analisar
            target_col: Nome da coluna alvo
            n_recommendations: Número de recomendações
            
        Returns:
            Lista de configurações de transformação recomendadas
        """
        # Detecta se temos alvo e seu tipo
        has_target = target_col is not None and target_col in df.columns
        target_type = None
        
        if has_target:
            y = df[target_col]
            target_type = 'categorical' if pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y) or y.nunique() <= 10 else 'numeric'
        
        # Lista de recomendações padrão
        recommendations = []
        
        # Recomendação básica para limpeza de dados
        recommendations.append({
            'missing_values_strategy': 'median',
            'outlier_method': 'iqr',
            'categorical_strategy': 'onehot',
            'scaling': 'standard',
            'generate_features': True
        })
        
        # Para problemas de classificação
        if has_target and target_type == 'categorical':
            # Verifica balanceamento de classes
            class_counts = df[target_col].value_counts(normalize=True)
            is_imbalanced = class_counts.max() / class_counts.min() > 3
            
            if is_imbalanced:
                # Recomendação para dados desbalanceados
                recommendations.append({
                    'missing_values_strategy': 'knn',
                    'outlier_method': 'isolation_forest',
                    'categorical_strategy': 'onehot',
                    'scaling': 'robust',
                    'generate_features': True,
                    'balance_classes': True,
                    'balance_method': 'smote'
                })
            else:
                # Recomendação para classificação balanceada
                recommendations.append({
                    'missing_values_strategy': 'knn',
                    'outlier_method': 'zscore',
                    'categorical_strategy': 'onehot',
                    'scaling': 'standard',
                    'generate_features': True,
                    'feature_selection': 'model_based'
                })
        
        # Para problemas de regressão
        elif has_target and target_type == 'numeric':
            # Verifica skewness
            try:
                skew = stats.skew(df[target_col].dropna())
                high_skew = abs(skew) > 1
            except:
                high_skew = False
            
            if high_skew:
                # Recomendação para regressão com target assimétrico
                recommendations.append({
                    'missing_values_strategy': 'iterative',
                    'outlier_method': 'isolation_forest',
                    'categorical_strategy': 'onehot',
                    'scaling': 'robust',
                    'generate_features': True,
                    'feature_selection': 'mutual_info'
                })
            else:
                # Recomendação para regressão padrão
                recommendations.append({
                    'missing_values_strategy': 'mean',
                    'outlier_method': 'zscore',
                    'categorical_strategy': 'onehot',
                    'scaling': 'standard',
                    'generate_features': True,
                    'dimensionality_reduction': 'pca'
                })
        
        # Sem alvo (clustering ou redução de dimensionalidade)
        else:
            # Verifica se tem muitas features
            has_many_features = df.shape[1] > 20
            
            if has_many_features:
                # Recomendação para redução de dimensionalidade
                recommendations.append({
                    'missing_values_strategy': 'median',
                    'outlier_method': 'isolation_forest',
                    'categorical_strategy': 'onehot',
                    'scaling': 'standard',
                    'dimensionality_reduction': 'pca',
                    'generate_features': False
                })
            else:
                # Recomendação para limpeza mais agressiva
                recommendations.append({
                    'missing_values_strategy': 'knn',
                    'outlier_method': 'isolation_forest',
                    'categorical_strategy': 'onehot',
                    'scaling': 'robust',
                    'remove_high_correlation': True,
                    'correlation_threshold': 0.85,
                    'generate_features': True
                })
        
        # Sempre incluir uma recomendação mínima
        recommendations.append({
            'missing_values_strategy': 'median',
            'categorical_strategy': 'onehot',
            'scaling': 'standard',
            'generate_features': False
        })
        
        # Limita ao número solicitado
        return recommendations[:n_recommendations]
    
    def record_result(self, dataset_profile: Dict, config: Dict, score: float,
                     metrics: Dict, dataset_idx: Optional[int] = None) -> None:
        """
        Registra o resultado de uma transformação para uso futuro.
        
        Args:
            dataset_profile: Perfil do dataset
            config: Configuração da transformação
            score: Pontuação obtida
            metrics: Métricas detalhadas
            dataset_idx: Índice do dataset (se já existir)
        """
        # Se o dataset ainda não está na base, adiciona
        if dataset_idx is None:
            self.dataset_profiles.append(dataset_profile)
            dataset_idx = len(self.dataset_profiles) - 1
        
        # Registra o resultado
        self.transformation_results.append({
            'dataset_idx': dataset_idx,
            'config': config,
            'score': score,
            'metrics': metrics,
            'timestamp': time.time()
        })
        
        self.logger.info(f"Registrado resultado para dataset {dataset_idx}: score={score:.4f}")
    
    def save(self, filepath: Optional[str] = None) -> None:
        """
        Salva a base de experiências em um arquivo.
        
        Args:
            filepath: Caminho do arquivo (usa self.experience_db se None)
        """
        filepath = filepath or self.experience_db
        if not filepath:
            self.logger.warning("Caminho de arquivo não especificado. Base de experiências não salva.")
            return
        
        data = {
            'profiles': self.dataset_profiles,
            'results': self.transformation_results
        }
        
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'wb') as f:
            joblib.dump(data, f)
        
        self.logger.info(f"Base de experiências salva em {filepath}")


class TransformationCombiner:
    """
    Gera e avalia combinações de transformações para encontrar o melhor pipeline.
    """
    def __init__(self, base_transformations: List[Dict] = None, max_depth: int = 3,
                beam_width: int = 5, parallel: bool = True, n_jobs: int = -1):
        """
        Inicializa o combinador de transformações.
        
        Args:
            base_transformations: Lista de transformações base a combinar
            max_depth: Profundidade máxima do pipeline
            beam_width: Largura do feixe na busca (número de melhores a manter)
            parallel: Se deve usar processamento paralelo
            n_jobs: Número de jobs para paralelização (-1 = todos os cores)
        """
        self.base_transformations = base_transformations or []
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.parallel = parallel
        self.n_jobs = n_jobs
        self._setup_logging()
        self.logger.info(f"TransformationCombiner inicializado com {len(self.base_transformations)} transformações base")
        self.logger.info("TransformationCombiner inicializado com sucesso.")

    def _setup_logging(self):
        self.logger = logging.getLogger("AutoFE.TransformationCombiner")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    
    def add_base_transformation(self, config: Dict) -> None:
        """
        Adiciona uma transformação base à lista.
        
        Args:
            config: Configuração da transformação
        """
        if config not in self.base_transformations:
            self.base_transformations.append(config)
            self.logger.info(f"Adicionada transformação base: {config}")
    
    def _combine_configs(self, config1: Dict, config2: Dict) -> Dict:
        """
        Combina duas configurações de transformação.
        
        Args:
            config1: Primeira configuração
            config2: Segunda configuração
            
        Returns:
            Configuração combinada
        """
        # Estratégia simples: para chaves em comum, usar valor de config2 (mais recente)
        combined = config1.copy()
        combined.update(config2)
        return combined
    
    def generate_combinations(self, recommendations: List[Dict] = None,
                             limit: int = None) -> List[Dict]:
        """
        Gera combinações de transformações.
        
        Args:
            recommendations: Lista de recomendações a incluir
            limit: Limite de combinações a gerar
            
        Returns:
            Lista de configurações combinadas
        """
        transformations = list(self.base_transformations)
        
        # Adiciona recomendações, se fornecidas
        if recommendations:
            for rec in recommendations:
                if rec not in transformations:
                    transformations.append(rec)
        
        # Verifica se há transformações disponíveis
        if not transformations:
            self.logger.warning("Nenhuma transformação disponível para combinar")
            return []
        
        # Inicializa lista de combinações com as transformações base
        combinations = transformations.copy()
        
        # Gera combinações extras
        for depth in range(2, self.max_depth + 1):
            pairs = list(itertools.combinations(transformations, depth))
            
            # Limita o número de pares para evitar explosão combinatória
            max_pairs = min(len(pairs), 20)
            if len(pairs) > max_pairs:
                # Seleciona aleatoriamente para manter diversidade
                import random
                random.shuffle(pairs)
                pairs = pairs[:max_pairs]
            
            for pair in pairs:
                # Combina as configurações sequencialmente
                combined = pair[0].copy()
                for config in pair[1:]:
                    combined = self._combine_configs(combined, config)
                
                combinations.append(combined)
        
        # Limita o número de combinações, se necessário
        if limit and len(combinations) > limit:
            # Seleciona aleatoriamente para diversidade
            import random
            random.shuffle(combinations)
            combinations = combinations[:limit]
        
        self.logger.info(f"Geradas {len(combinations)} combinações de transformações")
        return combinations
    
    def beam_search(self, df: pd.DataFrame, target_col: Optional[str] = None,
                   evaluator: Optional[TransformationEvaluator] = None,
                   recommendations: List[Dict] = None) -> TransformationTree:
        """
        Realiza uma busca em feixe para encontrar a melhor sequência de transformações.
        
        Args:
            df: DataFrame a transformar
            target_col: Nome da coluna alvo
            evaluator: Avaliador de transformações
            recommendations: Recomendações adicionais a incluir
            
        Returns:
            Árvore de transformações com os resultados
        """
        # Inicializa avaliador, se não fornecido
        if evaluator is None:
            evaluator = TransformationEvaluator(target_col=target_col)
        
        # Inicializa árvore de transformações
        tree = TransformationTree()
        
        # Adiciona nó raiz com dados originais
        root_metrics = evaluator.evaluate_transformation(df)
        root_score = evaluator.compute_overall_score(root_metrics)
        tree.nodes["root"].data = df
        tree.nodes["root"].score = root_score
        tree.nodes["root"].metrics = root_metrics
        
        # Gera combinações de transformações
        combinations = self.generate_combinations(recommendations)
        
        # Realiza busca em feixe por níveis
        frontier = ["root"]
        
        for level in range(1, self.max_depth + 1):
            self.logger.info(f"Buscando no nível {level}...")
            level_nodes = []
            
            # Processa cada nó na fronteira atual
            for parent_id in frontier:
                parent_node = tree.nodes[parent_id]
                parent_data = parent_node.data
                
                # Aplica cada transformação ao nó pai
                transformations_results = []
                
                # Executa transformações em paralelo ou sequencial
                if self.parallel:
                    # Converte -1 para o número de CPUs disponíveis (comportamento similar ao scikit-learn)
                    import os
                    num_workers = os.cpu_count() if self.n_jobs == -1 else self.n_jobs
                    # Garante que num_workers seja pelo menos 1
                    num_workers = max(1, num_workers)
                    
                    try:
                        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                            # Preparamos os argumentos para a função global
                            futures = [
                                executor.submit(
                                    process_transformation, 
                                    config, 
                                    parent_data.copy(), 
                                    target_col, 
                                    evaluator
                                ) for config in combinations
                            ]
                            
                            for future in concurrent.futures.as_completed(futures):
                                try:
                                    result = future.result()
                                    if result['data'] is not None:
                                        transformations_results.append(result)
                                except Exception as e:
                                    self.logger.error(f"Erro em worker: {e}")
                    except Exception as e:
                        self.logger.error(f"Erro ao configurar paralelização: {e}")
                        # Fallback para execução sequencial
                        self.logger.info("Usando execução sequencial como fallback")
                        for config in combinations:
                            result = process_transformation(config, parent_data.copy(), target_col, evaluator)
                            if result['data'] is not None:
                                transformations_results.append(result)
                else:
                    for config in combinations:
                        result = process_transformation(config, parent_data.copy(), target_col, evaluator)
                        if result['data'] is not None:
                            transformations_results.append(result)
                
                # Filtra resultados com erros
                valid_results = [r for r in transformations_results if r['data'] is not None]
                
                # Adiciona resultados à árvore
                for result in valid_results:
                    node_id = tree.add_transformation(
                        parent=parent_id, 
                        name=result['name'],
                        config=result['config'],
                        data=result['data'],
                        score=result['score'],
                        metrics=result['metrics'],
                        execution_time=result['execution_time'],
                        feature_importance=result['feature_importance']
                    )
                    level_nodes.append((node_id, result['score']))
            
            # Seleciona os melhores nós para a próxima iteração (beam search)
            if level_nodes:
                level_nodes.sort(key=lambda x: x[1], reverse=True)
                frontier = [node_id for node_id, _ in level_nodes[:self.beam_width]]
                self.logger.info(f"Selecionados {len(frontier)} nós para prosseguir: {frontier}")
            else:
                self.logger.warning(f"Nenhum nó válido encontrado no nível {level}. Parando a busca.")
                break
        
        # Poda a árvore para remover ramos de baixa qualidade
        tree.prune(keep_best=self.beam_width)
        
        return tree


class Explorer:
    """
    Explora e avalia transformações para encontrar a melhor configuração
    para um conjunto de dados.
    """
    def __init__(self, target_col: Optional[str] = None, problem_type: str = 'auto',
                 experience_db: Optional[str] = None, parallel: bool = True,
                 max_depth: int = 3, beam_width: int = 5, n_jobs: int = -1,
                 base_configs: List[Dict] = None):
        """
        Inicializa o explorador.
        
        Args:
            target_col: Nome da coluna alvo
            problem_type: Tipo de problema ('classification', 'regression', 'auto')
            experience_db: Caminho para o banco de dados de experiências
            parallel: Se deve usar processamento paralelo
            max_depth: Profundidade máxima do pipeline
            beam_width: Largura do feixe na busca
            n_jobs: Número de jobs para paralelização
            base_configs: Configurações base a incluir
        """
        self.target_col = target_col
        self.problem_type = problem_type
        self.experience_db = experience_db
        self.parallel = parallel
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.n_jobs = n_jobs
        
        # Instancia componentes
        self.evaluator = TransformationEvaluator(target_col=target_col, problem_type=problem_type)
        self.meta_learner = MetaLearner(experience_db=experience_db)
        
        # Configurações base
        self.base_configs = base_configs or self._get_default_configs()
        
        # Instancia o combinador
        self.combiner = TransformationCombiner(
            base_transformations=self.base_configs,
            max_depth=max_depth,
            beam_width=beam_width,
            parallel=parallel,
            n_jobs=n_jobs
        )
        
        # Armazena o resultado da exploração
        self.exploration_result = None
        self._setup_logging()
        self.logger.info(f"Explorer inicializado. Target: {target_col}, Tipo: {problem_type}")
    
    def _setup_logging(self):
        self.logger = logging.getLogger("AutoFE.MetaLearner")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _analyze_dataset(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict:
        """
        Analisa um dataset para determinar suas características principais e selecionar
        as melhores configurações.
        
        Args:
            df: DataFrame a ser analisado
            target_col: Nome da coluna alvo (opcional)
            
        Returns:
            Dicionário com características e configurações recomendadas
        """
        analysis = {
            'n_samples': len(df),
            'n_features': df.shape[1],
            'has_missing': df.isna().any().any(),
            'missing_ratio': df.isna().mean().mean(),
            'has_target': target_col is not None and target_col in df.columns,
            'problem_type': None,
            'is_imbalanced': False,
            'imbalance_ratio': 1.0,
            'target_cardinality': 0,
            'recommended_configs': []
        }
        
        # Analisar colunas por tipo
        analysis['n_numeric'] = len(df.select_dtypes(include=['number']).columns)
        analysis['n_categorical'] = len(df.select_dtypes(include=['object', 'category']).columns)
        
        # Se tiver alvo, analisar características
        if analysis['has_target']:
            y = df[target_col]
            
            # Detectar tipo de problema
            if pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y) or y.nunique() <= 10:
                analysis['problem_type'] = 'classification'
                analysis['target_cardinality'] = y.nunique()
                
                # Verificar balanceamento (para classificação)
                class_counts = y.value_counts()
                if len(class_counts) >= 2:
                    analysis['imbalance_ratio'] = class_counts.min() / class_counts.max()
                    analysis['is_imbalanced'] = analysis['imbalance_ratio'] < 0.2
                    analysis['minority_class_size'] = class_counts.min()
                    analysis['majority_class_size'] = class_counts.max()
            else:
                analysis['problem_type'] = 'regression'
                analysis['target_skew'] = float(stats.skew(y.dropna()) if len(y.dropna()) > 2 else 0)
                analysis['target_range'] = float(y.max() - y.min() if len(y) > 0 else 0)
        
        # Gerar configurações recomendadas com base na análise
        analysis['recommended_configs'] = self._generate_adaptive_configs(analysis)
        
        return analysis

    def _generate_adaptive_configs(self, analysis: Dict) -> List[Dict]:
        """
        Gera configurações adaptativas com base na análise do dataset.
        
        Args:
            analysis: Dicionário com características do dataset
            
        Returns:
            Lista de configurações recomendadas
        """
        configs = []
        
        # Configuração base adaptativa
        base_config = {
            'missing_values_strategy': 'knn' if analysis['missing_ratio'] > 0.05 else 'median',
            'categorical_strategy': 'onehot',
            'scaling': 'robust' if analysis['problem_type'] == 'regression' else 'standard',
            'generate_features': True
        }
        
        # Adapta configuração de outliers
        if analysis['n_samples'] < 1000:
            # Dataset pequeno - seja conservador com outliers
            base_config['outlier_method'] = None
        else:
            # Dataset grande - pode usar remoção de outliers
            base_config['outlier_method'] = 'iqr'
        
        # Problema de classificação
        if analysis['problem_type'] == 'classification':
            # Fortemente desbalanceado
            if analysis['is_imbalanced'] and analysis['imbalance_ratio'] < 0.01:
                # Extremamente desbalanceado (menos de 1%)
                configs.append({
                    **base_config,
                    'outlier_method': None,  # Não remove outliers
                    'scaling': 'robust',     # Escala robusta
                    'balance_classes': True,
                    'balance_method': 'smote',
                    'sampling_strategy': min(0.1, analysis['imbalance_ratio'] * 10)  # Equilibra sem criar muitos sintéticos
                })
                
                # Também adiciona configuração usando pesos
                configs.append({
                    **base_config,
                    'outlier_method': None,
                    'scaling': 'robust',
                    'balance_classes': False,
                    'use_sample_weights': True
                })
            elif analysis['is_imbalanced']:
                # Moderadamente desbalanceado
                configs.append({
                    **base_config,
                    'balance_classes': True,
                    'balance_method': 'smote',
                    'sampling_strategy': 'auto'
                })
            else:
                # Balanceado
                configs.append(base_config)
                
            # Configuração específica para classificação multiclasse
            if analysis['target_cardinality'] > 2:
                configs.append({
                    **base_config,
                    'feature_selection': 'model_based',
                    'balance_classes': False
                })
        
        # Problema de regressão
        elif analysis['problem_type'] == 'regression':
            # Verifica se target tem alta skewness
            if 'target_skew' in analysis and abs(analysis.get('target_skew', 0)) > 1.0:
                # Target assimétrico
                configs.append({
                    **base_config,
                    'outlier_method': 'isolation_forest',
                    'scaling': 'robust'
                })
            else:
                # Regressão padrão
                configs.append(base_config)
                
            # Adiciona configuração para regressão com seleção de features
            configs.append({
                **base_config,
                'feature_selection': 'model_based',
                'dimensionality_reduction': None if analysis['n_features'] < 50 else 'pca'
            })
        
        # Se não tiver alvo ou for desconhecido
        else:
            configs.append(base_config)
            
            # Se tiver muitas features, considere redução de dimensionalidade
            if analysis['n_features'] > 20:
                configs.append({
                    **base_config,
                    'outlier_method': None,
                    'dimensionality_reduction': 'pca'
                })
        
        # Se o dataset for pequeno, desativa geração de features em uma configuração
        if analysis['n_samples'] < 500:
            minimal_config = {**base_config, 'generate_features': False}
            configs.append(minimal_config)
        
        # Adicionar configuração mínima que deve funcionar para qualquer dataset
        configs.append({
            'missing_values_strategy': 'median',
            'outlier_method': None,
            'categorical_strategy': 'onehot',
            'scaling': 'standard',
            'generate_features': False
        })
        
        return configs

    def _get_default_configs(self) -> List[Dict]:
        """
        Retorna configurações padrão adaptativas baseadas no dataset atual.
        """
        # Se já temos uma análise do dataset, use as configurações recomendadas
        if hasattr(self, '_dataset_analysis') and self._dataset_analysis:
            return self._dataset_analysis.get('recommended_configs', [])
        
        # Configurações genéricas para diferentes tipos de problemas
        classification_configs = [
            # SMOTE básico
            {
                'missing_values_strategy': 'median',
                'outlier_method': 'zscore',
                'categorical_strategy': 'onehot',
                'scaling': 'standard',
                'generate_features': False,
                'balance_classes': True,
                'balance_method': 'smote',
                'sampling_strategy': 'auto'
            },
            
            # SMOTE com engenharia de features
            {
                'missing_values_strategy': 'knn',
                'outlier_method': 'isolation_forest',
                'categorical_strategy': 'onehot',
                'scaling': 'robust',
                'generate_features': True,
                'feature_selection': 'model_based',
                'balance_classes': True,
                'balance_method': 'smote',
                'sampling_strategy': 'auto'
            },
            
            # ADASYN para dados muito desbalanceados
            {
                'missing_values_strategy': 'knn',
                'outlier_method': 'isolation_forest',
                'categorical_strategy': 'onehot',
                'scaling': 'robust',
                'generate_features': True,
                'feature_selection': 'model_based',
                'balance_classes': True,
                'balance_method': 'adasyn',
                'sampling_strategy': 'auto'
            },
            
            # SMOTETomek para limpeza de fronteira
            {
                'missing_values_strategy': 'median',
                'outlier_method': 'zscore',
                'categorical_strategy': 'onehot',
                'scaling': 'standard',
                'generate_features': True,
                'balance_classes': True,
                'balance_method': 'smotetomek',
                'sampling_strategy': 'auto'
            },
            
            # NearMiss para subsampling controlado
            {
                'missing_values_strategy': 'median',
                'outlier_method': 'iqr',
                'categorical_strategy': 'onehot',
                'scaling': 'standard',
                'generate_features': False,
                'balance_classes': True,
                'balance_method': 'nearmiss',
                'sampling_strategy': 'auto'
            },
            # Configuração que preserva outliers e usa SMOTE
            {
                'missing_values_strategy': 'knn',
                'outlier_method': None,  # Desativa remoção de outliers
                'categorical_strategy': 'onehot',
                'scaling': 'robust',  # Mais adequado para dados financeiros
                'generate_features': True,
                'balance_classes': True,
                'balance_method': 'smote',
                'sampling_strategy': 'auto'
            },
            # Configuração que usa threshold de IQR mais permissivo
            {
                'missing_values_strategy': 'median',
                'outlier_method': 'iqr',
                'categorical_strategy': 'onehot',
                'scaling': 'standard',
                'generate_features': True,
                'feature_selection': 'model_based',
                'correlation_threshold': 0.9  # Permite mais correlação entre features
            },
            # Configuração focada em modelagem
            {
                'missing_values_strategy': 'median',
                'outlier_method': None,  # Sem remoção de outliers
                'categorical_strategy': 'onehot',
                'scaling': 'robust',
                'generate_features': True,
                'use_sample_weights': True  # Usa pesos em vez de balanceamento
            }
        ]
        
        regression_configs = [
            # Configuração base para regressão
            {
                'missing_values_strategy': 'median',
                'outlier_method': 'iqr',
                'categorical_strategy': 'onehot',
                'scaling': 'standard',
                'generate_features': True,
                'feature_selection': None
            },
            
            # Configuração para dados com outliers
            {
                'missing_values_strategy': 'knn',
                'outlier_method': 'isolation_forest',
                'categorical_strategy': 'onehot',
                'scaling': 'robust',
                'generate_features': True,
                'feature_selection': 'model_based'
            },
            
            # Configuração para dados com alta dimensionalidade
            {
                'missing_values_strategy': 'median',
                'outlier_method': 'zscore',
                'categorical_strategy': 'onehot',
                'scaling': 'standard',
                'dimensionality_reduction': 'pca',
                'generate_features': False,
                'remove_high_correlation': True,
                'correlation_threshold': 0.90
            },
            
            # Configuração para dados com valores extremos
            {
                'missing_values_strategy': 'median',
                'outlier_method': 'isolation_forest',
                'categorical_strategy': 'onehot',
                'scaling': 'robust',
                'generate_features': True,
                'feature_selection': 'mutual_info'
            },
            
            # Configuração para regressão com features polinomiais
            {
                'missing_values_strategy': 'median',
                'outlier_method': 'iqr',
                'categorical_strategy': 'onehot',
                'scaling': 'standard',
                'generate_features': True,
                'feature_selection': 'model_based',
                'polynomial_features': True,
                'polynomial_degree': 2,
                'interaction_only': True
            },
            
            # Configuração para dados com alta assimetria (skewed)
            {
                'missing_values_strategy': 'iterative',
                'outlier_method': 'isolation_forest',
                'categorical_strategy': 'onehot',
                'scaling': 'robust',
                'generate_features': True,
                'apply_log_transform': True,
                'feature_selection': 'mutual_info'
            },
            
            # Configuração para regressão com dados temporais
            {
                'missing_values_strategy': 'knn',
                'outlier_method': 'iqr',
                'categorical_strategy': 'onehot',
                'scaling': 'robust',
                'generate_features': True,
                'lag_features': True,
                'lag_order': [1, 2, 3],
                'rolling_features': True,
                'window_sizes': [3, 7, 14]
            },
            # Configuração que preserva outliers e usa SMOTE
            {
                'missing_values_strategy': 'knn',
                'outlier_method': None,  # Desativa remoção de outliers
                'categorical_strategy': 'onehot',
                'scaling': 'robust',  # Mais adequado para dados financeiros
                'generate_features': True,
                'balance_classes': True,
                'balance_method': 'smote',
                'sampling_strategy': 'auto'
            },
            # Configuração que usa threshold de IQR mais permissivo
            {
                'missing_values_strategy': 'median',
                'outlier_method': 'iqr',
                'categorical_strategy': 'onehot',
                'scaling': 'standard',
                'generate_features': True,
                'feature_selection': 'model_based',
                'correlation_threshold': 0.9  # Permite mais correlação entre features
            },
            # Configuração focada em modelagem
            {
                'missing_values_strategy': 'median',
                'outlier_method': None,  # Sem remoção de outliers
                'categorical_strategy': 'onehot',
                'scaling': 'robust',
                'generate_features': True,
                'use_sample_weights': True  # Usa pesos em vez de balanceamento
            }

        ]
        
        # Configuração mínima que deve funcionar para qualquer dataset
        minimal_config = {
            'missing_values_strategy': 'median',
            'outlier_method': None,
            'categorical_strategy': 'onehot',
            'scaling': 'standard',
            'generate_features': False
        }
        
        if self.problem_type == 'classification':
            return classification_configs + [minimal_config]
        elif self.problem_type == 'regression':
            return regression_configs + [minimal_config]
        else:
            return classification_configs[:1] + regression_configs[:1] + [minimal_config]

    def _get_default_configs(self) -> List[Dict]:
        """
        Retorna configurações padrão para iniciar a exploração.
        
        Returns:
            Lista de configurações
        """
        # Configurações básicas para diferentes cenários
        if self.problem_type == 'classification':
            return IMBALANCED_CONFIGS
        elif self.problem_type == 'regression':
            return REGRESSION_CONFIGS
        else:
            # Se o tipo de problema não for especificado ou for 'auto',
            # retorna uma combinação das configurações
            return REGRESSION_CONFIGS[:3] + IMBALANCED_CONFIGS[:2] + [
                # Configuração mínima (que funciona para qualquer tipo)
                {
                    'missing_values_strategy': 'median',
                    'categorical_strategy': 'onehot',
                    'scaling': 'standard',
                    'generate_features': False
                }
            ]
    
    def explore(self, df: pd.DataFrame) -> TransformationTree:
        """
        Explora o espaço de transformações para encontrar a melhor configuração.
        
        Args:
            df: DataFrame a explorar
            
        Returns:
            Árvore de transformações
        """
        self.logger.info(f"Iniciando exploração para DataFrame de dimensões {df.shape}")
        
        # Cria um perfil do dataset
        dataset_profile = self.meta_learner.profile_dataset(df, self.target_col)
        
        # Obtém recomendações com base em experiências anteriores
        recommendations = self.meta_learner.recommend_transformations(
            df, self.target_col, n_recommendations=5
        )
        
        # Adiciona recomendações ao combinador
        for config in recommendations:
            self.combiner.add_base_transformation(config)
        
        # Executa busca em feixe
        tree = self.combiner.beam_search(
            df=df,
            target_col=self.target_col,
            evaluator=self.evaluator,
            recommendations=recommendations
        )
        
        # Armazena resultado
        self.exploration_result = tree
        
        # Registra resultados no meta-learner
        try:
            best_nodes = tree.get_best_nodes(limit=3)
            for node_id in best_nodes:
                node = tree.nodes[node_id]
                self.meta_learner.record_result(
                    dataset_profile=dataset_profile,
                    config=node.config,
                    score=node.score,
                    metrics=node.metrics
                )
            
            # Salva a base de experiências
            if self.experience_db:
                self.meta_learner.save()
        except Exception as e:
            self.logger.warning(f"Erro ao registrar resultados no meta-learner: {e}")
        
        return tree
    
    def get_best_transformation(self) -> Dict:
        """
        Retorna a melhor transformação encontrada.
        
        Returns:
            Configuração da melhor transformação
        """
        if not self.exploration_result:
            self.logger.warning("Nenhuma exploração realizada. Retornando configuração padrão.")
            return self.base_configs[0]
        
        best_nodes = self.exploration_result.get_best_nodes(limit=1)
        if not best_nodes:
            self.logger.warning("Nenhum nó encontrado na exploração. Retornando configuração padrão.")
            return self.base_configs[0]
        
        best_node = self.exploration_result.nodes[best_nodes[0]]
        self.logger.info(f"Melhor transformação: {best_node.name} com score {best_node.score:.4f}")
        
        return best_node.config
    
    def get_transformation_report(self, top_k: int = 3) -> Dict:
        """
        Gera um relatório das melhores transformações.
        
        Args:
            top_k: Número de melhores transformações a incluir
            
        Returns:
            Relatório com detalhes das transformações
        """
        if not self.exploration_result:
            return {"status": "Nenhuma exploração realizada"}
        
        best_nodes = self.exploration_result.get_best_nodes(limit=top_k)
        
        report = {
            "status": "success",
            "n_transformations_explored": len(self.exploration_result.nodes) - 1,  # -1 para excluir a raiz
            "best_transformations": []
        }
        
        for i, node_id in enumerate(best_nodes):
            node = self.exploration_result.nodes[node_id]
            
            # Constrói caminho de transformações
            path = self.exploration_result.get_transformation_path(node_id)
            
            # Extrai principais métricas
            main_metrics = {}
            for key, value in node.metrics.items():
                if key in ['cv_accuracy', 'cv_f1', 'cv_roc_auc', 'cv_r2', 'cv_neg_rmse']:
                    main_metrics[key] = value
            
            # Obtém top features por importância
            top_features = sorted(
                node.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            transformation_info = {
                "rank": i + 1,
                "node_id": node_id,
                "name": node.name,
                "score": node.score,
                "config": node.config,
                "metrics": main_metrics,
                "execution_time": node.execution_time,
                "top_features": dict(top_features),
                "transformation_path": [p.get('name', 'unknown') for p in path if p.get('name') != 'root']
            }
            
            report["best_transformations"].append(transformation_info)
        
        return report
    
    def visualize_tree(self, output_file: Optional[str] = None) -> str:
        """
        Gera uma visualização da árvore de transformações.
        
        Args:
            output_file: Caminho para salvar a visualização (opcional)
            
        Returns:
            Código DOT da visualização
        """
        if not self.exploration_result:
            return "digraph { ROOT [label=\"No exploration data\"] }"
        
        try:
            import networkx as nx
            
            # Cria uma cópia do grafo para visualização
            viz_graph = nx.DiGraph()
            
            # Adiciona nós com atributos para visualização
            for node_id, data in self.exploration_result.graph.nodes(data=True):
                node = self.exploration_result.nodes.get(node_id)
                if not node:
                    continue
                
                # Cria label com informações relevantes
                if node_id == "root":
                    label = "Original Data"
                else:
                    # Extrai métricas principais
                    metrics_str = ""
                    for key in ['cv_accuracy', 'cv_f1', 'cv_r2', 'cv_neg_rmse']:
                        if key in node.metrics:
                            metrics_str += f"\\n{key}: {node.metrics[key]:.4f}"
                    
                    label = f"{node.name}\\nScore: {node.score:.4f}{metrics_str}"
                
                # Cores baseadas no score
                if node_id == "root":
                    color = "lightgrey"
                else:
                    # Escala de cores: vermelho (ruim) -> amarelo -> verde (bom)
                    score_norm = max(0, min(1, node.score))
                    if score_norm < 0.5:
                        # Vermelho para amarelo
                        r = 1.0
                        g = score_norm * 2
                        b = 0
                    else:
                        # Amarelo para verde
                        r = 1.0 - (score_norm - 0.5) * 2
                        g = 1.0
                        b = 0
                    
                    color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                
                viz_graph.add_node(node_id, label=label, color=color, style="filled")
            
            # Adiciona arestas
            for edge in self.exploration_result.graph.edges():
                viz_graph.add_edge(*edge)
            
            # Gera representação DOT
            dot_code = nx.nx_pydot.to_pydot(viz_graph).to_string()
            
            # Salva em arquivo, se especificado
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(dot_code)
                self.logger.info(f"Visualização salva em {output_file}")
            
            return dot_code
            
        except ImportError as e:
            self.logger.warning(f"Erro ao gerar visualização: {e}")
            return f"digraph {{ ROOT [label=\"Error: {e}\"] }}"
    
    def save(self, filepath: str) -> None:
        """
        Salva o explorador em um arquivo.
        
        Args:
            filepath: Caminho do arquivo
        """
        # Salva apenas os componentes necessários
        data = {
            'target_col': self.target_col,
            'problem_type': self.problem_type,
            'base_configs': self.base_configs,
            'exploration_result': self.exploration_result
        }
        
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'wb') as f:
            joblib.dump(data, f)
        
        self.logger.info(f"Explorer salvo em {filepath}")
    
    @classmethod
    def load(cls, filepath: str, experience_db: Optional[str] = None) -> 'Explorer':
        """
        Carrega um explorador de um arquivo.
        
        Args:
            filepath: Caminho do arquivo
            experience_db: Caminho para o banco de dados de experiências (opcional)
            
        Returns:
            Instância do Explorer carregada
        """
        with open(filepath, 'rb') as f:
            data = joblib.load(f)
        
        # Cria nova instância
        explorer = cls(
            target_col=data.get('target_col'),
            problem_type=data.get('problem_type'),
            experience_db=experience_db or data.get('experience_db'),
            base_configs=data.get('base_configs')
        )
        
        # Restaura resultado da exploração
        explorer.exploration_result = data.get('exploration_result')
        return explorer


def create_explorer(target_col: Optional[str] = None, problem_type: str = 'auto',
                   experience_db: Optional[str] = None, parallel: bool = True,
                   max_depth: int = 3, beam_width: int = 5, n_jobs: int = -1) -> Explorer:
    """
    Cria uma instância do Explorer com as configurações especificadas.
    
    Args:
        target_col: Nome da coluna alvo
        problem_type: Tipo de problema ('classification', 'regression', 'auto')
        experience_db: Caminho para o banco de dados de experiências
        parallel: Se deve usar processamento paralelo
        max_depth: Profundidade máxima do pipeline
        beam_width: Largura do feixe na busca
        n_jobs: Número de jobs para paralelização
        
    Returns:
        Instância configurada do Explorer
    """
    return Explorer(
        target_col=target_col,
        problem_type=problem_type,
        experience_db=experience_db,
        parallel=parallel,
        max_depth=max_depth,
        beam_width=beam_width,
        n_jobs=n_jobs
    )


def analyze_transformations(df: pd.DataFrame, target_col: Optional[str] = None,
                           problem_type: str = 'auto', parallel: bool = True) -> Dict:
    """
    Função auxiliar para analisar transformações para um DataFrame.
    
    Args:
        df: DataFrame a analisar
        target_col: Nome da coluna alvo (opcional)
        problem_type: Tipo de problema ('classification', 'regression', 'auto')
        parallel: Se deve usar processamento paralelo
        
    Returns:
        Dicionário com a melhor configuração e relatório
    """
    # Cria e executa o explorador
    explorer = create_explorer(
        target_col=target_col,
        problem_type=problem_type,
        parallel=parallel
    )
    
    # Explora transformações
    explorer.explore(df)
    
    # Obtém resultados
    best_config = explorer.get_best_transformation()
    report = explorer.get_transformation_report(top_k=3)
    
    return {
        'best_config': best_config,
        'report': report,
        'explorer': explorer  # Retorna o explorador para uso adicional
    }