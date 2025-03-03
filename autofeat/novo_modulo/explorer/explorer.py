# explorer.py

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression, f_classif, f_regression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
import joblib
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import logging
import os
from copy import deepcopy

class Explorer:
    """
    Módulo de exploração e engenharia de features do sistema AutoFE.
    Responsável por identificar features relevantes e gerar novas features
    através de diversas transformações.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o Explorer com configurações específicas.
        
        Args:
            config (dict, opcional): Dicionário de configurações personalizadas.
        """
        # Configurações padrão
        self.default_config = {
            'feature_selection_method': 'mutual_info',  # 'mutual_info', 'anova', 'chi2', 'recursive'
            'feature_selection_k': 'auto',  # 'auto' ou um inteiro para número específico de features
            'feature_selection_threshold': 0.01,  # threshold para filtragem por importância
            'feature_reduction_method': 'pca',  # 'pca', 'svd', 'none'
            'feature_reduction_components': 0.90,  # componentes a manter (float: variância explicada, int: número fixo)
            'polynomial_features': True,  # gerar features polinomiais
            'polynomial_degree': 2,  # grau máximo para features polinomiais
            'interaction_features': True,  # gerar features de interação
            'clustering_features': True,  # gerar features baseadas em clustering
            'n_clusters': 'auto',  # 'auto' ou um inteiro para número específico de clusters
            'agg_functions': ['mean', 'min', 'max', 'std'],  # funções de agregação para features de grupo
            'max_features_to_try': 1000,  # limite máximo de features a serem exploradas
            'evaluation_model': None,  # modelo para avaliar as features geradas
            'evaluation_metric': 'auto',  # 'auto', 'accuracy', 'roc_auc', 'f1', 'r2', 'rmse'
            'evaluation_cv': 5,  # número de folds para validação cruzada
            'problem_type': 'auto',  # 'auto', 'classification', 'regression'
            'random_state': 42,  # semente para reprodutibilidade
            'verbosity': 1  # nível de detalhamento dos logs
        }
        
        # Atualizar configurações com valores personalizados, se fornecidos
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # Inicializar atributos
        self.original_features = None
        self.original_feature_types = None
        self.feature_importances = {}
        self.feature_groups = {}
        self.transformations_applied = []
        self.feature_performance = {}
        self.best_features = None
        self.problem_type = self.config['problem_type']
        self.is_fitted = False
        self.interaction_features = []
        
        # Configurar logging
        self._setup_logging()
        
        self.logger.info("Explorer inicializado com sucesso.")
        
    def _setup_logging(self):
        """Configura o sistema de logging do Explorer."""
        self.logger = logging.getLogger("AutoFE.Explorer")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }
        self.logger.setLevel(log_levels.get(self.config['verbosity'], logging.INFO))

    # Modificação para o método fit() no preprocessor.py
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Ajusta o Explorer aos dados, identificando features importantes e
        descobrindo transformações úteis.
        
        Args:
            X (pandas.DataFrame): DataFrame com os dados de treinamento
            y (pandas.Series, opcional): Série com os valores alvo
            
        Returns:
            self: Instância do próprio Explorer
        """
        self.logger.info("Iniciando ajuste do Explorer...")
        
        # Verificar se o DataFrame está vazio
        if X is None or X.empty:
            raise ValueError("DataFrame vazio ou None fornecido para ajuste.")
        
        # Verificar o tipo dos dados de entrada
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X deve ser um pandas DataFrame, recebido: {type(X)}")
        
        # Verificar o tipo da variável alvo, se fornecida
        if y is not None and not isinstance(y, pd.Series):
            raise TypeError(f"y deve ser uma pandas Series, recebido: {type(y)}")
        
        # Fazer uma cópia dos dados
        X_copy = X.copy()
        
        # Guardar informações das features originais
        self.original_features = X_copy.columns.tolist()
        
        # Identificar os tipos de features
        self.original_feature_types = self._identify_column_types(X_copy)
        
        # Detectar o tipo de problema, se y for fornecido
        if y is not None:
            if self.problem_type == 'auto':
                self.problem_type = self._detect_problem_type(y)
                self.logger.info(f"Tipo de problema detectado: {self.problem_type}")
            
            # Analisar importância das features originais
            self._analyze_feature_importance(X_copy, y)
            
            # Agrupar features relacionadas
            self._group_related_features(X_copy)
            
            # Gerar novas features
            self.is_fitted = True

            # Gerar novas features
            self._generate_features(X_copy, y)

            # Selecionar as melhores features
            transformed_X = self.transform(X_copy)
            self._select_best_features(transformed_X, y)
        else:
            self.logger.warning("Variável alvo (y) não fornecida. Algumas transformações não serão realizadas.")
            # Sem o target, ainda podemos fazer algumas transformações básicas
            self._group_related_features(X_copy)
            self._generate_features(X_copy)
        
        self.is_fitted = True
        self.logger.info("Ajuste do Explorer concluído com sucesso.")
        return self
    
    # Modificação para o método fit_transform() no preprocessor.py
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Ajusta o Explorer e aplica as transformações ao mesmo conjunto de dados.
        
        Args:
            X (pandas.DataFrame): DataFrame com os dados
            y (pandas.Series, opcional): Série com os valores alvo
            
        Returns:
            pandas.DataFrame: DataFrame transformado
        """
        self.fit(X, y)
        return self.transform(X)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica as transformações descobertas durante o fit a um novo conjunto de dados.
        
        Args:
            X (pandas.DataFrame): Dados a serem transformados
            
        Returns:
            pandas.DataFrame: Dados com as transformações aplicadas
        """
        self.logger.info("Aplicando transformações aos dados...")
        
        if not self.is_fitted:
            # Alteração 1: Mudança de warning para exception para prevenir execução inadequada
            raise RuntimeError("Explorer não está ajustado! Execute fit() primeiro.")
        
        if not self.transformations_applied:
            self.logger.warning("Nenhuma transformação encontrada para aplicar.")
            return X
        
        result = X.copy()
        
        # Aplicar cada transformação na ordem em que foram descobertas
        for transform_info in self.transformations_applied:
            transform_type = transform_info['type']
            transform_params = transform_info['params']
            transform_func = transform_info['function']
            
            self.logger.debug(f"Aplicando transformação: {transform_type}")
            
            try:
                result = transform_func(result, **transform_params)
            except Exception as e:
                self.logger.error(f"Erro ao aplicar transformação {transform_type}: {str(e)}")
                # Continuar com as próximas transformações
        
        # Selecionar apenas as melhores features se já foram determinadas
        if self.best_features is not None:
            available_cols = set(result.columns).intersection(set(self.best_features))
            result = result[list(available_cols)]
            
            if len(available_cols) < len(self.best_features):
                missing = set(self.best_features) - available_cols
                self.logger.warning(f"Algumas features selecionadas não estão disponíveis: {missing}")
        
        self.logger.info(f"Transformação concluída. Dimensões resultantes: {result.shape}")
        return result
    
    def _detect_problem_type(self, y: pd.Series) -> str:
        """
        Detecta automaticamente o tipo de problema (classificação ou regressão)
        com base nos valores alvo.
        
        Args:
            y (pandas.Series): Valores alvo
            
        Returns:
            str: 'classification' ou 'regression'
        """
        # Se y for categórico ou tiver poucos valores únicos, é provavelmente classificação
        if y.dtype == 'object' or y.dtype == 'category' or len(y.unique()) / len(y) < 0.05:
            return 'classification'
        else:
            return 'regression'
    
    def _identify_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identifica os tipos de colunas no DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame a ser analisado
            
        Returns:
            dict: Dicionário com colunas categorizadas por tipo
        """
        column_types = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'text': []
        }
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Verificar se é numérico mas na verdade é categórico
                if len(df[col].unique()) < min(20, len(df[col]) * 0.05):
                    column_types['categorical'].append(col)
                else:
                    column_types['numeric'].append(col)
            elif pd.api.types.is_datetime64_dtype(df[col]):
                column_types['datetime'].append(col)
            elif df[col].dtype == 'object':
                # Verificar se parece ser texto ou categórico
                if df[col].str.len().mean() > 20:  # Heurística simples para texto
                    column_types['text'].append(col)
                else:
                    column_types['categorical'].append(col)
            else:
                column_types['categorical'].append(col)
        
        self.logger.debug(f"Tipos de colunas identificados: {column_types}")
        return column_types
    
    def _analyze_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Analisa a importância das features originais.
        
        Args:
            X (pandas.DataFrame): Features
            y (pandas.Series): Target
        """
        self.logger.info("Analisando importância das features...")
        
        # Selecionar método apropriado baseado no tipo de problema
        if self.problem_type == 'classification':
            score_func = mutual_info_classif if self.config['feature_selection_method'] == 'mutual_info' else f_classif
        else:
            score_func = mutual_info_regression if self.config['feature_selection_method'] == 'mutual_info' else f_regression
        
        # Separar análise por tipo de coluna
        numeric_cols = self.original_feature_types['numeric']
        if numeric_cols:
            X_numeric = X[numeric_cols]
            try:
                # Aplicar a função de pontuação
                scores = score_func(X_numeric, y)
                
                # Verificar o formato dos scores - podem ser um array ou uma tupla (F, p-value)
                if isinstance(scores, tuple):
                    # Para f_classif e f_regression que retornam (F, p-value)
                    importance_scores = scores[0]
                else:
                    # Para mutual_info que retorna um array diretamente
                    importance_scores = scores
                
                # Atribuir scores às features
                for col, score in zip(numeric_cols, importance_scores):
                    self.feature_importances[col] = float(score)
                    
            except Exception as e:
                self.logger.warning(f"Erro ao calcular importância para features numéricas: {str(e)}")
                # Registrar mais detalhes para debug
                self.logger.debug(f"Type of score_func: {type(score_func)}")
                self.logger.debug(f"Columns: {numeric_cols}")
                self.logger.debug(f"Exception details: {repr(e)}")
        
        # Análise para features categóricas
        categorical_cols = self.original_feature_types['categorical']
        if categorical_cols:
            try:
                # Precisamos garantir que as features categóricas estejam codificadas
                X_cat = pd.get_dummies(X[categorical_cols], drop_first=True)
                
                if not X_cat.empty:
                    # Aplicar a função de pontuação
                    scores = score_func(X_cat, y)
                    
                    # Verificar o formato dos scores
                    if isinstance(scores, tuple):
                        importance_scores = scores[0]
                    else:
                        importance_scores = scores
                    
                    # Mapear os resultados de volta para as colunas originais
                    col_mapping = {}
                    for col in X_cat.columns:
                        # Extrair o nome da coluna original da coluna one-hot
                        parts = col.split('_')
                        if len(parts) > 1:
                            orig_col = '_'.join(parts[:-1])
                        else:
                            orig_col = col
                            
                        if orig_col not in col_mapping:
                            col_mapping[orig_col] = []
                        col_mapping[orig_col].append(col)
                    
                    # Calcular a média das importâncias para cada feature original
                    for orig_col, one_hot_cols in col_mapping.items():
                        # Encontrar os índices das colunas one-hot no DataFrame
                        col_indices = [X_cat.columns.get_loc(c) for c in one_hot_cols if c in X_cat.columns]
                        
                        if col_indices:  # Só processar se houver índices válidos
                            # Calcular a média das importâncias
                            mean_importance = float(np.mean([importance_scores[i] for i in col_indices]))
                            self.feature_importances[orig_col] = mean_importance
            
            except Exception as e:
                self.logger.warning(f"Erro ao calcular importância para features categóricas: {str(e)}")
                # Registrar mais detalhes para debug
                self.logger.debug(f"Type of score_func: {type(score_func)}")
                self.logger.debug(f"Columns: {categorical_cols}")
                self.logger.debug(f"Exception details: {repr(e)}")
        
        # Normalizar scores para facilitar interpretação
        if self.feature_importances:
            max_score = max(self.feature_importances.values()) if self.feature_importances else 1.0
            if max_score > 0:
                for col in self.feature_importances:
                    self.feature_importances[col] /= max_score
        
        # Registrar as top features
        top_features = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)
        self.logger.info(f"Top 10 features por importância: {top_features[:10]}")
    
    def _group_related_features(self, X: pd.DataFrame) -> None:
        """
        Identifica e agrupa features relacionadas com base em correlações e outros critérios.
        
        Args:
            X (pandas.DataFrame): Features
        """
        self.logger.info("Agrupando features relacionadas...")
        
        # Inicializar grupos
        self.feature_groups = {
            'high_importance': [],
            'correlated': {},
            'low_variance': [],
            'numeric': self.original_feature_types['numeric'],
            'categorical': self.original_feature_types['categorical'],
            'datetime': self.original_feature_types['datetime'],
            'text': self.original_feature_types['text']
        }
        
        # Grupo de alta importância (baseado na análise anterior)
        if self.feature_importances:
            threshold = self.config['feature_selection_threshold']
            self.feature_groups['high_importance'] = [
                col for col, score in self.feature_importances.items() 
                if score >= threshold
            ]
            self.logger.debug(f"Features de alta importância: {len(self.feature_groups['high_importance'])}")
        
        # Identificar features correlacionadas (apenas numéricas)
        if len(self.original_feature_types['numeric']) > 1:
            try:
                corr_matrix = X[self.original_feature_types['numeric']].corr().abs()
                
                # Para cada feature, encontrar outras altamente correlacionadas
                for i, col in enumerate(corr_matrix.columns):
                    correlated_features = []
                    for j, other_col in enumerate(corr_matrix.columns):
                        if i != j and corr_matrix.iloc[i, j] > 0.8:  # Threshold para correlação alta
                            correlated_features.append(other_col)
                    
                    if correlated_features:
                        self.feature_groups['correlated'][col] = correlated_features
                
                self.logger.debug(f"Grupos de features correlacionadas: {len(self.feature_groups['correlated'])}")
            except Exception as e:
                self.logger.warning(f"Erro ao calcular correlações: {str(e)}")
        
        # Identificar features com baixa variância
        for col in self.original_feature_types['numeric']:
            if X[col].std() < 0.01 * X[col].mean() and X[col].mean() != 0:
                self.feature_groups['low_variance'].append(col)
        
        self.logger.debug(f"Features com baixa variância: {len(self.feature_groups['low_variance'])}")
    
    def _generate_features(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Gera novas features através de diversas transformações.
        
        Args:
            X (pandas.DataFrame): Features originais
            y (pandas.Series, opcional): Target para avaliação das novas features
        """
        self.logger.info("Gerando novas features...")
        
        # Inicializar lista de transformações
        self.transformations_applied = []
        feature_count = X.shape[1]
        
        # Aplicar transformações apenas se não excederem o limite de features
        if feature_count < self.config['max_features_to_try']:
            # 1. Gerar features polinomiais para variáveis numéricas
            if self.config['polynomial_features'] and self.original_feature_types['numeric']:
                self._generate_polynomial_features(X, y)
                
            # 2. Gerar features de interação entre variáveis importantes
            if self.config['interaction_features'] and len(self.feature_groups['high_importance']) >= 2:
                self._generate_interaction_features(X, y)
                
            # 3. Gerar features baseadas em clustering
            if self.config['clustering_features'] and len(self.original_feature_types['numeric']) >= 2:
                self._generate_clustering_features(X, y)
                
            # 4. Aplicar redução de dimensionalidade se necessário
            if self.config['feature_reduction_method'] != 'none':
                self._apply_dimensionality_reduction(X, y)
        else:
            self.logger.warning(f"Número de features ({feature_count}) excede o limite configurado. Pulando geração.")

    def _generate_polynomial_features(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Gera features polinomiais para variáveis numéricas.
        
        Args:
            X (pandas.DataFrame): Features
            y (pandas.Series, opcional): Target para avaliação
        """
        # Selecionar apenas as features numéricas mais importantes
        numeric_cols = self.original_feature_types['numeric']
        if not numeric_cols:
            return
        
        # Se tivermos informações de importância, usar apenas as mais importantes
        if self.feature_importances:
            numeric_importance = {col: self.feature_importances.get(col, 0) for col in numeric_cols}
            sorted_numeric = sorted(numeric_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Limitar o número de features para evitar explosão combinatorial
            max_poly_features = min(10, len(sorted_numeric))
            top_numeric = [col for col, _ in sorted_numeric[:max_poly_features]]
        else:
            # Sem informações de importância, usar todas as numéricas até um limite
            top_numeric = numeric_cols[:10]  # Limitar a 10 features
        
        if not top_numeric:
            return
        
        X_selected = X[top_numeric]
        
        # Configurar transformação polinomial
        poly = PolynomialFeatures(
            degree=self.config['polynomial_degree'],
            include_bias=False,
            interaction_only=False
        )
        
        try:
            # Aplicar transformação
            poly_features = poly.fit_transform(X_selected)
            feature_names = poly.get_feature_names_out(top_numeric)
            
            # Criar DataFrame com as novas features
            poly_df = pd.DataFrame(
                poly_features, 
                columns=feature_names,
                index=X.index
            )
            
            # Remover as colunas originais para evitar duplicação
            poly_df = poly_df.loc[:, ~poly_df.columns.isin(top_numeric)]
            
            # Criar DataFrame para avaliação
            # cluster_df = pd.DataFrame(cluster_data, index=X.index)
            
            # Avaliar as novas features se tivermos um target
            if y is not None and self.config['evaluation_model'] is not None:
                for col in poly_df.columns:
                    self._evaluate_feature(col, poly_df[col], y)
                        
            # Registrar a transformação
            self.transformations_applied.append({
                'type': 'polynomial',
                'params': {
                    'degree': self.config['polynomial_degree'],
                    'columns': top_numeric
                },
                'function': self._transform_polynomial
            })
            
            self.logger.info(f"Geradas {poly_df.shape[1]} features polinomiais.")
        except Exception as e:
            self.logger.warning(f"Erro ao gerar features polinomiais: {str(e)}")
            
    def _transform_polynomial(self, X: pd.DataFrame, degree: int, columns: List[str]) -> pd.DataFrame:
        """
        Aplica transformação polinomial em novos dados.
        
        Args:
            X (pandas.DataFrame): Dados de entrada
            degree (int): Grau do polinômio
            columns (list): Colunas a serem transformadas
            
        Returns:
            pandas.DataFrame: DataFrame com as features originais e as novas features
        """
        result = X.copy()
        
        # Verificar se todas as colunas necessárias estão presentes
        missing_cols = set(columns) - set(X.columns)
        if missing_cols:
            self.logger.warning(f"Colunas ausentes para transformação polinomial: {missing_cols}")
            # Usar apenas as colunas disponíveis
            available_cols = list(set(columns) - missing_cols)
            if not available_cols:
                return result
            columns = available_cols
        
        try:
            # Configurar transformação
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
            
            # Aplicar transformação
            poly_features = poly.fit_transform(X[columns])
            feature_names = poly.get_feature_names_out(columns)
            
            # Criar DataFrame com as novas features
            poly_df = pd.DataFrame(
                poly_features, 
                columns=feature_names,
                index=X.index
            )
            
            # Remover as colunas originais para evitar duplicação
            new_cols = [col for col in poly_df.columns if col not in X.columns]
            if new_cols:
                result = pd.concat([result, poly_df[new_cols]], axis=1)
        except Exception as e:
            self.logger.warning(f"Erro ao aplicar transformação polinomial: {str(e)}")
        
        return result
    
    def _generate_interaction_features(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Gera features de interação entre variáveis importantes.
        
        Args:
            X (pandas.DataFrame): Features
            y (pandas.Series, opcional): Target para avaliação
        """
        # Selecionar features importantes para interações
        important_features = self.feature_groups.get('high_importance', [])
        if len(important_features) < 2:
            # Se não temos features importantes identificadas, usar as numéricas
            important_features = self.original_feature_types['numeric'][:10]  # Limitando a 10
        
        if len(important_features) < 2:
            return
        
        # Limitar o número de features para evitar explosão combinatorial
        max_interact_features = min(10, len(important_features))
        self.interaction_features = important_features.copy()
        
        try:
            # Criar features de interação manualmente
            interaction_data = {}
            
            for i in range(len(self.interaction_features)):
                for j in range(i+1, len(self.interaction_features)):
                    col1 = self.interaction_features[i]
                    col2 = self.interaction_features[j]
                    
                    # Verificar tipos
                    col1_numeric = col1 in self.original_feature_types['numeric']
                    col2_numeric = col2 in self.original_feature_types['numeric']
                    
                    # Multiplicação (apenas para numéricas)
                    if col1_numeric and col2_numeric:
                        new_col = f"{col1}*{col2}"
                        interaction_data[new_col] = X[col1] * X[col2]
                        
                        # Divisão (com proteção contra divisão por zero)
                        if (X[col2] != 0).all():
                            new_col = f"{col1}/{col2}"
                            interaction_data[new_col] = X[col1] / (X[col2] + 1e-10)
                        
                        if (X[col1] != 0).all():
                            new_col = f"{col2}/{col1}"
                            interaction_data[new_col] = X[col2] / (X[col1] + 1e-10)
                        
                        # Diferença
                        new_col = f"{col1}-{col2}"
                        interaction_data[new_col] = X[col1] - X[col2]
                        
                        # Soma
                        new_col = f"{col1}+{col2}"
                        interaction_data[new_col] = X[col1] + X[col2]
            
            if interaction_data:
                # Criar DataFrame com as interações
                interactions_df = pd.DataFrame(interaction_data, index=X.index)
                
                # Avaliar as novas features se tivermos um target
                if y is not None and self.config['evaluation_model'] is not None:
                    for col in interactions_df.columns:
                        self._evaluate_feature(col, interactions_df[col], y)
                
                # Registrar a transformação
                self.transformations_applied.append({
                    'type': 'interaction',
                    'params': {
                        'columns': self.interaction_features
                    },
                    'function': self._transform_interaction
                })
                
                self.logger.info(f"Geradas {len(interaction_data)} features de interação.")
        except Exception as e:
            self.logger.warning(f"Erro ao gerar features de interação: {str(e)}")
    
    def _evaluate_feature(self, feature_name: str, feature_values: pd.Series, y: pd.Series) -> float:
        """
        Avalia a performance de uma feature individual.
        
        Args:
            feature_name (str): Nome da feature
            feature_values (pandas.Series): Valores da feature
            y (pandas.Series): Target
            
        Returns:
            float: Score de performance
        """
        # Implementação básica - pode ser expandida conforme necessidade
        try:
            # Usar o modelo de avaliação configurado ou um modelo simples
            model = self.config.get('evaluation_model')
            
            # Criar DataFrame para avaliação
            X_eval = pd.DataFrame({feature_name: feature_values})
            
            # Score padrão
            score = 0.0
            
            # Registrar o score
            self.feature_performance[feature_name] = score
            return score
        except Exception as e:
            self.logger.debug(f"Erro ao avaliar feature {feature_name}: {str(e)}")
            return 0.0
    
    def _transform_interaction(self, X: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Aplica transformações de interação em novos dados.
        
        Args:
            X (pandas.DataFrame): Dados de entrada
            columns (list): Colunas a serem utilizadas para interações
            
        Returns:
            pandas.DataFrame: DataFrame com as features originais e as novas features
        """
        result = X.copy()
        
        # Verificar se todas as colunas necessárias estão presentes
        available_cols = set(X.columns)
        interaction_cols = set(columns)
        missing_cols = interaction_cols - available_cols

        if missing_cols:
            self.logger.warning(f"Colunas ausentes para transformação de interação: {missing_cols}")
            # Usar apenas as colunas disponíveis
            available_cols = list(available_cols.intersection(interaction_cols))
            if len(available_cols) < 2:
                return result
            columns = available_cols
                
        try:
            
            if len(columns) < 2:
                self.logger.warning("Número insuficiente de colunas para interações.")
                return result
            
            # Criar features de interação de forma otimizada
            interaction_data = {}
            
            for i in range(len(columns)):
                for j in range(i+1, len(columns)):
                    col1 = columns[i]
                    col2 = columns[j]
                    
                    # Verificar tipos (usando a informação salva durante o fit)
                    col1_numeric = col1 in self.original_feature_types.get('numeric', [])
                    col2_numeric = col2 in self.original_feature_types.get('numeric', [])
                    
                    # Multiplicação (apenas para numéricas)
                    if col1_numeric and col2_numeric:
                        new_col = f"{col1}*{col2}"
                        interaction_data[new_col] = X[col1] * X[col2]
                        
                        # Divisão (com proteção contra divisão por zero)
                        if (X[col2] != 0).all():
                            new_col = f"{col1}/{col2}"
                            interaction_data[new_col] = X[col1] / (X[col2] + 1e-10)
                        
                        if (X[col1] != 0).all():
                            new_col = f"{col2}/{col1}"
                            interaction_data[new_col] = X[col2] / (X[col1] + 1e-10)
                        
                        # Diferença
                        new_col = f"{col1}-{col2}"
                        interaction_data[new_col] = X[col1] - X[col2]
                        
                        # Soma
                        new_col = f"{col1}+{col2}"
                        interaction_data[new_col] = X[col1] + X[col2]
            
            # Adicionar todas as interações de uma vez para evitar fragmentação
            if interaction_data:
                interactions_df = pd.DataFrame(interaction_data, index=X.index)
                result = pd.concat([result, interactions_df], axis=1)
        except Exception as e:
            self.logger.warning(f"Erro ao aplicar transformação de interação: {str(e)}")
        
        return result
    
    def _generate_clustering_features(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Gera features baseadas em clustering.
        
        Args:
            X (pandas.DataFrame): Features
            y (pandas.Series, opcional): Target para avaliação
        """
        # Selecionar features numéricas para clustering
        numeric_cols = self.original_feature_types['numeric']
        if len(numeric_cols) < 2:
            return
        
        # Se tivermos informações de importância, usar apenas as mais importantes
        if self.feature_importances:
            numeric_importance = {col: self.feature_importances.get(col, 0) for col in numeric_cols}
            sorted_numeric = sorted(numeric_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Limitar o número de features para clustering
            max_cluster_features = min(10, len(sorted_numeric))
            cluster_features = [col for col, _ in sorted_numeric[:max_cluster_features]]
        else:
            # Sem informações de importância, usar um subconjunto
            cluster_features = numeric_cols[:10]
        
        if len(cluster_features) < 2:
            return
            
        X_cluster = X[cluster_features]
        
        # Determinar número de clusters
        if self.config['n_clusters'] == 'auto':
            n_clusters = min(5, int(np.sqrt(len(X)) / 2))  # Heurística simples
        else:
            n_clusters = self.config['n_clusters']
        
        try:
            # Aplicar K-means
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.config['random_state'],
                n_init=10
            )
            cluster_labels = kmeans.fit_predict(X_cluster)
            
            # Criar features baseadas em clustering
            cluster_data = {
                'cluster_label': cluster_labels
            }
            
            
            # Calcular distância para cada centróide
            for i in range(n_clusters):
                centroid = kmeans.cluster_centers_[i]
                # Calcular distância euclidiana para o centróide
                dist = np.sqrt(((X_cluster - centroid) ** 2).sum(axis=1))
                cluster_data[f'distance_to_cluster_{i}'] = dist
                
            cluster_df = pd.DataFrame(cluster_data, index=X.index)
            if y is not None and self.config['evaluation_model'] is not None:
                for col in cluster_df.columns:
                    self._evaluate_feature(col, cluster_df[col], y)
            
            # Registrar a transformação
            self.transformations_applied.append({
                'type': 'clustering',
                'params': {
                    'columns': cluster_features,
                    'n_clusters': n_clusters
                },
                'function': self._transform_clustering,
                'model': kmeans
            })
            
            self.logger.info(f"Geradas {len(cluster_data)} features baseadas em clustering.")
        except Exception as e:
            self.logger.warning(f"Erro ao gerar features de clustering: {str(e)}")
    
    def _transform_clustering(self, X: pd.DataFrame, columns: List[str], n_clusters: int, model: Any = None) -> pd.DataFrame:
        """
        Aplica transformações de clustering em novos dados.
        
        Args:
            X (pandas.DataFrame): Dados de entrada
            columns (list): Colunas a serem utilizadas para clustering
            n_clusters (int): Número de clusters
            model (KMeans, opcional): Modelo KMeans ajustado
            
        Returns:
            pandas.DataFrame: DataFrame com as features originais e as novas features
        """
        result = X.copy()
        
        if hasattr(self, 'target_col') and isinstance(self.target_col, pd.Series):
            self.target_col = self.target_col.name if self.target_col.name is not None else "target"
        
        # Verificar se todas as colunas necessárias estão presentes
        missing_cols = set(columns) - set(X.columns)
        if missing_cols:
            self.logger.warning(f"Colunas ausentes para transformação de clustering: {missing_cols}")
            # Usar apenas as colunas disponíveis
            available_cols = list(set(columns) - missing_cols)
            if len(available_cols) < 2:
                return result
            columns = available_cols
        
        try:
            if hasattr(self, 'transformations_applied'):
                # Encontrar o modelo de clustering salvo
                for transform in self.transformations_applied:
                    if transform['type'] == 'clustering' and 'model' in transform:
                        model = transform['model']
                        break
            
            if model is None:
                self.logger.warning("Modelo de clustering não encontrado. Criando novo.")
                # Criar e ajustar novo modelo
                model = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.config['random_state'],
                    n_init=10
                )
                model.fit(X[columns])
            
            # Aplicar transformação de forma otimizada
            cluster_data = {}
            
            # Prever clusters para os novos dados
            cluster_labels = model.predict(X[columns])
            cluster_data['cluster_label'] = cluster_labels
            
            # Calcular distância para cada centróide
            for i in range(n_clusters):
                centroid = model.cluster_centers_[i]
                # Calcular distância euclidiana para o centróide
                dist = np.sqrt(((X[columns] - centroid) ** 2).sum(axis=1))
                cluster_data[f'distance_to_cluster_{i}'] = dist
            
            # Adicionar todas as features de clustering de uma vez
            cluster_df = pd.DataFrame(cluster_data, index=X.index)
            result = pd.concat([result, cluster_df], axis=1)
            
        except Exception as e:
            self.logger.warning(f"Erro ao aplicar transformação de clustering: {str(e)}")
        
        return result
    
    def _apply_dimensionality_reduction(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Aplica técnicas de redução de dimensionalidade.
        
        Args:
            X (pandas.DataFrame): Features
            y (pandas.Series, opcional): Target para avaliação
        """
        numeric_cols = self.original_feature_types['numeric']
        if len(numeric_cols) < 3:  # Redução de dimensionalidade só faz sentido com mais colunas
            return
        
        method = self.config['feature_reduction_method']
        components = self.config['feature_reduction_components']
        
        # Determinar componentes
        if isinstance(components, float) and 0 < components <= 1:
            n_components = components  # PCA vai interpretar como variância explicada se < 1
        else:
            n_components = min(int(components), len(numeric_cols) - 1)
        
        try:
            # Selecionar método de redução
            if method == 'pca':
                reducer = PCA(n_components=n_components, random_state=self.config['random_state'])
            elif method == 'svd':
                reducer = TruncatedSVD(n_components=n_components, random_state=self.config['random_state'])
            else:
                self.logger.warning(f"Método de redução desconhecido: {method}")
                return
            
            # Aplicar redução
            reduced_data = reducer.fit_transform(X[numeric_cols])
            
            # Registrar a transformação
            self.transformations_applied.append({
                'type': 'dimensionality_reduction',
                'params': {
                    'method': method,
                    'columns': numeric_cols,
                    'n_components': n_components
                },
                'function': self._transform_dimensionality_reduction,
                'model': reducer
            })
            
            # Se for PCA, registrar a variância explicada
            if method == 'pca':
                explained_variance = reducer.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance)
                self.logger.info(f"Variância explicada: {cumulative_variance[-1]:.2f}")
            
            self.logger.info(f"Geradas {reduced_data.shape[1]} componentes com {method}.")
        except Exception as e:
            self.logger.warning(f"Erro ao aplicar redução de dimensionalidade: {str(e)}")
    
    def _transform_dimensionality_reduction(self, X: pd.DataFrame, method: str, columns: List[str], 
                                          n_components: Union[int, float], model: Any = None) -> pd.DataFrame:
        """
        Aplica redução de dimensionalidade em novos dados.
        
        Args:
            X (pandas.DataFrame): Dados de entrada
            method (str): Método de redução ('pca' ou 'svd')
            columns (list): Colunas a serem utilizadas
            n_components (int/float): Número ou proporção de componentes
            model (objeto, opcional): Modelo de redução ajustado
            
        Returns:
            pandas.DataFrame: DataFrame com as features originais e as novas features
        """
        result = X.copy()
        
        # Verificar se todas as colunas necessárias estão presentes
        missing_cols = set(columns) - set(X.columns)
        if missing_cols:
            self.logger.warning(f"Colunas ausentes para redução de dimensionalidade: {missing_cols}")
            # Usar apenas as colunas disponíveis
            available_cols = list(set(columns) - missing_cols)
            if len(available_cols) < 2:
                return result
            columns = available_cols
        
        try:
            if hasattr(self, 'transformations_applied'):
                # Encontrar o modelo de redução salvo
                for transform in self.transformations_applied:
                    if transform['type'] == 'dimensionality_reduction' and 'model' in transform:
                        model = transform['model']
                        break
            
            if model is None:
                self.logger.warning(f"Modelo de redução {method} não encontrado. Criando novo.")
                # Criar e ajustar novo modelo
                if method == 'pca':
                    model = PCA(n_components=n_components, random_state=self.config['random_state'])
                elif method == 'svd':
                    model = TruncatedSVD(n_components=n_components, random_state=self.config['random_state'])
                else:
                    return result
                
                model.fit(X[columns])
            
            # Aplicar transformação de forma otimizada
            reduced_data = model.transform(X[columns])
            
            # Criar DataFrame com as componentes
            component_cols = [f"component_{i+1}" for i in range(reduced_data.shape[1])]
            reduced_df = pd.DataFrame(
                reduced_data,
                columns=component_cols,
                index=X.index
            )
            
            # Adicionar as componentes de uma vez
            result = pd.concat([result, reduced_df], axis=1)
            
        except Exception as e:
            self.logger.warning(f"Erro ao aplicar redução de dimensionalidade: {str(e)}")
        
        return result
    
    def _select_best_features(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Seleciona as melhores features com base na importância ou desempenho.
        
        Args:
            X (pandas.DataFrame): Features
            y (pandas.Series): Target
        """
        self.logger.info("Selecionando as melhores features...")
        
        # Escolher o número de features a selecionar
        if self.config['feature_selection_k'] == 'auto':
            k = min(X.shape[1], int(X.shape[1] * 0.8))  # Heurística: selecionar 80% das features
        else:
            k = min(int(self.config['feature_selection_k']), X.shape[1])
        
        method = self.config['feature_selection_method']
        
        try:
            # Escolher o método de seleção
            if method == 'mutual_info':
                if self.problem_type == 'classification':
                    selector = SelectKBest(mutual_info_classif, k=k)
                else:
                    selector = SelectKBest(mutual_info_regression, k=k)
            elif method == 'anova':
                if self.problem_type == 'classification':
                    selector = SelectKBest(f_classif, k=k)
                else:
                    selector = SelectKBest(f_regression, k=k)
            else:
                self.logger.warning(f"Método de seleção desconhecido: {method}")
                return
            
            # Aplicar seleção
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            selected_cols = X.columns[selected_indices].tolist()
            
            self.best_features = selected_cols
            self.logger.info(f"Selecionadas {len(selected_cols)} melhores features.")
        except Exception as e:
            self.logger.warning(f"Erro ao selecionar melhores features: {str(e)}")
    
    def get_feature_importances(self) -> Dict[str, float]:
        """
        Retorna as importâncias das features.
        
        Returns:
            dict: Dicionário com nomes de features e suas importâncias
        """
        if not self.feature_importances:
            self.logger.warning("Importâncias das features não calculadas. Execute fit() primeiro.")
        
        return self.feature_importances
    
    def get_best_features(self) -> List[str]:
        """
        Retorna a lista das melhores features selecionadas.
        
        Returns:
            list: Lista com nomes das melhores features
        """
        if self.best_features is None:
            self.logger.warning("Melhores features não selecionadas. Execute fit() com um target.")
        
        return self.best_features or []
    
    def save(self, filepath: str) -> None:
        """
        Salva o Explorer em um arquivo para uso posterior.
        
        Args:
            filepath (str): Caminho para salvar o modelo
        """
        self.logger.info(f"Salvando Explorer em {filepath}...")
        
        if not hasattr(self, 'is_fitted') or not self.is_fitted:
            self.logger.warning("Tentativa de salvar um Explorer não ajustado. Recomenda-se executar fit() primeiro.")
        
        try:
            joblib.dump(self, filepath)
            self.logger.info("Explorer salvo com sucesso.")
        except Exception as e:
            self.logger.error(f"Erro ao salvar Explorer: {str(e)}")
            raise
    
    @classmethod
    def load(cls, filepath: str) -> 'Explorer':
        """
        Carrega um Explorer previamente salvo.
        
        Args:
            filepath (str): Caminho para o arquivo salvo
            
        Returns:
            Explorer: Instância carregada do Explorer
        """
        try:
            explorer = joblib.load(filepath)
            
            # Garantir que é uma instância válida de Explorer
            if not isinstance(explorer, cls):
                raise TypeError(f"O arquivo não contém uma instância válida de {cls.__name__}.")
            
            if not hasattr(explorer, 'interaction_features'):
                explorer.interaction_features = []
                
            if hasattr(explorer, 'is_fitted') and explorer.is_fitted:
                explorer.logger.info("Explorer carregado com sucesso. O modelo já está ajustado.")
            else:
                explorer.logger.warning("Explorer carregado, mas não está ajustado. Execute fit() antes de transformar dados.")
                explorer.is_fitted = False
                        
            return explorer
        except Exception as e:
            logging.error(f"Erro ao carregar Explorer de {filepath}: {str(e)}")
            raise


def create_explorer(config: Optional[Dict[str, Any]] = None) -> Explorer:
    """
    Função auxiliar para criar uma instância do Explorer com configurações opcionais.
    
    Args:
        config (dict, opcional): Configurações personalizadas
        
    Returns:
        Explorer: Instância configurada do Explorer
    """
    return Explorer(config)