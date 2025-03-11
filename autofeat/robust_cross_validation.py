import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Union, List, Tuple, Callable
from sklearn.model_selection import (
    cross_val_score, 
    cross_validate,
    KFold, 
    StratifiedKFold, 
    RepeatedKFold, 
    RepeatedStratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
    GroupKFold
)
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score,
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)
from sklearn.base import BaseEstimator
import warnings

class RobustCrossValidator:
    """
    Implementa métodos de validação cruzada robustos e adaptáveis para diferentes
    tipos e tamanhos de datasets, com foco em evitar overfitting.
    """
    def __init__(self, 
                 problem_type: str = 'auto',
                 n_splits: int = 5, 
                 n_repeats: int = 3,
                 random_state: int = 42,
                 cv_strategy: str = 'auto',
                 verbosity: int = 1):
        """
        Inicializa o validador cruzado robusto.
        
        Args:
            problem_type: Tipo de problema ('classification', 'regression', 'timeseries', 'auto')
            n_splits: Número de divisões para validação cruzada
            n_repeats: Número de repetições para métodos repetidos
            random_state: Semente aleatória para reprodutibilidade
            cv_strategy: Estratégia de validação cruzada ('kfold', 'stratified', 'repeated', 
                         'repeated_stratified', 'timeseries', 'group', 'auto')
            verbosity: Nível de verbosidade para logging (0=silencioso, 1=normal, 2=detalhado)
        """
        self.problem_type = problem_type
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.cv_strategy = cv_strategy
        self.verbosity = verbosity
        
        # Configuração de logging
        self.logger = logging.getLogger("AutoFE.RobustCV")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel({0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbosity, logging.INFO))
        
        self.logger.info("RobustCrossValidator inicializado com sucesso")
        
    def _detect_problem_type(self, y: pd.Series) -> str:
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
    
    def _select_cv_strategy(self, X: pd.DataFrame, y: pd.Series, groups=None, time_column=None) -> str:
        """
        Seleciona automaticamente a melhor estratégia de validação cruzada
        com base nas características dos dados.
        
        Args:
            X: Features
            y: Variável alvo
            groups: Grupos para validação cruzada agrupada
            time_column: Coluna temporal para séries temporais
            
        Returns:
            Estratégia de validação cruzada
        """
        n_samples = len(X)
        problem_type = self.problem_type
        
        if problem_type == 'auto':
            problem_type = self._detect_problem_type(y)
        
        # Para datasets muito pequenos (<100 amostras)
        if n_samples < 100:
            if problem_type == 'classification':
                # Verifica se tem classes muito pequenas
                min_class_count = y.value_counts().min()
                if min_class_count < 10:
                    self.logger.info("Dataset pequeno com classes pequenas. Usando StratifiedKFold com n_splits reduzido.")
                    # Reduz o número de splits para garantir representação adequada
                    self.n_splits = min(3, self.n_splits)
                    return 'stratified'
                else:
                    self.logger.info("Dataset pequeno. Usando RepeatedStratifiedKFold para maior robustez.")
                    return 'repeated_stratified'
            else:  # regressão
                self.logger.info("Dataset pequeno para regressão. Usando RepeatedKFold.")
                return 'repeated'
        
        # Para datasets de tamanho médio (100-1000 amostras)
        elif n_samples < 1000:
            if problem_type == 'classification':
                # Verifica balanceamento de classes
                class_counts = y.value_counts(normalize=True)
                is_imbalanced = class_counts.min() < 0.1  # <10% é desbalanceado
                
                if is_imbalanced:
                    self.logger.info("Dataset médio com classes desbalanceadas. Usando RepeatedStratifiedKFold.")
                    return 'repeated_stratified'
                else:
                    self.logger.info("Dataset médio com classes balanceadas. Usando StratifiedKFold.")
                    return 'stratified'
            else:  # regressão
                self.logger.info("Dataset médio para regressão. Usando KFold padrão.")
                return 'kfold'
        
        # Para datasets grandes (>1000 amostras)
        else:
            if problem_type == 'classification':
                self.logger.info("Dataset grande para classificação. Usando StratifiedKFold.")
                return 'stratified'
            else:  # regressão
                self.logger.info("Dataset grande para regressão. Usando KFold.")
                return 'kfold'
        
        # Casos especiais
        if groups is not None:
            self.logger.info("Dados com estrutura de grupos. Usando GroupKFold.")
            return 'group'
            
        if time_column is not None or problem_type == 'timeseries':
            self.logger.info("Dados com estrutura temporal. Usando TimeSeriesSplit.")
            return 'timeseries'
    
    def _create_cv_splitter(self, strategy: str, X: pd.DataFrame, y: pd.Series = None, 
                          groups=None, time_column=None):
        """
        Cria o objeto de divisão de validação cruzada com base na estratégia.
        
        Args:
            strategy: Estratégia de validação cruzada
            X: Features
            y: Variável alvo
            groups: Grupos para validação cruzada agrupada
            time_column: Coluna temporal para séries temporais
            
        Returns:
            Objeto de divisão de validação cruzada
        """
        if strategy == 'auto':
            strategy = self._select_cv_strategy(X, y, groups, time_column)
            
        if strategy == 'kfold':
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            
        elif strategy == 'stratified':
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            
        elif strategy == 'repeated':
            return RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
            
        elif strategy == 'repeated_stratified':
            return RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
            
        elif strategy == 'timeseries':
            # Para séries temporais, usamos divisão temporal
            # max_train_size determina o tamanho máximo do conjunto de treino para evitar
            # usar dados muito antigos em séries temporais
            if time_column is not None and time_column in X.columns:
                # Se fornecido time_column, ordenamos o DataFrame por essa coluna
                X_sorted = X.sort_values(by=time_column)
                # Retorna os índices ordenados para usar na validação cruzada
                sorted_indices = X_sorted.index
                
                # Usando TimeSeriesSplit com os índices ordenados
                ts_split = TimeSeriesSplit(n_splits=self.n_splits)
                
                # Retorna um iterador personalizado que usa os índices ordenados
                def time_series_split():
                    for train_idx, test_idx in ts_split.split(X):
                        yield sorted_indices[train_idx], sorted_indices[test_idx]
                
                return time_series_split()
            else:
                return TimeSeriesSplit(n_splits=self.n_splits)
            
        elif strategy == 'group':
            if groups is None:
                self.logger.warning("GroupKFold selecionado, mas nenhum grupo fornecido. Usando KFold padrão.")
                return KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            return GroupKFold(n_splits=self.n_splits)
            
        else:
            self.logger.warning(f"Estratégia {strategy} não reconhecida. Usando KFold padrão.")
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
    
    def _get_metrics(self, problem_type: str) -> Dict[str, Callable]:
        """
        Retorna as métricas apropriadas para o tipo de problema.
        
        Args:
            problem_type: Tipo de problema ('classification', 'regression')
            
        Returns:
            Dicionário com métricas
        """
        if problem_type == 'classification':
            return {
                'accuracy': 'accuracy',
                'f1_weighted': 'f1_weighted',
                'precision_weighted': 'precision_weighted',
                'recall_weighted': 'recall_weighted',
                'roc_auc_ovr_weighted': 'roc_auc_ovr_weighted'
            }
        else:  # regressão
            return {
                'neg_mean_squared_error': 'neg_mean_squared_error',
                'neg_mean_absolute_error': 'neg_mean_absolute_error',
                'r2': 'r2'
            }
    
    def _sanitize_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Sanitiza os dados para validação cruzada, tratando valores ausentes e inválidos.
        
        Args:
            X: Features
            y: Variável alvo
            
        Returns:
            Tupla (X_sanitized, y_sanitized)
        """
        # Verifica valores ausentes nas features
        if X.isnull().any().any():
            self.logger.warning("Detectados valores ausentes nas features. Preenchendo com medianas/modas.")
            X_sanitized = X.copy()
            
            # Preenche valores ausentes em colunas numéricas com mediana
            for col in X_sanitized.select_dtypes(include=['number']).columns:
                X_sanitized[col] = X_sanitized[col].fillna(X_sanitized[col].median())
            
            # Preenche valores ausentes em colunas categóricas com moda
            for col in X_sanitized.select_dtypes(include=['object', 'category']).columns:
                X_sanitized[col] = X_sanitized[col].fillna(X_sanitized[col].mode()[0] if not X_sanitized[col].mode().empty else "missing")
        else:
            X_sanitized = X
        
        # Verifica valores ausentes na variável alvo
        if y.isnull().any():
            self.logger.warning("Detectados valores ausentes na variável alvo. Removendo linhas correspondentes.")
            valid_mask = ~y.isnull()
            y_sanitized = y[valid_mask]
            X_sanitized = X_sanitized.loc[valid_mask]
        else:
            y_sanitized = y
            
        # Verifica valores infinitos nas features
        if (X_sanitized.select_dtypes(include=['number']) == np.inf).any().any() or \
           (X_sanitized.select_dtypes(include=['number']) == -np.inf).any().any():
            self.logger.warning("Detectados valores infinitos nas features. Substituindo por valores extremos.")
            X_sanitized = X_sanitized.replace([np.inf, -np.inf], [1e9, -1e9])
        
        return X_sanitized, y_sanitized
    
    def cross_validate(self, 
                      model: BaseEstimator, 
                      X: pd.DataFrame, 
                      y: pd.Series,
                      groups=None,
                      time_column=None,
                      return_train_score: bool = False,
                      scoring=None,
                      fit_params=None,
                      error_score='raise') -> Dict[str, np.ndarray]:
        """
        Realiza validação cruzada robusta com tratamento de erros e adaptações.
        
        Args:
            model: Modelo a ser avaliado
            X: Features
            y: Variável alvo
            groups: Grupos para validação cruzada agrupada
            time_column: Coluna temporal para séries temporais
            return_train_score: Se deve calcular métricas no conjunto de treino
            scoring: Métricas específicas para avaliação (None=automático)
            fit_params: Parâmetros adicionais para model.fit()
            error_score: Tratamento de erros ('raise'=lança exceção, valor numérico=uso como score em erro)
            
        Returns:
            Dicionário com os resultados da validação cruzada
        """
        self.logger.info(f"Iniciando validação cruzada robusta com modelo {model.__class__.__name__}")
        
        # Sanitiza os dados
        X_cv, y_cv = self._sanitize_data(X, y)
        
        # Detecta o tipo de problema
        problem_type = self.problem_type
        if problem_type == 'auto':
            problem_type = self._detect_problem_type(y_cv)
        
        # Ajusta parâmetros com base no tamanho do dataset
        n_samples = len(X_cv)
        if n_samples < 100:
            self.logger.info(f"Dataset pequeno ({n_samples} amostras). Ajustando parâmetros de validação.")
            
            # Para datasets muito pequenos, reduz número de folds
            original_n_splits = self.n_splits
            self.n_splits = min(3, self.n_splits)
            
            # Aumenta repetições para maior robustez
            original_n_repeats = self.n_repeats
            self.n_repeats = max(5, self.n_repeats)
            
            # Restaura configurações originais ao final
            cleanup_needed = True
        else:
            cleanup_needed = False
            
        # Seleciona estratégia de CV
        if self.cv_strategy == 'auto':
            strategy = self._select_cv_strategy(X_cv, y_cv, groups, time_column)
        else:
            strategy = self.cv_strategy
            
        # Cria divisor de CV
        cv_splitter = self._create_cv_splitter(strategy, X_cv, y_cv, groups, time_column)
        
        # Define métricas se não especificadas
        if scoring is None:
            scoring = self._get_metrics(problem_type)
            
        try:
            # Executa validação cruzada
            self.logger.info(f"Executando validação cruzada com estratégia {strategy}, {self.n_splits} splits, {self.n_repeats} repetições")
            
            with warnings.catch_warnings():
                # Silencia avisos durante a validação cruzada
                warnings.simplefilter("ignore")
                
                cv_results = cross_validate(
                    model, 
                    X_cv, 
                    y_cv,
                    groups=groups,
                    cv=cv_splitter,
                    scoring=scoring,
                    return_train_score=return_train_score,
                    fit_params=fit_params,
                    error_score=error_score,
                    n_jobs=-1  # Usa todos os núcleos disponíveis
                )
            
            # Calcula métricas adicionais
            # Média e desvio padrão para cada métrica
            results = {}
            
            for metric, values in cv_results.items():
                results[f"{metric}_mean"] = np.mean(values)
                results[f"{metric}_std"] = np.std(values)
                results[f"{metric}_values"] = values
            
            # Adiciona informações sobre a validação cruzada
            results['cv_strategy'] = strategy
            results['n_splits'] = self.n_splits
            if strategy in ['repeated', 'repeated_stratified']:
                results['n_repeats'] = self.n_repeats
            
            # Calcula confiabilidade dos resultados (maior variância = menor confiabilidade)
            test_metrics = [key for key in cv_results.keys() if key.startswith('test_')]
            if test_metrics:
                # Calcula o coeficiente de variação médio (desvio padrão / média)
                cv_values = []
                for metric in test_metrics:
                    mean_val = np.mean(cv_results[metric])
                    std_val = np.std(cv_results[metric])
                    # Evita divisão por zero
                    if mean_val != 0:
                        cv_values.append(abs(std_val / mean_val))
                    else:
                        cv_values.append(0)
                
                results['reliability_score'] = 1.0 - min(1.0, sum(cv_values) / len(cv_values))
            
            self.logger.info(f"Validação cruzada concluída com sucesso. Confiabilidade: {results.get('reliability_score', 'N/A'):.4f}")
            
            # Dados sobre possível overfitting
            if return_train_score:
                # Compara scores de treino e teste para métricas comuns
                overfitting_scores = {}
                for metric in test_metrics:
                    if f"train_{metric[5:]}" in cv_results:
                        train_mean = np.mean(cv_results[f"train_{metric[5:]}"])
                        test_mean = np.mean(cv_results[metric])
                        # Razão entre score de treino e teste (>1.2 pode indicar overfitting)
                        if test_mean != 0:
                            ratio = train_mean / test_mean
                            overfitting_scores[f"{metric[5:]}_overfit_ratio"] = ratio
                
                results['overfitting_scores'] = overfitting_scores
                
                # Score global de overfitting (média das razões)
                if overfitting_scores:
                    avg_ratio = np.mean(list(overfitting_scores.values()))
                    # Normaliza para [0, 1] onde 0=sem overfitting, 1=overfitting severo
                    # Considera >2.0 como overfitting severo
                    overfitting_grade = min(1.0, max(0.0, (avg_ratio - 1.0) / 1.0))
                    results['overfitting_grade'] = overfitting_grade
                    
                    if overfitting_grade > 0.3:
                        self.logger.warning(f"Possível overfitting detectado (grau: {overfitting_grade:.2f})")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Erro durante validação cruzada: {str(e)}")
            # Retorna um resultado básico com erro
            return {
                'error': str(e),
                'cv_strategy': strategy,
                'n_splits': self.n_splits,
                'success': False
            }
        
        finally:
            # Restaura configurações originais se foram alteradas
            if cleanup_needed:
                self.n_splits = original_n_splits
                self.n_repeats = original_n_repeats
    
    def nested_cross_validate(self, 
                             model_factory: Callable[[], BaseEstimator],
                             param_grid: Dict,
                             X: pd.DataFrame, 
                             y: pd.Series,
                             groups=None,
                             time_column=None,
                             n_outer_splits: int = 3,
                             scoring=None) -> Dict:
        """
        Realiza validação cruzada aninhada para avaliação mais precisa e imparcial.
        
        Args:
            model_factory: Função que retorna uma nova instância do modelo
            param_grid: Grade de parâmetros para otimização
            X: Features
            y: Variável alvo
            groups: Grupos para validação cruzada agrupada
            time_column: Coluna temporal para séries temporais
            n_outer_splits: Número de divisões para o loop externo
            scoring: Métricas específicas para avaliação (None=automático)
            
        Returns:
            Dicionário com os resultados da validação cruzada aninhada
        """
        from sklearn.model_selection import GridSearchCV
        
        self.logger.info("Iniciando validação cruzada aninhada")
        
        # Sanitiza os dados
        X_cv, y_cv = self._sanitize_data(X, y)
        
        # Detecta o tipo de problema
        problem_type = self.problem_type
        if problem_type == 'auto':
            problem_type = self._detect_problem_type(y_cv)
        
        # Define métricas se não especificadas
        if scoring is None:
            # Pega a primeira métrica do dicionário para otimização
            metrics_dict = self._get_metrics(problem_type)
            scoring = next(iter(metrics_dict.values()))
        
        # Seleciona estratégia para loop externo e interno
        outer_strategy = self._select_cv_strategy(X_cv, y_cv, groups, time_column)
        
        # Se for uma estratégia repetida, usa a versão não repetida para o loop externo
        if outer_strategy == 'repeated_stratified':
            outer_strategy = 'stratified'
        elif outer_strategy == 'repeated':
            outer_strategy = 'kfold'
        
        # Cria divisor de CV para loop externo
        # Usa menos splits para o loop externo para reduzir tempo computacional
        original_n_splits = self.n_splits
        self.n_splits = n_outer_splits
        outer_cv = self._create_cv_splitter(outer_strategy, X_cv, y_cv, groups, time_column)
        
        # Restaura configuração original
        self.n_splits = original_n_splits
        
        # Armazena scores para cada fold externo
        outer_scores = []
        best_params_list = []
        
        # Inicializa resultados
        results = {
            'outer_scores': [],
            'best_params': [],
            'outer_cv_strategy': outer_strategy,
            'inner_cv_strategy': self.cv_strategy,
            'n_outer_splits': n_outer_splits,
            'n_inner_splits': self.n_splits
        }
        
        try:
            # Loop externo
            fold_idx = 0
            for train_idx, test_idx in outer_cv.split(X_cv, y_cv, groups):
                fold_idx += 1
                self.logger.info(f"Executando fold externo {fold_idx}/{n_outer_splits}")
                
                # Divide em conjuntos de treino e teste para este fold
                X_train, X_test = X_cv.iloc[train_idx], X_cv.iloc[test_idx]
                y_train, y_test = y_cv.iloc[train_idx], y_cv.iloc[test_idx]
                
                # Seletor para loop interno
                inner_strategy = self.cv_strategy
                if inner_strategy == 'auto':
                    inner_strategy = self._select_cv_strategy(X_train, y_train)
                
                inner_cv = self._create_cv_splitter(inner_strategy, X_train, y_train)
                
                # Configura pesquisa de hiperparâmetros
                estimator = model_factory()
                
                # GridSearchCV para otimização de hiperparâmetros no loop interno
                grid_search = GridSearchCV(
                    estimator,
                    param_grid,
                    cv=inner_cv,
                    scoring=scoring,
                    n_jobs=-1,
                    refit=True
                )
                
                # Otimiza hiperparâmetros
                grid_search.fit(X_train, y_train)
                
                # Avalia no conjunto de teste
                best_model = grid_search.best_estimator_
                
                # Armazena resultados
                if problem_type == 'classification':
                    y_pred = best_model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # Probabilidades para ROC-AUC se o modelo suportar
                    roc_auc = None
                    if hasattr(best_model, 'predict_proba'):
                        try:
                            if len(np.unique(y_test)) > 2:  # Multiclasse
                                # One-vs-Rest AUC
                                y_proba = best_model.predict_proba(X_test)
                                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                            else:  # Binário
                                y_proba = best_model.predict_proba(X_test)[:, 1]
                                roc_auc = roc_auc_score(y_test, y_proba)
                        except:
                            pass
                    
                    fold_score = {
                        'accuracy': accuracy,
                        'f1_weighted': f1,
                        'best_params': grid_search.best_params_
                    }
                    
                    if roc_auc is not None:
                        fold_score['roc_auc'] = roc_auc
                else:  # regressão
                    y_pred = best_model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    fold_score = {
                        'neg_mean_squared_error': -mse,
                        'neg_mean_absolute_error': -mae,
                        'r2': r2,
                        'best_params': grid_search.best_params_
                    }
                
                outer_scores.append(fold_score)
                best_params_list.append(grid_search.best_params_)
                
                self.logger.info(f"Concluído fold {fold_idx}. Melhores parâmetros: {grid_search.best_params_}")
            
            # Calcula médias e desvios padrão
            final_scores = {}
            for metric in outer_scores[0].keys():
                if metric == 'best_params':
                    continue
                
                values = [score[metric] for score in outer_scores]
                final_scores[f"{metric}_mean"] = np.mean(values)
                final_scores[f"{metric}_std"] = np.std(values)
                final_scores[f"{metric}_values"] = values
            
            # Identifica os parâmetros mais frequentes
            from collections import Counter
            
            # Conta ocorrências de cada configuração de parâmetros
            param_occurrences = Counter([str(params) for params in best_params_list])
            most_common_params_str, count = param_occurrences.most_common(1)[0]
            
            # Converte de volta para dicionário (eval é seguro aqui porque sabemos que é um dicionário serializado)
            import ast
            most_common_params = next((params for params in best_params_list 
                                    if str(params) == most_common_params_str), {})
            
            # Adiciona aos resultados
            results.update(final_scores)
            results['best_params'] = most_common_params
            results['params_consensus'] = count / len(best_params_list)
            results['all_best_params'] = best_params_list
            
            self.logger.info(f"Validação cruzada aninhada concluída. Consenso de parâmetros: {results['params_consensus']:.2f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erro durante validação cruzada aninhada: {str(e)}")
            # Retorna um resultado básico com erro
            return {
                'error': str(e),
                'outer_cv_strategy': outer_strategy,
                'inner_cv_strategy': self.cv_strategy,
                'n_outer_splits': n_outer_splits,
                'success': False
            }
    
    def adaptive_train_test_split(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                                stratify: bool = True, groups=None, time_column=None) -> Tuple:
        """
        Realiza divisão treino/teste adaptativa com base nas características dos dados.
        
        Args:
            X: Features
            y: Variável alvo
            test_size: Proporção do conjunto de teste (adaptado automaticamente se necessário)
            stratify: Se deve estratificar por classes (apenas para classificação)
            groups: Grupos para validação cruzada agrupada
            time_column: Coluna temporal para séries temporais
            
        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
        # Sanitiza os dados
        X_clean, y_clean = self._sanitize_data(X, y)
        
        # Detecta o tipo de problema
        problem_type = self.problem_type
        if problem_type == 'auto':
            problem_type = self._detect_problem_type(y_clean)
        
        # Ajusta o tamanho do teste com base no tamanho do dataset
        n_samples = len(X_clean)
        adjusted_test_size = test_size
        
        # Para datasets muito pequenos, aumenta proporção do treino
        if n_samples < 100:
            # Mantém ao menos 80% para treino em datasets pequenos
            adjusted_test_size = min(test_size, 0.2)
            self.logger.info(f"Dataset pequeno ({n_samples} amostras). Ajustando test_size para {adjusted_test_size}")
        elif n_samples < 1000:
            # Para datasets médios, usa o padrão
            adjusted_test_size = test_size
        else:
            # Para datasets grandes, pode usar mais dados para teste
            adjusted_test_size = max(test_size, 0.2)
        
        # Casos especiais - dados temporais
        if time_column is not None and time_column in X_clean.columns:
            self.logger.info(f"Realizando divisão temporal usando coluna {time_column}")
            
            # Ordena por tempo
            sorted_indices = X_clean[time_column].argsort()
            X_sorted = X_clean.iloc[sorted_indices]
            y_sorted = y_clean.iloc[sorted_indices]
            
            # Usa os últimos N% para teste
            split_idx = int(len(X_sorted) * (1 - adjusted_test_size))
            
            X_train = X_sorted.iloc[:split_idx]
            X_test = X_sorted.iloc[split_idx:]
            y_train = y_sorted.iloc[:split_idx]
            y_test = y_sorted.iloc[split_idx:]
            
            return X_train, X_test, y_train, y_test
        
        # Casos especiais - dados agrupados
        elif groups is not None:
            from sklearn.model_selection import GroupShuffleSplit
            
            self.logger.info("Realizando divisão preservando grupos")
            
            # Usando GroupShuffleSplit para preservar grupos
            gss = GroupShuffleSplit(n_splits=1, test_size=adjusted_test_size, random_state=self.random_state)
            train_idx, test_idx = next(gss.split(X_clean, y_clean, groups))
            
            X_train = X_clean.iloc[train_idx]
            X_test = X_clean.iloc[test_idx]
            y_train = y_clean.iloc[train_idx]
            y_test = y_clean.iloc[test_idx]
            
            return X_train, X_test, y_train, y_test
        
        # Caso padrão - estratificado ou simples
        else:
            should_stratify = stratify and problem_type == 'classification'
            
            # Para classificação, verifica se é possível estratificar
            if should_stratify:
                # Verifica se todas as classes têm amostras suficientes para estratificação
                class_counts = y_clean.value_counts()
                min_class_count = class_counts.min()
                
                # Precisa de mais de 1 exemplo por classe no conjunto de teste
                if min_class_count * adjusted_test_size < 2:
                    should_stratify = False
                    self.logger.warning(f"Classe minoritária tem apenas {min_class_count} exemplos. Desabilitando estratificação.")
            
            strat_y = y_clean if should_stratify else None
            
            return train_test_split(
                X_clean, y_clean, 
                test_size=adjusted_test_size, 
                stratify=strat_y, 
                random_state=self.random_state
            )
    
    def bootstrap_evaluate(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series, 
                         n_iterations: int = 100, sample_size: float = 0.8,
                         stratify: bool = True) -> Dict:
        """
        Realiza avaliação bootstrap para uma estimativa mais robusta do desempenho.
        
        Args:
            model: Modelo a ser avaliado
            X: Features
            y: Variável alvo
            n_iterations: Número de iterações bootstrap
            sample_size: Tamanho de cada amostra bootstrap (proporção do dataset original)
            stratify: Se deve estratificar por classes (para classificação)
            
        Returns:
            Dicionário com resultados da avaliação bootstrap
        """
        self.logger.info(f"Iniciando avaliação bootstrap com {n_iterations} iterações")
        
        # Sanitiza os dados
        X_clean, y_clean = self._sanitize_data(X, y)
        
        # Detecta o tipo de problema
        problem_type = self.problem_type
        if problem_type == 'auto':
            problem_type = self._detect_problem_type(y_clean)
        
        # Inicializa coletores de métricas
        bootstrap_scores = {}
        
        # Define as métricas a serem coletadas
        if problem_type == 'classification':
            metrics = {
                'accuracy': accuracy_score,
                'f1_weighted': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
                'precision_weighted': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
                'recall_weighted': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted')
            }
            
            # Adiciona ROC-AUC se o modelo suporta predict_proba
            if hasattr(model, 'predict_proba'):
                if len(np.unique(y_clean)) > 2:  # Multiclasse
                    metrics['roc_auc_ovr'] = lambda y_true, y_pred_proba: roc_auc_score(
                        y_true, y_pred_proba, multi_class='ovr', average='weighted'
                    )
                else:  # Binário
                    metrics['roc_auc'] = roc_auc_score
            
        else:  # regressão
            metrics = {
                'neg_mean_squared_error': lambda y_true, y_pred: -mean_squared_error(y_true, y_pred),
                'neg_mean_absolute_error': lambda y_true, y_pred: -mean_absolute_error(y_true, y_pred),
                'r2': r2_score
            }
        
        # Inicializa arrays para cada métrica
        for metric_name in metrics.keys():
            bootstrap_scores[metric_name] = []
        
        # Executa bootstrap
        for i in range(n_iterations):
            try:
                # Amostragem bootstrap com reposição
                n_samples = int(len(X_clean) * sample_size)
                
                if stratify and problem_type == 'classification':
                    # Amostragem estratificada
                    train_indices = []
                    for cls in np.unique(y_clean):
                        cls_indices = np.where(y_clean == cls)[0]
                        # Amostra com reposição dentro de cada classe
                        cls_sample = np.random.choice(
                            cls_indices, 
                            size=int(len(cls_indices) * sample_size),
                            replace=True
                        )
                        train_indices.extend(cls_sample)
                else:
                    # Amostragem aleatória
                    train_indices = np.random.choice(
                        len(X_clean), 
                        size=n_samples,
                        replace=True
                    )
                
                # Cria a amostra bootstrap
                X_boot = X_clean.iloc[train_indices]
                y_boot = y_clean.iloc[train_indices]
                
                # Ajusta o modelo
                model.fit(X_boot, y_boot)
                
                # Out-of-bag sample (exemplos não utilizados no treino)
                oob_indices = np.setdiff1d(np.arange(len(X_clean)), np.unique(train_indices))
                
                if len(oob_indices) > 0:
                    X_oob = X_clean.iloc[oob_indices]
                    y_oob = y_clean.iloc[oob_indices]
                    
                    # Faz predições
                    y_pred = model.predict(X_oob)
                    
                    # Coleta métricas
                    for metric_name, metric_func in metrics.items():
                        if 'roc_auc' in metric_name and hasattr(model, 'predict_proba'):
                            # Para ROC-AUC, precisa de probabilidades
                            y_pred_proba = model.predict_proba(X_oob)
                            if len(np.unique(y_oob)) <= 2:  # Binário
                                # Extrai probabilidade da classe positiva
                                y_pred_proba = y_pred_proba[:, 1]
                            score = metric_func(y_oob, y_pred_proba)
                        else:
                            score = metric_func(y_oob, y_pred)
                        
                        bootstrap_scores[metric_name].append(score)
                
                if (i + 1) % 10 == 0 or i == 0 or i == n_iterations - 1:
                    self.logger.info(f"Concluída iteração bootstrap {i+1}/{n_iterations}")
                    
            except Exception as e:
                self.logger.warning(f"Erro na iteração bootstrap {i+1}: {str(e)}")
                continue
        
        # Calcula estatísticas para cada métrica
        results = {}
        
        for metric_name, scores in bootstrap_scores.items():
            if len(scores) > 0:
                results[f"{metric_name}_mean"] = np.mean(scores)
                results[f"{metric_name}_std"] = np.std(scores)
                results[f"{metric_name}_median"] = np.median(scores)
                
                # Intervalo de confiança de 95%
                alpha = 0.05
                lower_percentile = alpha / 2 * 100
                upper_percentile = (1 - alpha / 2) * 100
                results[f"{metric_name}_ci_lower"] = np.percentile(scores, lower_percentile)
                results[f"{metric_name}_ci_upper"] = np.percentile(scores, upper_percentile)
                
                # Armazena todos os valores
                results[f"{metric_name}_values"] = scores
        
        # Adiciona informações sobre a avaliação
        results['n_iterations'] = len(bootstrap_scores[next(iter(bootstrap_scores))])
        results['sample_size'] = sample_size
        results['success_rate'] = results['n_iterations'] / n_iterations
        
        self.logger.info(f"Avaliação bootstrap concluída com {results['n_iterations']} iterações bem-sucedidas")
        
        return results
    
    def permutation_importance(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                             n_repeats: int = 10, scoring=None, n_jobs: int = -1) -> Dict:
        """
        Calcula a importância de cada feature por permutação.
        
        Args:
            model: Modelo treinado
            X: Features
            y: Variável alvo
            n_repeats: Número de repetições para estimativa mais robusta
            scoring: Função de pontuação
            n_jobs: Número de jobs paralelos
            
        Returns:
            Dicionário com importância de cada feature
        """
        from sklearn.inspection import permutation_importance
        
        self.logger.info("Calculando importância de features por permutação")
        
        # Sanitiza os dados
        X_clean, y_clean = self._sanitize_data(X, y)
        
        # Detecta o tipo de problema
        problem_type = self.problem_type
        if problem_type == 'auto':
            problem_type = self._detect_problem_type(y_clean)
        
        # Define scoring se não especificado
        if scoring is None:
            if problem_type == 'classification':
                if hasattr(model, 'predict_proba'):
                    scoring = 'roc_auc_ovr' if len(np.unique(y_clean)) > 2 else 'roc_auc'
                else:
                    scoring = 'f1_weighted'
            else:  # regressão
                scoring = 'r2'
        
        try:
            # Calcula importância por permutação
            result = permutation_importance(
                model, X_clean, y_clean, 
                n_repeats=n_repeats,
                random_state=self.random_state,
                scoring=scoring,
                n_jobs=n_jobs
            )
            
            # Formata resultados
            importances = {}
            for i, col in enumerate(X_clean.columns):
                importances[col] = {
                    'mean_importance': result.importances_mean[i],
                    'std_importance': result.importances_std[i],
                    'raw_values': result.importances[i, :].tolist()
                }
            
            # Ordena features por importância
            sorted_features = sorted(
                importances.keys(), 
                key=lambda x: importances[x]['mean_importance'],
                reverse=True
            )
            
            # Calcula importância normalizada (soma = 1)
            means = np.array([importances[col]['mean_importance'] for col in X_clean.columns])
            
            # Normaliza apenas valores positivos
            positive_sum = np.sum(np.maximum(0, means))
            if positive_sum > 0:
                for col in importances:
                    importances[col]['normalized_importance'] = max(0, importances[col]['mean_importance']) / positive_sum
            
            self.logger.info(f"Importância por permutação concluída. Feature mais importante: {sorted_features[0]}")
            
            return {
                'importances': importances,
                'sorted_features': sorted_features,
                'scoring': scoring,
                'n_repeats': n_repeats
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular importância por permutação: {str(e)}")
            return {
                'error': str(e),
                'success': False
            }
    
    def learning_curve_analysis(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                              train_sizes: np.ndarray = None, cv: int = 5, 
                              scoring=None) -> Dict:
        """
        Analisa a curva de aprendizado para verificar underfitting/overfitting e 
        estimar a necessidade de mais dados.
        
        Args:
            model: Modelo a ser avaliado
            X: Features
            y: Variável alvo
            train_sizes: Tamanhos de treino a serem avaliados (proporção do dataset)
            cv: Número de folds ou objeto de CV
            scoring: Métrica para avaliação
            
        Returns:
            Dicionário com resultados da análise
        """
        from sklearn.model_selection import learning_curve
        
        self.logger.info("Iniciando análise de curva de aprendizado")
        
        # Sanitiza os dados
        X_clean, y_clean = self._sanitize_data(X, y)
        
        # Detecta o tipo de problema
        problem_type = self.problem_type
        if problem_type == 'auto':
            problem_type = self._detect_problem_type(y_clean)
        
        # Define train_sizes se não especificado
        if train_sizes is None:
            # Adapta ao tamanho do dataset
            n_samples = len(X_clean)
            
            if n_samples < 100:
                # Para datasets pequenos, usa poucas frações
                train_sizes = np.linspace(0.3, 1.0, 3)
            elif n_samples < 1000:
                # Para datasets médios
                train_sizes = np.linspace(0.1, 1.0, 5)
            else:
                # Para datasets grandes
                train_sizes = np.linspace(0.01, 1.0, 10)
        
        # Define scoring se não especificado
        if scoring is None:
            if problem_type == 'classification':
                scoring = 'f1_weighted'
            else:  # regressão
                scoring = 'r2'
        
        # Define CV
        if isinstance(cv, int):
            if problem_type == 'classification':
                # Estratificado para classificação
                cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            else:
                # KFold padrão para regressão
                cv = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        try:
            # Calcula curva de aprendizado
            train_sizes_abs, train_scores, test_scores = learning_curve(
                model, X_clean, y_clean,
                train_sizes=train_sizes,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                random_state=self.random_state
            )
            
            # Calcula estatísticas
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            # Armazena resultados
            results = {
                'train_sizes_abs': train_sizes_abs.tolist(),
                'train_sizes_rel': (train_sizes_abs / len(X_clean)).tolist(),
                'train_mean': train_mean.tolist(),
                'train_std': train_std.tolist(),
                'test_mean': test_mean.tolist(),
                'test_std': test_std.tolist(),
                'scoring': scoring
            }
            
            # Analisa a curva para diagnóstico
            # Verifica overfitting - diferença entre treino e teste na maior fração
            max_diff = train_mean[-1] - test_mean[-1]
            
            # Verifica se o score de teste ainda está melhorando na maior fração
            # Calcula a inclinação das últimas 30% das frações
            n_points = len(test_mean)
            cutoff = max(1, int(n_points * 0.7))
            last_points = test_mean[cutoff:]
            
            if len(last_points) >= 2:
                # Inclinação da reta entre o primeiro e último ponto
                slope = (last_points[-1] - last_points[0]) / (len(last_points) - 1)
            else:
                slope = 0
            
            # Determina o diagnóstico
            if max_diff > 0.1:  # Diferença significativa entre treino e teste
                diagnosis = 'overfitting'
            elif train_mean[-1] < 0.7 and test_mean[-1] < 0.7:  # Scores baixos tanto em treino quanto teste
                diagnosis = 'underfitting'
            elif slope > 0.01:  # Score de teste ainda melhorando significativamente
                diagnosis = 'need_more_data'
            else:  # Bom desempenho, curva estabilizada
                diagnosis = 'good_fit'
            
            # Adiciona diagnóstico e recomendações
            results['diagnosis'] = diagnosis
            
            if diagnosis == 'overfitting':
                results['recommendation'] = (
                    "O modelo apresenta sinais de overfitting. "
                    "Recomendações: considere regularização mais forte, redução da complexidade "
                    "do modelo ou técnicas como early stopping."
                )
            elif diagnosis == 'underfitting':
                results['recommendation'] = (
                    "O modelo apresenta sinais de underfitting. "
                    "Recomendações: considere modelos mais complexos, redução da regularização, "
                    "ou geração de features mais informativas."
                )
            elif diagnosis == 'need_more_data':
                results['recommendation'] = (
                    "A curva de aprendizado ainda está crescendo, indicando que mais dados "
                    "provavelmente melhorariam o desempenho do modelo. "
                    "Considere coletar mais dados ou usar técnicas de data augmentation."
                )
            else:  # good_fit
                results['recommendation'] = (
                    "O modelo parece bem ajustado, com boa generalização. "
                    "A curva de aprendizado está estabilizada, sugerindo que "
                    "o tamanho atual do dataset é adequado."
                )
            
            self.logger.info(f"Análise de curva de aprendizado concluída. Diagnóstico: {diagnosis}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erro na análise de curva de aprendizado: {str(e)}")
            return {
                'error': str(e),
                'success': False
            }
    
    def evaluate_with_confidence(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                               method: str = 'auto') -> Dict:
        """
        Avalia o modelo com estimativas de confiança, escolhendo a técnica mais 
        apropriada para o dataset.
        
        Args:
            model: Modelo a ser avaliado
            X: Features
            y: Variável alvo
            method: Método de avaliação ('cv', 'bootstrap', 'nested_cv', 'auto')
            
        Returns:
            Dicionário com resultados da avaliação e intervalos de confiança
        """
        # Sanitiza os dados
        X_clean, y_clean = self._sanitize_data(X, y)
        
        # Detecta o tipo de problema
        problem_type = self.problem_type
        if problem_type == 'auto':
            problem_type = self._detect_problem_type(y_clean)
        
        # Determina o melhor método para o dataset
        if method == 'auto':
            n_samples = len(X_clean)
            
            if n_samples < 100:
                # Para datasets muito pequenos, bootstrap é mais adequado
                method = 'bootstrap'
                self.logger.info(f"Dataset pequeno ({n_samples} amostras). Selecionando método 'bootstrap'.")
            elif n_samples < 500:
                # Para datasets pequenos a médios, validação cruzada repetida
                method = 'cv'
                # Usa estratégia repetida para maior robustez
                self.cv_strategy = 'repeated_stratified' if problem_type == 'classification' else 'repeated'
                self.logger.info(f"Dataset médio ({n_samples} amostras). Selecionando método 'cv' com estratégia {self.cv_strategy}.")
            else:
                # Para datasets grandes, validação cruzada padrão
                method = 'cv'
                self.logger.info(f"Dataset grande ({n_samples} amostras). Selecionando método 'cv'.")
        
        # Executa a avaliação com o método escolhido
        if method == 'bootstrap':
            self.logger.info("Executando avaliação bootstrap")
            return self.bootstrap_evaluate(model, X_clean, y_clean, n_iterations=100)
            
        elif method == 'nested_cv':
            self.logger.info("Executando validação cruzada aninhada")
            # Para validação aninhada, precisamos de uma factory function e um grid
            # Como aqui estamos avaliando um modelo já definido, usamos um grid mínimo
            model_factory = lambda: model.__class__(**model.get_params())
            param_grid = {'dummy': [1]}  # Grid trivial apenas para satisfazer a API
            
            return self.nested_cross_validate(
                model_factory=model_factory,
                param_grid=param_grid,
                X=X_clean, 
                y=y_clean
            )
            
        else:  # 'cv' (padrão)
            self.logger.info(f"Executando validação cruzada com estratégia {self.cv_strategy}")
            return self.cross_validate(model, X_clean, y_clean, return_train_score=True)
    
    def detect_drift(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                   return_feature_drifts: bool = True) -> Dict:
        """
        Detecta drift estatístico entre conjuntos de treino e teste.
        
        Args:
            X_train: Features do conjunto de treino
            X_test: Features do conjunto de teste
            return_feature_drifts: Se deve retornar análise por feature
            
        Returns:
            Dicionário com resultados da detecção de drift
        """
        self.logger.info("Iniciando detecção de drift entre conjuntos de treino e teste")
        
        # Sanitiza os dados
        X_train_clean = X_train.copy()
        X_test_clean = X_test.copy()
        
        # Garante as mesmas colunas
        common_cols = list(set(X_train_clean.columns) & set(X_test_clean.columns))
        if len(common_cols) < len(X_train_clean.columns):
            self.logger.warning(f"Algumas colunas diferem entre treino e teste. Usando apenas as {len(common_cols)} colunas em comum.")
        
        X_train_clean = X_train_clean[common_cols]
        X_test_clean = X_test_clean[common_cols]
        
        # Resultados globais
        results = {
            'drift_detected': False,
            'drift_score': 0.0,
            'feature_drifts': {},
            'n_features': len(common_cols),
            'n_drifting_features': 0
        }
        
        try:
            # Lista para armazenar scores de drift por feature
            feature_drift_scores = []
            drifting_features = []
            
            # Avalia cada feature numericamente
            for col in common_cols:
                # Pula colunas não numéricas para esta análise
                if X_train_clean[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                    continue
                
                # Extrai valores não nulos
                train_values = X_train_clean[col].dropna().values
                test_values = X_test_clean[col].dropna().values
                
                if len(train_values) < 5 or len(test_values) < 5:
                    # Poucos dados para análise estatística confiável
                    continue
                
                # Testes estatísticos para detectar drift
                try:
                    # Teste de Kolmogorov-Smirnov para distribuições
                    from scipy.stats import ks_2samp
                    ks_stat, ks_pval = ks_2samp(train_values, test_values)
                    
                    # Diferença nas médias, normalizada pela desvio padrão combinado
                    mean_diff = abs(np.mean(train_values) - np.mean(test_values))
                    pooled_std = np.sqrt((np.var(train_values) + np.var(test_values)) / 2)
                    normalized_mean_diff = mean_diff / pooled_std if pooled_std > 0 else 0
                    
                    # Diferença na distribuição (percentis)
                    train_percentiles = np.percentile(train_values, [25, 50, 75])
                    test_percentiles = np.percentile(test_values, [25, 50, 75])
                    percentile_diffs = np.abs(train_percentiles - test_percentiles)
                    max_percentile_diff = np.max(percentile_diffs) / pooled_std if pooled_std > 0 else 0
                    
                    # Combina métricas em um score de drift
                    drift_score = 0.5 * ks_stat + 0.3 * normalized_mean_diff + 0.2 * max_percentile_diff
                    
                    # Determina se há drift significativo
                    has_drift = ks_pval < 0.05 and drift_score > 0.3
                    
                    # Armazena resultados por feature
                    if return_feature_drifts:
                        results['feature_drifts'][col] = {
                            'drift_detected': has_drift,
                            'drift_score': drift_score,
                            'ks_stat': ks_stat,
                            'ks_pval': ks_pval,
                            'mean_diff': mean_diff,
                            'normalized_mean_diff': normalized_mean_diff,
                            'percentile_diffs': percentile_diffs.tolist()
                        }
                    
                    # Coleta para score global
                    feature_drift_scores.append(drift_score)
                    if has_drift:
                        drifting_features.append(col)
                
                except Exception as e:
                    self.logger.warning(f"Erro ao analisar drift para feature {col}: {str(e)}")
                    continue
            
            # Resultado global
            if feature_drift_scores:
                # Score global como média ponderada dos top 30% piores features
                if len(feature_drift_scores) > 0:
                    sorted_scores = sorted(feature_drift_scores, reverse=True)
                    top_n = max(1, int(len(sorted_scores) * 0.3))
                    global_drift_score = np.mean(sorted_scores[:top_n])
                    
                    # Define drift global com base no número de features afetadas
                    results['drift_score'] = global_drift_score
                    results['n_drifting_features'] = len(drifting_features)
                    results['drift_detected'] = len(drifting_features) > len(common_cols) * 0.1
                    results['drifting_features'] = drifting_features
            
            self.logger.info(f"Detecção de drift concluída. Drift detectado: {results['drift_detected']}, "
                          f"Score: {results['drift_score']:.4f}, Features com drift: {results['n_drifting_features']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erro na detecção de drift: {str(e)}")
            return {
                'error': str(e),
                'success': False
            }
    
    def calibration_analysis(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series, 
                           cv: int = 5) -> Dict:
        """
        Analisa a calibração de probabilidades do modelo para classificação.
        
        Args:
            model: Modelo a ser avaliado
            X: Features
            y: Variável alvo
            cv: Número de folds para validação cruzada
            
        Returns:
            Dicionário com resultados da análise de calibração
        """
        from sklearn.calibration import calibration_curve
        
        self.logger.info("Iniciando análise de calibração de probabilidades")
        
        # Sanitiza os dados
        X_clean, y_clean = self._sanitize_data(X, y)
        
        # Verifica se é um problema de classificação
        problem_type = self.problem_type
        if problem_type == 'auto':
            problem_type = self._detect_problem_type(y_clean)
        
        if problem_type != 'classification':
            return {'error': 'Análise de calibração só é aplicável a problemas de classificação',
                   'success': False}
        
        # Verifica se o modelo possui predict_proba
        if not hasattr(model, 'predict_proba'):
            return {'error': 'O modelo não suporta previsão de probabilidades',
                   'success': False}
        
        try:
            # Para problemas binários
            if len(np.unique(y_clean)) == 2:
                # Gera probabilidades via validação cruzada
                from sklearn.model_selection import cross_val_predict
                
                # Seleciona o método de CV
                if isinstance(cv, int):
                    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
                else:
                    cv_obj = cv
                
                # Obter probabilidades via validação cruzada
                y_probs = cross_val_predict(
                    model, 
                    X_clean, 
                    y_clean, 
                    cv=cv_obj, 
                    method='predict_proba'
                )[:, 1]
                
                # Calcula curva de calibração
                prob_true, prob_pred = calibration_curve(y_clean, y_probs, n_bins=10)
                
                # Brier score (menor é melhor)
                from sklearn.metrics import brier_score_loss
                brier = brier_score_loss(y_clean, y_probs)
                
                # Resultados
                results = {
                    'prob_true': prob_true.tolist(),
                    'prob_pred': prob_pred.tolist(),
                    'brier_score': brier,
                    'is_binary': True
                }
                
                # Análise de qualidade da calibração
                # Calcula desvio médio absoluto entre prob_true e prob_pred
                calibration_error = np.mean(np.abs(prob_true - prob_pred))
                results['calibration_error'] = calibration_error
                
                # Diagnostico
                if calibration_error < 0.05:
                    calibration_quality = "excelente"
                elif calibration_error < 0.1:
                    calibration_quality = "boa"
                elif calibration_error < 0.2:
                    calibration_quality = "razoável"
                else:
                    calibration_quality = "ruim"
                
                results['calibration_quality'] = calibration_quality
                
                # Verifica padrão de over/underconfidence
                confidence_pattern = "balanceada"
                if np.mean(prob_pred - prob_true) > 0.05:
                    confidence_pattern = "overconfidence"
                elif np.mean(prob_true - prob_pred) > 0.05:
                    confidence_pattern = "underconfidence"
                
                results['confidence_pattern'] = confidence_pattern
                
                # Recomendações
                if calibration_quality in ["ruim", "razoável"]:
                    if confidence_pattern == "overconfidence":
                        results['recommendation'] = (
                            "O modelo tende a superestimar as probabilidades. "
                            "Considere usar métodos de calibração como Platt Scaling "
                            "ou Isotonic Regression."
                        )
                    elif confidence_pattern == "underconfidence":
                        results['recommendation'] = (
                            "O modelo tende a subestimar as probabilidades. "
                            "Considere usar métodos de calibração como Platt Scaling "
                            "ou Isotonic Regression."
                        )
                    else:
                        results['recommendation'] = (
                            "A calibração do modelo é inconsistente. "
                            "Considere usar métodos de calibração como Isotonic Regression."
                        )
                else:
                    results['recommendation'] = (
                        "A calibração do modelo é adequada. Não são necessárias ações específicas."
                    )
                
                self.logger.info(f"Análise de calibração concluída. Qualidade: {calibration_quality}, "
                              f"Padrão: {confidence_pattern}, Brier score: {brier:.4f}")
                
                return results
                
            else:
                # Para problemas multiclasse, reporta métricas gerais
                self.logger.info("Análise de calibração para problemas multiclasse não é tão detalhada")
                
                # Implementação simplificada para multiclasse
                results = {
                    'is_binary': False,
                    'message': 'Análise detalhada de calibração disponível apenas para classificação binária',
                    'success': True
                }
                
                return results
                
        except Exception as e:
            self.logger.error(f"Erro na análise de calibração: {str(e)}")
            return {
                'error': str(e),
                'success': False
            }
    
    def cross_validation_report(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                              method: str = 'auto', output_format: str = 'dict') -> Dict:
        """
        Gera um relatório abrangente com validação cruzada e análises adicionais.
        
        Args:
            model: Modelo a ser avaliado
            X: Features
            y: Variável alvo
            method: Método de validação ('cv', 'bootstrap', 'nested_cv', 'auto')
            output_format: Formato de saída ('dict' ou 'markdown')
            
        Returns:
            Relatório com resultados da validação e análises adicionais
        """
        self.logger.info(f"Gerando relatório abrangente de validação cruzada usando método '{method}'")
        
        # Sanitiza os dados
        X_clean, y_clean = self._sanitize_data(X, y)
        
        # Detecta o tipo de problema
        problem_type = self.problem_type
        if problem_type == 'auto':
            problem_type = self._detect_problem_type(y_clean)
        
        # Executa validação cruzada com confiança
        cv_results = self.evaluate_with_confidence(model, X_clean, y_clean, method=method)
        
        # Inicializa relatório
        report = {
            'validation_results': cv_results,
            'problem_type': problem_type,
            'model_type': model.__class__.__name__,
            'dataset_size': len(X_clean),
            'n_features': X_clean.shape[1],
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'method': method if method != 'auto' else cv_results.get('cv_strategy', method)
        }
        
        # Adiciona análises adicionais se não houve erro na validação cruzada
        if 'error' not in cv_results:
            # Análise de importance
            try:
                if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                    self.logger.info("Calculando importância de features")
                    importance_results = self.permutation_importance(
                        model, X_clean, y_clean, n_repeats=5
                    )
                    report['feature_importance'] = importance_results
            except Exception as e:
                self.logger.warning(f"Erro ao calcular importância de features: {str(e)}")
            
            # Análise de curva de aprendizado
            try:
                if method != 'bootstrap':  # Evita cálculo redundante com bootstrap
                    self.logger.info("Calculando curva de aprendizado")
                    learning_curve_results = self.learning_curve_analysis(
                        model, X_clean, y_clean, cv=3
                    )
                    report['learning_curve'] = learning_curve_results
            except Exception as e:
                self.logger.warning(f"Erro ao calcular curva de aprendizado: {str(e)}")
            
            # Análise de calibração para classificação
            if (problem_type == 'classification' and hasattr(model, 'predict_proba') 
                and len(np.unique(y_clean)) <= 10):  # Limita para casos práticos
                try:
                    self.logger.info("Calculando análise de calibração")
                    calibration_results = self.calibration_analysis(
                        model, X_clean, y_clean, cv=3
                    )
                    report['calibration'] = calibration_results
                except Exception as e:
                    self.logger.warning(f"Erro ao calcular análise de calibração: {str(e)}")
        
        # Formata o relatório
        if output_format == 'markdown':
            markdown_report = self._format_report_as_markdown(report)
            return {'markdown': markdown_report, 'data': report}
        else:
            return report
    
    def _format_report_as_markdown(self, report: Dict) -> str:
        """
        Formata o relatório como markdown para facilitar a visualização.
        
        Args:
            report: Dicionário com o relatório
            
        Returns:
            Relatório formatado em markdown
        """
        problem_type = report.get('problem_type', 'unknown')
        
        # Cabeçalho
        markdown = "# Relatório de Validação Cruzada Robusta\n\n"
        
        markdown += f"**Modelo:** {report.get('model_type', 'N/A')}  \n"
        markdown += f"**Tipo de problema:** {problem_type}  \n"
        markdown += f"**Tamanho do dataset:** {report.get('dataset_size', 'N/A')} amostras, {report.get('n_features', 'N/A')} features  \n"
        markdown += f"**Método de validação:** {report.get('method', 'N/A')}  \n"
        markdown += f"**Data:** {report.get('timestamp', 'N/A')}  \n\n"
        
        # Resultados principais da validação
        markdown += "## Resultados da Validação\n\n"
        
        if 'error' in report.get('validation_results', {}):
            markdown += f"**ERRO:** {report['validation_results']['error']}\n\n"
        else:
            validation_results = report.get('validation_results', {})
            
            # Tabela de métricas principais
            markdown += "| Métrica | Valor | Desvio padrão |\n"
            markdown += "|---------|-------|---------------|\n"
            
            # Obtém métricas relevantes com base no tipo de problema
            metrics_to_show = []
            
            if problem_type == 'classification':
                metrics_to_show = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
            else:  # regressão
                metrics_to_show = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
            
            for base_metric in metrics_to_show:
                # Procura variantes do nome da métrica
                for key in validation_results.keys():
                    if key.endswith(f"{base_metric}_mean") or key == f"test_{base_metric}_mean":
                        metric_name = base_metric
                        metric_value = validation_results[key]
                        
                        # Localiza desvio padrão correspondente
                        std_key = key.replace("_mean", "_std")
                        metric_std = validation_results.get(std_key, "N/A")
                        
                        markdown += f"| {metric_name} | {metric_value:.4f} | {metric_std:.4f} |\n"
                        break
            
            # Informações adicionais
            if 'reliability_score' in validation_results:
                markdown += f"\n**Confiabilidade das métricas:** {validation_results['reliability_score']:.4f}\n"
            
            if 'params_consensus' in validation_results:
                markdown += f"\n**Consenso de hiperparâmetros:** {validation_results['params_consensus']:.2f}\n"
            
            if 'overfitting_grade' in validation_results:
                overfitting = validation_results['overfitting_grade']
                markdown += f"\n**Grau de overfitting:** {overfitting:.2f} "
                
                if overfitting < 0.2:
                    markdown += "(Baixo)"
                elif overfitting < 0.5:
                    markdown += "(Moderado)"
                else:
                    markdown += "(Alto)"
            
        # Curva de Aprendizado
        if 'learning_curve' in report and 'error' not in report['learning_curve']:
            markdown += "\n## Análise da Curva de Aprendizado\n\n"
            
            learning_curve = report['learning_curve']
            diagnosis = learning_curve.get('diagnosis', 'unknown')
            markdown += f"**Diagnóstico:** {diagnosis.replace('_', ' ').title()}\n\n"
            markdown += f"**Recomendação:** {learning_curve.get('recommendation', 'N/A')}\n\n"
            
            # Poderia adicionar código para geração de gráfico da curva de aprendizado
            # usando matplotlib ou plotly, mas isso depende do ambiente de execução
        
        # Importância de Features
        if 'feature_importance' in report and 'error' not in report['feature_importance']:
            markdown += "\n## Importância das Features\n\n"
            
            importance = report['feature_importance']
            sorted_features = importance.get('sorted_features', [])
            
            if sorted_features:
                markdown += "| Feature | Importância | Desvio padrão |\n"
                markdown += "|---------|------------|---------------|\n"
                
                # Mostra as top 10 features mais importantes
                for i, feature in enumerate(sorted_features[:10]):
                    if feature in importance.get('importances', {}):
                        feat_imp = importance['importances'][feature]
                        mean_imp = feat_imp.get('mean_importance', 0)
                        std_imp = feat_imp.get('std_importance', 0)
                        
                        markdown += f"| {feature} | {mean_imp:.4f} | {std_imp:.4f} |\n"
            
        # Calibração (apenas para classificação)
        if 'calibration' in report and 'error' not in report['calibration']:
            calibration = report['calibration']
            
            if calibration.get('is_binary', False):
                markdown += "\n## Análise de Calibração\n\n"
                
                markdown += f"**Qualidade da calibração:** {calibration.get('calibration_quality', 'N/A')}\n\n"
                markdown += f"**Padrão de confiança:** {calibration.get('confidence_pattern', 'N/A')}\n\n"
                markdown += f"**Brier score:** {calibration.get('brier_score', 'N/A'):.4f}\n\n"
                markdown += f"**Recomendação:** {calibration.get('recommendation', 'N/A')}\n\n"
                
                # Poderia adicionar código para geração do gráfico de calibração
        
        return markdown

def create_validator(problem_type: str = 'auto', cv_strategy: str = 'auto',
                    n_splits: int = 5, n_repeats: int = 3,
                    random_state: int = 42, verbosity: int = 1) -> RobustCrossValidator:
    """
    Cria uma instância do RobustCrossValidator com as configurações especificadas.
    
    Args:
        problem_type: Tipo de problema ('classification', 'regression', 'timeseries', 'auto')
        cv_strategy: Estratégia de validação cruzada
        n_splits: Número de divisões para validação cruzada
        n_repeats: Número de repetições para métodos repetidos
        random_state: Semente aleatória para reprodutibilidade
        verbosity: Nível de verbosidade para logging
        
    Returns:
        Instância configurada do RobustCrossValidator
    """
    return RobustCrossValidator(
        problem_type=problem_type,
        cv_strategy=cv_strategy,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
        verbosity=verbosity
    )

def quick_validate(model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                 problem_type: str = 'auto', method: str = 'auto',
                 scoring=None, markdown_report: bool = False) -> Dict:
    """
    Função auxiliar para validação rápida de modelos.
    
    Args:
        model: Modelo a ser validado
        X: Features
        y: Variável alvo
        problem_type: Tipo de problema ('classification', 'regression', 'timeseries', 'auto')
        method: Método de validação ('cv', 'bootstrap', 'nested_cv', 'auto')
        scoring: Função ou nome de pontuação
        markdown_report: Se deve retornar relatório em markdown
        
    Returns:
        Resultados da validação
    """
    validator = create_validator(problem_type=problem_type)
    
    if markdown_report:
        return validator.cross_validation_report(
            model, X, y, method=method, output_format='markdown'
        )
    else:
        return validator.evaluate_with_confidence(model, X, y, method=method)
