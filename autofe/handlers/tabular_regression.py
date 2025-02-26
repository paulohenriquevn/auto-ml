"""
Handler para datasets de regressão tabular.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Importações internas
from config import DATASET_HANDLERS_CONFIG
from utils.transformations import apply_transformation


class TabularRegressionHandler:
    """
    Handler específico para datasets de regressão tabular.
    Responsável por preparar dados, aplicar transformações e avaliar resultados.
    """
    
    def __init__(self):
        """
        Inicializa o handler para regressão tabular.
        """
        self.logger = logging.getLogger(__name__)
        self.config = DATASET_HANDLERS_CONFIG['tabular_regression']
        self.feature_scalers = {}
        self.target_scaler = None
        self.feature_importance = None
    
    def is_classification(self) -> bool:
        """
        Verifica se o handler é para classificação.
        
        Returns:
            False, pois este handler é para regressão
        """
        return False
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        target: Union[str, pd.Series],
        **kwargs
    ) -> pd.DataFrame:
        """
        Prepara os dados para o processo de engenharia de features.
        
        Args:
            data: DataFrame com os dados
            target: Nome da coluna alvo ou Series com valores alvo
            **kwargs: Parâmetros adicionais
            
        Returns:
            DataFrame preparado
        """
        self.logger.info("Preparando dados para regressão tabular")
        
        # Separar features e alvo
        if isinstance(target, str):
            y = data[target]
            X = data.drop(columns=[target])
        else:
            y = target
            X = data
        
        # Verificar se o alvo é numérico
        if not pd.api.types.is_numeric_dtype(y):
            self.logger.warning("Alvo não é numérico, convertendo para numérico")
            y = pd.to_numeric(y, errors='coerce')
        
        # Normalizar alvo (opcional)
        if kwargs.get('normalize_target', False):
            self.logger.info("Normalizando variável alvo")
            self.target_scaler = StandardScaler()
            y = pd.Series(self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten())
        
        # Preparar variáveis categóricas
        X_prepared = self._prepare_categorical_features(X)
        
        # Retornar dados preparados com alvo
        prepared_data = X_prepared.copy()
        if isinstance(target, str):
            prepared_data[target] = y
        
        return prepared_data
    
    def _prepare_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara variáveis categóricas para modelagem.
        
        Args:
            X: DataFrame com features
            
        Returns:
            DataFrame com features preparadas
        """
        X_prepared = X.copy()
        
        for column in X.columns:
            # Verificar se a coluna é categórica ou data
            if (pd.api.types.is_object_dtype(X[column]) or 
                pd.api.types.is_categorical_dtype(X[column]) or
                pd.api.types.is_datetime64_any_dtype(X[column])):
                
                # Codificar usando one-hot encoding
                self.logger.info(f"Codificando feature categórica: {column}")
                
                # Para datas, converter para string primeiro
                if pd.api.types.is_datetime64_any_dtype(X[column]):
                    X_prepared[column] = X_prepared[column].dt.strftime('%Y-%m-%d')
                
                try:
                    # Limitar categorias para evitar explosão de dimensionalidade
                    if X_prepared[column].nunique() > 20:
                        # Para muitas categorias, manter apenas as 19 mais frequentes
                        top_cats = X_prepared[column].value_counts().nlargest(19).index.tolist()
                        X_prepared[column] = X_prepared[column].apply(
                            lambda x: x if x in top_cats else 'Other'
                        )
                    
                    # Aplicar one-hot encoding
                    dummies = pd.get_dummies(X_prepared[column], prefix=column, drop_first=True)
                    
                    # Adicionar colunas dummies ao DataFrame
                    X_prepared = pd.concat([X_prepared, dummies], axis=1)
                    
                    # Remover coluna original
                    X_prepared = X_prepared.drop(columns=[column])
                except Exception as e:
                    self.logger.warning(f"Erro ao codificar {column}: {str(e)}")
                    # Se falhar, simplesmente remover a coluna
                    X_prepared = X_prepared.drop(columns=[column])
        
        return X_prepared
    
    def _fill_missing_values(self, X: pd.DataFrame) -> None:
        """
        Preenche valores ausentes em um DataFrame, considerando os tipos de dados.
        
        Args:
            X: DataFrame com features a serem preenchidas (modificado in-place)
        """
        for col in X.columns:
            if X[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    # Para colunas numéricas, usar mediana
                    X[col] = X[col].fillna(X[col].median())
                elif pd.api.types.is_datetime64_any_dtype(X[col]):
                    # Para datas, usar a data mais frequente
                    X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else pd.NaT)
                else:
                    # Para categóricas, usar o valor mais frequente
                    X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "MISSING")
    
    def _ensure_numeric_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Garante que o DataFrame contenha apenas features numéricas.
        
        Args:
            X: DataFrame com features
            
        Returns:
            DataFrame contendo apenas colunas numéricas
        """
        # Selecionar apenas colunas numéricas
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            self.logger.warning("Nenhuma coluna numérica encontrada. Criando feature dummy.")
            # Se não houver colunas numéricas, criar uma dummy para evitar erros
            X_numeric = pd.DataFrame({
                'dummy_feature': np.ones(len(X))
            }, index=X.index)
            return X_numeric
        
        return X[numeric_cols]
    
    def evaluate_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: Optional[str] = None
    ) -> float:
        """
        Avalia a qualidade de um conjunto de features.
        
        Args:
            X: DataFrame com features
            y: Série com alvo
            metric: Métrica de avaliação (None = usar configuração)
            
        Returns:
            Valor numérico representando a qualidade
        """
        # Definir métrica de avaliação
        if metric is None:
            metric = self.config.get('regression_metric', 'rmse')
        
        # Remover colunas com muitos valores ausentes
        X = X.copy()
        for col in X.columns:
            if X[col].isnull().mean() > 0.5:
                X = X.drop(columns=[col])
                self.logger.warning(f"Removida coluna {col} com >50% valores ausentes")
        
        if len(X.columns) == 0:
            self.logger.warning("Nenhuma feature válida para avaliação")
            return 0.0
        
        # Preencher valores ausentes considerando o tipo de dados
        self._fill_missing_values(X)
        
        # Garantir que temos apenas features numéricas
        X_numeric = self._ensure_numeric_features(X)
        
        if X_numeric.shape[1] == 0:
            self.logger.warning("Nenhuma feature numérica válida após pré-processamento")
            return 0.0
        
        # Configurar validação cruzada
        n_folds = self.config.get('n_folds', 5)
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Inicializar modelo
        model_type = self.config.get('default_model', 'random_forest')
        
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            self.logger.warning(f"Modelo {model_type} não reconhecido, usando Random Forest")
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        try:
            # Selecionar métrica de scoring
            if metric == 'rmse':
                # Para RMSE, usar MSE negativo e depois tirar raiz quadrada
                scoring = 'neg_mean_squared_error'
            elif metric == 'mae':
                scoring = 'neg_mean_absolute_error'
            elif metric == 'r2':
                scoring = 'r2'
            else:
                scoring = 'neg_mean_squared_error'
            
            # Executar validação cruzada
            scores = cross_val_score(model, X_numeric, y, cv=cv, scoring=scoring, n_jobs=-1)
            
            # Processar scores conforme a métrica
            if metric == 'rmse':
                avg_score = np.sqrt(-np.mean(scores))  # Converter para RMSE e inverter sinal
                # Normalizar para [0, 1] onde 1 é melhor
                max_error = np.max(np.abs(y - y.mean())) if len(y) > 0 else 1.0
                normalized_score = 1 - min(avg_score / (max_error + 1e-10), 1)
            elif metric == 'mae':
                avg_score = -np.mean(scores)  # Inverter sinal
                # Normalizar para [0, 1] onde 1 é melhor
                max_error = np.max(np.abs(y - y.mean())) if len(y) > 0 else 1.0
                normalized_score = 1 - min(avg_score / (max_error + 1e-10), 1)
            elif metric == 'r2':
                avg_score = np.mean(scores)
                # R² já está entre -inf e 1, normalizar para [0, 1]
                normalized_score = (avg_score + 1) / 2 if avg_score < 0 else avg_score
            else:
                avg_score = np.sqrt(-np.mean(scores))  # Assumir RMSE
                max_error = np.max(np.abs(y - y.mean())) if len(y) > 0 else 1.0
                normalized_score = 1 - min(avg_score / (max_error + 1e-10), 1)
            
            # Treinar modelo em todos os dados para obter importância de features
            model.fit(X_numeric, y)
            self.feature_importance = pd.DataFrame({
                'feature': X_numeric.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return normalized_score
            
        except Exception as e:
            self.logger.error(f"Erro ao avaliar features: {str(e)}")
            return 0.0
    
    def apply_transformations(
        self,
        data: pd.DataFrame,
        transformations: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Aplica um conjunto de transformações aos dados.
        
        Args:
            data: DataFrame original
            transformations: Lista de transformações a aplicar
            
        Returns:
            DataFrame com transformações aplicadas
        """
        self.logger.info(f"Aplicando {len(transformations)} transformações")
        
        # Criar cópia dos dados originais
        transformed_data = data.copy()
        
        # Aplicar cada transformação
        for transform in transformations:
            try:
                transformed_feature = self.apply_single_transformation(
                    data,
                    transform['transformation_type'],
                    transform['transformation_params']
                )
                
                if transformed_feature is not None:
                    transformed_data[transform['name']] = transformed_feature
            except Exception as e:
                self.logger.warning(f"Erro ao aplicar transformação {transform['name']}: {str(e)}")
        
        return transformed_data
    
    def apply_single_transformation(
        self,
        data: pd.DataFrame,
        transformation_type: str,
        transformation_params: Dict[str, Any]
    ) -> Optional[pd.Series]:
        """
        Aplica uma única transformação aos dados.
        
        Args:
            data: DataFrame original
            transformation_type: Tipo de transformação
            transformation_params: Parâmetros da transformação
            
        Returns:
            Série com valores transformados ou None em caso de erro
        """
        try:
            return apply_transformation(data, transformation_type, transformation_params)
        except Exception as e:
            self.logger.warning(f"Erro ao aplicar transformação {transformation_type}: {str(e)}")
            return None
    
    def evaluate_transformations(
        self,
        transformed_data: pd.DataFrame,
        target: Union[str, pd.Series]
    ) -> Dict[str, float]:
        """
        Avalia a qualidade das transformações aplicadas.
        
        Args:
            transformed_data: DataFrame com transformações aplicadas
            target: Nome da coluna alvo ou Series com valores alvo
            
        Returns:
            Dicionário com métricas de performance
        """
        # Separar features e alvo
        if isinstance(target, str):
            y = transformed_data[target]
            X = transformed_data.drop(columns=[target])
        else:
            y = target
            X = transformed_data
        
        # Inicializar modelo
        model_type = self.config.get('default_model', 'random_forest')
        
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Configurar validação cruzada
        n_folds = self.config.get('n_folds', 5)
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Remover colunas com muitos valores ausentes
        X_clean = X.copy()
        for col in X_clean.columns:
            if X_clean[col].isnull().mean() > 0.5:
                X_clean = X_clean.drop(columns=[col])
        
        # Preencher valores ausentes considerando o tipo de dados
        self._fill_missing_values(X_clean)
        
        # Garantir que temos apenas features numéricas
        X_numeric = self._ensure_numeric_features(X_clean)
        
        if X_numeric.shape[1] == 0:
            self.logger.warning("Nenhuma feature numérica válida após pré-processamento")
            return {'rmse': float('inf'), 'mae': float('inf'), 'r2': -float('inf')}
        
        try:
            # Calcular métricas via validação cruzada
            rmse_scores = -cross_val_score(model, X_numeric, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
            rmse = np.sqrt(np.mean(rmse_scores))
            
            mae_scores = -cross_val_score(model, X_numeric, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
            mae = np.mean(mae_scores)
            
            r2_scores = cross_val_score(model, X_numeric, y, cv=cv, scoring='r2', n_jobs=-1)
            r2 = np.mean(r2_scores)
            
            # Compilar métricas
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erro ao avaliar transformações: {str(e)}")
            return {'rmse': float('inf'), 'mae': float('inf'), 'r2': -float('inf')}
    
    def extract_data_properties(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Extrai propriedades relevantes do dataset.
        
        Args:
            data: DataFrame com os dados
            
        Returns:
            Dicionário com propriedades do dataset
        """
        properties = {
            'num_samples': len(data),
            'num_features': len(data.columns),
            'num_numeric': sum(pd.api.types.is_numeric_dtype(data[col]) for col in data.columns),
            'num_categorical': sum(pd.api.types.is_object_dtype(data[col]) or 
                                 pd.api.types.is_categorical_dtype(data[col]) for col in data.columns),
            'num_datetime': sum(pd.api.types.is_datetime64_any_dtype(data[col]) for col in data.columns),
            'num_text': sum(pd.api.types.is_string_dtype(data[col]) and data[col].str.len().mean() > 10
                           for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])),
            'missing_ratio': data.isnull().mean().mean()
        }
        
        # Calcular estatísticas para features numéricas
        numeric_features = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        if numeric_features:
            skewness_values = []
            kurtosis_values = []
            variance_values = []
            
            for col in numeric_features:
                # Ignorar colunas com valores ausentes
                if data[col].isnull().any():
                    continue
                
                try:
                    skewness_values.append(data[col].skew())
                    kurtosis_values.append(data[col].kurtosis())
                    variance_values.append(data[col].var())
                except Exception:
                    pass
            
            properties['feature_stats'] = {
                'mean_skewness': np.mean(skewness_values) if skewness_values else 0,
                'mean_kurtosis': np.mean(kurtosis_values) if kurtosis_values else 0,
                'mean_variance': np.mean(variance_values) if variance_values else 0
            }
        
        # Detectar tipo de alvo (contínuo)
        target_column = None
        for col in data.columns:
            if col.lower() in ['target', 'y', 'value', 'response']:
                target_column = col
                break
        
        if target_column and pd.api.types.is_numeric_dtype(data[target_column]):
            properties['target_type'] = 'continuous'
            
            # Calcular estatísticas do alvo
            target_data = data[target_column].dropna()
            if len(target_data) > 0:
                properties['target_stats'] = {
                    'min': float(target_data.min()),  # Converter para float para serialização JSON
                    'max': float(target_data.max()),
                    'mean': float(target_data.mean()),
                    'median': float(target_data.median()),
                    'std': float(target_data.std()),
                    'skewness': float(target_data.skew()),
                    'kurtosis': float(target_data.kurtosis())
                }
        
        return properties