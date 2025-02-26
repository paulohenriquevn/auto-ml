"""
Handler para datasets de previsão de séries temporais.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings

# Importações internas
from config import DATASET_HANDLERS_CONFIG
from utils.transformations import apply_transformation


class TimeSeriesHandler:
    """
    Handler específico para datasets de previsão de séries temporais.
    Responsável por preparar dados, aplicar transformações e avaliar resultados.
    """
    
    def __init__(self):
        """
        Inicializa o handler para séries temporais.
        """
        self.logger = logging.getLogger(__name__)
        self.config = DATASET_HANDLERS_CONFIG['time_series']
        self.date_column = None
        self.target_scaler = None
        self.feature_scalers = {}
        self.feature_importance = None
    
    def is_classification(self) -> bool:
        """
        Verifica se o handler é para classificação.
        
        Returns:
            False, pois este handler é para regressão/previsão de séries temporais
        """
        return False
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        target: Union[str, pd.Series],
        date_column: Optional[str] = None,
        forecast_horizon: int = 1,
        **kwargs
    ) -> pd.DataFrame:
        """
        Prepara os dados para o processo de engenharia de features.
        
        Args:
            data: DataFrame com os dados
            target: Nome da coluna alvo ou Series com valores alvo
            date_column: Nome da coluna de data/tempo (opcional)
            forecast_horizon: Horizonte de previsão (número de períodos à frente)
            **kwargs: Parâmetros adicionais
            
        Returns:
            DataFrame preparado
        """
        self.logger.info("Preparando dados para previsão de séries temporais")
        
        # Identificar coluna de data/tempo
        if date_column:
            self.date_column = date_column
        else:
            # Tentar identificar automaticamente
            for col in data.columns:
                if pd.api.types.is_datetime64_any_dtype(data[col]):
                    self.date_column = col
                    break
        
        # Se ainda não encontrou, tentar converter colunas candidatas
        if not self.date_column:
            date_candidates = ['date', 'time', 'datetime', 'timestamp', 'ds']
            for candidate in date_candidates:
                if candidate in data.columns:
                    try:
                        data[candidate] = pd.to_datetime(data[candidate])
                        self.date_column = candidate
                        break
                    except:
                        pass
        
        # Verificar se encontrou coluna de data
        if not self.date_column:
            self.logger.warning("Coluna de data/tempo não encontrada")
        else:
            self.logger.info(f"Usando coluna de data/tempo: {self.date_column}")
            # Ordenar por data
            data = data.sort_values(by=self.date_column)
        
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
            y = pd.Series(self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten(), index=y.index)
        
        # Preparar features temporais
        X_prepared = self._prepare_temporal_features(X, y, forecast_horizon)
        
        # Retornar dados preparados com alvo
        prepared_data = X_prepared.copy()
        if isinstance(target, str):
            prepared_data[target] = y
        
        return prepared_data
    
    def _prepare_temporal_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        forecast_horizon: int
    ) -> pd.DataFrame:
        """
        Prepara features temporais básicas.
        
        Args:
            X: DataFrame com features
            y: Série com alvo
            forecast_horizon: Horizonte de previsão
            
        Returns:
            DataFrame com features temporais
        """
        X_prepared = X.copy()
        
        # Se temos coluna de data, extrair componentes temporais
        if self.date_column and self.date_column in X_prepared.columns:
            date_col = X_prepared[self.date_column]
            
            # Extrair componentes temporais
            X_prepared['year'] = date_col.dt.year
            X_prepared['month'] = date_col.dt.month
            X_prepared['day'] = date_col.dt.day
            X_prepared['dayofweek'] = date_col.dt.dayofweek
            X_prepared['quarter'] = date_col.dt.quarter
            X_prepared['is_month_start'] = date_col.dt.is_month_start.astype(int)
            X_prepared['is_month_end'] = date_col.dt.is_month_end.astype(int)
            
            # Se tem informação de hora, extrair componentes
            if hasattr(date_col.dt, 'hour'):
                X_prepared['hour'] = date_col.dt.hour
                X_prepared['minute'] = date_col.dt.minute
                
                # Adicionar indicador de horário comercial
                X_prepared['is_business_hour'] = ((X_prepared['hour'] >= 9) & 
                                                (X_prepared['hour'] <= 17) & 
                                                (X_prepared['dayofweek'] < 5)).astype(int)
            
            # Adicionar recursos cíclicos para mês e dia da semana
            X_prepared['month_sin'] = np.sin(2 * np.pi * X_prepared['month'] / 12)
            X_prepared['month_cos'] = np.cos(2 * np.pi * X_prepared['month'] / 12)
            X_prepared['dayofweek_sin'] = np.sin(2 * np.pi * X_prepared['dayofweek'] / 7)
            X_prepared['dayofweek_cos'] = np.cos(2 * np.pi * X_prepared['dayofweek'] / 7)
        
        # Criar lags do alvo como features
        max_lag = min(12, len(y) // 10)  # Limitar número de lags
        
        # Usar apenas lags maiores que o horizonte de previsão para evitar vazamento
        for lag in range(forecast_horizon, forecast_horizon + max_lag):
            lag_name = f'lag_{lag}'
            X_prepared[lag_name] = y.shift(lag)
        
        # Criar médias móveis
        for window in [2, 3, 5, 7, 14, 30]:
            if window < len(y) // 5:  # Limitar tamanho da janela
                window_name = f'rolling_mean_{window}'
                X_prepared[window_name] = y.shift(forecast_horizon).rolling(window=window).mean()
        
        return X_prepared
    
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
        
        # Remover colunas de data e colunas com muitos valores ausentes
        X = X.copy()
        
        if self.date_column and self.date_column in X.columns:
            X = X.drop(columns=[self.date_column])
        
        for col in X.columns:
            if X[col].isnull().mean() > 0.3:  # Mais tolerante com valores ausentes em séries temporais
                X = X.drop(columns=[col])
                self.logger.warning(f"Removida coluna {col} com >30% valores ausentes")
        
        if len(X.columns) == 0:
            self.logger.warning("Nenhuma feature válida para avaliação")
            return 0.0
        
        # Preencher valores ausentes
        for col in X.columns:
            if X[col].isnull().any():
                # Para séries temporais, usar forward fill primeiro, depois backward fill
                X[col] = X[col].fillna(method='ffill')
                X[col] = X[col].fillna(method='bfill')
                # Se ainda houver NaN, usar mediana
                X[col] = X[col].fillna(X[col].median() if not pd.isna(X[col].median()) else 0)
        
        try:
            # Para séries temporais, usar validação temporal
            result = self._time_series_cv(X, y, metric=metric)
            return result
        except Exception as e:
            self.logger.error(f"Erro ao avaliar features: {str(e)}")
            return 0.0
    
    def _time_series_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = 'rmse'
    ) -> float:
        """
        Executa validação cruzada específica para séries temporais.
        
        Args:
            X: DataFrame com features
            y: Série com alvo
            metric: Métrica de avaliação
            
        Returns:
            Score de qualidade normalizado
        """
        # Ignorar warnings de dados np.nan
        warnings.filterwarnings("ignore")
        
        # Configurações
        n_splits = self.config.get('n_splits', 3)
        gap = self.config.get('gap', 0)
        
        # Total de dados
        n_samples = len(y)
        
        # Verificar se há dados suficientes para validação
        if n_samples < n_splits * 5:
            # Poucos dados, usar avaliação simples
            train_size = int(n_samples * 0.7)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            # Treinar modelo
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Avaliar no conjunto de teste
            y_pred = model.predict(X_test)
            
            # Calcular erro
            if metric == 'rmse':
                error = np.sqrt(mean_squared_error(y_test, y_pred))
            elif metric == 'mae':
                error = mean_absolute_error(y_test, y_pred)
            else:
                error = np.sqrt(mean_squared_error(y_test, y_pred))  # Padrão é RMSE
            
            # Salvar importância de features
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Normalizar erro para score em [0, 1]
            max_error = np.max(np.abs(y - y.mean()))
            normalized_score = 1 - min(error / (max_error + 1e-10), 1)
            
            return normalized_score
        
        # Configurar validação com múltiplos splits temporais
        errors = []
        feature_importances = []
        
        # Tamanho do teste é aproximadamente n_samples / (n_splits + 1)
        test_size = n_samples // (n_splits + 1)
        
        for i in range(n_splits):
            # Calcular índices para training e testing
            train_end = n_samples - (n_splits - i) * test_size
            test_start = train_end + gap
            test_end = min(test_start + test_size, n_samples)
            
            # Verificar se há dados suficientes
            if train_end <= 0 or test_start >= n_samples:
                continue
            
            # Dividir dados
            X_train, X_test = X.iloc[:train_end], X.iloc[test_start:test_end]
            y_train, y_test = y.iloc[:train_end], y.iloc[test_start:test_end]
            
            # Treinar modelo
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Avaliar no conjunto de teste
            y_pred = model.predict(X_test)
            
            # Calcular erro
            if metric == 'rmse':
                error = np.sqrt(mean_squared_error(y_test, y_pred))
            elif metric == 'mae':
                error = mean_absolute_error(y_test, y_pred)
            else:
                error = np.sqrt(mean_squared_error(y_test, y_pred))  # Padrão é RMSE
            
            errors.append(error)
            feature_importances.append(model.feature_importances_)
        
        # Calcular erro médio
        if not errors:
            return 0.0
        
        mean_error = np.mean(errors)
        
        # Calcular importância média das features
        if feature_importances:
            mean_importance = np.mean(feature_importances, axis=0)
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': mean_importance
            }).sort_values('importance', ascending=False)
        
        # Normalizar erro para score em [0, 1]
        max_error = np.max(np.abs(y - y.mean()))
        normalized_score = 1 - min(mean_error / (max_error + 1e-10), 1)
        
        return normalized_score
    
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
        
        # Remover coluna de data
        if self.date_column and self.date_column in X.columns:
            X = X.drop(columns=[self.date_column])
        
        # Remover colunas com muitos valores ausentes
        for col in X.columns:
            if X[col].isnull().mean() > 0.3:  # Mais tolerante com valores ausentes em séries temporais
                X = X.drop(columns=[col])
        
        # Preencher valores ausentes
        for col in X.columns:
            if X[col].isnull().any():
                # Para séries temporais, usar forward fill primeiro, depois backward fill
                X[col] = X[col].fillna(method='ffill')
                X[col] = X[col].fillna(method='bfill')
                # Se ainda houver NaN, usar mediana
                X[col] = X[col].fillna(X[col].median() if not pd.isna(X[col].median()) else 0)
        
        try:
            # Avaliar usando validação temporal
            n_samples = len(y)
            train_size = int(n_samples * 0.7)
            
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            # Treinar modelo
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Avaliar no conjunto de teste
            y_pred = model.predict(X_test)
            
            # Calcular métricas
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Calcular MAPE se não houver zeros
            if (y_test != 0).all():
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            else:
                mape = np.nan
            
            # Calcular R²
            ss_total = np.sum((y_test - y_test.mean()) ** 2)
            ss_residual = np.sum((y_test - y_pred) ** 2)
            r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            
            # Compilar métricas
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            if not np.isnan(mape):
                metrics['mape'] = mape
            
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
            'is_time_series': True,
            'num_numeric': sum(pd.api.types.is_numeric_dtype(data[col]) for col in data.columns),
            'num_categorical': sum(pd.api.types.is_object_dtype(data[col]) or 
                                 pd.api.types.is_categorical_dtype(data[col]) for col in data.columns),
            'num_datetime': sum(pd.api.types.is_datetime64_any_dtype(data[col]) for col in data.columns),
            'missing_ratio': data.isnull().mean().mean()
        }
        
        # Verificar se temos coluna de data identificada
        if self.date_column and self.date_column in data.columns:
            # Extrair propriedades da série temporal
            date_series = data[self.date_column]
            
            # Frequência aproximada dos dados
            try:
                freq = pd.infer_freq(date_series)
                properties['frequency'] = str(freq) if freq else 'irregular'
            except:
                properties['frequency'] = 'irregular'
            
            # Calcular intervalo médio em segundos
            try:
                intervals = date_series.diff().dropna()
                avg_interval = intervals.mean().total_seconds()
                properties['avg_interval_seconds'] = avg_interval
            except:
                properties['avg_interval_seconds'] = None
            
            # Verificar se há sazonalidade
            has_hourly = False
            has_daily = False
            has_weekly = False
            has_monthly = False
            has_quarterly = False
            has_yearly = False
            
            try:
                # Verificar spans temporais
                min_date = date_series.min()
                max_date = date_series.max()
                span = max_date - min_date
                
                span_seconds = span.total_seconds()
                span_days = span_seconds / (3600 * 24)
                
                if span_days >= 365:
                    has_yearly = True
                if span_days >= 90:
                    has_quarterly = True
                if span_days >= 30:
                    has_monthly = True
                if span_days >= 7:
                    has_weekly = True
                if span_days >= 1:
                    has_daily = True
                if span_seconds >= 3600:
                    has_hourly = True
            except:
                pass
            
            properties['seasonality'] = {
                'hourly': has_hourly,
                'daily': has_daily,
                'weekly': has_weekly,
                'monthly': has_monthly,
                'quarterly': has_quarterly,
                'yearly': has_yearly
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
            
            # Verificar estacionariedade da série alvo
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_result = adfuller(data[target_column].dropna())
                properties['target_stationary'] = adf_result[1] < 0.05  # p-valor < 0.05 indica série estacionária
            except:
                properties['target_stationary'] = None
        
        return properties
