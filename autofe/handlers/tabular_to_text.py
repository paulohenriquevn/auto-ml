"""
Handler para datasets de tabular para texto (onde o alvo é texto).
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Importações internas
from config import DATASET_HANDLERS_CONFIG
from utils.transformations import apply_transformation


class TabularToTextHandler:
    """
    Handler específico para datasets tabular para texto.
    Responsável por preparar dados, aplicar transformações e avaliar resultados.
    """
    
    def __init__(self):
        """
        Inicializa o handler para tabular para texto.
        """
        self.logger = logging.getLogger(__name__)
        self.config = DATASET_HANDLERS_CONFIG['tabular_to_text']
        self.text_vectorizer = None
        self.feature_encoders = {}
        self.feature_importance = None
    
    def is_classification(self) -> bool:
        """
        Verifica se o handler é para classificação.
        
        Returns:
            False, pois este handler não é para classificação tradicional
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
            target: Nome da coluna alvo ou Series com valores alvo (texto)
            **kwargs: Parâmetros adicionais
            
        Returns:
            DataFrame preparado
        """
        self.logger.info("Preparando dados para tabular para texto")
        
        # Separar features e alvo
        if isinstance(target, str):
            y = data[target]
            X = data.drop(columns=[target])
        else:
            y = target
            X = data
        
        # Verificar se o alvo é texto
        if not pd.api.types.is_object_dtype(y) and not pd.api.types.is_string_dtype(y):
            self.logger.warning("Alvo não é texto, convertendo para string")
            y = y.astype(str)
        
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
            # Verificar se a coluna é categórica
            if pd.api.types.is_object_dtype(X[column]) or pd.api.types.is_categorical_dtype(X[column]):
                # Codificar com LabelEncoder
                self.logger.info(f"Codificando feature categórica: {column}")
                encoder = LabelEncoder()
                
                # Tratar valores ausentes
                missing_mask = X[column].isna()
                non_missing_values = X[column][~missing_mask]
                
                if len(non_missing_values) > 0:
                    # Codificar valores não ausentes
                    encoded_values = encoder.fit_transform(non_missing_values)
                    
                    # Criar série com valores codificados
                    encoded_series = pd.Series(index=X.index, dtype=float)
                    encoded_series[~missing_mask] = encoded_values
                    encoded_series[missing_mask] = np.nan
                    
                    # Substituir coluna original
                    X_prepared[column] = encoded_series
                    
                    # Armazenar encoder
                    self.feature_encoders[column] = encoder
        
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
            y: Série com alvo (texto)
            metric: Métrica de avaliação (None = usar configuração)
            
        Returns:
            Valor numérico representando a qualidade
        """
        # Para alvo de texto, vamos converter o texto em vetores numéricos
        # e avaliar a capacidade das features de prever os principais componentes do texto
        
        # Remover colunas com muitos valores ausentes
        X = X.copy()
        for col in X.columns:
            if X[col].isnull().mean() > 0.5:
                X = X.drop(columns=[col])
                self.logger.warning(f"Removida coluna {col} com >50% valores ausentes")
        
        if len(X.columns) == 0:
            self.logger.warning("Nenhuma feature válida para avaliação")
            return 0.0
        
        # Preencher valores ausentes
        X = X.fillna(X.median())
        
        try:
            # Vetorizar o texto alvo
            self.text_vectorizer = TfidfVectorizer(max_features=100)
            y_vectors = self.text_vectorizer.fit_transform(y.fillna("")).toarray()
            
            # Calcular componentes principais do texto
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(5, y_vectors.shape[1]))
            y_components = pca.fit_transform(y_vectors)
            
            # Dividir em treino e teste
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_components, test_size=0.2, random_state=42
            )
            
            # Inicializar modelo
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Treinar para prever os componentes principais
            model.fit(X_train, y_train)
            
            # Fazer previsões
            y_pred = model.predict(X_test)
            
            # Calcular MSE para cada componente
            mse_per_component = np.mean((y_test - y_pred) ** 2, axis=0)
            
            # Calcular RMSE médio
            avg_rmse = np.sqrt(np.mean(mse_per_component))
            
            # Armazenar importância de features
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Normalizar para [0, 1] onde 1 é melhor
            # O RMSE está entre 0 e infinito, então precisamos normalizar
            max_value = np.max(y_components)
            min_value = np.min(y_components)
            range_value = max_value - min_value
            
            # Se o range é 0, retornar 0.5 (valor neutro)
            if range_value == 0:
                return 0.5
            
            # Normalizar, de forma que menor RMSE dê um score melhor
            normalized_score = 1 - min(avg_rmse / range_value, 1)
            
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
        
        # Remover colunas com muitos valores ausentes
        for col in X.columns:
            if X[col].isnull().mean() > 0.5:
                X = X.drop(columns=[col])
        
        # Preencher valores ausentes
        X = X.fillna(X.median())
        
        try:
            # Vetorizar o texto alvo
            if self.text_vectorizer is None:
                self.text_vectorizer = TfidfVectorizer(max_features=100)
                self.text_vectorizer.fit(y.fillna(""))
            
            y_vectors = self.text_vectorizer.transform(y.fillna("")).toarray()
            
            # Calcular componentes principais do texto
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(5, y_vectors.shape[1]))
            y_components = pca.fit_transform(y_vectors)
            
            # Dividir em treino e teste
            test_size = self.config.get('test_size', 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_components, test_size=test_size, random_state=42
            )
            
            # Inicializar modelo
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Treinar para prever os componentes principais
            model.fit(X_train, y_train)
            
            # Fazer previsões
            y_pred = model.predict(X_test)
            
            # Calcular MSE para cada componente
            mse_per_component = np.mean((y_test - y_pred) ** 2, axis=0)
            
            # Calcular RMSE médio
            avg_rmse = np.sqrt(np.mean(mse_per_component))
            
            # Calcular R² para cada componente
            r2_per_component = []
            for i in range(y_test.shape[1]):
                ss_total = np.sum((y_test[:, i] - np.mean(y_test[:, i])) ** 2)
                ss_residual = np.sum((y_test[:, i] - y_pred[:, i]) ** 2)
                r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
                r2_per_component.append(r2)
            
            avg_r2 = np.mean(r2_per_component)
            
            # Calcular similaridade de cosseno
            from sklearn.metrics.pairwise import cosine_similarity
            avg_cosine = np.mean([
                cosine_similarity(y_test[i].reshape(1, -1), y_pred[i].reshape(1, -1))[0, 0]
                for i in range(len(y_test))
            ])
            
            # Compilar métricas
            metrics = {
                'rmse': avg_rmse,
                'r2': avg_r2,
                'cosine_similarity': avg_cosine
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erro ao avaliar transformações: {str(e)}")
            return {'rmse': float('inf'), 'r2': -float('inf'), 'cosine_similarity': 0.0}
    
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
        
        # Detectar e analisar coluna de texto (alvo)
        text_column = None
        for col in data.columns:
            if pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_string_dtype(data[col]):
                # Verificar se parece texto (comprimento médio > 20 caracteres)
                if data[col].astype(str).str.len().mean() > 20:
                    text_column = col
                    break
        
        if text_column:
            properties['target_type'] = 'text'
            
            # Calcular estatísticas do texto
            text_data = data[text_column].astype(str)
            properties['text_stats'] = {
                'avg_length': text_data.str.len().mean(),
                'avg_word_count': text_data.str.split().str.len().mean(),
                'max_length': text_data.str.len().max(),
                'min_length': text_data.str.len().min()
            }
        
        return properties
