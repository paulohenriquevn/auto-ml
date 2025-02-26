"""
Handler para datasets de classificação tabular.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Importações internas
from config import DATASET_HANDLERS_CONFIG
from utils.transformations import apply_transformation


class TabularClassificationHandler:
    """
    Handler específico para datasets de classificação tabular.
    Responsável por preparar dados, aplicar transformações e avaliar resultados.
    """
    
    def __init__(self):
        """
        Inicializa o handler para classificação tabular.
        """
        self.logger = logging.getLogger(__name__)
        self.config = DATASET_HANDLERS_CONFIG['tabular_classification']
        self.label_encoders = {}
        self.feature_encoders = {}
        self.target_encoder = None
        self.feature_importance = None
    
    def is_classification(self) -> bool:
        """
        Verifica se o handler é para classificação.
        
        Returns:
            True, pois este handler é para classificação
        """
        return True
    
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
        self.logger.info("Preparando dados para classificação tabular")
        
        # Separar features e alvo
        if isinstance(target, str):
            y = data[target]
            X = data.drop(columns=[target])
        else:
            y = target
            X = data
        
        # Codificar variável alvo categórica
        if not pd.api.types.is_numeric_dtype(y):
            self.logger.info("Codificando variável alvo categórica")
            self.target_encoder = LabelEncoder()
            y = pd.Series(self.target_encoder.fit_transform(y))
        
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
            y: Série com alvo
            metric: Métrica de avaliação (None = usar configuração)
            
        Returns:
            Valor numérico representando a qualidade
        """
        # Definir métrica de avaliação
        if metric is None:
            metric = self.config.get('classification_metric', 'auc')
        
        # Remover colunas com muitos valores ausentes
        X = X.copy()
        for col in X.columns:
            if X[col].isnull().mean() > 0.5:
                X = X.drop(columns=[col])
                self.logger.warning(f"Removida coluna {col} com >50% valores ausentes")
        
        if len(X.columns) == 0:
            self.logger.warning("Nenhuma feature válida para avaliação")
            return 0.0
        
        # Preencher valores ausentes - só nas colunas numéricas
        self._fill_missing_values(X)
        
        # Configurar validação cruzada
        n_folds = self.config.get('n_folds', 5)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Inicializar modelo
        model_type = self.config.get('default_model', 'random_forest')
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            self.logger.warning(f"Modelo {model_type} não reconhecido, usando Random Forest")
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        try:
            # Verificar se há classes suficientes
            if len(np.unique(y)) < 2:
                self.logger.warning("Menos de 2 classes únicas, impossível avaliar")
                return 0.0
            
            # Selecionar métrica de scoring
            if metric == 'auc':
                scoring = 'roc_auc'
            elif metric == 'accuracy':
                scoring = 'accuracy'
            elif metric == 'f1':
                scoring = 'f1_weighted'
            else:
                scoring = 'roc_auc'
            
            # Executar validação cruzada
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            avg_score = np.mean(scores)
            
            # Treinar modelo em todos os dados para obter importância de features
            model.fit(X, y)
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return avg_score
            
        except Exception as e:
            self.logger.error(f"Erro ao avaliar features: {str(e)}")
            return 0.0
    
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
        
        # Codificar variável alvo categórica se necessário
        if not pd.api.types.is_numeric_dtype(y) and self.target_encoder:
            y = pd.Series(self.target_encoder.transform(y))
        
        # Inicializar modelo
        model_type = self.config.get('default_model', 'random_forest')
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Configurar validação cruzada
        n_folds = self.config.get('n_folds', 5)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Remover colunas com muitos valores ausentes
        for col in X.columns:
            if X[col].isnull().mean() > 0.5:
                X = X.drop(columns=[col])
        
        # Preencher valores ausentes, considerando os tipos de dados
        X_filled = X.copy()
        self._fill_missing_values(X_filled)
        
        try:
            # Calcular métricas via validação cruzada
            accuracy_scores = cross_val_score(model, X_filled, y, cv=cv, scoring='accuracy', n_jobs=-1)
            
            # Para AUC, verificar se é binário ou multiclasse
            n_classes = len(np.unique(y))
            
            if n_classes == 2:
                # Classificação binária
                auc_scores = cross_val_score(model, X_filled, y, cv=cv, scoring='roc_auc', n_jobs=-1)
                avg_auc = np.mean(auc_scores)
            else:
                # Classificação multiclasse
                avg_auc = 0.0  # Não calcular AUC para multiclasse
                
            # Calcular F1 score
            f1_scores = cross_val_score(model, X_filled, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
            
            # Compilar métricas
            metrics = {
                'accuracy': np.mean(accuracy_scores),
                'f1_score': np.mean(f1_scores)
            }
            
            if n_classes == 2:
                metrics['auc'] = avg_auc
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erro ao avaliar transformações: {str(e)}")
            return {'accuracy': 0.0, 'f1_score': 0.0, 'auc': 0.0}
    
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
        
        # Detectar tipo de alvo (multiclasse ou binário)
        target_column = None
        for col in data.columns:
            if col.lower() in ['target', 'label', 'class', 'y']:
                target_column = col
                break
        
        if target_column:
            unique_values = data[target_column].nunique()
            properties['target_type'] = 'binary' if unique_values == 2 else 'multiclass'
        
        return properties