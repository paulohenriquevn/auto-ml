"""
Implementação do preditor de transformações para o Learner-Predictor.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import pandas as pd
import numpy as np

# Importações internas
from config import LEARNER_PREDICTOR_CONFIG, TRANSFORMATIONS
from predictor.meta_learning import MetaLearner
from predictor.feature_imagification import FeatureImagification


class TransformationPredictor:
    """
    Preditor de transformações que combina meta-aprendizado e imagificação de features
    para recomendar as transformações mais promissoras para um novo dataset.
    """
    
    def __init__(self, meta_learner: Optional[MetaLearner] = None):
        """
        Inicializa o preditor de transformações.
        
        Args:
            meta_learner: Instância do MetaLearner (opcional)
        """
        self.logger = logging.getLogger(__name__)
        self.meta_learner = meta_learner if meta_learner else MetaLearner()
        self.feature_imagifier = FeatureImagification()
        self.recommendation_threshold = LEARNER_PREDICTOR_CONFIG['recommendation_threshold']
    
    def predict_transformations(
        self,
        data: pd.DataFrame,
        dataset_type: str,
        target: Optional[Union[str, pd.Series]] = None,
        max_recommendations_per_feature: int = 5
    ) -> Dict[str, List[str]]:
        """
        Prediz as transformações mais promissoras para um novo dataset.
        
        Args:
            data: DataFrame com os dados
            dataset_type: Tipo de dataset ('tabular_classification', 'tabular_regression', etc.)
            target: Nome da coluna alvo ou Series com valores alvo (opcional)
            max_recommendations_per_feature: Número máximo de recomendações por feature
            
        Returns:
            Dicionário com {nome_da_feature: [transformações_recomendadas]}
        """
        self.logger.info(f"Predizendo transformações para dataset do tipo {dataset_type}")
        
        # Extrair propriedades do dataset
        data_properties = self._extract_data_properties(data, target, dataset_type)
        
        # Encontrar datasets similares no histórico
        similar_datasets = self.meta_learner.find_similar_datasets(
            dataset_type, data_properties
        )
        
        # Se não há datasets similares, usar abordagem baseada em tipos de dados
        if not similar_datasets:
            self.logger.info("Nenhum dataset similar encontrado, usando abordagem baseada em tipos")
            return self._recommend_by_data_types(data)
        
        self.logger.info(f"Encontrados {len(similar_datasets)} datasets similares")
        
        # Imagificar dataset para comparação visual
        imagified_features = self.feature_imagifier.imagify_dataset(data, target)
        
        # Inicializar dicionário de recomendações
        recommendations = {}
        
        # Para cada coluna, recomendar transformações
        for column in data.columns:
            if isinstance(target, str) and column == target:
                continue  # Pular a coluna alvo
            
            # Detectar tipo da coluna
            column_type = self._detect_column_type(data[column])
            
            # Obter transformações candidatas para este tipo
            candidate_transformations = self._get_candidate_transformations(column_type)
            
            # Avaliar cada transformação candidata
            transformation_scores = []
            
            for transformation in candidate_transformations:
                # Criar objeto de transformação
                transform_obj = {
                    'transformation_type': transformation,
                    'transformation_params': {'column': column}
                }
                
                # Predizer eficácia da transformação
                score = self.meta_learner.predict_transformation_effectiveness(
                    dataset_type, data_properties, transform_obj
                )
                
                transformation_scores.append((transformation, score))
            
            # Ordenar transformações por score
            transformation_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Selecionar as melhores transformações acima do limiar
            recommended_transforms = []
            
            for transform, score in transformation_scores:
                if score >= self.recommendation_threshold and len(recommended_transforms) < max_recommendations_per_feature:
                    recommended_transforms.append(transform)
            
            # Se nenhuma transformação atender ao limiar, incluir pelo menos uma
            if not recommended_transforms and transformation_scores:
                best_transform, _ = transformation_scores[0]
                recommended_transforms.append(best_transform)
            
            # Adicionar ao dicionário de recomendações
            recommendations[column] = recommended_transforms
        
        self.logger.info(f"Recomendadas transformações para {len(recommendations)} features")
        return recommendations
    
    def _extract_data_properties(
        self,
        data: pd.DataFrame,
        target: Optional[Union[str, pd.Series]],
        dataset_type: str
    ) -> Dict[str, Any]:
        """
        Extrai propriedades de um dataset.
        
        Args:
            data: DataFrame com os dados
            target: Nome da coluna alvo ou Series com valores alvo
            dataset_type: Tipo de dataset
            
        Returns:
            Dicionário com propriedades do dataset
        """
        # Extrair informações básicas
        properties = {
            'num_samples': len(data),
            'num_features': len(data.columns),
            'dataset_type': dataset_type,
        }
        
        # Contar tipos de colunas
        column_types = {
            'numeric': 0,
            'categorical': 0,
            'datetime': 0,
            'text': 0
        }
        
        for column in data.columns:
            col_type = self._detect_column_type(data[column])
            column_types[col_type] += 1
        
        properties.update({
            'num_numeric': column_types['numeric'],
            'num_categorical': column_types['categorical'],
            'num_datetime': column_types['datetime'],
            'num_text': column_types['text']
        })
        
        # Calcular proporção de valores ausentes
        properties['missing_ratio'] = data.isnull().mean().mean()
        
        # Extrair tipo de alvo
        if isinstance(target, str) and target in data.columns:
            properties['target_type'] = self._detect_column_type(data[target])
        elif isinstance(target, pd.Series):
            properties['target_type'] = self._detect_column_type(target)
        else:
            properties['target_type'] = 'unknown'
        
        # Calcular estatísticas para colunas numéricas
        numeric_stats = {
            'mean_skewness': 0,
            'mean_kurtosis': 0,
            'mean_variance': 0
        }
        
        numeric_columns = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        
        if numeric_columns:
            skewness_values = []
            kurtosis_values = []
            variance_values = []
            
            for col in numeric_columns:
                # Ignorar colunas com valores ausentes
                if data[col].isnull().any():
                    continue
                
                try:
                    skewness_values.append(data[col].skew())
                    kurtosis_values.append(data[col].kurtosis())
                    variance_values.append(data[col].var())
                except Exception:
                    pass
            
            if skewness_values:
                numeric_stats['mean_skewness'] = np.mean(skewness_values)
            if kurtosis_values:
                numeric_stats['mean_kurtosis'] = np.mean(kurtosis_values)
            if variance_values:
                numeric_stats['mean_variance'] = np.mean(variance_values)
        
        properties['feature_stats'] = numeric_stats
        
        return properties
    
    def _detect_column_type(self, column: pd.Series) -> str:
        """
        Detecta o tipo de dados de uma coluna.
        
        Args:
            column: Série do pandas com os dados da coluna
            
        Returns:
            Tipo de dados ('numeric', 'categorical', 'datetime', 'text')
        """
        if pd.api.types.is_numeric_dtype(column):
            return 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(column):
            return 'datetime'
        elif column.dtype == 'object':
            # Verificar se é texto (mais de 10 caracteres em média)
            if column.astype(str).str.len().mean() > 10:
                return 'text'
            else:
                return 'categorical'
        else:
            return 'categorical'
    
    def _get_candidate_transformations(self, column_type: str) -> List[str]:
        """
        Retorna transformações candidatas para um tipo de coluna.
        
        Args:
            column_type: Tipo de coluna ('numeric', 'categorical', 'datetime', 'text')
            
        Returns:
            Lista de transformações candidatas
        """
        return TRANSFORMATIONS.get(column_type, [])
    
    def _recommend_by_data_types(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Recomenda transformações baseadas apenas nos tipos de dados das colunas.
        
        Args:
            data: DataFrame com os dados
            
        Returns:
            Dicionário com {nome_da_coluna: [transformações_recomendadas]}
        """
        recommendations = {}
        
        for column in data.columns:
            column_type = self._detect_column_type(data[column])
            transformations = self._get_default_transformations(column_type, data[column])
            
            if transformations:
                recommendations[column] = transformations
        
        return recommendations
    
    def _get_default_transformations(self, column_type: str, column: pd.Series) -> List[str]:
        """
        Retorna transformações padrão para um tipo de coluna.
        
        Args:
            column_type: Tipo de coluna
            column: Dados da coluna
            
        Returns:
            Lista de transformações padrão
        """
        # Obter todas as transformações possíveis para este tipo
        all_transforms = TRANSFORMATIONS.get(column_type, [])
        
        # Selecionar as transformações mais comuns para cada tipo
        if column_type == 'numeric':
            # Verificar estatísticas da coluna para decidir transformações
            try:
                skewness = column.skew()
                has_zeros = (column == 0).any()
                has_negatives = (column < 0).any()
                
                default_transforms = []
                
                # Para dados assimétricos positivos, log e sqrt são úteis
                if skewness > 1 and not has_negatives and not has_zeros:
                    default_transforms.extend(['log', 'sqrt'])
                
                # Para dados assimétricos negativos, considerar outras transformações
                elif skewness < -1 and not has_zeros:
                    default_transforms.extend(['square', 'reciprocal'])
                
                # Transformações básicas sempre úteis
                default_transforms.extend(['standardize', 'min_max_scale'])
                
                # Limitar a 3 transformações
                return default_transforms[:3]
                
            except Exception:
                # Em caso de erro, usar transformações padrão
                return ['standardize', 'min_max_scale', 'boxcox']
        
        elif column_type == 'categorical':
            return ['one_hot_encode', 'label_encode', 'target_encode']
        
        elif column_type == 'datetime':
            return ['extract_year', 'extract_month', 'extract_dayofweek', 'time_since_reference']
        
        elif column_type == 'text':
            return ['word_count', 'char_count', 'tfidf']
        
        else:
            return []
    
    def validate_recommendations(
        self,
        recommendations: Dict[str, List[str]],
        data: pd.DataFrame,
        dataset_handler: Any
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Valida rapidamente as recomendações de transformações.
        
        Args:
            recommendations: Dicionário com recomendações
            data: DataFrame com os dados
            dataset_handler: Handler específico para o tipo de dataset
            
        Returns:
            Dicionário com {nome_da_coluna: [(transformação, score)]}
        """
        self.logger.info("Validando recomendações de transformações")
        
        validated_recommendations = {}
        
        for column, transforms in recommendations.items():
            if column not in data.columns:
                continue
            
            scores = []
            
            for transform in transforms:
                try:
                    # Aplicar transformação usando o handler
                    transformed = dataset_handler.apply_single_transformation(
                        data,
                        transformation_type=transform,
                        transformation_params={'column': column}
                    )
                    
                    # Verificar se a transformação gerou resultados válidos
                    if transformed is not None and not transformed.isnull().all():
                        # Avaliar qualidade da transformação
                        quality_score = self._evaluate_transformation_quality(
                            transformed, data[column]
                        )
                        scores.append((transform, quality_score))
                except Exception as e:
                    self.logger.warning(f"Erro ao validar transformação {transform} para {column}: {str(e)}")
            
            # Ordenar por score
            scores.sort(key=lambda x: x[1], reverse=True)
            
            if scores:
                validated_recommendations[column] = scores
        
        return validated_recommendations
    
    def _evaluate_transformation_quality(
        self,
        transformed: pd.Series,
        original: pd.Series
    ) -> float:
        """
        Avalia a qualidade de uma transformação.
        
        Args:
            transformed: Série transformada
            original: Série original
            
        Returns:
            Score de qualidade entre 0 e 1
        """
        try:
            # Verificar se ambas são numéricas
            if not pd.api.types.is_numeric_dtype(transformed) or transformed.isnull().all():
                return 0.5  # Score neutro para transformações não numéricas
            
            # Calcular estatísticas da transformação
            skewness_original = abs(original.skew()) if pd.api.types.is_numeric_dtype(original) else float('inf')
            skewness_transformed = abs(transformed.skew())
            
            # Melhora na assimetria (valores mais próximos de 0 são melhores)
            skewness_improvement = max(0, min(1, (skewness_original - skewness_transformed) / max(1, skewness_original)))
            
            # Verificar valores ausentes
            missing_ratio_original = original.isnull().mean()
            missing_ratio_transformed = transformed.isnull().mean()
            
            # Penalizar aumento de valores ausentes
            missing_penalty = max(0, missing_ratio_transformed - missing_ratio_original)
            
            # Calcular score final
            quality_score = 0.7 * skewness_improvement - 0.3 * missing_penalty + 0.5
            
            # Limitar entre 0 e 1
            return max(0, min(1, quality_score))
            
        except Exception:
            return 0.5  # Score neutro em caso de erro
