"""
Implementação do refinamento adaptativo para o Explorer.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.cluster import KMeans

# Importações internas
from config import EXPLORER_CONFIG


class FeatureRefinement:
    """
    Responsável pelo refinamento adaptativo de features, incluindo:
    - Eliminação de features irrelevantes
    - Priorização de features interpretáveis
    - Redução automática de dimensionalidade
    """
    
    def __init__(self):
        """
        Inicializa o módulo de refinamento de features.
        """
        self.logger = logging.getLogger(__name__)
    
    def refine(
        self,
        candidate_features: List[Dict[str, Any]],
        data: pd.DataFrame,
        target: Union[str, pd.Series],
        dataset_handler: Any,
        max_features: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Refina o conjunto de features candidatas.
        
        Args:
            candidate_features: Lista de features candidatas
            data: DataFrame com os dados
            target: Variável alvo
            dataset_handler: Handler específico para o tipo de dataset
            max_features: Número máximo de features a selecionar (None = sem limite)
            
        Returns:
            Lista refinada de features
        """
        self.logger.info(f"Iniciando refinamento de {len(candidate_features)} features candidatas")
        
        # Extrair dados de entrada e alvo
        if isinstance(target, str):
            y = data[target]
            X = data.drop(columns=[target])
        else:
            y = target
            X = data
        
        # 1. Eliminar features irrelevantes (baixo ganho de performance)
        refined_features = self._remove_irrelevant_features(candidate_features)
        self.logger.info(f"Após remoção de irrelevantes: {len(refined_features)} features")
        
        # Se não sobrou nenhuma feature após filtragem, retornar lista vazia
        if not refined_features:
            self.logger.warning("Nenhuma feature relevante encontrada")
            return []
        
        # 2. Aplicar transformações selecionadas para criar o DataFrame aumentado
        augmented_data = self._apply_candidate_transformations(refined_features, X, dataset_handler)
        
        # 3. Eliminar redundâncias (alta correlação entre features)
        refined_features = self._remove_redundant_features(refined_features, augmented_data)
        self.logger.info(f"Após remoção de redundantes: {len(refined_features)} features")
        
        # 4. Priorizar features interpretáveis
        refined_features = self._prioritize_interpretable_features(refined_features)
        
        # 5. Aplicar redução de dimensionalidade se necessário
        if max_features and len(refined_features) > max_features:
            refined_features = self._apply_dimensionality_reduction(
                refined_features, augmented_data, y, dataset_handler, max_features
            )
            self.logger.info(f"Após redução de dimensionalidade: {len(refined_features)} features")
        
        self.logger.info(f"Refinamento concluído, {len(refined_features)} features selecionadas")
        return refined_features
    
    def _remove_irrelevant_features(
        self,
        candidate_features: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove features com baixo ganho de performance.
        
        Args:
            candidate_features: Lista de features candidatas
            
        Returns:
            Lista filtrada de features
        """
        min_gain = EXPLORER_CONFIG['min_performance_gain']
        return [
            feature for feature in candidate_features
            if feature['performance_gain'] >= min_gain
        ]
    
    def _apply_candidate_transformations(
        self,
        candidate_features: List[Dict[str, Any]],
        X: pd.DataFrame,
        dataset_handler: Any
    ) -> pd.DataFrame:
        """
        Aplica as transformações candidatas para criar um DataFrame aumentado.
        
        Args:
            candidate_features: Lista de features candidatas
            X: DataFrame original
            dataset_handler: Handler específico para o tipo de dataset
            
        Returns:
            DataFrame aumentado com as transformações aplicadas
        """
        # Criar cópia do DataFrame original
        augmented_data = X.copy()
        
        # Para cada feature candidata, aplicar a transformação
        for feature in candidate_features:
            try:
                # Usar o handler para aplicar a transformação
                transformed_feature = dataset_handler.apply_single_transformation(
                    X,
                    transformation_type=feature['transformation_type'],
                    transformation_params=feature['transformation_params']
                )
                
                # Adicionar ao DataFrame aumentado se válido
                if transformed_feature is not None and not transformed_feature.isnull().all():
                    augmented_data[feature['name']] = transformed_feature
                else:
                    self.logger.warning(f"Transformação {feature['name']} produziu valores inválidos")
            except Exception as e:
                self.logger.warning(f"Erro ao aplicar transformação {feature['name']}: {str(e)}")
        
        return augmented_data
    
    def _remove_redundant_features(
        self,
        candidate_features: List[Dict[str, Any]],
        augmented_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Remove features redundantes com base na correlação.
        
        Args:
            candidate_features: Lista de features candidatas
            augmented_data: DataFrame com todas as features (originais e transformadas)
            
        Returns:
            Lista filtrada de features
        """
        # Ordenar features por importância
        sorted_features = sorted(
            candidate_features,
            key=lambda f: f['performance_gain'] * f['importance'],
            reverse=True
        )
        
        # Inicializar conjunto de features selecionadas
        selected_features = []
        selected_names = set()
        
        # Limiar de redundância
        redundancy_threshold = EXPLORER_CONFIG['redundancy_threshold']
        
        # Para cada feature, verificar redundância com as já selecionadas
        for feature in sorted_features:
            feature_name = feature['name']
            
            # Verificar se a feature está no DataFrame aumentado
            if feature_name not in augmented_data.columns:
                continue
            
            # Para features numéricas, verificar correlação
            if pd.api.types.is_numeric_dtype(augmented_data[feature_name]):
                # Verificar correlação com features já selecionadas
                is_redundant = False
                
                for selected_name in selected_names:
                    if selected_name in augmented_data.columns and pd.api.types.is_numeric_dtype(augmented_data[selected_name]):
                        # Calcular correlação absoluta
                        correlation = augmented_data[[feature_name, selected_name]].corr().abs().iloc[0, 1]
                        
                        if np.isnan(correlation):
                            continue
                        
                        if correlation >= redundancy_threshold:
                            is_redundant = True
                            break
                
                if not is_redundant:
                    selected_features.append(feature)
                    selected_names.add(feature_name)
            else:
                # Para features não numéricas, adicionar diretamente
                selected_features.append(feature)
                selected_names.add(feature_name)
        
        return selected_features
    
    def _prioritize_interpretable_features(
        self,
        candidate_features: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Prioriza features mais interpretáveis.
        
        Args:
            candidate_features: Lista de features candidatas
            
        Returns:
            Lista ordenada de features
        """
        # Definir pesos de interpretabilidade para diferentes tipos de transformação
        interpretability_weights = {
            # Transformações simples (alta interpretabilidade)
            'sqrt': 1.0,
            'log': 0.9,
            'square': 0.9,
            'reciprocal': 0.8,
            'standardize': 0.9,
            'min_max_scale': 0.9,
            
            # Transformações estatísticas (média interpretabilidade)
            'rolling_mean': 0.7,
            'rolling_std': 0.7,
            'rolling_min': 0.7,
            'rolling_max': 0.7,
            'lag': 0.7,
            
            # Transformações complexas (baixa interpretabilidade)
            'fourier_features': 0.4,
            'polynomial': 0.5,
            'boxcox': 0.6,
            'quantile_transform': 0.5,
            
            # Valor padrão para transformações não listadas
            'default': 0.6
        }
        
        # Atribuir peso de interpretabilidade a cada feature
        for feature in candidate_features:
            transformation_type = feature['transformation_type']
            interpretability = interpretability_weights.get(transformation_type, interpretability_weights['default'])
            feature['interpretability'] = interpretability
        
        # Ordenar por uma combinação de performance, importância e interpretabilidade
        return sorted(
            candidate_features,
            key=lambda f: (
                f['performance_gain'] * 0.4 +
                f['importance'] * 0.3 +
                f['interpretability'] * 0.3
            ),
            reverse=True
        )
    
    def _apply_dimensionality_reduction(
        self,
        candidate_features: List[Dict[str, Any]],
        augmented_data: pd.DataFrame,
        target: pd.Series,
        dataset_handler: Any,
        max_features: int
    ) -> List[Dict[str, Any]]:
        """
        Aplica técnicas de redução de dimensionalidade.
        
        Args:
            candidate_features: Lista de features candidatas
            augmented_data: DataFrame com todas as features
            target: Variável alvo
            dataset_handler: Handler específico para o tipo de dataset
            max_features: Número máximo de features a selecionar
            
        Returns:
            Lista reduzida de features
        """
        # Verificar se o número de features já é menor que o máximo
        if len(candidate_features) <= max_features:
            return candidate_features
        
        # Extrair nomes das features candidatas
        feature_names = [f['name'] for f in candidate_features if f['name'] in augmented_data.columns]
        
        # Selecionar apenas features numéricas para PCA
        numeric_features = [
            name for name in feature_names 
            if pd.api.types.is_numeric_dtype(augmented_data[name])
        ]
        
        if len(numeric_features) <= max_features:
            # Se temos poucas features numéricas, selecionar as melhores features diretamente
            return candidate_features[:max_features]
        
        # 1. Método 1: Seleção baseada em importância
        # Se estamos usando classificação ou regressão, usar SelectKBest
        if dataset_handler.is_classification():
            selector = SelectKBest(
                score_func=mutual_info_classif,
                k=max_features
            )
        else:
            selector = SelectKBest(
                score_func=mutual_info_regression,
                k=max_features
            )
        
        # Preparar dados para seleção
        X_select = augmented_data[numeric_features].copy()
        
        # Preencher valores ausentes
        for col in X_select.columns:
            if X_select[col].isnull().any():
                X_select[col] = X_select[col].fillna(X_select[col].mean())
        
        # Aplicar seleção
        try:
            X_new = selector.fit_transform(X_select, target)
            selected_indices = selector.get_support(indices=True)
            selected_features = [numeric_features[i] for i in selected_indices]
            
            # Filtrar candidate_features para manter apenas as selecionadas
            selected_candidates = []
            name_to_feature = {f['name']: f for f in candidate_features}
            
            for name in selected_features:
                if name in name_to_feature:
                    selected_candidates.append(name_to_feature[name])
            
            # Adicionar quaisquer features não numéricas que não foram consideradas
            for feature in candidate_features:
                if feature['name'] not in numeric_features and feature not in selected_candidates:
                    selected_candidates.append(feature)
            
            # Limitar ao número máximo
            return selected_candidates[:max_features]
            
        except Exception as e:
            self.logger.warning(f"Erro ao aplicar seleção de features: {str(e)}")
            # Em caso de erro, voltar ao método simples
            return candidate_features[:max_features]
