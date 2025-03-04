import pandas as pd
import numpy as np
from typing import List, Dict, Optional

def get_imbalanced_configs() -> List[Dict]:
    """
    Gera configurações específicas para lidar com datasets desbalanceados.
    
    Returns:
        Lista de configurações de transformação
    """
    # Configurações para desbalanceamento
    imbalanced_configs = [
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
        
        # Borderline SMOTE - melhor para fronteiras de decisão complexas
        {
            'missing_values_strategy': 'knn',
            'outlier_method': 'zscore',
            'categorical_strategy': 'onehot',
            'scaling': 'standard',
            'generate_features': True,
            'balance_classes': True,
            'balance_method': 'borderline',
            'sampling_strategy': 'auto'
        },
        
        # ADASYN - adapta densidade de acordo com a dificuldade de aprendizado
        {
            'missing_values_strategy': 'median',
            'outlier_method': 'isolation_forest',
            'categorical_strategy': 'onehot',
            'scaling': 'standard',
            'generate_features': False,
            'balance_classes': True,
            'balance_method': 'adasyn',
            'sampling_strategy': 'auto'
        },
        
        # Under-sampling com NearMiss
        {
            'missing_values_strategy': 'median',
            'outlier_method': 'zscore',
            'categorical_strategy': 'onehot',
            'scaling': 'standard',
            'generate_features': False,
            'balance_classes': True,
            'balance_method': 'nearmiss',
            'sampling_strategy': 'auto'
        },
        
        # Random Under-sampling - mais rápido, menos inteligente
        {
            'missing_values_strategy': 'median',
            'outlier_method': 'iqr',
            'categorical_strategy': 'onehot',
            'scaling': 'standard',
            'generate_features': False,
            'balance_classes': True,
            'balance_method': 'random_under',
            'sampling_strategy': 'auto'
        },
        
        # Random Over-sampling - mais rápido, menos inteligente que SMOTE
        {
            'missing_values_strategy': 'median',
            'outlier_method': 'iqr',
            'categorical_strategy': 'onehot',
            'scaling': 'standard',
            'generate_features': False,
            'balance_classes': True,
            'balance_method': 'random_over',
            'sampling_strategy': 'auto'
        },
        
        # Combinação SMOTE + ENN (limpeza)
        {
            'missing_values_strategy': 'knn',
            'outlier_method': 'zscore',
            'categorical_strategy': 'onehot',
            'scaling': 'robust',
            'generate_features': True,
            'balance_classes': True,
            'balance_method': 'smoteenn',
            'sampling_strategy': 'auto'
        },
        
        # Combinação SMOTE + Tomek Links (limpeza)
        {
            'missing_values_strategy': 'knn',
            'outlier_method': 'zscore',
            'categorical_strategy': 'onehot',
            'scaling': 'robust',
            'generate_features': True,
            'balance_classes': True,
            'balance_method': 'smotetomek',
            'sampling_strategy': 'auto'
        },
        
        # Usando pesos de amostra em vez de balanceamento
        {
            'missing_values_strategy': 'median',
            'outlier_method': 'zscore',
            'categorical_strategy': 'onehot',
            'scaling': 'standard',
            'generate_features': False,
            'balance_classes': False,
            'use_sample_weights': True
        }
    ]
    
    return imbalanced_configs

def detect_imbalance(df: pd.DataFrame, target_col: str, threshold: float = 0.2) -> bool:
    """
    Detecta se um dataset está desbalanceado.
    
    Args:
        df: DataFrame com os dados
        target_col: Nome da coluna alvo
        threshold: Limite para definir desbalanceamento (razão classe minoritária/majoritária)
        
    Returns:
        True se o dataset estiver desbalanceado, False caso contrário
    """
    if target_col not in df.columns:
        return False
    
    # Verifica se é um problema de classificação
    y = df[target_col]
    if not (pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y) or y.nunique() <= 10):
        return False
    
    # Calcula proporção da classe minoritária
    class_counts = y.value_counts()
    min_class_ratio = class_counts.min() / class_counts.max()
    
    # Considera desbalanceado se a razão for menor que o threshold
    is_imbalanced = min_class_ratio < threshold
    
    if is_imbalanced:
        print(f"Dataset desbalanceado detectado: razão minoritária/majoritária = {min_class_ratio:.4f}")
        print(f"Distribuição de classes: {class_counts.to_dict()}")
    
    return is_imbalanced

def update_explorer_with_imbalanced_configs(df: pd.DataFrame, target_col: str, explorer) -> None:
    """
    Atualiza o Explorer com configurações específicas para dados desbalanceados,
    se o dataset for desbalanceado.
    
    Args:
        df: DataFrame com os dados
        target_col: Nome da coluna alvo
        explorer: Instância do Explorer
    """
    # Verifica se o dataset está desbalanceado
    if detect_imbalance(df, target_col):
        # Adiciona configurações para dados desbalanceados
        imbalanced_configs = get_imbalanced_configs()
        
        for config in imbalanced_configs:
            explorer.combiner.add_base_transformation(config)
            
        print(f"Adicionadas {len(imbalanced_configs)} configurações para lidar com dados desbalanceados")