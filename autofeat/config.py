REGRESSION_CONFIGS = [
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

# Também podemos definir o IMBALANCED_CONFIGS para completude
IMBALANCED_CONFIGS = [
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