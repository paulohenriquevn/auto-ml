"""
Configurações globais para o sistema de automação de engenharia de features.
"""

# Configurações do Explorer
EXPLORER_CONFIG = {
    # Número máximo de transformações a serem aplicadas em sequência
    'max_transformation_depth': 3,
    
    # Número máximo de features a serem consideradas em cada nível
    'max_features_per_level': 30,
    
    # Limite mínimo de ganho de performance para manter uma feature
    'min_performance_gain': 0.01,
    
    # Tempo máximo (em segundos) para exploração
    'exploration_timeout': 3600,
    
    # Métrica de avaliação padrão para regressão
    'regression_metric': 'rmse',
    
    # Métrica de avaliação padrão para classificação
    'classification_metric': 'auc',
    
    # Critério para eliminação de features redundantes (correlação)
    'redundancy_threshold': 0.95,
}

# Configurações do Learner-Predictor
LEARNER_PREDICTOR_CONFIG = {
    # Caminho para salvar o histórico de transformações
    'history_path': './transformation_history/',
    
    # Número de datasets similares a considerar para recomendações
    'n_similar_datasets': 5,
    
    # Limite de confiança para recomendar transformações
    'recommendation_threshold': 0.7,
    
    # Número de bins para imagificação de features
    'imagification_bins': 30,
    
    # Modelo de meta-aprendizado (options: 'random_forest', 'xgboost', 'neural_network')
    'meta_model': 'random_forest',
}

# Configurações específicas para diferentes tipos de datasets
DATASET_HANDLERS_CONFIG = {
    'tabular_classification': {
        'default_model': 'random_forest',
        'validation_strategy': 'stratified_cv',
        'n_folds': 5,
    },
    'tabular_regression': {
        'default_model': 'random_forest',
        'validation_strategy': 'cv',
        'n_folds': 5,
    },
    'tabular_to_text': {
        'default_model': 'neural_network',
        'validation_strategy': 'train_test_split',
        'test_size': 0.2,
    },
    'time_series': {
        'default_model': 'xgboost',
        'validation_strategy': 'time_based_split',
        'n_splits': 3,
        'gap': 0,
    }
}

# Lista de transformações disponíveis para diferentes tipos de dados
TRANSFORMATIONS = {
    'numeric': [
        'log', 'sqrt', 'square', 'cube', 'reciprocal', 
        'sin', 'cos', 'tan', 'sigmoid', 'tanh',
        'standardize', 'normalize', 'min_max_scale',
        'quantile_transform', 'power_transform', 'boxcox'
    ],
    'categorical': [
        'one_hot_encode', 'label_encode', 'target_encode', 
        'count_encode', 'frequency_encode', 'mean_encode',
        'hash_encode', 'weight_of_evidence'
    ],
    'datetime': [
        'extract_year', 'extract_month', 'extract_day', 
        'extract_hour', 'extract_minute', 'extract_second',
        'extract_dayofweek', 'extract_quarter', 'is_weekend',
        'time_since_reference', 'time_to_event'
    ],
    'text': [
        'word_count', 'char_count', 'stop_word_count',
        'unique_word_count', 'uppercase_count', 'lowercase_count',
        'punctuation_count', 'tfidf', 'word_embeddings',
        'sentiment_score', 'readability_score'
    ],
    'time_series': [
        'lag', 'rolling_mean', 'rolling_std', 'rolling_min', 
        'rolling_max', 'rolling_median', 'exponential_moving_average',
        'differencing', 'decompose_trend', 'decompose_seasonal',
        'fourier_features', 'autocorrelation'
    ],
    'interaction': [
        'sum', 'difference', 'product', 'ratio', 'polynomial'
    ],
}

# Configurações de logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': './logs/autofeature.log',
}
