#!/usr/bin/env python3
"""
Script de validação simplificado para testar as funcionalidades básicas do AutoFE.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import logging
import warnings

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutoFE-Simple-Validation")
warnings.filterwarnings("ignore")

# Importe os módulos do AutoFE
from preprocessor import PreProcessor

def test_with_iris():
    """
    Testa o AutoFE com o dataset Iris (classificação).
    """
    logger.info("=== Testando com dataset Iris (Classificação) ===")
    
    # Carrega o dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    logger.info(f"Dataset carregado: {df.shape}")
    
    # Define a coluna alvo
    target_col = 'target'
    
    # Divide em conjuntos de treino e teste
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    # Configura o preprocessador
    config = {
        'missing_values_strategy': 'median',
        'outlier_method': 'iqr',
        'categorical_strategy': 'onehot',
        'scaling': 'standard',
        'generate_features': True,
        'correlation_threshold': 0.9,
        'verbosity': 1
    }
    
    # Cria e ajusta o preprocessador
    preprocessor = PreProcessor(config)
    preprocessor.fit(train_df, target_col=target_col)
    
    # Transforma os dados
    train_transformed = preprocessor.transform(train_df, target_col=target_col)
    test_transformed = preprocessor.transform(test_df, target_col=target_col)
    
    logger.info(f"Dados transformados: {train_transformed.shape}, {test_transformed.shape}")
    
    # Treina um modelo
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(train_transformed.drop(target_col, axis=1), train_transformed[target_col])
    
    # Avalia o modelo
    y_pred = model.predict(test_transformed.drop(target_col, axis=1))
    accuracy = accuracy_score(test_transformed[target_col], y_pred)
    
    logger.info(f"Acurácia: {accuracy:.4f}")
    
    return {
        'accuracy': accuracy,
        'original_features': df.shape[1] - 1,
        'transformed_features': train_transformed.shape[1] - 1
    }

def test_with_california():
    """
    Testa o AutoFE com o dataset California Housing (regressão).
    """
    logger.info("=== Testando com dataset California Housing (Regressão) ===")
    
    # Carrega o dataset
    california = fetch_california_housing()
    df = pd.DataFrame(data=california.data, columns=california.feature_names)
    df['target'] = california.target
    
    # Usa uma amostra para processamento mais rápido
    df = df.sample(1000, random_state=42)
    
    logger.info(f"Dataset carregado: {df.shape}")
    
    # Define a coluna alvo
    target_col = 'target'
    
    # Divide em conjuntos de treino e teste
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    # Configura o preprocessador
    config = {
        'missing_values_strategy': 'median',
        'outlier_method': 'iqr',
        'categorical_strategy': 'onehot',
        'scaling': 'robust',
        'generate_features': True,
        'correlation_threshold': 0.9,
        'verbosity': 1
    }
    
    # Cria e ajusta o preprocessador
    preprocessor = PreProcessor(config)
    preprocessor.fit(train_df, target_col=target_col)
    
    # Transforma os dados
    train_transformed = preprocessor.transform(train_df, target_col=target_col)
    test_transformed = preprocessor.transform(test_df, target_col=target_col)
    
    logger.info(f"Dados transformados: {train_transformed.shape}, {test_transformed.shape}")
    
    # Treina um modelo
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(train_transformed.drop(target_col, axis=1), train_transformed[target_col])
    
    # Avalia o modelo
    y_pred = model.predict(test_transformed.drop(target_col, axis=1))
    r2 = r2_score(test_transformed[target_col], y_pred)
    
    logger.info(f"R²: {r2:.4f}")
    
    return {
        'r2': r2,
        'original_features': df.shape[1] - 1,
        'transformed_features': train_transformed.shape[1] - 1
    }

def test_with_mixed_data():
    """
    Testa o AutoFE com um dataset misto criado artificialmente.
    """
    logger.info("=== Testando com dataset misto artificial ===")
    
    # Cria um dataset misto com colunas numéricas e categóricas
    np.random.seed(42)
    n_samples = 1000
    
    # Colunas numéricas
    num_data = {
        'age': np.random.normal(35, 10, n_samples).astype(int),
        'income': np.random.lognormal(10, 0.5, n_samples),
        'score': np.random.uniform(0, 100, n_samples)
    }
    
    # Colunas categóricas
    cat_data = {
        'gender': np.random.choice(['M', 'F'], n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    }
    
    # Criando o target (binário)
    age_factor = (num_data['age'] > 30).astype(int)
    income_factor = (num_data['income'] > np.median(num_data['income'])).astype(int)
    gender_factor = (cat_data['gender'] == 'F').astype(int)
    
    target = (age_factor + income_factor + gender_factor >= 2).astype(int)
    
    # Combina tudo em um DataFrame
    df = pd.DataFrame({**num_data, **cat_data, 'target': target})
    
    # Adiciona alguns valores ausentes
    for col in df.columns:
        if col != 'target':
            mask = np.random.choice([True, False], df.shape[0], p=[0.05, 0.95])
            df.loc[mask, col] = np.nan
    
    logger.info(f"Dataset misto criado: {df.shape}")
    logger.info(f"Tipos de colunas: {df.dtypes}")
    logger.info(f"Valores ausentes: {df.isnull().sum().sum()}")
    
    # Define a coluna alvo
    target_col = 'target'
    
    # Divide em conjuntos de treino e teste
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    # Configura o preprocessador
    config = {
        'missing_values_strategy': 'median',
        'outlier_method': 'iqr',
        'categorical_strategy': 'onehot',
        'scaling': 'standard',
        'generate_features': True,
        'correlation_threshold': 0.9,
        'verbosity': 1
    }
    
    # Cria e ajusta o preprocessador
    preprocessor = PreProcessor(config)
    preprocessor.fit(train_df, target_col=target_col)
    
    # Transforma os dados
    train_transformed = preprocessor.transform(train_df, target_col=target_col)
    test_transformed = preprocessor.transform(test_df, target_col=target_col)
    
    logger.info(f"Dados transformados: {train_transformed.shape}, {test_transformed.shape}")
    
    # Treina um modelo
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(train_transformed.drop(target_col, axis=1), train_transformed[target_col])
    
    # Avalia o modelo
    y_pred = model.predict(test_transformed.drop(target_col, axis=1))
    accuracy = accuracy_score(test_transformed[target_col], y_pred)
    
    logger.info(f"Acurácia: {accuracy:.4f}")
    
    return {
        'accuracy': accuracy,
        'original_features': df.shape[1] - 1,
        'transformed_features': train_transformed.shape[1] - 1
    }

if __name__ == "__main__":
    logger.info("Iniciando validação simplificada do AutoFE...")
    
    try:
        # Testa com dataset Iris
        iris_results = test_with_iris()
        
        # Testa com dataset California Housing
        california_results = test_with_california()
        
        # Testa com dataset misto artificial
        mixed_results = test_with_mixed_data()
        
        # Imprime um resumo
        logger.info("\n=== Resumo dos Resultados ===")
        logger.info(f"Iris (Classificação):")
        logger.info(f"  - Acurácia: {iris_results['accuracy']:.4f}")
        logger.info(f"  - Features originais: {iris_results['original_features']}")
        logger.info(f"  - Features após transformação: {iris_results['transformed_features']}")
        
        logger.info(f"\nCalifornia Housing (Regressão):")
        logger.info(f"  - R²: {california_results['r2']:.4f}")
        logger.info(f"  - Features originais: {california_results['original_features']}")
        logger.info(f"  - Features após transformação: {california_results['transformed_features']}")
        
        logger.info(f"\nDataset Misto (Classificação):")
        logger.info(f"  - Acurácia: {mixed_results['accuracy']:.4f}")
        logger.info(f"  - Features originais: {mixed_results['original_features']}")
        logger.info(f"  - Features após transformação: {mixed_results['transformed_features']}")
        
    except Exception as e:
        logger.error(f"Erro durante a validação: {e}", exc_info=True)
    
    logger.info("Validação simplificada concluída!")