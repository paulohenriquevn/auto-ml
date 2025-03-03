import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import fetch_california_housing, load_iris
from preprocessor import PreProcessor

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutoFE-Testing")


def test_classification_iris():
    """Teste de classificação com dataset Iris."""
    logger.info("=== Testando AutoFE com dataset Iris ===")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    target_col = 'target'
    
    config = {'missing_values_strategy': 'median', 'outlier_method': 'iqr', 'scaling': 'standard'}
    preprocessor = PreProcessor(config)
    preprocessor.fit(train_df, target_col)
    train_transformed = preprocessor.transform(train_df, target_col)
    test_transformed = preprocessor.transform(test_df, target_col)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(train_transformed.drop(columns=[target_col]), train_transformed[target_col])
    y_pred = model.predict(test_transformed.drop(columns=[target_col]))
    
    accuracy = accuracy_score(test_transformed[target_col], y_pred)
    logger.info(f"Acurácia: {accuracy:.4f}")
    return accuracy


def test_regression_california():
    """Teste de regressão com o dataset California Housing."""
    logger.info("=== Testando AutoFE com dataset California Housing ===")
    california = fetch_california_housing()
    df = pd.DataFrame(data=california.data, columns=california.feature_names)
    df['target'] = california.target
    
    df = df.sample(1000, random_state=42)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    target_col = 'target'
    
    config = {'missing_values_strategy': 'median', 'outlier_method': 'iqr', 'scaling': 'robust'}
    preprocessor = PreProcessor(config)
    preprocessor.fit(train_df, target_col)
    train_transformed = preprocessor.transform(train_df, target_col)
    test_transformed = preprocessor.transform(test_df, target_col)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(train_transformed.drop(columns=[target_col]), train_transformed[target_col])
    y_pred = model.predict(test_transformed.drop(columns=[target_col]))
    
    r2 = r2_score(test_transformed[target_col], y_pred)
    logger.info(f"R²: {r2:.4f}")
    return r2


def test_synthetic_classification():
    """Teste com dados sintéticos para classificação."""
    logger.info("=== Testando AutoFE com dataset sintético ===")
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.uniform(0, 10, n_samples),
        'feature3': np.random.randint(0, 100, n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    target_col = 'target'
    
    config = {'missing_values_strategy': 'most_frequent', 'outlier_method': 'zscore', 'scaling': 'minmax'}
    preprocessor = PreProcessor(config)
    preprocessor.fit(train_df, target_col)
    train_transformed = preprocessor.transform(train_df, target_col)
    test_transformed = preprocessor.transform(test_df, target_col)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(train_transformed.drop(columns=[target_col]), train_transformed[target_col])
    y_pred = model.predict(test_transformed.drop(columns=[target_col]))
    
    accuracy = accuracy_score(test_transformed[target_col], y_pred)
    logger.info(f"Acurácia: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    logger.info("Iniciando testes do AutoFE...")
    
    acc_iris = test_classification_iris()
    r2_california = test_regression_california()
    acc_synthetic = test_synthetic_classification()
    
    logger.info("\n=== Resumo dos Testes ===")
    logger.info(f"Acurácia - Iris: {acc_iris:.4f}")
    logger.info(f"R² - California Housing: {r2_california:.4f}")
    logger.info(f"Acurácia - Sintético: {acc_synthetic:.4f}")
