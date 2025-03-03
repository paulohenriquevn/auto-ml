import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import fetch_california_housing, load_iris
from preprocessor import PreProcessor
from preprocessor import Explorer

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutoFE-Testing")

def test_classification_iris():
    """Teste de classificação com dataset Iris."""
    logger.info("=== Testando AutoFE com dataset Iris ===")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    target_col = 'target'
    explorador = Explorer(target_col=target_col)
    df_transformed = explorador.analyze_transformations(df)
    
    train_df, test_df = train_test_split(df_transformed, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(train_df.drop(columns=[target_col]), train_df[target_col])
    y_pred = model.predict(test_df.drop(columns=[target_col]))
    
    accuracy = accuracy_score(test_df[target_col], y_pred)
    logger.info(f"Acurácia: {accuracy:.4f}")
    return accuracy

def test_fraud_classification():
    """Teste com o dataset de fraude em cartões de crédito."""
    logger.info("=== Testando AutoFE com dataset de fraude em cartões ===")
    df = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")
    target_col = 'Class'
    
    # Aplicar o Explorer antes da separação para garantir consistência
    explorador = Explorer(target_col=target_col)
    df_transformed = explorador.analyze_transformations(df)

    # Garantir que df_transformed inclui apenas as amostras válidas do target
    df_target_filtered = df.loc[df_transformed.index, target_col]

    # Agora o train_test_split terá dados consistentes
    train_df, test_df, y_train, y_test = train_test_split(
        df_transformed, df_target_filtered, test_size=0.3, random_state=42, stratify=df_target_filtered
    )

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(train_df, y_train)
    y_pred = model.predict(test_df)

    auc_roc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"AUC-ROC: {auc_roc:.4f}, F1-Score: {f1:.4f}")
    return auc_roc, f1


def test_heart_disease_classification():
    """Teste com o dataset de doenças cardíacas."""
    logger.info("=== Testando AutoFE com dataset de doenças cardíacas ===")
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header=None)
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = df.replace('?', np.nan).dropna()
    df['target'] = (df['target'] > 0).astype(int)
    
    target_col = 'target'
    explorador = Explorer(target_col=target_col)
    df_transformed = explorador.analyze_transformations(df)
    
    train_df, test_df = train_test_split(df_transformed, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(train_df.drop(columns=[target_col]), train_df[target_col])
    y_pred = model.predict(test_df.drop(columns=[target_col]))
    auc_roc = roc_auc_score(test_df[target_col], y_pred)
    f1 = f1_score(test_df[target_col], y_pred)
    
    logger.info(f"AUC-ROC: {auc_roc:.4f}, F1-Score: {f1:.4f}")
    return auc_roc, f1

if __name__ == "__main__":
    logger.info("Iniciando testes do AutoFE...")
    
    acc_iris = test_classification_iris()
    auc_fraud, f1_fraud = test_fraud_classification()
    auc_heart, f1_heart = test_heart_disease_classification()
    
    logger.info("\n=== Resumo dos Testes ===")
    logger.info(f"Acurácia - Iris: {acc_iris:.4f}")
    logger.info(f"AUC-ROC - Fraude: {auc_fraud:.4f}, F1-Score: {f1_fraud:.4f}")
    logger.info(f"AUC-ROC - Doenças Cardíacas: {auc_heart:.4f}, F1-Score: {f1_heart:.4f}")