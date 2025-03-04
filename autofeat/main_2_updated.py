import pandas as pd
import numpy as np
import logging
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import fetch_california_housing, load_iris
from preprocessor import PreProcessor
# --- INTEGRAÇÃO COM BALANCEAMENTO E MÉTRICAS PARA DESBALANCEAMENTO ---
from imbalanced_metrics_evaluator import ImbalancedMetricsEvaluator
from desbalanceamento import detect_imbalance, get_imbalanced_configs


# Adicionar o diretório atual ao path para garantir que o módulo explorer seja encontrado
sys.path.append(os.path.abspath('.'))

# Importar o Explorer melhorado (assumindo que o módulo explorer.py contém a implementação)
from explorer import Explorer, process_transformation

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutoFE-Testing")

# Atualizando os testes para aplicar técnicas de balanceamento automaticamente
def apply_best_balance_strategy(df, target_col):
    """
    Verifica se os dados são desbalanceados e aplica a melhor estratégia de balanceamento.
    """
    if detect_imbalance(df, target_col):
        logger.info("Dados desbalanceados detectados! Aplicando técnica de balanceamento adequada.")
        balance_configs = get_imbalanced_configs()
        best_config = balance_configs[0]  # Inicialmente usa a primeira opção
        
        best_score = float("-inf")
        evaluator = ImbalancedMetricsEvaluator()

        for config in balance_configs:
            temp_preprocessor = PreProcessor(config)
            transformed_df = temp_preprocessor.fit_transform(df)
            metrics = evaluator.evaluate(df[target_col], transformed_df[target_col])

            score = metrics.get('balanced_accuracy', 0)
            if score > best_score:
                best_score = score
                best_config = config

        logger.info(f"Técnica de balanceamento selecionada: {best_config['balance_method']}")
        return PreProcessor(best_config).fit_transform(df)
    
    return df  # Retorna os dados originais se não forem desbalanceados

def test_classification_iris():
    """Teste de classificação com dataset Iris."""
    logger.info("=== Testando AutoFE com dataset Iris ===")
    
    # Carrega o dataset Iris
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    target_col = 'target'
    
    # Usa o Explorer melhorado com paralelização
    explorador = Explorer(
        target_col=target_col,
        problem_type='classification',
        parallel=True,  # Habilitamos a paralelização
        max_depth=2,    # Profundidade limitada para teste mais rápido
        beam_width=3    # Largura limitada para teste mais rápido
    )
    
    # Executa a exploração
    transformation_tree = explorador.explore(df)
    
    # Obtém a melhor transformação
    best_config = explorador.get_best_transformation()
    logger.info(f"Melhor configuração: {best_config}")
    
    # Aplica a transformação usando o PreProcessor
    preprocessor = PreProcessor(best_config)
    preprocessor.fit(df, target_col=target_col)
    df_transformed = preprocessor.transform(df, target_col=target_col)
    
    # Divisão treino/teste
    train_df, test_df = train_test_split(df_transformed, test_size=0.3, random_state=42)
    
    # Treinamento e avaliação
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(train_df.drop(columns=[target_col]), train_df[target_col])
    y_pred = model.predict(test_df.drop(columns=[target_col]))
    
    accuracy = accuracy_score(test_df[target_col], y_pred)
    logger.info(f"Acurácia: {accuracy:.4f}")
    
    # Gera um relatório detalhado
    report = explorador.get_transformation_report(top_k=3)
    logger.info(f"Top 3 transformações: {[t['name'] for t in report['best_transformations']]}")
    
    return accuracy, explorador

def test_fraud_classification():
    """Teste com o dataset de fraude em cartões de crédito."""
    logger.info("=== Testando AutoFE com dataset de fraude em cartões ===")
    try:
        df = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")
        # Usar uma amostra para teste mais rápido
        df = df.sample(frac=0.1, random_state=42)
    except Exception as e:
        logger.error(f"Erro ao carregar dataset de fraude: {e}")
        return None, None
    
    target_col = 'Class'
    
    # Usa o Explorer melhorado
    explorador = Explorer(
        target_col=target_col,
        problem_type='classification',
        parallel=True,
        max_depth=2,
        beam_width=3
    )
    
    # Executa a exploração e obtém a melhor configuração
    explorador.explore(df)
    best_config = explorador.get_best_transformation()
    logger.info(f"Melhor configuração: {best_config}")
    
    # Aplica a transformação
    preprocessor = PreProcessor(best_config)
    preprocessor.fit(df, target_col=target_col)
    df_transformed = preprocessor.transform(df, target_col=target_col)

    # Divisão treino/teste
    X = df_transformed.drop(columns=[target_col])
    y = df_transformed[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Treinamento e avaliação
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    auc_roc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"AUC-ROC: {auc_roc:.4f}, F1-Score: {f1:.4f}")
    return auc_roc, f1


def test_heart_disease_classification():
    """Teste com o dataset de doenças cardíacas."""
    logger.info("=== Testando AutoFE com dataset de doenças cardíacas ===")
    try:
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header=None)
        df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        df = df.replace('?', np.nan).dropna()
        df['target'] = (df['target'] > 0).astype(int)
    except Exception as e:
        logger.error(f"Erro ao carregar dataset de doenças cardíacas: {e}")
        return None, None
    
    target_col = 'target'
    
    # Usa o Explorer melhorado com modo sequencial para garantir compatibilidade
    explorador = Explorer(
        target_col=target_col,
        problem_type='classification',
        parallel=True,  # Podemos testar com e sem paralelização
        max_depth=2,
        beam_width=3
    )
    
    # Executa a exploração e obtém a melhor configuração
    explorador.explore(df)
    best_config = explorador.get_best_transformation()
    logger.info(f"Melhor configuração: {best_config}")
    
    # Aplica a transformação
    preprocessor = PreProcessor(best_config)
    preprocessor.fit(df, target_col=target_col)
    df_transformed = preprocessor.transform(df, target_col=target_col)
    
    # Divisão treino/teste
    train_df, test_df = train_test_split(df_transformed, test_size=0.3, random_state=42)
    
    # Treinamento e avaliação
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(train_df.drop(columns=[target_col]), train_df[target_col])
    y_pred = model.predict(test_df.drop(columns=[target_col]))
    
    auc_roc = roc_auc_score(test_df[target_col], y_pred)
    f1 = f1_score(test_df[target_col], y_pred)
    
    logger.info(f"AUC-ROC: {auc_roc:.4f}, F1-Score: {f1:.4f}")
    return auc_roc, f1


def test_sequential_vs_parallel():
    """Compara o desempenho do modo sequencial vs paralelo."""
    logger.info("=== Comparando modo sequencial vs paralelo ===")
    
    # Carrega o dataset Iris
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    target_col = 'target'
    
    # Teste com modo sequencial
    import time
    start_time = time.time()
    seq_explorer = Explorer(
        target_col=target_col,
        problem_type='classification',
        parallel=False,
        max_depth=2,
        beam_width=3
    )
    seq_explorer.explore(df)
    seq_time = time.time() - start_time
    seq_config = seq_explorer.get_best_transformation()
    
    # Teste com paralelização
    start_time = time.time()
    par_explorer = Explorer(
        target_col=target_col,
        problem_type='classification',
        parallel=True,
        max_depth=2,
        beam_width=3
    )
    par_explorer.explore(df)
    par_time = time.time() - start_time
    par_config = par_explorer.get_best_transformation()
    
    # Resultados
    logger.info(f"Tempo em modo sequencial: {seq_time:.2f}s")
    logger.info(f"Tempo em modo paralelo: {par_time:.2f}s")
    logger.info(f"Speedup: {seq_time/par_time:.2f}x")
    logger.info(f"Configuração sequencial: {seq_config}")
    logger.info(f"Configuração paralela: {par_config}")
    
    return {
        'sequential_time': seq_time,
        'parallel_time': par_time,
        'speedup': seq_time/par_time,
        'sequential_config': seq_config,
        'parallel_config': par_config
    }


if __name__ == "__main__":
    logger.info("Iniciando testes do AutoFE Melhorado...")
    
    # Teste do Explorer melhorado com Iris
    acc_iris, iris_explorer = test_classification_iris()
    
    # Executa os outros testes
    try:
        auc_fraud, f1_fraud = test_fraud_classification()
    except Exception as e:
        logger.error(f"Erro no teste de fraude: {e}")
        auc_fraud, f1_fraud = None, None
    
    try:
        # Use modo sequencial para o dataset de doenças cardíacas para evitar problemas
        auc_heart, f1_heart = test_heart_disease_classification()
    except Exception as e:
        logger.error(f"Erro no teste de doenças cardíacas: {e}")
        auc_heart, f1_heart = None, None
    
    # Resumo de todos os testes
    logger.info("\n=== Resumo dos Testes com Explorer Melhorado ===")
    logger.info(f"Acurácia - Iris: {acc_iris:.4f}")
    if auc_fraud is not None:
        logger.info(f"AUC-ROC - Fraude: {auc_fraud:.4f}, F1-Score: {f1_fraud:.4f}")
    if auc_heart is not None:
        logger.info(f"AUC-ROC - Doenças Cardíacas: {auc_heart:.4f}, F1-Score: {f1_heart:.4f}")

# Atualizando testes para usar essa função antes da transformação
def test_fraud_classification():
    """Teste com o dataset de fraude em cartões de crédito."""
    logger.info("=== Testando AutoFE com dataset de fraude em cartões ===")
    try:
        df = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")
        df = df.sample(frac=0.1, random_state=42)
    except Exception as e:
        logger.error(f"Erro ao carregar dataset de fraude: {e}")
        return None, None

    target_col = 'Class'
    
    # Aplicando melhor estratégia de balanceamento
    df = apply_best_balance_strategy(df, target_col)
    
    # Explorador atualizado
    explorador = Explorer(
        target_col=target_col,
        problem_type='classification',
        parallel=True,
        max_depth=2,
        beam_width=3
    )
    
    explorador.explore(df)
    best_config = explorador.get_best_transformation()
    logger.info(f"Melhor configuração: {best_config}")

    # Aplicação da transformação com balanceamento
    preprocessor = PreProcessor(best_config)
    preprocessor.fit(df, target_col=target_col)
    df_transformed = preprocessor.transform(df, target_col=target_col)

    # Divisão treino/teste
    X = df_transformed.drop(columns=[target_col])
    y = df_transformed[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Treinamento e avaliação
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    evaluator = ImbalancedMetricsEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred)
    
    logger.info(f"AUC-ROC: {metrics['auprc']:.4f}, F1-Score: {metrics['macro_f1']:.4f}, Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    
    return metrics['auprc'], metrics['macro_f1']
