import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
import logging
import os
import sys

# Adiciona o diretório atual ao caminho para garantir importações corretas
sys.path.append(os.path.abspath('.'))

# Importe o módulo Explorer melhorado
from explorer import create_explorer, analyze_transformations

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutoFE.Example")


def exemplo_classificacao():
    # Carrega o dataset Iris para classificação
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    
    # Combina features e target
    df = pd.concat([X, y], axis=1)
    
    logger.info(f"Dataset Iris carregado com {df.shape[0]} amostras e {df.shape[1]} features")
    
    # Analisa transformações para classificação
    results = analyze_transformations(
        df=df,
        target_col='target',
        problem_type='classification',
        parallel=True
    )
    
    # Exibe a melhor configuração encontrada
    logger.info(f"Melhor configuração para classificação: {results['best_config']}")
    
    # Exibe relatório resumido
    best_transformations = results['report']['best_transformations']
    for i, t in enumerate(best_transformations):
        logger.info(f"Rank {i+1}: {t['name']} (Score: {t['score']:.4f})")
        if 'cv_accuracy' in t['metrics']:
            logger.info(f"  Accuracy: {t['metrics']['cv_accuracy']:.4f}")
        if 'cv_f1' in t['metrics']:
            logger.info(f"  F1-Score: {t['metrics']['cv_f1']:.4f}")
    
    # Salva a visualização da árvore
    explorer = results['explorer']
    dot_code = explorer.visualize_tree(output_file="iris_transformation_tree.dot")
    logger.info("Visualização da árvore salva em 'iris_transformation_tree.dot'")
    
    return results

def exemplo_regressao():
    # Carrega o dataset Diabetes para regressão
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target, name='target')
    
    # Combina features e target
    df = pd.concat([X, y], axis=1)
    
    # Adiciona algumas colunas categóricas para teste
    df['cat1'] = np.random.choice(['A', 'B', 'C'], size=len(df))
    df['cat2'] = np.random.choice(['X', 'Y', 'Z'], size=len(df))
    
    # Adiciona valores ausentes para teste
    mask = np.random.rand(*df.shape) < 0.1
    df_missing = df.mask(mask)
    
    logger.info(f"Dataset Diabetes carregado com {df_missing.shape[0]} amostras e {df_missing.shape[1]} features")
    
    # Cria um explorador customizado
    explorer = create_explorer(
        target_col='target',
        problem_type='regression',
        max_depth=2,
        beam_width=3,
        parallel=True
    )
    
    # Executa a exploração
    tree = explorer.explore(df_missing)
    
    # Obtém a melhor configuração
    best_config = explorer.get_best_transformation()
    logger.info(f"Melhor configuração para regressão: {best_config}")
    
    # Gera e exibe relatório
    report = explorer.get_transformation_report(top_k=3)
    
    # Exibe relatório resumido
    best_transformations = report['best_transformations']
    for i, t in enumerate(best_transformations):
        logger.info(f"Rank {i+1}: {t['name']} (Score: {t['score']:.4f})")
        if 'cv_r2' in t['metrics']:
            logger.info(f"  R²: {t['metrics']['cv_r2']:.4f}")
        if 'cv_neg_rmse' in t['metrics']:
            logger.info(f"  RMSE: {abs(t['metrics']['cv_neg_rmse']):.4f}")
    
    # Salva o explorador para uso futuro
    os.makedirs("models", exist_ok=True)
    explorer.save("models/diabetes_explorer.pkl")
    logger.info("Explorer salvo em 'models/diabetes_explorer.pkl'")
    
    return explorer

def exemplo_pipeline_completo():
    """
    Exemplo de um pipeline completo de AutoFE usando o Explorer melhorado.
    """
    # Carrega o dataset Iris para classificação
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    
    # Combina features e target
    df = pd.concat([X, y], axis=1)
    
    # Divide em treino e teste
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    logger.info(f"Dataset dividido em treino ({train_df.shape[0]} amostras) e teste ({test_df.shape[0]} amostras)")
    
    # Cria e executa o explorador
    explorer = create_explorer(
        target_col='target',
        problem_type='classification',
        experience_db="models/experience_db.pkl",
        max_depth=2,
        beam_width=3
    )
    
    # Explora transformações no conjunto de treino
    tree = explorer.explore(train_df)
    
    # Obtém a melhor configuração
    best_config = explorer.get_best_transformation()
    logger.info(f"Melhor configuração encontrada: {best_config}")
    
    # Aplica a melhor transformação no conjunto de teste
    from preprocessor import PreProcessor
    
    # Cria e ajusta o preprocessador com a melhor configuração
    preprocessor = PreProcessor(best_config)
    preprocessor.fit(train_df, target_col='target')
    
    # Transforma os conjuntos de treino e teste
    train_transformed = preprocessor.transform(train_df, target_col='target')
    test_transformed = preprocessor.transform(test_df, target_col='target')
    
    logger.info(f"Dados transformados: treino {train_transformed.shape}, teste {test_transformed.shape}")
    
    # Treina um modelo com os dados transformados
    from sklearn.ensemble import RandomForestClassifier
    
    # Separa features e target
    X_train = train_transformed.drop(columns=['target'])
    y_train = train_transformed['target']
    X_test = test_transformed.drop(columns=['target'])
    y_test = test_transformed['target']
    
    # Treina o modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Avalia no conjunto de teste
    accuracy = model.score(X_test, y_test)
    logger.info(f"Acurácia no conjunto de teste: {accuracy:.4f}")
    
    # Salva o pipeline completo
    os.makedirs("models", exist_ok=True)
    preprocessor.save("models/best_preprocessor.pkl")
    
    return {
        'explorer': explorer,
        'preprocessor': preprocessor,
        'model': model,
        'accuracy': accuracy
    }

def exemplo_com_fallback_sequencial():
    """
    Exemplo usando o Explorer com fallback para execução sequencial.
    """
    # Carrega o dataset Iris
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    df = pd.concat([X, y], axis=1)
    
    logger.info(f"Dataset carregado com {df.shape[0]} amostras e {df.shape[1]} features")
    
    # Analisa transformações com modo sequencial (sem paralelização)
    logger.info("Iniciando análise de transformações em modo sequencial...")
    results = analyze_transformations(
        df=df, 
        target_col='target',
        problem_type='classification',
        parallel=False  # Importante: desabilita paralelização
    )
    
    # Verifica a melhor configuração
    logger.info(f"Melhor configuração encontrada: {results['best_config']}")
    
    # Se há um relatório, exibe as melhores transformações
    if 'report' in results and 'best_transformations' in results['report']:
        for i, trans in enumerate(results['report']['best_transformations']):
            logger.info(f"Transformação #{i+1}: {trans['name']} (Score: {trans['score']:.4f})")
    
    return results

if __name__ == "__main__":
    logger.info("Executando exemplo de exemplo_com_fallback_sequencial...")
    exemplo_com_fallback_sequencial()
    
    # logger.info("Executando exemplo de classificação...")
    # exemplo_classificacao()
    
    # logger.info("\nExecutando exemplo de regressão...")
    # exemplo_regressao()
    
    # logger.info("\nExecutando exemplo de pipeline completo...")
    # exemplo_pipeline_completo()