import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes, load_digits, load_files, load_iris, load_linnerud, load_sample_image, load_sample_images, load_svmlight_file, load_svmlight_files, load_wine, load_breast_cancer
from preprocessor import Explorer, PreProcessor, create_preprocessor

def load_dataset(dataset_name):
    """Carrega um dataset do Scikit-learn."""
    if dataset_name.lower() == "iris":
        data = load_iris()
    elif dataset_name.lower() == "wine":
        data = load_wine()
    elif dataset_name.lower() == "breast_cancer":
        data = load_breast_cancer()
    elif dataset_name.lower() == "diabetes":
        data = load_diabetes()
    elif dataset_name.lower() == "digits":
        data = load_digits()
    elif dataset_name.lower() == "linnerud":
        data = load_linnerud()
    else:
        raise ValueError(f"Dataset {dataset_name} não suportado.")
    
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Verifica se 'target' existe e se tem o formato correto
    if hasattr(data, 'target'):
        target = np.array(data.target).ravel()  # Garante que seja um array 1D
        if len(target) == df.shape[0]:  # Confirma que os tamanhos correspondem
            df['target'] = target
        else:
            ValueError(f"Aviso: O dataset {dataset_name} tem um número incompatível de labels ({len(target)}) em relação às amostras ({df.shape[0]}). Ignorando 'target'.")
    else:
        ValueError(f"Aviso: O dataset {dataset_name} não possui um alvo ('target').")

    return df


def explore_dataset(df, sample_size=5):
    """
    Realiza uma exploração básica do dataset.
    
    Args:
        df (pd.DataFrame): DataFrame a ser explorado
        sample_size (int): Número de amostras a serem exibidas
        
    Returns:
        pd.DataFrame: O mesmo DataFrame de entrada
    """
    print("\n=== EXPLORAÇÃO DO DATASET ===")
    
    print(f"\nPrimeiras {sample_size} linhas:")
    print(df.head(sample_size))
    
    print("\nInformações das colunas:")
    print(df.info())
    
    print("\nEstatísticas descritivas (colunas numéricas):")
    print(df.describe())
    
    print("\nValores ausentes por coluna:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0].sort_values(ascending=False))
    
    # Contar tipos de colunas
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    
    print("\nDistribuição de tipos de colunas:")
    print(f"- Numéricas: {len(numeric_cols)}")
    print(f"- Categóricas: {len(categorical_cols)}")
    print(f"- Booleanas: {len(bool_cols)}")
    
    return df

def visualize_dataset(df, output_dir="output"):
    """Gera histogramas das variáveis numéricas."""
    os.makedirs(output_dir, exist_ok=True)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    plt.figure(figsize=(12, 6))
    df[numeric_cols].hist(bins=30, figsize=(12, 8))
    plt.savefig(f"{output_dir}/dataset_histograms.png")
    print(f"Histogramas salvos em {output_dir}/dataset_histograms.png")

def preprocess_data(df, target_col="target"):
    """Executa o pré-processamento dos dados."""
    config = {
        'missing_values_strategy': 'median',
        'scaling': 'standard',
        'categorical_strategy': 'onehot',
        'dimensionality_reduction': 'pca',
        'feature_selection': 'variance'
    }
    preprocessor = create_preprocessor(config)
    # Ajustando o preprocessador corretamente
    preprocessor.fit(df)
    X_processed = preprocessor.transform(df)
    
    return X_processed, preprocessor

def visualize_dataset(df, processed=False, output_dir="output"):
    """
    Cria visualizações básicas do dataset.
    
    Args:
        df (pd.DataFrame): DataFrame a ser visualizado
        output_dir (str): Diretório para salvar as visualizações
    """
    print("\n=== VISUALIZAÇÃO DO DATASET ===")
    
    input = "processado" if processed else "original"
    
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Selecionar algumas colunas numéricas para visualizar (até 5)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) > 5:
        numeric_cols = numeric_cols[:5]
    
    if len(numeric_cols) > 0:
        print(f"\nCriando histogramas para {len(numeric_cols)} colunas numéricas...")
        
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(len(numeric_cols), 1, i)
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribuição de {col}')
            plt.tight_layout()
        
        plt.savefig(f"{output_dir}/numeric_distributions_{input}.png")
        print(f"Histogramas salvos em {output_dir}/numeric_distributions_{input}.png")
    
    # Visualizar correlações entre colunas numéricas
    if len(numeric_cols) > 1:
        print("\nCriando matriz de correlação...")
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Matriz de Correlação')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_matrix_{input}.png")
        print(f"Matriz de correlação salva em {output_dir}/correlation_matrix_{input}.png")
    
    # Selecionar algumas colunas categóricas para visualizar (até 3)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(categorical_cols) > 3:
        categorical_cols = categorical_cols[:3]
    
    if len(categorical_cols) > 0:
        print(f"\nCriando gráficos de barras para {len(categorical_cols)} colunas categóricas...")
        
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(categorical_cols, 1):
            plt.subplot(len(categorical_cols), 1, i)
            # Limitar a 10 categorias mais frequentes
            value_counts = df[col].value_counts().nlargest(10)
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f'Top 10 categorias em {col}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
        
        plt.savefig(f"{output_dir}/categorical_distributions_{input}.png")
        print(f"Gráficos de barras salvos em {output_dir}/categorical_distributions_{input}.png")

def heuristic_example(data):
    """Exemplo de heurística que prioriza conjuntos de features menores."""
    return -len(data) if data is not None else float('-inf')

import os

def preprocess_and_save(df, best_config, dataset_name, output_dir="output"):
    """
    Aplica o pré-processamento com a melhor configuração encontrada e salva os resultados.

    Args:
        df (pd.DataFrame): O DataFrame original.
        best_config (dict): A melhor configuração de pré-processamento encontrada.
        output_dir (str): Diretório onde os arquivos serão salvos.
    """
    # Criando o diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Aplicar o pré-processador com a melhor configuração
    preprocessor = PreProcessor(best_config).fit(df)
    X_processed = preprocessor.transform(df)

    # Salvar dataset processado
    dataset_path = os.path.join(output_dir, f"{dataset_name}_processado.csv")
    X_processed.to_csv(dataset_path, index=False)
    print(f"Dataset processado salvo em {dataset_path}")

    # Salvar o objeto do preprocessador
    preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")
    preprocessor.save(preprocessor_path)
    print(f"Preprocessador salvo em {preprocessor_path}")

    return X_processed, preprocessor


def main():
    available_datasets = [
            "iris",
            "wine", 
            "breast_cancer",
            "diabetes",
            "digits",
            "iris",
            "breast_cancer",
            "linnerud",
        ]
    print("Datasets disponíveis para teste:")
    for i, dataset_name in enumerate(available_datasets, 1):
        output = f"output/{dataset_name}"
        print(f"\nUsando dataset: {dataset_name}")
        explorer = Explorer(target_col="target")
        sample_data = load_dataset(dataset_name)
        visualize_dataset(sample_data, output_dir=output)
         # Salvar dataset antes de processar
        os.makedirs(output, exist_ok=True) 
        dataset_path = os.path.join(output, f"{dataset_name}.csv")
        sample_data.to_csv(dataset_path, index=False)
        
        print(f"Dataset processado salvo em {dataset_path}")
    
        best_data = explorer.analyze_transformations(sample_data)

        # Obtém a melhor configuração usada no PreProcessor
        best_config = explorer.tree.graph.nodes[explorer.find_best_transformation()].get("config", {})

        # Aplica a melhor configuração e salva o dataset
        X_processed, preprocessor = preprocess_and_save(sample_data, best_config, dataset_name, output_dir=output)

        print("\n=== RESUMO DO PRE-PROCESSAMENTO ===")
        print(f"Formato original: {sample_data.shape}")
        print(f"Formato processado: {X_processed.shape}")
        print(f"Melhor conjunto de features resultante tem dimensão: {best_data.shape}")
        
        explore_dataset(X_processed)
        visualize_dataset(X_processed, True, output_dir=output)
        print("\n=== FIM DO SCRIPT ===")
        

if __name__ == "__main__":
    main()
