#!/usr/bin/env python3
"""
Exemplo de uso do módulo PreProcessor com datasets pré-carregados.

Este script demonstra como usar o módulo PreProcessor para limpar e transformar
dados de datasets publicamente disponíveis, sem realizar treinamento de modelos.
"""

# Imports principais
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.datasets import load_diabetes, load_digits, load_files, load_iris, load_linnerud, load_sample_image, load_sample_images, load_svmlight_file, load_svmlight_files, load_wine, load_breast_cancer, fetch_california_housing
from sklearn.datasets import make_classification

# Importar o PreProcessor
from preprocessor import PreProcessor, create_preprocessor

def load_and_prepare_dataset(dataset_name, subset=None, split="train"):
    """
    Carrega um dataset usando a biblioteca Hugging Face Datasets ou scikit-learn.
    
    Args:
        dataset_name (str): Nome do dataset a ser carregado
        subset (str, opcional): Subconjunto do dataset
        split (str, opcional): Split do dataset (train, test, validation)
        
    Returns:
        pd.DataFrame: DataFrame com os dados carregados
    """
    print(f"Carregando dataset '{dataset_name}'...")
    
    try:
        # Datasets populares do scikit-learn
        if dataset_name.lower() == "iris":
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['target_name'] = [data.target_names[i] for i in data.target]
            print(f"Dataset Iris carregado com sucesso via scikit-learn. Formato: {df.shape}")
            return df
            
        elif dataset_name.lower() == "wine":
            data = load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['target_name'] = [data.target_names[i] for i in data.target]
            print(f"Dataset Wine carregado com sucesso via scikit-learn. Formato: {df.shape}")
            return df
            
        elif dataset_name.lower() == "breast_cancer":
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['target_name'] = [data.target_names[i] for i in data.target]
            print(f"Dataset Breast Cancer carregado com sucesso via scikit-learn. Formato: {df.shape}")
            return df
        
        elif dataset_name.lower() == "diabetes":
            data = load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['target_name'] = [data.target_names[i] for i in data.target]
            print(f"Dataset Breast Cancer carregado com sucesso via scikit-learn. Formato: {df.shape}")
            return df
        
        elif dataset_name.lower() == "digits":
            data = load_digits()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['target_name'] = [data.target_names[i] for i in data.target]
            print(f"Dataset Breast Cancer carregado com sucesso via scikit-learn. Formato: {df.shape}")
            return df
        
        elif dataset_name.lower() == "files":
            data = load_files()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['target_name'] = [data.target_names[i] for i in data.target]
            print(f"Dataset Breast Cancer carregado com sucesso via scikit-learn. Formato: {df.shape}")
            return df
        
        elif dataset_name.lower() == "linnerud":
            data = load_linnerud()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['target_name'] = [data.target_names[i] for i in data.target]
            print(f"Dataset Breast Cancer carregado com sucesso via scikit-learn. Formato: {df.shape}")
            return df
        
        elif dataset_name.lower() == "sample_image":
            data = load_sample_image()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['target_name'] = [data.target_names[i] for i in data.target]
            print(f"Dataset Breast Cancer carregado com sucesso via scikit-learn. Formato: {df.shape}")
            return df
        
        elif dataset_name.lower() == "sample_images":
            data = load_sample_images()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['target_name'] = [data.target_names[i] for i in data.target]
            print(f"Dataset Breast Cancer carregado com sucesso via scikit-learn. Formato: {df.shape}")
            return df
        
        elif dataset_name.lower() == "svmlight_file":
            data = load_svmlight_file()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['target_name'] = [data.target_names[i] for i in data.target]
            print(f"Dataset Breast Cancer carregado com sucesso via scikit-learn. Formato: {df.shape}")
            return df
        
        elif dataset_name.lower() == "svmlight_files":
            data = load_svmlight_files()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['target_name'] = [data.target_names[i] for i in data.target]
            print(f"Dataset Breast Cancer carregado com sucesso via scikit-learn. Formato: {df.shape}")
            return df
        
        # Tentar carregar do Hugging Face Datasets
        print(f"Tentando carregar do Hugging Face Hub...")
        if subset:
            dataset = load_dataset(dataset_name, subset, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        # Converter para pandas DataFrame
        df = pd.DataFrame(dataset)
        print(f"Dataset carregado com sucesso do Hugging Face Hub. Formato: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Erro ao carregar o dataset: {str(e)}")
        print("Tentando usar dataset alternativo...")
        
        # Fallback para um dataset conhecido do Hugging Face
        try:
            print("Carregando dataset 'titanic' como alternativa...")
            dataset = load_dataset("haukevs/titanic", split="train")
            df = pd.DataFrame(dataset)
            print(f"Dataset Titanic carregado com sucesso. Formato: {df.shape}")
            return df
        except Exception as e2:
            print(f"Também não foi possível carregar o dataset alternativo: {str(e2)}")
            
            # Último recurso: criar um dataset sintético
            print("Criando dataset sintético para demonstração...")
            X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                                      n_redundant=2, n_repeated=0, n_classes=2, 
                                      random_state=42)
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feature_names)
            df['target'] = y
            print(f"Dataset sintético criado. Formato: {df.shape}")
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
    """
    Cria visualizações básicas do dataset.
    
    Args:
        df (pd.DataFrame): DataFrame a ser visualizado
        output_dir (str): Diretório para salvar as visualizações
    """
    print("\n=== VISUALIZAÇÃO DO DATASET ===")
    
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
        
        plt.savefig(f"{output_dir}/numeric_distributions.png")
        print(f"Histogramas salvos em {output_dir}/numeric_distributions.png")
    
    # Visualizar correlações entre colunas numéricas
    if len(numeric_cols) > 1:
        print("\nCriando matriz de correlação...")
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Matriz de Correlação')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_matrix.png")
        print(f"Matriz de correlação salva em {output_dir}/correlation_matrix.png")
    
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
        
        plt.savefig(f"{output_dir}/categorical_distributions.png")
        print(f"Gráficos de barras salvos em {output_dir}/categorical_distributions.png")

def preprocess_data(df, target_col=None):
    """
    Aplica o PreProcessor nos dados.
    
    Args:
        df (pd.DataFrame): DataFrame a ser processado
        target_col (str, opcional): Nome da coluna alvo/target
        
    Returns:
        tuple: (DataFrame processado, Series com target, PreProcessor)
    """
    print("\n=== PRÉ-PROCESSAMENTO DE DADOS ===")
    
    # Preparar dados (remover colunas que não serão úteis para análise)
    data = df.copy()
    
    # Identificar colunas com grande número de valores únicos (possíveis IDs)
    high_cardinality_cols = []
    for col in data.select_dtypes(include=['object']).columns:
        if data[col].nunique() > 0.5 * len(data):
            high_cardinality_cols.append(col)
    
    print(f"Identificadas {len(high_cardinality_cols)} colunas com alta cardinalidade (possíveis IDs)")
    
    # Remover colunas identificadas como IDs ou com alta cardinalidade
    if high_cardinality_cols:
        print(f"Removendo colunas: {', '.join(high_cardinality_cols)}")
        data = data.drop(columns=high_cardinality_cols)
    
    # Separar features e target
    if target_col and target_col in data.columns:
        y = data[target_col].copy()
        X = data.drop(columns=[target_col])
        print(f"Separada coluna alvo: '{target_col}'")
    else:
        y = None
        X = data
        print("Nenhuma coluna alvo especificada")
    
    # Configuração do preprocessador
    config = {
        'missing_values_strategy': 'median',
        'outlier_strategy': 'clip',
        'categorical_strategy': 'onehot',
        'normalization': True,
        'outlier_threshold': 3.0,
        'max_categories': 15  # Limitar o número de categorias para evitar explosão dimensional
    }
    
    # Criar e treinar o preprocessador
    print("Criando e treinando o preprocessador...")
    preprocessor = create_preprocessor(config)
    
    # Aplicar o preprocessador
    print("Aplicando preprocessamento...")
    try:
        X_processed = preprocessor.fit_transform(X)
        print(f"Preprocessamento concluído com sucesso!")
        print(f"Formato original dos dados: {X.shape}")
        print(f"Formato dos dados processados: {X_processed.shape}")
        
        # Identificar tipos de colunas encontradas
        print(f"\nTipos de colunas identificadas pelo preprocessador:")
        for col_type, cols in preprocessor.column_types.items():
            print(f"- {col_type}: {len(cols)} colunas")
            if cols and len(cols) <= 5:  # Mostrar nomes se forem poucas colunas
                print(f"  {', '.join(cols)}")
        
        # Salvar o preprocessador para uso futuro
        preprocessor.save("dataset_preprocessor.joblib")
        print("Preprocessador salvo como 'dataset_preprocessor.joblib'")
        
        return X_processed, y, preprocessor
    
    except Exception as e:
        print(f"Erro durante o preprocessamento: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, y, None

def analyze_preprocessed_data(X_processed, y=None, output_dir="output"):
    """
    Analisa os dados pré-processados e salva em arquivo.
    
    Args:
        X_processed (pd.DataFrame): DataFrame pré-processado
        y (pd.Series, opcional): Valores alvo
        output_dir (str): Diretório para salvar as visualizações e dados
    """
    print("\n=== ANÁLISE DOS DADOS PRÉ-PROCESSADOS ===")
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nPrimeiras 5 linhas dos dados processados:")
    print(X_processed.head())
    
    print("\nEstatísticas dos dados processados:")
    print(X_processed.describe())
    
    # Verificar valores ausentes
    missing_after = X_processed.isnull().sum().sum()
    print(f"\nValores ausentes após processamento: {missing_after}")
    
    # Verificar variância das features
    variance = X_processed.var()
    low_variance_features = variance[variance < 0.01].shape[0]
    print(f"Features com baixa variância (< 0.01): {low_variance_features}")
    
    # Criar visualização da distribuição de valores processados
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=X_processed.iloc[:, :min(10, X_processed.shape[1])])
    plt.title('Distribuição das primeiras 10 features processadas')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/processed_features_distribution.png")
    print(f"Boxplot das features processadas salvo em {output_dir}/processed_features_distribution.png")
    
    # Salvar os dados pré-processados em CSV
    try:
        # Se tivermos a variável alvo, combinamos com as features processadas
        if y is not None:
            # Verificar se y é uma Series ou um array
            if isinstance(y, pd.Series):
                dataset_processed = pd.concat([X_processed, y], axis=1)
            else:
                # Se y for um array numpy ou lista
                y_series = pd.Series(y, name='target')
                dataset_processed = pd.concat([X_processed, y_series], axis=1)
        else:
            dataset_processed = X_processed
            
        # Salvar em formato CSV
        csv_path = f"{output_dir}/processed_dataset.csv"
        dataset_processed.to_csv(csv_path, index=False)
        print(f"Dataset processado salvo em {csv_path}")
        
        # Salvar também em formato Excel para facilitar a visualização
        try:
            excel_path = f"{output_dir}/processed_dataset.xlsx"
            dataset_processed.to_excel(excel_path, index=False)
            print(f"Dataset processado também salvo em {excel_path}")
        except Exception as e:
            print(f"Não foi possível salvar em Excel: {str(e)}")
            
        return dataset_processed
            
    except Exception as e:
        print(f"Erro ao salvar o dataset processado: {str(e)}")
        return X_processed

def main():
    """Função principal."""
    try:
        # Lista de datasets disponíveis para teste
        available_datasets = [
            "iris",
            "wine", 
            "breast_cancer",
            "titanic",
            "diabetes",
            "digits",
            "files",
            "iris",
            "breast_cancer",
            "linnerud",
            "sample_image",
            "sample_images",
            "svmlight_file",
            "svmlight_files",
            "wine",
            "pirocheto/phishing-url",
            "wwydmanski/blog-feedback",
            "RUCAIBox/Data-to-text-Generation",
            "YuRiVeRTi/V1Q",
            "dijihax/Dataset"
        ]
        
        print("Datasets disponíveis para teste:")
        for i, dataset_name in enumerate(available_datasets, 1):
            print(f"{i}. {dataset_name}")
            
            # Configurar o dataset a ser usado (pode ser alterado conforme necessário)
            print(f"\nUsando dataset: {dataset_name}")
            
            # Criar pasta de output
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Carregar o dataset
            df = load_and_prepare_dataset(dataset_name)
            
            # Explorar o dataset
            df = explore_dataset(df)
            
            # Visualizar o dataset
            visualize_dataset(df, output_dir=output_dir)
            
            # Determinar coluna alvo com base no dataset
            target_col = None
            if dataset_name.lower() in ["iris", "wine", "breast_cancer"]:
                target_col = "target"
            elif dataset_name.lower() == "titanic":
                # No titanic do Hugging Face, a coluna pode ter nome diferente
                if "survived" in df.columns:
                    target_col = "survived"
                elif "Survived" in df.columns:
                    target_col = "Survived"
            
            if target_col:
                print(f"Coluna alvo identificada: '{target_col}'")
            else:
                print("Nenhuma coluna alvo identificada automaticamente")
            
            # Pré-processar os dados
            X_processed, y, preprocessor = preprocess_data(df, target_col)
            
            if X_processed is not None:
                # Analisar os dados pré-processados e salvar em arquivo
                dataset_processed = analyze_preprocessed_data(X_processed, y, output_dir=output_dir)
                
                # Salvar também o dataset original para referência
                original_csv_path = f"{output_dir}/original_dataset.csv"
                df.to_csv(original_csv_path, index=False)
                print(f"Dataset original salvo em {original_csv_path}")
                
                # Salvar metadados do dataset processado
                metadata = {
                    "dataset_name": dataset_name,
                    "original_shape": df.shape,
                    "processed_shape": X_processed.shape,
                    "target_column": target_col,
                    "preprocessing_config": preprocessor.config if preprocessor else {},
                    "column_types": {k: list(v) for k, v in preprocessor.column_types.items()} if preprocessor else {},
                    "transformation_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Salvar metadados em formato JSON
                try:
                    with open(f"{output_dir}/dataset_metadata.json", "w") as f:
                        json.dump(metadata, f, indent=2, default=str)
                    print(f"Metadados do dataset salvos em {output_dir}/dataset_metadata.json")
                except Exception as e:
                    print(f"Erro ao salvar metadados: {str(e)}")
                
                print("\n=== RESUMO DO PRÉ-PROCESSAMENTO ===")
                print(f"Dataset: {dataset_name}")
                print(f"Registros originais: {df.shape[0]}")
                print(f"Features originais: {df.shape[1] - (1 if target_col else 0)}")
                print(f"Features após pré-processamento: {X_processed.shape[1]}")
                
                # Mostrar algumas transformações realizadas
                if preprocessor:
                    print("\nTransformações realizadas:")
                    print(f"- Tratamento de valores ausentes usando '{preprocessor.config['missing_values_strategy']}'")
                    print(f"- Tratamento de outliers usando '{preprocessor.config['outlier_strategy']}'")
                    print(f"- Codificação de variáveis categóricas usando '{preprocessor.config['categorical_strategy']}'")
                    print(f"- Normalização de variáveis numéricas: {'Sim' if preprocessor.config['normalization'] else 'Não'}")
                    
                    # Mostrar tamanho do preprocessador salvo
                    if os.path.exists("dataset_preprocessor.joblib"):
                        size_mb = os.path.getsize("dataset_preprocessor.joblib") / (1024 * 1024)
                        print(f"Tamanho do preprocessador salvo: {size_mb:.2f} MB")
                
                # Gerar um relatório HTML básico
                try:
                    html_report = f"""
                    <html>
                    <head>
                        <title>Relatório de Pré-processamento - {dataset_name}</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            h1, h2 {{ color: #2c3e50; }}
                            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                            th {{ background-color: #f2f2f2; }}
                            .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
                            img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                        </style>
                    </head>
                    <body>
                        <h1>Relatório de Pré-processamento - {dataset_name}</h1>
                        <div class="summary">
                            <h2>Resumo</h2>
                            <p>Dataset: {dataset_name}</p>
                            <p>Registros: {df.shape[0]}</p>
                            <p>Features originais: {df.shape[1] - (1 if target_col else 0)}</p>
                            <p>Features após pré-processamento: {X_processed.shape[1]}</p>
                            <p>Data de processamento: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                        </div>
                        
                        <h2>Configurações de Pré-processamento</h2>
                        <ul>
                            <li>Tratamento de valores ausentes: {preprocessor.config['missing_values_strategy']}</li>
                            <li>Tratamento de outliers: {preprocessor.config['outlier_strategy']}</li>
                            <li>Codificação de variáveis categóricas: {preprocessor.config['categorical_strategy']}</li>
                            <li>Normalização: {'Sim' if preprocessor.config['normalization'] else 'Não'}</li>
                        </ul>
                        
                        <h2>Visualizações</h2>
                        <p>Distribuição das Features Processadas:</p>
                        <img src="processed_features_distribution.png" alt="Distribuição das Features">
                        
                        <h2>Links para Arquivos</h2>
                        <ul>
                            <li><a href="original_dataset.csv">Dataset Original (CSV)</a></li>
                            <li><a href="processed_dataset.csv">Dataset Processado (CSV)</a></li>
                            <li><a href="dataset_metadata.json">Metadados do Processamento (JSON)</a></li>
                        </ul>
                    </body>
                    </html>
                    """
                    
                    with open(f"{output_dir}/preprocessing_report.html", "w") as f:
                        f.write(html_report)
                    print(f"Relatório HTML gerado em {output_dir}/preprocessing_report.html")
                except Exception as e:
                    print(f"Erro ao gerar relatório HTML: {str(e)}")
            
            print("\nExemplo concluído com sucesso!")
            print("Resultado do pré-processamento disponível na pasta 'output/'")
            print("Para usar outro dataset, altere a variável 'dataset_name' na função main()")
        
    except Exception as e:
        print(f"\nErro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()

# Bloco de execução principal
if __name__ == "__main__":
    main()