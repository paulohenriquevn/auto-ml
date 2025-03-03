"""
Exemplo de como usar o módulo PreProcessor do AutoFE.
"""

import pandas as pd
import numpy as np
from preprocessor import PreProcessor, create_preprocessor

def main():
    """
    Exemplo prático de uso do PreProcessor.
    """
    # Criar dados de exemplo
    print("Criando dados de exemplo...")
    data = {
        'idade': [25, 30, np.nan, 42, 38, 35, 45, 27, 65, 18],
        'salario': [5000, 7500, 8200, np.nan, 12000, 4500, 6800, 3200, 15000, 2800],
        'cidade': ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'São Paulo', 
                  'Curitiba', np.nan, 'Rio de Janeiro', 'Salvador', 'Brasília', 'Recife'],
        'nivel_educacao': ['Superior', 'Pós-graduação', 'Superior', 'Ensino Médio',
                          'Pós-graduação', 'Superior', 'Doutorado', 'Superior', 'Mestrado', np.nan],
        'score_credito': [700, 820, 680, 590, 900, 720, 650, 750, np.nan, 600],
        # Adicionando alguns outliers
        'dias_atraso': [0, 2, 0, 15, 0, 0, 3, 0, 1, 150]  # 150 é um outlier
    }
    
    df = pd.DataFrame(data)
    print("Dados originais:")
    print(df.head())
    print("\nInformações dos dados:")
    print(df.info())
    print("\nEstatísticas descritivas:")
    print(df.describe())
    
    # Verificar valores ausentes
    print("\nValores ausentes por coluna:")
    print(df.isnull().sum())
    
    # Criar e configurar o preprocessador
    print("\nCriando preprocessador...")
    config = {
        'missing_values_strategy': 'median',
        'outlier_strategy': 'clip',
        'categorical_strategy': 'onehot',
        'normalization': True,
        'outlier_threshold': 3.0
    }
    
    preprocessor = create_preprocessor(config)
    
    # Ajustar e transformar os dados
    print("\nAplicando preprocessamento...")
    df_transformed = preprocessor.fit_transform(df)
    
    print("\nDados após preprocessamento:")
    print(df_transformed.head())
    print("\nDimensões dos dados transformados:", df_transformed.shape)
    
    # Salvar o preprocessador para uso futuro
    preprocessor.save("preprocessor_model.joblib")
    print("\nPreprocessador salvo em 'preprocessor_model.joblib'")
    
    # Simular novos dados para transformação
    print("\nSimulando novos dados para transformação...")
    novos_dados = {
        'idade': [33, 29, 51],
        'salario': [6200, 5800, 9500],
        'cidade': ['São Paulo', 'Porto Alegre', 'Florianópolis'],
        'nivel_educacao': ['Superior', 'Mestrado', 'Pós-graduação'],
        'score_credito': [680, 710, 850],
        'dias_atraso': [1, 0, 5]
    }
    
    df_novos = pd.DataFrame(novos_dados)
    print(df_novos)
    
    # Transformar os novos dados usando o mesmo preprocessador
    print("\nTransformando novos dados...")
    df_novos_transformed = preprocessor.transform(df_novos)
    print(df_novos_transformed.head())
    
    # Carregar um preprocessador salvo
    print("\nCarregando preprocessador salvo...")
    loaded_preprocessor = PreProcessor.load("preprocessor_model.joblib")
    
    # Verificar se o preprocessador carregado funciona corretamente
    print("\nVerificando preprocessador carregado...")
    df_loaded_transform = loaded_preprocessor.transform(df_novos)
    print(df_loaded_transform.head())
    
    # Verificar se as transformações são iguais
    print("\nVerificando se as transformações são iguais:")
    are_equal = np.allclose(df_novos_transformed.values, df_loaded_transform.values)
    print(f"Transformações iguais: {are_equal}")

if __name__ == "__main__":
    main()