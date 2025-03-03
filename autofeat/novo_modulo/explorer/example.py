"""
Exemplo de uso do AutoFE para automatização de engenharia de features

Este script demonstra o uso dos módulos AutoExplorer e PreProcessor da
biblioteca AutoFE para processamento de dados e otimização de features.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Configurar path para importação de módulos
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Importar módulos do AutoFE
from auto_explorer import AutoExplorer
from preprocessing import PreProcessor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AutoFE-Example')

def main():
    """Função principal do exemplo."""
    try:
        logger.info("=== Exemplo de Uso do AutoExplorer ===")
        
        # Carregar dataset
        logger.info("Carregando dataset: diabetes")
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        y = pd.Series(diabetes.target, name="target")
        
        logger.info(f"Dataset carregado com sucesso. Dimensões: {X.shape}")
        
        # Aplicar pré-processamento
        logger.info("Aplicando pré-processamento aos dados")
        preprocessor_config = {
            'missing_values_strategy': 'median',
            'outlier_method': 'clip',
            'normalization': True
        }
        preprocessor = PreProcessor(preprocessor_config)
        
        X_cleaned = preprocessor.fit_transform(X)
        
        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_cleaned, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Divisão treino/teste: {X_train.shape[0]} amostras para treino, {X_test.shape[0]} para teste")
        
        # Instanciar o AutoExplorer
        explorer = AutoExplorer(
            max_iterations=15,
            improvement_threshold=0.0005
        )
        
        logger.info("Iniciando exploração automática de features")
        X_train_transformed = explorer.fit_transform(X_train, y_train)
        
        # Transformar dados de teste
        X_test_transformed = X_test[explorer.best_features]
        
        logger.info(f"Exploração concluída. Dimensões após AutoExplorer: {X_train_transformed.shape}")
        
        # Treinar modelo final
        logger.info("Treinando modelo final (Ridge)")
        model = Ridge(alpha=1.0)
        model.fit(X_train_transformed, y_train)
        
        # Avaliar o modelo
        y_pred = model.predict(X_test_transformed)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Modelo Final - MSE: {mse:.2f}, R²: {r2:.4f}")
        
        # Mostrar features selecionadas
        logger.info(f"Features selecionadas: {explorer.best_features}")
        
        return {
            "final_mse": mse,
            "final_r2": r2
        }
        
    except Exception as e:
        logger.error(f"Erro durante a execução: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    main()