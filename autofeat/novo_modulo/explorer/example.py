# example_explorer.py

import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Adicionar o diretório novo_modulo ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing import create_preprocessor, PreProcessor

# Importar o PreProcessor e o Explorer
from explorer import create_explorer, Explorer
# Carregar os modelos salvos
from explorer import Explorer
    

def main():
    """
    Exemplo completo de uso do sistema AutoFE com os módulos PreProcessor e Explorer.
    Demonstra o fluxo de trabalho típico e os benefícios da engenharia de features automática.
    """
    print("=== Exemplo de Uso do Sistema AutoFE ===")
    print("\nCarregando dataset de diabetes...")
    
    # Carregar o dataset de diabetes (regressão)
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target)
    
    print(f"Dimensões do dataset: {X.shape}")
    print(f"Features originais: {X.columns.tolist()}")
    
    # Adicionar alguns valores ausentes para demonstrar o preprocessamento
    rows_to_modify = np.random.choice(X.index, size=int(X.shape[0] * 0.1), replace=False)
    cols_to_modify = np.random.choice(X.columns, size=3, replace=False)
    
    for row in rows_to_modify:
        for col in cols_to_modify:
            X.loc[row, col] = np.nan
    
    print(f"\nAdicionados valores ausentes: {X.isna().sum().sum()} valores")
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nDados de treino: {X_train.shape}")
    print(f"Dados de teste: {X_test.shape}")
    
    # ------ Abordagem 1: Sem Engenharia de Features ------
    print("\n=== Abordagem 1: Apenas Preprocessamento Básico ===")
    
    # Configurar o preprocessador 
    preprocessor_config = {
        'missing_values_strategy': 'median',
        'outlier_method': 'clip',
        'normalization': True
    }
    preprocessor = create_preprocessor(preprocessor_config)
    
    # Aplicar preprocessamento
    X_train_clean = preprocessor.fit_transform(X_train)
    X_test_clean = preprocessor.transform(X_test)
    
    print(f"Dados pré-processados: {X_train_clean.shape}")
    
    # Treinar um modelo básico como baseline
    baseline_model = LinearRegression()
    baseline_model.fit(X_train_clean, y_train)
    
    # Avaliar o modelo baseline
    y_pred_baseline = baseline_model.predict(X_test_clean)
    baseline_mse = mean_squared_error(y_test, y_pred_baseline)
    baseline_r2 = r2_score(y_test, y_pred_baseline)
    
    print(f"Modelo Baseline - MSE: {baseline_mse:.2f}, R²: {baseline_r2:.4f}")
    
    # ------ Abordagem 2: Com Engenharia de Features Automática ------
    print("\n=== Abordagem 2: Com Engenharia de Features Automática ===")
    
    # Configurar o Explorer
    explorer_config = {
        'polynomial_features': True,
        'polynomial_degree': 2,
        'interaction_features': True,
        'clustering_features': True,
        'n_clusters': 3,
        'feature_reduction_method': 'pca',
        'feature_reduction_components': 0.95,
        'problem_type': 'regression',
        'verbosity': 1
    }
    explorer = create_explorer(explorer_config)
    
    # Aplicar engenharia de features
    X_train_enhanced = explorer.fit_transform(X_train_clean, y_train)
    X_test_enhanced = explorer.transform(X_test_clean)
    
    print(f"Dados após engenharia de features: {X_train_enhanced.shape}")
    
    # Examinar as features mais importantes identificadas
    importances = explorer.get_feature_importances()
    print("\nTop 5 features por importância:")
    for feature, score in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {feature}: {score:.4f}")
    
    # Treinar o mesmo tipo de modelo, mas com as features melhoradas
    enhanced_model = LinearRegression()
    enhanced_model.fit(X_train_enhanced, y_train)
    
    # Avaliar o modelo com features melhoradas
    y_pred_enhanced = enhanced_model.predict(X_test_enhanced)
    enhanced_mse = mean_squared_error(y_test, y_pred_enhanced)
    enhanced_r2 = r2_score(y_test, y_pred_enhanced)
    
    print(f"Modelo com Features Engenheiradas - MSE: {enhanced_mse:.2f}, R²: {enhanced_r2:.4f}")
    
    # Calcular a melhoria
    mse_improvement = ((baseline_mse - enhanced_mse) / baseline_mse) * 100
    r2_improvement = ((enhanced_r2 - baseline_r2) / abs(baseline_r2)) * 100
    
    print(f"\nMelhoria com AutoFE:")
    print(f"  Redução de MSE: {mse_improvement:.2f}%")
    print(f"  Aumento de R²: {r2_improvement:.2f}%")
    
    # ------ Demonstrar uso em pipeline scikit-learn ------
    print("\n=== Usando AutoFE em um Pipeline Scikit-learn ===")
    
    # Criar um pipeline completo
    pipeline = Pipeline([
        ('preprocessor', create_preprocessor(preprocessor_config)),
        ('explorer', create_explorer(explorer_config)),
        ('model', LinearRegression())
    ])
    
    # Treinar o pipeline diretamente nos dados originais
    pipeline.fit(X_train, y_train)
    
    # Avaliar o pipeline
    y_pred_pipeline = pipeline.predict(X_test)
    pipeline_mse = mean_squared_error(y_test, y_pred_pipeline)
    pipeline_r2 = r2_score(y_test, y_pred_pipeline)
    
    print(f"Pipeline AutoFE - MSE: {pipeline_mse:.2f}, R²: {pipeline_r2:.4f}")
    
    # ------ Demonstrar persistência ------
    print("\n=== Demonstrando Persistência dos Modelos ===")
    
    # Salvar o preprocessador e o explorer para uso futuro
    preprocessor.save("diabetes_preprocessor.joblib")
    explorer.save("diabetes_explorer.joblib")
    
    print("Modelos salvos como 'diabetes_preprocessor.joblib' e 'diabetes_explorer.joblib'")
    

    loaded_preprocessor = PreProcessor.load("diabetes_preprocessor.joblib")
    loaded_explorer = Explorer.load("diabetes_explorer.joblib")
    
    print("Modelos carregados com sucesso!")
    
    # Verificar se os modelos carregados produzem os mesmos resultados
    X_test_clean_loaded = loaded_preprocessor.transform(X_test)
    X_test_enhanced_loaded = loaded_explorer.transform(X_test_clean_loaded)
    
    # Treinar um novo modelo com os dados transformados pelos modelos carregados
    loaded_model = LinearRegression()
    loaded_model.fit(loaded_explorer.transform(loaded_preprocessor.transform(X_train)), y_train)
    y_pred_loaded = loaded_model.predict(X_test_enhanced_loaded)
    loaded_r2 = r2_score(y_test, y_pred_loaded)
    
    print(f"Modelo treinado com modelos carregados - R²: {loaded_r2:.4f}")
    print(f"Mesmos resultados que o modelo original? {abs(loaded_r2 - enhanced_r2) < 1e-10}")
    
    print("\n=== Resumo de Resultados ===")
    print(f"Baseline (sem AutoFE): R² = {baseline_r2:.4f}")
    print(f"Com AutoFE: R² = {enhanced_r2:.4f}")
    print(f"Melhoria: {r2_improvement:.2f}%")
    
    return {
        "baseline_r2": baseline_r2,
        "enhanced_r2": enhanced_r2,
        "improvement": r2_improvement
    }

if __name__ == "__main__":
    main()