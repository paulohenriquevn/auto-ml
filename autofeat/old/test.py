# tests/test_autofe.py

"""
Suíte de Testes para o Sistema AutoFE (Automated Feature Engineering)

Este módulo de testes abrange vários cenários de machine learning para validar
a robustez e eficácia do sistema de engenharia automática de features.

Objetivos dos Testes:
1. Validar o processamento de diferentes tipos de problemas de ML
2. Demonstrar a capacidade de melhoria de features
3. Testar a adaptabilidade do sistema a diversos datasets
"""

import sys
import os
import logging
import numpy as np
import pandas as pd

# Importações de Machine Learning
from sklearn.datasets import (
    load_iris, load_diabetes, fetch_california_housing, 
    load_wine, make_classification, make_regression
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, 
    mean_squared_error, r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importações do AutoFE
from common.data_types import ProblemType
from preprocessing.preprocessor import PreProcessor
from explorer.explorer import Explorer
from predictor.predictor import Predictor
from postprocessor.postprocessor import PosProcessor

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("autofe_tests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AutoFE.Tests")

class ModelPerformanceComparator:
    """
    Utilitário para comparação de desempenho de modelos de machine learning
    
    Responsabilidades:
    1. Avaliar modelos em diferentes configurações de features
    2. Calcular métricas de desempenho
    3. Comparar performance antes e depois da engenharia de features
    """
    
    @staticmethod
    def get_models(problem_type):
        """
        Seleciona modelos apropriados com base no tipo de problema
        
        Args:
            problem_type (ProblemType): Tipo do problema de machine learning
        
        Returns:
            list: Lista de modelos para avaliação
        """
        if problem_type == ProblemType.CLASSIFICATION:
            return [
                ('Logistic Regression', Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(max_iter=1000))
                ])),
                ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('SVM', Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', SVC(probability=True, random_state=42))
                ]))
            ]
        elif problem_type == ProblemType.REGRESSION:
            return [
                ('Ridge Regression', Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', Ridge())
                ])),
                ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('SVR', Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', SVR())
                ]))
            ]
        else:
            raise ValueError("Tipo de problema não suportado")
    
    @classmethod
    def evaluate_models(cls, X, y, problem_type, test_size=0.2):
        """
        Avalia o desempenho dos modelos
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Variável alvo
            problem_type (ProblemType): Tipo do problema
            test_size (float): Proporção do conjunto de teste
        
        Returns:
            dict: Resultados de desempenho dos modelos
        """
        # Divisão dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Obtém modelos para o problema
        models = cls.get_models(problem_type)
        
        # Dicionário para armazenar resultados
        results = {}
        
        for name, model in models:
            try:
                # Treina o modelo
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calcula métricas baseado no tipo de problema
                if problem_type == ProblemType.CLASSIFICATION:
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'f1_score': f1_score(y_test, y_pred, average='weighted')
                    }
                else:  # Regressão
                    metrics = {
                        'mse': mean_squared_error(y_test, y_pred),
                        'r2': r2_score(y_test, y_pred)
                    }
                
                results[name] = metrics
            except Exception as e:
                logger.error(f"Erro ao avaliar modelo {name}: {e}")
        
        return results

class AutoFETestSuite:
    """
    Suíte de testes abrangente para o sistema AutoFE
    
    Cobre múltiplos cenários de machine learning e valida
    a eficácia do pipeline de engenharia de features
    """
    
    def __init__(self):
        """
        Inicializa componentes do sistema AutoFE para teste
        """
        self.preprocessor = PreProcessor()
        self.explorer = Explorer()
        self.predictor = Predictor()
        self.postprocessor = PosProcessor()
    
    def _run_autofe_pipeline(self, df, target_column, problem_type):
        """
        Executa o pipeline completo de AutoFE
        
        Args:
            df (pd.DataFrame): Dataset para processamento
            target_column (str): Nome da coluna alvo
            problem_type (ProblemType): Tipo do problema
        
        Returns:
            tuple: DataFrame processado e relatórios de cada etapa
        """
        # Pré-processamento
        df_preprocessed, preprocess_report = self.preprocessor.process(
            df, target_column, problem_type
        )
        
        # Exploração de features
        df_explored, explore_report = self.explorer.process(
            df_preprocessed, target_column, problem_type
        )
        
        # Predição e transformação de features
        df_predicted, predict_report = self.predictor.process(
            df_explored, target_column, problem_type
        )
        
        # Seleção final de features
        df_final, final_report = self.postprocessor.process(
            df_predicted, target_column, problem_type,
            [preprocess_report, explore_report, predict_report]
        )
        
        self.save_model_results(problem_type, "dataset", final_report)
        return df_final, {
            'preprocess': preprocess_report,
            'explore': explore_report,
            'predict': predict_report,
            'final': final_report
        }
    
    def _compare_feature_engineering_impact(self, df, target_column, problem_type):
        """
        Compara o impacto da engenharia de features
        
        Args:
            df (pd.DataFrame): Dataset original
            target_column (str): Nome da coluna alvo
            problem_type (ProblemType): Tipo do problema
        
        Returns:
            dict: Resultados comparativos
        """
        # Features originais
        X_original = df.drop(columns=[target_column])
        y_original = df[target_column]
        
        # Avaliação inicial
        original_results = ModelPerformanceComparator.evaluate_models(
            X_original, y_original, problem_type
        )
        
        # Executa pipeline de AutoFE
        df_processed, _ = self._run_autofe_pipeline(df, target_column, problem_type)
        
        # Features processadas
        X_processed = df_processed.drop(columns=[target_column])
        y_processed = df_processed[target_column]
        
        # Avaliação após processamento
        processed_results = ModelPerformanceComparator.evaluate_models(
            X_processed, y_processed, problem_type
        )
        
        return {
            'original_results': original_results,
            'processed_results': processed_results,
            'original_features': X_original.shape[1],
            'processed_features': X_processed.shape[1]
        }
    
    def test_classification_scenario(self):
        """
        Teste de cenário de classificação usando dataset Iris
        """
        logger.info("=== Teste de Cenário de Classificação ===")
        
        # Carrega dataset Iris
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        
        # Define parâmetros do problema
        target_column = 'target'
        problem_type = ProblemType.CLASSIFICATION
        
        # Compara impacto da engenharia de features
        comparison_results = self._compare_feature_engineering_impact(
            df, target_column, problem_type
        )
        
        # Registra resultados
        self._log_performance_comparison(comparison_results)
        
    def test_regression_scenario(self):
        """
        Teste de cenário de regressão usando dataset Diabetes
        """
        logger.info("=== Teste de Cenário de Regressão ===")
        
        # Carrega dataset Diabetes
        diabetes = load_wine()
        df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
        df['target'] = diabetes.target
        
        # Define parâmetros do problema
        target_column = 'Classes'
        problem_type = ProblemType.CLASSIFICATION
        
        # Compara impacto da engenharia de features
        comparison_results = self._compare_feature_engineering_impact(
            df, target_column, problem_type
        )
        
        # Registra resultados
        self._log_performance_comparison(comparison_results)
    
    
    
    def test_regression_scenario(self):
        """
        Teste de cenário de regressão usando dataset Diabetes
        """
        logger.info("=== Teste de Cenário de Regressão ===")
        
        # Carrega dataset Diabetes
        diabetes = fetch_california_housing()
        df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
        df['target'] = diabetes.target
        
        # Define parâmetros do problema
        target_column = 'target'
        problem_type = ProblemType.REGRESSION
        
        # Compara impacto da engenharia de features
        comparison_results = self._compare_feature_engineering_impact(
            df, target_column, problem_type
        )
        
        # Registra resultados
        self._log_performance_comparison(comparison_results)
        
    
    def test_regression_scenario(self):
        """
        Teste de cenário de regressão usando dataset Diabetes
        """
        logger.info("=== Teste de Cenário de Regressão ===")
        
        # Carrega dataset Diabetes
        diabetes = load_diabetes()
        df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
        df['target'] = diabetes.target
        
        # Define parâmetros do problema
        target_column = 'target'
        problem_type = ProblemType.REGRESSION
        
        # Compara impacto da engenharia de features
        comparison_results = self._compare_feature_engineering_impact(
            df, target_column, problem_type
        )
        
        # Registra resultados
        self._log_performance_comparison(comparison_results)
        
        
    def test_regression_scenario_2(self):
        """
        Teste de cenário de regressão usando dataset Diabetes
        """
        logger.info("=== Teste de Cenário de Regressão ===")
        
        # Carrega dataset Diabetes
        df = pd.read_csv("arquivo.csv")
        df['target'] = df["FLT_TOT_1"]
        
        # Define parâmetros do problema
        target_column = 'target'
        problem_type = ProblemType.REGRESSION
        
        # Compara impacto da engenharia de features
        comparison_results = self._compare_feature_engineering_impact(
            df, target_column, problem_type
        )
        
        # Registra resultados
        self._log_performance_comparison(comparison_results)
    
    def test_synthetic_datasets(self):
        """
        Testa o sistema com datasets sintéticos controlados
        """
        logger.info("=== Teste com Datasets Sintéticos ===")
        
        # Dataset sintético de classificação
        X_class, y_class = make_classification(
            n_samples=1000, n_features=20, 
            n_informative=10, n_redundant=5, 
            random_state=42
        )
        df_class = pd.DataFrame(X_class, columns=[f'feature_{i}' for i in range(20)])
        df_class['target'] = y_class
        
        # Dataset sintético de regressão
        X_reg, y_reg = make_regression(
            n_samples=1000, n_features=20, 
            n_informative=10, noise=0.1, 
            random_state=42
        )
        df_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(20)])
        df_reg['target'] = y_reg
        
        # Testes com datasets sintéticos
        logger.info("Teste de Classificação com Dataset Sintético")
        df_class_processed, _ = self._run_autofe_pipeline(
            df_class, 'target', ProblemType.CLASSIFICATION
        )
        
        logger.info("Teste de Regressão com Dataset Sintético")
        df_reg_processed, _ = self._run_autofe_pipeline(
            df_reg, 'target', ProblemType.REGRESSION
        )
    
    
    def _log_performance_comparison(self, comparison_results):
        """
        Registra comparação de desempenho dos modelos
        
        Args:
            comparison_results (dict): Resultados da comparação de performance
        """
        logger.info("Resultados de Performance Original:")
        for model, metrics in comparison_results['original_results'].items():
            logger.info(f"{model}: {metrics}")
        
        logger.info("\nResultados de Performance Após Processamento:")
        for model, metrics in comparison_results['processed_results'].items():
            logger.info(f"{model}: {metrics}")
        
        logger.info("\nResumo de Features:")
        logger.info(f"Features Originais: {comparison_results['original_features']}")
        logger.info(f"Features Processadas: {comparison_results['processed_features']}")
    
    def run_comprehensive_tests(self):
        """
        Executa todos os testes disponíveis
        """
        test_methods = [
            # self.test_classification_scenario,
            # self.test_regression_scenario,
            # self.test_synthetic_datasets,
            self.test_regression_scenario_2
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                logger.error(f"Teste {test_method.__name__} falhou: {e}")
                logger.exception("Detalhes do erro:")

    
    def save_model_results(self, problem_type, dataset_name, results):
        import json
        from datetime import datetime
        """
        Salva resultados dos modelos em formato JSON
        
        Args:
            problem_type (ProblemType): Tipo do problema
            dataset_name (str): Nome do dataset
            results (dict): Resultados dos modelos
        """
        # Caminho para o arquivo de resultados
        results_file = os.path.join(".", f"{dataset_name}_{problem_type.name.lower()}-{datetime.now().isoformat()}_results.json")
        
        # Adiciona metadados
        full_results = {
            'timestamp': datetime.now().isoformat(),
            'problem_type': problem_type.name,
            'dataset': dataset_name,
            'models': results
        }
        
        # Salva resultados em JSON
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=4)
        
        logger.info(f"Resultados salvos em {results_file}")
        
        
def main():
    """
    Ponto de entrada para execução dos testes
    """
    test_suite = AutoFETestSuite()
    test_suite.run_comprehensive_tests()

if __name__ == "__main__":
    main()
