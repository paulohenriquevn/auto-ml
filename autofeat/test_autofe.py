# tests/test_autofe.py
import sys
import os
import pandas as pd
import numpy as np
import logging
from sklearn.datasets import load_iris, load_diabetes, fetch_california_housing, load_wine
from sklearn.model_selection import train_test_split

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data_types import ProblemType
from preprocessing.preprocessor import PreProcessor
from explorer.explorer import Explorer
from predictor.predictor import Predictor
from postprocessor.postprocessor import PosProcessor

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tests.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AutoFE.Tests")

class TestAutoFE:
    """
    Classe para testar os componentes do sistema AutoFE.
    """
    
    def __init__(self):
        """
        Inicializa a classe de testes.
        """
        self.preprocessor = PreProcessor()
        self.explorer = Explorer()
        self.predictor = Predictor()
        self.postprocessor = PosProcessor()
        
        logger.info("Testes inicializados")
    
    def run_all_tests(self):
        """
        Executa todos os testes disponíveis.
        """
        try:
            self.test_classification()
        except Exception as e:
            logger.error(f"Teste de CLASSIFICAÇÃO falhou: {e}")
            
        try:
            self.test_regression()
        except Exception as e:
            logger.error(f"Teste de REGRESSÃO falhou: {e}")
            
        try:
            self.test_time_series()
        except Exception as e:
            logger.error(f"Teste de SÉRIES TEMPORAIS falhou: {e}")
            
        try:
            self.test_text()
        except Exception as e:
            logger.error(f"Teste de TEXTO falhou: {e}")
            
        logger.info("TODOS OS TESTES CONCLUÍDOS")
    
    def test_classification(self):
        """
        Testa o sistema em um problema de classificação.
        """
        logger.info("=== Testando problema de CLASSIFICAÇÃO ===")
        
        # Carrega o dataset Iris
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        
        # Define parâmetros do problema
        target_column = 'target'
        problem_type = ProblemType.CLASSIFICATION
        
        # Executa o pipeline completo
        self._test_pipeline(df, target_column, problem_type, "CLASSIFICAÇÃO")
    
    def test_regression(self):
        """
        Testa o sistema em um problema de regressão.
        """
        logger.info("=== Testando problema de REGRESSÃO ===")
        
        # Carrega o dataset Diabetes
        diabetes = load_diabetes()
        df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
        df['target'] = diabetes.target
        
        # Converte categorias para strings antes de aplicar o processamento
        # Isso evita problemas com categorias numéricas
        for col in df.columns:
            if col != 'target' and len(df[col].unique()) < 10:
                df[col] = df[col].astype(str)
        
        # Define parâmetros do problema
        target_column = 'target'
        problem_type = ProblemType.REGRESSION
        
        # Executa o pipeline completo
        self._test_pipeline(df, target_column, problem_type, "REGRESSÃO")


    def test_time_series(self):
        """
        Testa o sistema em um problema de séries temporais.
        """
        logger.info("=== Testando problema de SÉRIES TEMPORAIS ===")
        
        # Cria um dataset sintético de séries temporais
        np.random.seed(42)
        
        # Cria datas para 2 anos com frequência diária
        dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
        
        # Cria features - primeiro vamos criar um DataFrame sem a coluna de data
        df = pd.DataFrame({
            'temp': np.random.normal(20, 5, len(dates)),  # Temperatura
            'humidity': np.random.normal(60, 10, len(dates)),  # Umidade
            'pressure': np.random.normal(1013, 5, len(dates)),  # Pressão atmosférica
            'wind_speed': np.random.normal(10, 3, len(dates))  # Velocidade do vento
        })
        
        # Adiciona sazonalidade e tendência à temperatura
        df['temp'] += 10 * np.sin(2 * np.pi * (np.arange(len(dates)) / 365))  # Sazonalidade anual
        df['temp'] += np.arange(len(dates)) * 0.01  # Tendência crescente
        
        # Adiciona ruído
        df['temp'] += np.random.normal(0, 1, len(dates))
        
        # Define o alvo como a temperatura do dia seguinte
        df['target'] = df['temp'].shift(-1)
        
        # Adiciona características temporais já extraídas da data
        df['day'] = dates.day
        df['month'] = dates.month
        df['year'] = dates.year
        df['dayofweek'] = dates.dayofweek
        
        # Remove a última linha (que terá NaN no target)
        df = df.dropna()
        
        # Define parâmetros do problema
        target_column = 'target'
        problem_type = ProblemType.TIME_SERIES
        
        # Executa o pipeline completo - sem passar a coluna de data
        self._test_pipeline(df, target_column, problem_type, "SÉRIES TEMPORAIS")


    def test_text(self):
        """
        Testa o sistema em um problema envolvendo texto.
        """
        logger.info("=== Testando problema com TEXTO ===")
        
        # Cria um dataset sintético com texto
        np.random.seed(42)
        
        # Lista de frases para cada categoria
        positive_phrases = [
            "Excelente produto, super recomendo!",
            "Amei a compra, valeu cada centavo.",
            "Produto de alta qualidade, atendeu todas expectativas.",
            "Superou minhas expectativas, muito satisfeito.",
            "Ótimo custo-benefício, recomendo para todos."
        ]
        
        negative_phrases = [
            "Produto de péssima qualidade, não recomendo.",
            "Decepcionado com a compra, não vale o preço.",
            "Quebrou no primeiro uso, péssimo produto.",
            "Atendimento terrível e produto abaixo da expectativa.",
            "Não comprem, é uma furada total."
        ]
        
        neutral_phrases = [
            "Produto dentro do esperado para o preço.",
            "Atendeu às necessidades básicas, nada especial.",
            "Produto comum, sem grandes surpresas.",
            "Qualidade mediana, cumpre o que promete.",
            "Atendimento normal, sem problemas ou elogios."
        ]
        
        # Cria o dataset
        reviews = []
        sentiments = []
        
        # Adiciona frases positivas
        for phrase in positive_phrases:
            reviews.append(phrase)
            sentiments.append(2)  # 2 = Positivo
        
        # Adiciona frases negativas
        for phrase in negative_phrases:
            reviews.append(phrase)
            sentiments.append(0)  # 0 = Negativo
        
        # Adiciona frases neutras
        for phrase in neutral_phrases:
            reviews.append(phrase)
            sentiments.append(1)  # 1 = Neutro
        
        # Adiciona algumas duplicatas para testar remoção de duplicatas
        reviews.append(positive_phrases[0])
        sentiments.append(2)
        
        # Cria o DataFrame
        df = pd.DataFrame({
            'review_text': reviews,
            'sentiment': sentiments,
            'length': [len(text) for text in reviews],
            'word_count': [len(text.split()) for text in reviews]
        })
        
        # Define parâmetros do problema
        target_column = 'sentiment'
        problem_type = ProblemType.TEXT
        
        # Executa o pipeline completo
        self._test_pipeline(df, target_column, problem_type, "TEXTO")
    
    def _test_pipeline(self, df, target_column, problem_type, problem_name, time_column=None):
        """
        Executa o pipeline completo em um dataset.
        
        Args:
            df (pandas.DataFrame): DataFrame a ser processado.
            target_column (str): Nome da coluna alvo.
            problem_type (ProblemType): Tipo do problema.
            problem_name (str): Nome descritivo do problema.
            time_column (str, optional): Nome da coluna temporal.
        """
        logger.info(f"Testando pipeline completo para problema: {problem_name}")
        logger.info(f"Shape original: {df.shape}")
        
        try:
            # Etapa 1: Pré-processamento
            logger.info("Executando pré-processador...")
            df_preprocessed, preprocess_report = self.preprocessor.process(
                df, target_column, problem_type, time_column
            )
            logger.info(f"Pré-processamento concluído. Shape: {df_preprocessed.shape}")
            
            # Etapa 2: Exploração e geração de features
            logger.info("Executando explorer...")
            df_explored, explore_report = self.explorer.process(
                df_preprocessed, target_column, problem_type, time_column
            )
            logger.info(f"Explorer concluído. Shape: {df_explored.shape}")
            
            # Etapa 3: Predição de melhores transformações
            logger.info("Executando predictor...")
            df_predicted, predict_report = self.predictor.process(
                df_explored, target_column, problem_type, time_column
            )
            logger.info(f"Predictor concluído. Shape: {df_predicted.shape}")
            
            # Etapa 4: Pós-processamento
            logger.info("Executando pós-processador...")
            df_final, final_report = self.postprocessor.process(
                df_predicted, target_column, problem_type, time_column,
                [preprocess_report, explore_report, predict_report]
            )
            logger.info(f"Pós-processamento concluído. Shape: {df_final.shape}")
            
            # Exibe informações do relatório final
            if 'dataset_score' in final_report:
                logger.info(f"Score final do dataset: {final_report['dataset_score']}/10")
            logger.info(f"Features originais: {df.shape[1] - 1}")
            logger.info(f"Features finais: {df_final.shape[1] - 1}")
            
            logger.info(f"Teste para {problem_name} CONCLUÍDO COM SUCESSO")
            logger.info("-" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"Erro no teste para {problem_name}: {str(e)}")
            logger.exception("Detalhes do erro:")
            logger.info("-" * 80)
            
            return False

if __name__ == "__main__":
    # Executa os testes
    tester = TestAutoFE()
    tester.run_all_tests()