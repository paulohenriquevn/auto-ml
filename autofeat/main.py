# main.py
from preprocessing.preprocessor import PreProcessor
from explorer.explorer import Explorer
from predictor.predictor import Predictor
from postprocessor.postprocessor import PosProcessor
from common.data_types import DataType, ProblemType
import pandas as pd
import os
import json
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("autofe.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AutoFE")

class AutoFE:
    """
    Sistema principal para automação de engenharia de features.
    
    Este sistema integra os quatro módulos principais (PreProcessor, Explorer,
    Predictor e PosProcessor) para automatizar completamente o processo de
    engenharia de features em conjuntos de dados.
    """
    
    def __init__(self, config_path=None):
        """
        Inicializa o sistema AutoFE.
        
        Args:
            config_path (str, optional): Caminho para o arquivo de configuração. 
                                        Se None, usa configurações padrão.
        """
        self.config = self._load_config(config_path)
        self.preprocessor = PreProcessor(self.config.get('preprocessing', {}))
        self.explorer = Explorer(self.config.get('explorer', {}))
        self.predictor = Predictor(self.config.get('predictor', {}))
        self.postprocessor = PosProcessor(self.config.get('postprocessor', {}))
        
        logger.info("Sistema AutoFE inicializado com sucesso")
    
    def _load_config(self, config_path):
        """
        Carrega configurações do sistema a partir de um arquivo JSON.
        
        Args:
            config_path (str): Caminho para o arquivo de configuração.
            
        Returns:
            dict: Configurações carregadas ou configurações padrão.
        """
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Erro ao carregar configuração: {e}")
                logger.info("Usando configurações padrão")
        
        # Configurações padrão
        return {
            'preprocessing': {
                'remove_duplicates': True,
                'handle_missing_values': True,
                'handle_outliers': True,
                'normalize_data': True
            },
            'explorer': {
                'max_depth': 3,
                'min_feature_importance': 0.01,
                'max_features': 100
            },
            'predictor': {
                'use_meta_learning': True,
                'evaluation_metric': 'auto'
            },
            'postprocessor': {
                'feature_selection': True,
                'min_importance_threshold': 0.05,
                'report_detail_level': 'detailed'
            }
        }
    
    def process(self, df, target_column, problem_type, time_column=None):
        """
        Processa um DataFrame aplicando todos os passos de engenharia de features.
        
        Args:
            df (pandas.DataFrame): O DataFrame a ser processado.
            target_column (str): Nome da coluna alvo para predição.
            problem_type (ProblemType): Tipo do problema (classificação, regressão, etc.).
            time_column (str, optional): Nome da coluna temporal para séries temporais.
            
        Returns:
            tuple: (DataFrame processado, relatório de processamento)
        """
        logger.info(f"Iniciando processamento para problema do tipo: {problem_type}")
        
        # Etapa 1: Pré-processamento
        df_preprocessed, preprocess_report = self.preprocessor.process(
            df, target_column, problem_type, time_column
        )
        logger.info("Pré-processamento concluído")
        
        # Etapa 2: Exploração e geração de features
        df_explored, explore_report = self.explorer.process(
            df_preprocessed, target_column, problem_type, time_column
        )
        logger.info("Exploração de features concluída")
        
        # Etapa 3: Predição de melhores transformações
        df_predicted, predict_report = self.predictor.process(
            df_explored, target_column, problem_type, time_column
        )
        logger.info("Predição de transformações concluída")
        
        # Etapa 4: Pós-processamento (seleção final e relatório)
        df_final, final_report = self.postprocessor.process(
            df_predicted, target_column, problem_type, time_column,
            [preprocess_report, explore_report, predict_report]
        )
        logger.info("Pós-processamento concluído")
        
        return df_final, final_report
    
    def save_results(self, df, report, output_dir="./output"):
        """
        Salva os resultados do processamento.
        
        Args:
            df (pandas.DataFrame): DataFrame processado final.
            report (dict): Relatório gerado pelo sistema.
            output_dir (str): Diretório para salvar os resultados.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Salva o DataFrame processado
        df.to_csv(f"{output_dir}/processed_data.csv", index=False)
        
        # Salva o relatório como JSON
        with open(f"{output_dir}/report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Salva uma versão legível do relatório
        with open(f"{output_dir}/report.txt", 'w') as f:
            f.write(self._format_report(report))
        
        logger.info(f"Resultados salvos em: {output_dir}")
    
    def _format_report(self, report):
        """
        Formata o relatório JSON em texto legível.
        
        Args:
            report (dict): Relatório em formato de dicionário.
            
        Returns:
            str: Relatório formatado como texto.
        """
        formatted = "=== RELATÓRIO DE ENGENHARIA DE FEATURES ===\n\n"
        
        # Adiciona score geral
        if 'dataset_score' in report:
            formatted += f"Pontuação do Dataset: {report['dataset_score']}/10\n\n"
        
        # Formata as diferentes seções do relatório
        for section, content in report.items():
            if section == 'dataset_score':
                continue
                
            formatted += f"== {section.replace('_', ' ').title()} ==\n"
            
            if isinstance(content, dict):
                for key, value in content.items():
                    formatted += f"  - {key}: {value}\n"
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            formatted += f"  - {k}: {v}\n"
                    else:
                        formatted += f"  - {item}\n"
            else:
                formatted += f"  {content}\n"
            
            formatted += "\n"
        
        return formatted


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AutoFE: Sistema de Engenharia de Features Automatizado")
    parser.add_argument("--data", required=True, help="Caminho para o arquivo CSV de entrada")
    parser.add_argument("--target", required=True, help="Nome da coluna alvo")
    parser.add_argument("--problem", required=True, choices=[p.name for p in ProblemType], 
                       help="Tipo de problema (CLASSIFICATION, REGRESSION, TEXT, TIME_SERIES)")
    parser.add_argument("--time-col", help="Coluna de tempo para problemas de séries temporais")
    parser.add_argument("--config", help="Caminho para arquivo de configuração")
    parser.add_argument("--output", default="./output", help="Diretório de saída")
    
    args = parser.parse_args()
    
    # Carrega os dados
    df = pd.read_csv(args.data)
    
    # Inicializa o sistema
    autofe = AutoFE(args.config)
    
    # Converte o tipo de problema de string para enum
    problem_type = ProblemType[args.problem]
    
    # Processa os dados
    processed_df, report = autofe.process(df, args.target, problem_type, args.time_col)
    
    # Salva os resultados
    autofe.save_results(processed_df, report, args.output)