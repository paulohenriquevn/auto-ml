# preprocessing/preprocessor.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from typing import Dict, List, Tuple, Any, Optional
import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importações locais
from common.data_types import DataType, ProblemType, DatasetInfo, ColumnInfo
from preprocessing.data_utils import DataPreprocessor

# Configuração de Logging
logger = logging.getLogger("AutoFE.PreProcessor")

class PreProcessor:
    """
    Módulo responsável pela limpeza e preparação inicial dos dados.
    
    Este módulo realiza as seguintes operações:
    - Identificação e correção de erros nos dados
    - Tratamento de valores ausentes
    - Remoção de duplicatas
    - Detecção e tratamento de outliers
    - Conversão de tipos de dados
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o módulo de pré-processamento.
        
        Args:
            config (dict): Configurações para o pré-processamento.
        """
        self.config = config or {}
        self.remove_duplicates = self.config.get('remove_duplicates', True)
        self.handle_missing_values = self.config.get('handle_missing_values', True)
        self.handle_outliers = self.config.get('handle_outliers', True)
        self.normalize_data = self.config.get('normalize_data', True)
        self.max_categories = self.config.get('max_categories', 20)
        
        logger.info("Módulo de pré-processamento inicializado")
    
    def process(self, df: pd.DataFrame, target_column: str, 
               problem_type: ProblemType, time_column: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Processa o DataFrame aplicando todos os passos de pré-processamento.
        
        Args:
            df (pd.DataFrame): DataFrame a ser processado.
            target_column (str): Nome da coluna alvo.
            problem_type (ProblemType): Tipo do problema.
            time_column (str, optional): Nome da coluna temporal.
            
        Returns:
            tuple: DataFrame processado e relatório do processamento.
        """
        # Inicializa o relatório de processamento
        report = {
            "original_shape": df.shape,
            "steps": [],
            "column_types": {}
        }
        
        # Cria uma cópia de trabalho do DataFrame
        df_working = df.copy()
        
        # Passo 1: Análise inicial do dataset
        dataset_info = DatasetInfo.from_dataframe(df_working, target_column, time_column, problem_type)
        
        # Registra informações iniciais no relatório
        report["initial_analysis"] = self._get_dataset_analysis(dataset_info)
        
        # Passo 2: Utiliza o novo DataPreprocessor para processamento robusto
        try:
            # Pré-processamento avançado
            df_working = DataPreprocessor.preprocess_dataframe(
                df_working, 
                target_column=target_column,
                max_categories=self.max_categories
            )
            
            # Identifica tipos de colunas após processamento
            column_types = DataPreprocessor.identify_column_types(df_working)
            report['column_types'] = column_types
            
            # Adiciona passos de processamento ao relatório
            report['steps'].append({
                'action': 'advanced_preprocessing',
                'details': {
                    'categorical_normalization': f'Máximo de {self.max_categories} categorias',
                    'numeric_conversion': 'Conversão robusta',
                    'missing_values_handling': 'Tratamento inteligente'
                }
            })
        except Exception as e:
            logger.error(f"Erro no pré-processamento avançado: {e}")
            logger.warning("Usando métodos de processamento padrão")
            
            # Métodos originais de processamento como fallback
            # (mantém a lógica original dos métodos abaixo)
            if self.remove_duplicates:
                df_working = self._remove_duplicates(df_working, report)
            
            if self.handle_missing_values:
                df_working, missing_report = self._handle_missing_values(df_working, dataset_info)
                report['steps'].append({"missing_values": missing_report})
            
            if self.handle_outliers:
                df_working, outlier_report = self._handle_outliers(df_working, dataset_info)
                report['steps'].append({"outliers": outlier_report})
            
            if self.normalize_data:
                df_working, norm_report = self._normalize_data(df_working, dataset_info)
                report['steps'].append({"normalization": norm_report})
        
        # Gera relatório final
        report['final_shape'] = df_working.shape
        
        # Adiciona informações sobre transformações
        columns_before = set(df.columns)
        columns_after = set(df_working.columns)
        
        report['columns_added'] = list(columns_after - columns_before)
        report['columns_removed'] = list(columns_before - columns_after)
        
        # Adiciona informações sobre tipos de dados
        report['data_types_after'] = {
            col: str(df_working[col].dtype) for col in df_working.columns
        }
        
        logger.info(f"Pré-processamento concluído. Formato final: {df_working.shape}")
        
        return df_working, report
    
    # Os outros métodos da classe (como _remove_duplicates, etc.) permanecem inalterados
    # Incluindo _get_dataset_analysis, _handle_missing_values, etc.
    
    def _get_dataset_analysis(self, dataset_info: DatasetInfo) -> Dict[str, Any]:
        """
        Gera um relatório de análise inicial do dataset.
        
        Args:
            dataset_info (DatasetInfo): Informações sobre o dataset.
            
        Returns:
            dict: Análise do dataset.
        """
        analysis = {
            "num_rows": dataset_info.num_rows,
            "num_columns": dataset_info.num_columns,
            "target_column": dataset_info.target_column,
            "problem_type": dataset_info.problem_type.name if dataset_info.problem_type else None,
            "column_types": {},
            "missing_values": {},
            "unique_values": {}
        }
        
        for col in dataset_info.columns:
            analysis["column_types"][col.name] = col.data_type.name
            
            if col.missing_percentage > 0:
                analysis["missing_values"][col.name] = f"{col.missing_percentage:.2f}%"
                
            analysis["unique_values"][col.name] = col.num_unique_values
            
        return analysis
    
    # Os demais métodos da classe original (_remove_duplicates, 
    # _handle_missing_values, _handle_outliers, _normalize_data) 
    # permanecem inalterados