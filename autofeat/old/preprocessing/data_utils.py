# preprocessing/data_utils.py

"""
Utilitários avançados para pré-processamento de dados
Fornece métodos robustos para identificação, normalização 
e conversão de tipos de dados.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger("AutoFE.DataUtils")

class DataPreprocessor:
    """
    Utilitário avançado para pré-processamento de dados
    
    Responsabilidades:
    1. Identificar tipos de dados
    2. Normalizar e converter dados
    3. Preparar dados para machine learning
    """
    
    @staticmethod
    def identify_column_types(df: pd.DataFrame) -> Dict[str, str]:
        """
        Identifica os tipos de colunas de forma inteligente
        
        Args:
            df (pd.DataFrame): DataFrame para análise
        
        Returns:
            dict: Mapeamento de tipos de colunas
        """
        column_types = {}
        
        for column in df.columns:
            # Análise de tipos
            if pd.api.types.is_numeric_dtype(df[column]):
                column_types[column] = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                column_types[column] = 'datetime'
            elif df[column].nunique() / len(df) < 0.05:  # Categorias limitadas
                column_types[column] = 'categorical'
            elif df[column].dtype == 'object':
                # Tenta converter para numérico
                try:
                    pd.to_numeric(df[column], errors='raise')
                    column_types[column] = 'numeric'
                except:
                    column_types[column] = 'text'
            else:
                column_types[column] = 'other'
        
        return column_types
    
    @staticmethod
    def normalize_categorical_columns(
        df: pd.DataFrame, 
        max_categories: int = 20, 
        target_column: str = None
    ) -> pd.DataFrame:
        """
        Normaliza colunas categóricas limitando número de categorias
        
        Args:
            df (pd.DataFrame): DataFrame para processamento
            max_categories (int): Número máximo de categorias
            target_column (str, opcional): Coluna alvo para referência
        
        Returns:
            pd.DataFrame: DataFrame com categorias normalizadas
        """
        df_normalized = df.copy()
        
        # Colunas para processar (excluindo coluna alvo se especificada)
        columns_to_process = [
            col for col in df.select_dtypes(include=['object', 'category']).columns
            if col != target_column
        ]
        
        for column in columns_to_process:
            # Obtém categorias mais frequentes
            top_categories = df[column].value_counts().nlargest(max_categories).index
            
            # Agrupa categorias menos frequentes
            df_normalized[column] = df_normalized[column].apply(
                lambda x: x if x in top_categories else 'Outros'
            )
            
            # Converte para categoria
            df_normalized[column] = df_normalized[column].astype('category')
        
        return df_normalized
    
    @staticmethod
    def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converte colunas para numérico de forma robusta
        
        Args:
            df (pd.DataFrame): DataFrame para conversão
        
        Returns:
            pd.DataFrame: DataFrame com colunas convertidas
        """
        df_converted = df.copy()
        
        for column in df.columns:
            # Pula se já for numérico
            if pd.api.types.is_numeric_dtype(df[column]):
                continue
            
            # Tenta converter removendo caracteres não numéricos
            try:
                # Remove caracteres não numéricos, exceto ponto e sinal de menos
                cleaned_series = df[column].astype(str).str.replace(
                    r'[^\d.-]', '', 
                    regex=True
                )
                
                # Converte para numérico
                numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                
                # Substitui a coluna se conversão for bem-sucedida
                if not numeric_series.isna().all():
                    df_converted[column] = numeric_series
            except Exception as e:
                logger.warning(f"Erro ao converter coluna {column}: {e}")
        
        return df_converted
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Trata valores ausentes de forma inteligente
        
        Args:
            df (pd.DataFrame): DataFrame para tratamento
        
        Returns:
            pd.DataFrame: DataFrame com valores ausentes tratados
        """
        df_handled = df.copy()
        
        for column in df.columns:
            # Estratégia de preenchimento baseada no tipo de dados
            if pd.api.types.is_numeric_dtype(df[column]):
                # Preenche com mediana para numéricas
                df_handled[column].fillna(df[column].median(), inplace=True)
            elif pd.api.types.is_categorical_dtype(df[column]):
                # Preenche com modo para categóricas
                df_handled[column].fillna(df[column].mode()[0], inplace=True)
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                # Preenche com data mais próxima para datetime
                df_handled[column].fillna(method='ffill', inplace=True)
                df_handled[column].fillna(method='bfill', inplace=True)
            else:
                # Outros tipos: preenche com 'Unknown' ou valor mais comum
                df_handled[column].fillna('Unknown', inplace=True)
        
        return df_handled
    
    @classmethod
    def preprocess_dataframe(
        cls, 
        df: pd.DataFrame, 
        target_column: str = None,
        max_categories: int = 20
    ) -> pd.DataFrame:
        """
        Pipeline completo de pré-processamento
        
        Args:
            df (pd.DataFrame): DataFrame original
            target_column (str, opcional): Coluna alvo
            max_categories (int): Máximo de categorias para normalização
        
        Returns:
            pd.DataFrame: DataFrame completamente processado
        """
        # Normaliza colunas categóricas
        df_normalized = cls.normalize_categorical_columns(
            df, 
            max_categories=max_categories, 
            target_column=target_column
        )
        
        # Converte para numérico
        df_numeric = cls.convert_to_numeric(df_normalized)
        
        # Trata valores ausentes
        df_clean = cls.handle_missing_values(df_numeric)
        
        # Remove colunas com todos valores ausentes
        df_final = df_clean.dropna(axis=1, how='all')
        
        return df_final