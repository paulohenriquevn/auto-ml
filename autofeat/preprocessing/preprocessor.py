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

from common.data_types import DataType, ProblemType, DatasetInfo, ColumnInfo

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
        Processa o DataFrame aplicando todas as etapas de pré-processamento.
        
        Args:
            df (pd.DataFrame): DataFrame a ser processado.
            target_column (str): Nome da coluna alvo.
            problem_type (ProblemType): Tipo do problema.
            time_column (str, optional): Nome da coluna temporal.
            
        Returns:
            tuple: DataFrame processado e relatório do processamento.
        """
        report = {
            "original_shape": df.shape,
            "steps": []
        }
        
        # Passo 1: Análise inicial do dataset
        dataset_info = DatasetInfo.from_dataframe(df, target_column, time_column, problem_type)
        report["dataset_analysis"] = self._get_dataset_analysis(dataset_info)
        
        # Guarda as colunas originais para comparação posterior
        original_columns = set(df.columns)
        
        # Passo 2: Conversão de tipos de dados
        df, type_conversion_report = self._convert_data_types(df, dataset_info)
        report["steps"].append({"type_conversion": type_conversion_report})
        
        # Passo 3: Remover duplicatas
        if self.remove_duplicates:
            original_rows = len(df)
            df = df.drop_duplicates()
            duplicates_removed = original_rows - len(df)
            report["steps"].append({
                "duplicates_removed": {
                    "count": duplicates_removed,
                    "percentage": round(duplicates_removed / original_rows * 100, 2) if original_rows > 0 else 0
                }
            })
            logger.info(f"Removidas {duplicates_removed} linhas duplicadas")
        
        # Passo 4: Tratamento de valores ausentes
        if self.handle_missing_values:
            df, missing_report = self._handle_missing_values(df, dataset_info)
            report["steps"].append({"missing_values": missing_report})
            
        # Passo 5: Tratamento de outliers
        if self.handle_outliers:
            df, outlier_report = self._handle_outliers(df, dataset_info)
            report["steps"].append({"outliers": outlier_report})
            
        # Passo 6: Normalização dos dados (se aplicável)
        if self.normalize_data:
            df, norm_report = self._normalize_data(df, dataset_info)
            report["steps"].append({"normalization": norm_report})
            
        # Atualizar informações do dataset após o processamento
        final_columns = set(df.columns)
        
        # Identifica colunas modificadas (adicionadas ou removidas)
        added_columns = final_columns - original_columns
        removed_columns = original_columns - final_columns
        
        report["final_shape"] = df.shape
        report["columns_added"] = list(added_columns)
        report["columns_removed"] = list(removed_columns)
        
        logger.info(f"Pré-processamento concluído. Formato final: {df.shape}")
        
        return df, report
    
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
    
    def _convert_data_types(self, df: pd.DataFrame, dataset_info: DatasetInfo) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Converte os tipos de dados das colunas para os tipos apropriados.
        
        Args:
            df (pd.DataFrame): DataFrame a ser processado.
            dataset_info (DatasetInfo): Informações sobre o dataset.
            
        Returns:
            tuple: DataFrame com tipos convertidos e relatório da conversão.
        """
        conversion_report = []
        df_converted = df.copy()
        
        for col_info in dataset_info.columns:
            col_name = col_info.name
            
            try:
                # Tenta converter para datetime
                if col_info.data_type == DataType.UNKNOWN:
                    try:
                        df_converted[col_name] = pd.to_datetime(df[col_name], errors='raise')
                        conversion_report.append({
                            "column": col_name,
                            "from_type": str(df[col_name].dtype),
                            "to_type": "datetime64",
                            "success": True
                        })
                        logger.info(f"Coluna {col_name} convertida para datetime")
                        continue
                    except:
                        pass
                    
                # Tenta converter para numérico
                if col_info.data_type == DataType.UNKNOWN:
                    try:
                        df_converted[col_name] = pd.to_numeric(df[col_name], errors='raise')
                        conversion_report.append({
                            "column": col_name,
                            "from_type": str(df[col_name].dtype),
                            "to_type": str(df_converted[col_name].dtype),
                            "success": True
                        })
                        logger.info(f"Coluna {col_name} convertida para numérico")
                        continue
                    except:
                        pass
                
                # Verifica se deve converter para categórico
                if col_info.data_type == DataType.CATEGORICAL or (
                    col_info.num_unique_values and 
                    col_info.num_unique_values < self.max_categories and 
                    col_info.num_unique_values < len(df) * 0.05
                ):
                    df_converted[col_name] = df[col_name].astype('category')
                    conversion_report.append({
                        "column": col_name,
                        "from_type": str(df[col_name].dtype),
                        "to_type": "category",
                        "success": True
                    })
                    logger.info(f"Coluna {col_name} convertida para categórica")
            except Exception as e:
                conversion_report.append({
                    "column": col_name,
                    "from_type": str(df[col_name].dtype),
                    "to_type": "failed",
                    "success": False,
                    "error": str(e)
                })
                logger.warning(f"Falha ao converter coluna {col_name}: {str(e)}")
        
        return df_converted, conversion_report
    
    def _handle_missing_values(self, df: pd.DataFrame, dataset_info: DatasetInfo) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Trata valores ausentes no DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame a ser processado.
            dataset_info (DatasetInfo): Informações sobre o dataset.
            
        Returns:
            tuple: DataFrame com valores ausentes tratados e relatório do tratamento.
        """
        missing_report = {}
        df_processed = df.copy()
        
        # Colunas com valores ausentes
        columns_with_missing = [col.name for col in dataset_info.columns 
                               if col.missing_percentage > 0]
        
        if not columns_with_missing:
            return df, {"columns_processed": 0, "method": "none"}
        
        missing_report["columns_with_missing"] = {}
        
        for col_name in columns_with_missing:
            col_info = next(col for col in dataset_info.columns if col.name == col_name)
            missing_count = df[col_name].isna().sum()
            missing_percentage = col_info.missing_percentage
            
            # Registra informações sobre valores ausentes
            missing_report["columns_with_missing"][col_name] = {
                "missing_count": int(missing_count),
                "missing_percentage": round(missing_percentage, 2)
            }
            
            # Define a estratégia de imputação com base no tipo de dados
            if col_info.data_type == DataType.NUMERIC:
                # Usa mediana para valores numéricos (mais robusta a outliers)
                imputer = SimpleImputer(strategy='median')
                df_processed[col_name] = imputer.fit_transform(df[[col_name]])
                missing_report["columns_with_missing"][col_name]["method"] = "median_imputation"
                logger.info(f"Valores ausentes na coluna {col_name} preenchidos com mediana")
                
            elif col_info.data_type == DataType.CATEGORICAL:
                # Usa o valor mais frequente para categóricos
                imputer = SimpleImputer(strategy='most_frequent')
                df_processed[col_name] = imputer.fit_transform(df[[col_name]])
                missing_report["columns_with_missing"][col_name]["method"] = "most_frequent_imputation"
                logger.info(f"Valores ausentes na coluna {col_name} preenchidos com valor mais frequente")
                
            elif col_info.data_type == DataType.DATETIME:
                # Para timestamps, usamos a média se possível, ou o valor anterior
                try:
                    df_processed[col_name] = df[col_name].fillna(method='ffill')
                    if df_processed[col_name].isna().sum() > 0:
                        df_processed[col_name] = df_processed[col_name].fillna(method='bfill')
                    missing_report["columns_with_missing"][col_name]["method"] = "time_interpolation"
                    logger.info(f"Valores ausentes na coluna {col_name} preenchidos com interpolação temporal")
                except:
                    df_processed[col_name] = df[col_name].fillna(df[col_name].mode()[0])
                    missing_report["columns_with_missing"][col_name]["method"] = "mode_imputation"
                    logger.info(f"Valores ausentes na coluna {col_name} preenchidos com moda")
            
            elif col_info.data_type == DataType.TEXT:
                # Para texto, substituímos por uma string vazia ou 'desconhecido'
                df_processed[col_name] = df[col_name].fillna("desconhecido")
                missing_report["columns_with_missing"][col_name]["method"] = "constant_imputation"
                logger.info(f"Valores ausentes na coluna {col_name} preenchidos com 'desconhecido'")
                
            else:
                # Para outros tipos, usamos o valor mais frequente
                df_processed[col_name] = df[col_name].fillna(df[col_name].mode()[0] if not df[col_name].mode().empty else "missing")
                missing_report["columns_with_missing"][col_name]["method"] = "mode_imputation"
                logger.info(f"Valores ausentes na coluna {col_name} preenchidos com moda")
                
            # Adiciona uma coluna indicadora de valores ausentes se houver muitos valores ausentes
            if missing_percentage > 10:
                indicator_name = f"{col_name}_missing_indicator"
                df_processed[indicator_name] = df[col_name].isna().astype(int)
                missing_report["columns_with_missing"][col_name]["indicator_added"] = indicator_name
                logger.info(f"Adicionado indicador de valores ausentes para coluna {col_name}")
        
        missing_report["total_imputed_values"] = sum(info["missing_count"] 
                                                  for _, info in missing_report["columns_with_missing"].items())
        missing_report["columns_processed"] = len(missing_report["columns_with_missing"])
        
        return df_processed, missing_report
    
    def _handle_outliers(self, df: pd.DataFrame, dataset_info: DatasetInfo) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detecta e trata outliers no DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame a ser processado.
            dataset_info (DatasetInfo): Informações sobre o dataset.
            
        Returns:
            tuple: DataFrame com outliers tratados e relatório do tratamento.
        """
        outlier_report = {}
        df_processed = df.copy()
        
        # Apenas colunas numéricas (exceto target e tempo) são consideradas para tratamento de outliers
        numeric_columns = [col.name for col in dataset_info.columns 
                          if col.data_type == DataType.NUMERIC
                          and not col.is_target
                          and not col.is_time_column]
        
        if not numeric_columns:
            return df, {"columns_processed": 0, "method": "none"}
        
        outlier_report["columns_with_outliers"] = {}
        
        for col_name in numeric_columns:
            # Verificar se a coluna é categórica e convertê-la para numérica
            if pd.api.types.is_categorical_dtype(df[col_name]):
                try:
                    # Tenta converter para numérico apenas para o cálculo (não modifica o DataFrame)
                    temp_series = df[col_name].astype(float)
                except (ValueError, TypeError):
                    # Se não conseguir converter, pule esta coluna
                    continue
            else:
                temp_series = df[col_name]
            
            # Calcula os limites usando o método IQR (Intervalo Interquartil)
            try:
                Q1 = temp_series.quantile(0.25)
                Q3 = temp_series.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identifica outliers
                outliers = df[(temp_series < lower_bound) | (temp_series > upper_bound)][col_name]
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / len(df)) * 100
                
                if outlier_count > 0:
                    outlier_report["columns_with_outliers"][col_name] = {
                        "outlier_count": outlier_count,
                        "outlier_percentage": round(outlier_percentage, 2),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound)
                    }
                    
                    # Cria indicador de outlier
                    indicator_name = f"{col_name}_outlier_indicator"
                    df_processed[indicator_name] = ((temp_series < lower_bound) | 
                                                 (temp_series > upper_bound)).astype(int)
                    
                    # Trata outliers com capping apenas se a coluna não for categórica
                    if not pd.api.types.is_categorical_dtype(df[col_name]):
                        df_processed[col_name] = np.where(
                            temp_series < lower_bound,
                            lower_bound,
                            np.where(
                                temp_series > upper_bound,
                                upper_bound,
                                df[col_name]
                            )
                        )
                    
                    outlier_report["columns_with_outliers"][col_name]["method"] = "capping"
                    outlier_report["columns_with_outliers"][col_name]["indicator_added"] = indicator_name
                    logger.info(f"Tratados {outlier_count} outliers na coluna {col_name} usando capping")
            except Exception as e:
                logger.warning(f"Erro ao tratar outliers na coluna {col_name}: {str(e)}")
                continue
        
        outlier_report["total_outliers"] = sum(info["outlier_count"] 
                                            for _, info in outlier_report["columns_with_outliers"].items())
        outlier_report["columns_processed"] = len(outlier_report["columns_with_outliers"])
        
        return df_processed, outlier_report
    
    def _normalize_data(self, df: pd.DataFrame, dataset_info: DatasetInfo) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Normaliza as colunas numéricas do DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame a ser processado.
            dataset_info (DatasetInfo): Informações sobre o dataset.
            
        Returns:
            tuple: DataFrame com valores normalizados e relatório da normalização.
        """
        normalization_report = {}
        df_processed = df.copy()
        
        # Apenas colunas numéricas (exceto target e tempo) são normalizadas
        numeric_columns = [col.name for col in dataset_info.columns 
                          if col.data_type == DataType.NUMERIC
                          and not col.is_target
                          and not col.is_time_column]
        
        # Filtra apenas colunas que não são categóricas
        numeric_columns = [col for col in numeric_columns if not pd.api.types.is_categorical_dtype(df[col])]
        
        if not numeric_columns:
            return df, {"columns_processed": 0, "method": "none"}
        
        # Decide entre StandardScaler ou MinMaxScaler com base no problema
        if dataset_info.problem_type in [ProblemType.REGRESSION, ProblemType.TIME_SERIES]:
            # Para regressão e séries temporais, geralmente StandardScaler é melhor
            scaler = StandardScaler()
            method = "standardization"
        else:
            # Para classificação, MinMaxScaler geralmente é melhor
            scaler = MinMaxScaler()
            method = "min_max_scaling"
        
        # Aplica o scaler às colunas numéricas
        df_processed[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        normalization_report["method"] = method
        normalization_report["columns_processed"] = len(numeric_columns)
        normalization_report["columns"] = numeric_columns
        
        logger.info(f"Normalizadas {len(numeric_columns)} colunas numéricas usando {method}")
        
        return df_processed, normalization_report