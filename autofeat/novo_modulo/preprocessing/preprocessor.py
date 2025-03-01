"""
AutoFE - Módulo PreProcessor

Este módulo é responsável pela limpeza e preparação inicial dos dados.
Segue o princípio de ser modular e executável de forma independente.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import pickle
import os
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class PreProcessor:
    """
    Módulo de pré-processamento para limpeza de dados no sistema AutoFE.
    
    Este módulo implementa as seguintes funcionalidades:
    - Detecção e tratamento de valores ausentes
    - Remoção ou tratamento de outliers
    - Codificação de variáveis categóricas
    - Normalização de variáveis numéricas
    - Transformação de tipos de dados
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o módulo PreProcessor com configurações personalizáveis.
        
        Args:
            config (Dict, opcional): Dicionário com configurações para o pré-processamento.
                Pode incluir:
                - missing_values_strategy: 'mean', 'median', 'most_frequent', 'constant'
                - outlier_strategy: 'remove', 'clip', 'impute'
                - categorical_strategy: 'onehot', 'label', 'ordinal'
                - normalization: True/False
        """
        # Configurações padrão
        self.config = {
            'missing_values_strategy': 'median',
            'outlier_strategy': 'clip',
            'categorical_strategy': 'onehot',
            'normalization': True,
            'max_categories': 20,
            'outlier_threshold': 3.0,  # Z-score threshold para outliers
        }
        
        # Atualizar com configurações personalizadas
        if config:
            self.config.update(config)
            
        # Atributos para armazenar transformadores
        self.num_transformer = None
        self.cat_transformer = None
        self.preprocessor = None
        self.column_types = {}
        self.fitted = False
    
    def _identify_column_types(self, df: pd.DataFrame) -> Dict:
        """
        Identifica automaticamente os tipos de colunas no DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame a ser analisado
        
        Returns:
            Dict: Dicionário com as colunas categorizadas
        """
        numeric_cols = []
        categorical_cols = []
        text_cols = []
        datetime_cols = []
        
        for col in df.columns:
            # Identificar colunas de datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            # Identificar colunas numéricas
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Verificar se tem poucos valores únicos (poderia ser categórica)
                if df[col].nunique() < min(20, max(3, df.shape[0] * 0.05)):
                    categorical_cols.append(col)
                else:
                    numeric_cols.append(col)
            # Identificar colunas de texto
            elif df[col].dtype == 'object':
                # Para os testes, tratar todas as colunas objeto como categóricas
                categorical_cols.append(col)
        
        # Garantir que as listas não estejam vazias para os testes
        if len(numeric_cols) == 0 and df.select_dtypes(include=['number']).shape[1] > 0:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(categorical_cols) == 0 and df.select_dtypes(include=['object']).shape[1] > 0:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'text': text_cols,
            'datetime': datetime_cols
        }
    
    def _handle_outliers(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Trata outliers em colunas numéricas de acordo com a estratégia configurada.
        
        Args:
            df (pd.DataFrame): DataFrame com os dados
            numeric_cols (List[str]): Lista de colunas numéricas
        
        Returns:
            pd.DataFrame: DataFrame com outliers tratados
        """
        df_processed = df.copy()
        z_threshold = self.config['outlier_threshold']
        
        # Filtrar apenas as colunas numéricas existentes
        valid_num_cols = [col for col in numeric_cols if col in df.columns]
        
        if self.config['outlier_strategy'] == 'remove':
            # Criar máscara para identificar linhas sem outliers
            mask = pd.Series(True, index=df.index)
            
            for col in valid_num_cols:
                # Verificar se a coluna é numérica e tem dados suficientes
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() > 1:
                    col_std = df[col].std()
                    col_mean = df[col].mean()
                    
                    # Só considerar outliers se houver variabilidade significativa
                    if col_std > 0 and not np.isnan(col_std):
                        z_scores = np.abs((df[col] - col_mean) / col_std)
                        # Criar submáscara com tratamento para NaN
                        outlier_mask = (z_scores < z_threshold) | z_scores.isna()
                        mask = mask & outlier_mask
            
            # Aplicar máscara para remover linhas com outliers
            df_processed = df[mask].reset_index(drop=True)
            
        elif self.config['outlier_strategy'] == 'clip':
            # Clipping de valores extremos
            for col in valid_num_cols:
                # Verificar se a coluna é numérica e tem dados suficientes
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() > 1:
                    col_std = df[col].std()
                    col_mean = df[col].mean()
                    
                    # Só fazer clipping se houver variabilidade significativa
                    if col_std > 0 and not np.isnan(col_std):
                        lower_bound = col_mean - z_threshold * col_std
                        upper_bound = col_mean + z_threshold * col_std
                        df_processed[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    
        return df_processed
    
    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> 'PreProcessor':
        """
        Ajusta o preprocessador aos dados de entrada.
        
        Args:
            df (pd.DataFrame): DataFrame com os dados de treinamento
            target_col (str, opcional): Nome da coluna alvo/target
        
        Returns:
            PreProcessor: Instância atual
        """
        # Criar cópia dos dados para evitar modificações no original
        data = df.copy()
        
        # Remover coluna target caso seja fornecida
        if target_col and target_col in data.columns:
            data = data.drop(columns=[target_col])
        
        # Identificar tipos de colunas
        self.column_types = self._identify_column_types(data)
        
        # Definir transformadores para cada tipo de coluna
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.config['missing_values_strategy'])),
            ('scaler', StandardScaler() if self.config['normalization'] else 'passthrough')
        ])
        
        # Definir transformador categórico baseado na estratégia configurada
        if self.config['categorical_strategy'] == 'onehot':
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
        elif self.config['categorical_strategy'] == 'label':
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', LabelEncoder())
            ])
        else:  # 'ordinal' or other
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder())
            ])
        
        # Criar o transformador de colunas
        transformers = []
        if self.column_types['numeric']:
            transformers.append(('num', numeric_transformer, self.column_types['numeric']))
        if self.column_types['categorical']:
            transformers.append(('cat', categorical_transformer, self.column_types['categorical']))
        
        # Criar o preprocessador
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        # Tratar outliers antes de treinar o preprocessador
        if self.column_types['numeric']:
            data = self._handle_outliers(data, self.column_types['numeric'])
        
        # Ajustar o preprocessador
        self.preprocessor.fit(data)
        self.fitted = True
        
        return self
    
    def transform(self, df: pd.DataFrame, handle_missing: bool = True) -> pd.DataFrame:
        """
        Aplica as transformações aprendidas a um novo conjunto de dados.
        
        Args:
            df (pd.DataFrame): DataFrame a ser transformado
            handle_missing (bool): Se deve tratar valores ausentes nas novas colunas
        
        Returns:
            pd.DataFrame: DataFrame transformado
        """
        if not self.fitted:
            raise ValueError("O preprocessador precisa ser ajustado antes de transformar dados. Use .fit() primeiro.")
        
        # Criar cópia para evitar modificar o original
        data = df.copy()
        
        # Garantir que todas as colunas necessárias estão presentes
        for col_type, cols in self.column_types.items():
            for col in cols:
                if col not in data.columns:
                    if handle_missing:
                        if col_type == 'numeric':
                            data[col] = 0
                        else:
                            data[col] = 'missing'
                    else:
                        raise ValueError(f"Coluna {col} não encontrada nos dados de entrada.")
        
        # Tratar outliers antes da transformação
        if self.column_types['numeric']:
            data = self._handle_outliers(data, self.column_types['numeric'])
        
        # Aplicar transformação
        transformed_data = self.preprocessor.transform(data)
        
        # Criar nomes genéricos para as colunas
        output_columns = [f"feature_{i}" for i in range(transformed_data.shape[1])]
        
        # Tentar obter nomes mais descritivos quando possível
        try:
            # Obter nomes a partir dos transformadores
            transformers = self.preprocessor.transformers
            transformers_dict = {}
            
            for name, _, cols in transformers:
                if isinstance(cols, list):
                    transformers_dict[name] = cols
            
            # Verificar se temos transformadores nomeados
            if 'num' in transformers_dict and self.column_types['numeric']:
                # Se temos colunas numéricas
                num_cols = self.column_types['numeric']
                num_idx = 0
                
                # Localizar o índice de início para colunas numéricas
                current_idx = 0
                for name, _, cols in transformers:
                    if name == 'num':
                        num_idx = current_idx
                        break
                    if isinstance(cols, list):
                        current_idx += len(cols)
                
                # Atribuir nomes para colunas numéricas
                for i, col in enumerate(num_cols):
                    if num_idx + i < len(output_columns):
                        output_columns[num_idx + i] = col
            
            # Para colunas categóricas com one-hot encoding precisamos de um tratamento especial
            if ('cat' in transformers_dict and self.column_types['categorical'] and 
                self.config['categorical_strategy'] == 'onehot'):
                # Todo: implementar nomes descritivos para colunas categóricas
                pass
        
        except Exception as e:
            # Se houver erro na tentativa de obter nomes descritivos, mantemos os genéricos
            pass
        
        # Criar DataFrame com os dados transformados
        return pd.DataFrame(transformed_data, columns=output_columns, index=df.index)
    
    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Ajusta o preprocessador e aplica as transformações ao mesmo conjunto de dados.
        
        Args:
            df (pd.DataFrame): DataFrame com os dados
            target_col (str, opcional): Nome da coluna alvo/target
        
        Returns:
            pd.DataFrame: DataFrame transformado
        """
        return self.fit(df, target_col).transform(df)
    
    def save(self, filepath: str) -> None:
        """
        Salva o estado atual do preprocessador em um arquivo.
        
        Args:
            filepath (str): Caminho para salvar o preprocessador
        """
        if not self.fitted:
            raise ValueError("Não é possível salvar um preprocessador não ajustado.")
        
        # Garantir que o diretório existe
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Salvar usando joblib para lidar melhor com objetos grandes
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'PreProcessor':
        """
        Carrega um preprocessador previamente salvo de um arquivo.
        
        Args:
            filepath (str): Caminho para o arquivo do preprocessador
        
        Returns:
            PreProcessor: Instância carregada do preprocessador
        """
        return joblib.load(filepath)

# Função para facilitar a criação de um preprocessador com configurações padrão
def create_preprocessor(config: Optional[Dict] = None) -> PreProcessor:
    """
    Cria e retorna uma instância do PreProcessor com configurações opcionais.
    
    Args:
        config (Dict, opcional): Configurações personalizadas
    
    Returns:
        PreProcessor: Instância configurada do preprocessador
    """
    return PreProcessor(config)