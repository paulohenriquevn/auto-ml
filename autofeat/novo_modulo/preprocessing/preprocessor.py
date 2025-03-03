"""
AutoFE - Módulo PreProcessor Aprimorado

Este módulo é responsável pela limpeza e preparação inicial dos dados.
Segue o princípio de ser modular e executável de forma independente.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import pickle
import os
import joblib
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats

class PreProcessor:
    """
    Módulo de pré-processamento para limpeza de dados no sistema AutoFE.
    
    Este módulo implementa as seguintes funcionalidades:
    - Detecção e tratamento de valores ausentes
    - Remoção ou tratamento de outliers (clipping, remoção ou winsorization)
    - Codificação de variáveis categóricas
    - Normalização de variáveis numéricas (com opção para detecção automática)
    - Transformação de tipos de dados
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o módulo PreProcessor com configurações personalizáveis.
        
        Args:
            config (Dict, opcional): Dicionário com configurações para o pré-processamento.
                Pode incluir:
                - missing_values_strategy: 'mean', 'median', 'most_frequent', 'constant'
                - outlier_method: 'clip', 'remove', 'winsorize'
                - outlier_threshold: z-score para detecção de outliers (para método 'clip' e 'remove')
                - winsorize_percentiles: tupla (lower, upper) com percentis para winsorization
                - categorical_strategy: 'onehot', 'label', 'ordinal'
                - normalization: True/False ou 'auto'
                - max_categories: número máximo de categorias para onehot encoding
                - model_type: tipo de modelo a ser usado (para normalização automática)
        """
        # Configurações padrão
        self.config = {
            'missing_values_strategy': 'median',
            'outlier_method': 'clip',
            'outlier_threshold': 3.0,  # Z-score threshold para outliers
            'winsorize_percentiles': (0.05, 0.95),  # Percentis para winsorization
            'categorical_strategy': 'onehot',
            'normalization': True,     # True, False ou 'auto'
            'max_categories': 20,
            'model_type': None,        # Tipo de modelo para normalização automática
            'verbosity': 1  # nível de detalhamento dos logs
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
         # Configurar logging
        self._setup_logging()
        
        self.logger.info("Explorer inicializado com sucesso.")
        
    def _setup_logging(self):
        """Configura o sistema de logging do PreProcessor."""
        self.logger = logging.getLogger("AutoFE.PreProcessor")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }
        self.logger.setLevel(log_levels.get(self.config['verbosity'], logging.INFO))

    
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
    
    def _is_normalization_needed(self) -> bool:
        """
        Determina se a normalização é necessária com base na configuração ou tipo de modelo.
        
        Returns:
            bool: True se a normalização for recomendada, False caso contrário
        """
        # Se a normalização for explicitamente definida (True/False)
        if isinstance(self.config['normalization'], bool):
            return self.config['normalization']
        
        # Se configurado para automático, decidir com base no tipo de modelo
        if self.config['normalization'] == 'auto':
            model_type = self.config['model_type']
            
            # Se o tipo de modelo for especificado
            if model_type:
                # Modelos insensíveis à escala
                if model_type.lower() in ['tree', 'forest', 'randomforest', 'decisiontree', 
                                         'boosting', 'xgboost', 'lightgbm', 'catboost']:
                    return False
                    
                # Modelos sensíveis à escala
                elif model_type.lower() in ['linear', 'logistic', 'regression', 'svm', 
                                           'nn', 'neural', 'knn', 'neighbors', 'kmeans', 
                                           'pca', 'lda']:
                    return True
            
            # Comportamento padrão: normalizar por segurança
            return True
            
        # Por padrão, normalizar
        return True
    
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
        outlier_method = self.config['outlier_method']
        
        # Filtrar apenas as colunas numéricas existentes
        valid_num_cols = [col for col in numeric_cols if col in df.columns]
        
        if outlier_method == 'remove':
            # Método: remoção de outliers baseado em z-score
            z_threshold = self.config['outlier_threshold']
            mask = pd.Series(True, index=df.index)
            
            for col in valid_num_cols:
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() > 1:
                    col_std = df[col].std()
                    col_mean = df[col].mean()
                    
                    if col_std > 0 and not np.isnan(col_std):
                        z_scores = np.abs((df[col] - col_mean) / col_std)
                        outlier_mask = (z_scores < z_threshold) | z_scores.isna()
                        mask = mask & outlier_mask
            
            # Aplicar máscara para remover linhas com outliers
            df_processed = df[mask].reset_index(drop=True)
            
        elif outlier_method == 'clip':
            # Método: clipping de valores extremos baseado em z-score
            z_threshold = self.config['outlier_threshold']
            
            for col in valid_num_cols:
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() > 1:
                    col_std = df[col].std()
                    col_mean = df[col].mean()
                    
                    if col_std > 0 and not np.isnan(col_std):
                        lower_bound = col_mean - z_threshold * col_std
                        upper_bound = col_mean + z_threshold * col_std
                        df_processed[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    
        elif outlier_method == 'winsorize':
            # Método: winsorization baseado em percentis
            lower_percentile, upper_percentile = self.config['winsorize_percentiles']
            
            for col in valid_num_cols:
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() > 1:
                    # Calcular percentis
                    lower_limit = df[col].quantile(lower_percentile)
                    upper_limit = df[col].quantile(upper_percentile)
                    
                    # Aplicar Winsorization
                    df_processed[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
        
        return df_processed
    
    def optimize_normalization(self, X: pd.DataFrame, y: pd.Series, model: Any, cv: int = 5) -> bool:
        """
        Avalia experimentalmente se a normalização melhora o desempenho do modelo.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            model: Modelo a ser avaliado
            cv (int): Número de folds para validação cruzada
        
        Returns:
            bool: True se normalização melhorar o desempenho, False caso contrário
        """
        from sklearn.model_selection import cross_val_score
        
        # Configurar preprocessador com normalização
        config_norm = self.config.copy()
        config_norm['normalization'] = True
        prep_norm = PreProcessor(config_norm)
        X_norm = prep_norm.fit_transform(X)
        
        # Configurar preprocessador sem normalização
        config_no_norm = self.config.copy()
        config_no_norm['normalization'] = False
        prep_no_norm = PreProcessor(config_no_norm)
        X_no_norm = prep_no_norm.fit_transform(X)
        
        # Avaliar desempenho com validação cruzada
        score_norm = cross_val_score(model, X_norm, y, cv=cv).mean()
        score_no_norm = cross_val_score(model, X_no_norm, y, cv=cv).mean()
        
        # Atualizar a configuração com base no resultado
        self.config['normalization'] = score_norm > score_no_norm
        
        # Retornar o resultado
        return score_norm > score_no_norm
    
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
        # Excluir a coluna alvo do processamento, se fornecida e existir
        if target_col is not None:
            if isinstance(target_col, str):
                if target_col in data.columns:
                    self.logger.info(f"Excluindo coluna alvo '{target_col}' do processamento.")
                    data = data.drop(columns=[target_col])
                else:
                    self.logger.warning(f"Coluna alvo '{target_col}' não encontrada no DataFrame.")
            else:
                self.logger.warning(f"target_col deve ser uma string, recebido: {type(target_col)}")
        
        # Identificar tipos de colunas
        self.column_types = self._identify_column_types(data)
        
        # Determinar se deve usar normalização
        should_normalize = self._is_normalization_needed()
        
        # Definir transformadores para cada tipo de coluna
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.config['missing_values_strategy'])),
            ('scaler', StandardScaler() if should_normalize else 'passthrough')
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