"""
Módulo Datasets para AutoFE

Este módulo implementa funcionalidades para carregamento, divisão e
gerenciamento de datasets no sistema AutoFE, semelhante ao módulo
datasets do sklearn.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Callable
from sklearn.model_selection import train_test_split
import joblib
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Datasets')

class Dataset:
    """
    Classe para gerenciamento de datasets no AutoFE.
    
    Attributes:
        X (pd.DataFrame): Features do dataset
        y (pd.Series): Variável alvo
        target_name (str): Nome da coluna alvo
        X_train (pd.DataFrame): Features de treino
        X_test (pd.DataFrame): Features de teste
        X_val (pd.DataFrame): Features de validação
        y_train (pd.Series): Variável alvo de treino
        y_test (pd.Series): Variável alvo de teste
        y_val (pd.Series): Variável alvo de validação
        metadata (Dict): Metadados sobre o dataset
        preprocessor: Objeto preprocessador associado ao dataset
    """
    
    def __init__(
        self, 
        X: pd.DataFrame, 
        y: pd.Series = None, 
        target_name: str = None,
        metadata: Dict = None
    ):
        """
        Inicializa um novo Dataset.
        
        Args:
            X: DataFrame com as features
            y: Series com a variável alvo (opcional)
            target_name: Nome da coluna alvo (opcional)
            metadata: Informações adicionais sobre o dataset
        """
        self.X = X
        self.y = y
        self.target_name = target_name or (y.name if y is not None else None)
        self.metadata = metadata or {}
        
        # Conjuntos divididos (inicialmente None)
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None 
        self.y_val = None
        
        # Preprocessador associado (opcional)
        self.preprocessor = None
        
        logger.info(f"Dataset criado com {X.shape[0]} amostras e {X.shape[1]} features")
        
    def split_data(
        self, 
        test_size: float = 0.2, 
        val_size: float = 0.0, 
        random_state: int = 42,
        stratify: bool = False
    ) -> None:
        """
        Divide o dataset em conjuntos de treino, teste e validação.
        
        Args:
            test_size: Proporção do conjunto de teste
            val_size: Proporção do conjunto de validação
            random_state: Semente para reprodutibilidade
            stratify: Se deve estratificar pela variável alvo
        """
        if self.y is None and stratify:
            logger.warning("Não é possível estratificar sem variável alvo. Desabilitando stratify.")
            stratify = False
            
        stratify_data = self.y if stratify else None
            
        if val_size > 0:
            # Primeiro separar teste
            X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                self.X, self.y, 
                test_size=test_size, 
                random_state=random_state,
                stratify=stratify_data
            )
            
            # Depois separar validação do restante
            val_adjusted_size = val_size / (1 - test_size)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_adjusted_size,
                random_state=random_state,
                stratify=y_temp if stratify else None
            )
            
            logger.info(f"Dataset dividido em treino ({self.X_train.shape[0]} amostras), "
                       f"validação ({self.X_val.shape[0]} amostras) e "
                       f"teste ({self.X_test.shape[0]} amostras)")
        else:
            # Apenas treino e teste
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, 
                test_size=test_size, 
                random_state=random_state,
                stratify=stratify_data
            )
            
            self.X_val = None
            self.y_val = None
            
            logger.info(f"Dataset dividido em treino ({self.X_train.shape[0]} amostras) e "
                       f"teste ({self.X_test.shape[0]} amostras)")
            
    def get_train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Retorna os dados de treino.
        
        Returns:
            Tupla contendo (X_train, y_train)
        """
        if self.X_train is None:
            raise ValueError("Os dados ainda não foram divididos. Use split_data() primeiro.")
            
        return self.X_train, self.y_train
        
    def get_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Retorna os dados de teste.
        
        Returns:
            Tupla contendo (X_test, y_test)
        """
        if self.X_test is None:
            raise ValueError("Os dados ainda não foram divididos. Use split_data() primeiro.")
            
        return self.X_test, self.y_test
        
    def get_val_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Retorna os dados de validação.
        
        Returns:
            Tupla contendo (X_val, y_val)
        """
        if self.X_val is None:
            raise ValueError("Não há dados de validação. Use split_data() com val_size > 0.")
            
        return self.X_val, self.y_val
        
    def apply_preprocessor(self, preprocessor) -> None:
        """
        Aplica um preprocessador aos dados e atualiza os conjuntos.
        
        Args:
            preprocessor: Objeto preprocessador com métodos fit_transform e transform
        """
        if self.X_train is None:
            raise ValueError("Os dados devem ser divididos antes de aplicar o preprocessador.")
            
        # Salvar o preprocessador para uso futuro
        self.preprocessor = preprocessor
            
        # Aplicar aos dados de treino (fit_transform)
        self.X_train = preprocessor.fit_transform(self.X_train)
        
        # Aplicar aos dados de teste (apenas transform)
        self.X_test = preprocessor.transform(self.X_test)
        
        # Aplicar aos dados de validação, se existirem
        if self.X_val is not None:
            self.X_val = preprocessor.transform(self.X_val)
            
        logger.info("Preprocessador aplicado aos dados com sucesso")
    
    def prepare_new_data(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara novos dados para inferência aplicando o mesmo preprocessamento.
        
        Args:
            X_new: DataFrame com novos dados
            
        Returns:
            DataFrame com dados preparados para inferência
        """
        if self.preprocessor is None:
            raise ValueError("Nenhum preprocessador foi aplicado ao dataset.")
            
        return self.preprocessor.transform(X_new)
        
    def save(self, path: str) -> None:
        """
        Salva o dataset completo para uso posterior.
        
        Args:
            path: Caminho onde o dataset será salvo
        """
        # Verificar se o diretório existe
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Salvar usando joblib
        joblib.dump(self, path)
        logger.info(f"Dataset salvo em {path}")
        
    @classmethod
    def load(cls, path: str) -> 'Dataset':
        """
        Carrega um dataset previamente salvo.
        
        Args:
            path: Caminho onde o dataset foi salvo
            
        Returns:
            Instância carregada do Dataset
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
            
        # Carregar usando joblib
        dataset = joblib.load(path)
        logger.info(f"Dataset carregado de {path}")
        
        return dataset
        
    def describe(self) -> Dict:
        """
        Retorna um resumo descritivo do dataset.
        
        Returns:
            Dicionário com informações sobre o dataset
        """
        info = {
            'total_samples': self.X.shape[0],
            'num_features': self.X.shape[1],
            'feature_names': self.X.columns.tolist(),
            'has_target': self.y is not None,
            'is_split': self.X_train is not None
        }
        
        if info['is_split']:
            info['train_samples'] = self.X_train.shape[0]
            info['test_samples'] = self.X_test.shape[0]
            
            if self.X_val is not None:
                info['val_samples'] = self.X_val.shape[0]
                
        if info['has_target']:
            info['target_name'] = self.target_name
            
            if hasattr(self.y, 'nunique'):
                info['unique_targets'] = self.y.nunique()
                
                # Detectar tipo de problema
                if info['unique_targets'] == 2:
                    info['problem_type'] = 'binary_classification'
                elif info['unique_targets'] > 2 and info['unique_targets'] < 50:
                    info['problem_type'] = 'multiclass_classification'
                else:
                    info['problem_type'] = 'regression'
                    
        return info

# Funções para carregamento de datasets comuns
def load_csv(
    filepath: str, 
    target_column: str = None, 
    sep: str = ',',
    encoding: str = 'utf-8',
    **kwargs
) -> Dataset:
    """
    Carrega um dataset a partir de um arquivo CSV.
    
    Args:
        filepath: Caminho para o arquivo CSV
        target_column: Nome da coluna alvo (opcional)
        sep: Separador usado no CSV
        encoding: Codificação do arquivo
        **kwargs: Argumentos adicionais para pd.read_csv
        
    Returns:
        Instância de Dataset
    """
    try:
        # Carregar o CSV
        data = pd.read_csv(filepath, sep=sep, encoding=encoding, **kwargs)
        
        # Separar features e alvo, se especificado
        if target_column is not None and target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
        else:
            X = data
            y = None
            
        # Criar metadados
        metadata = {
            'source': filepath,
            'source_type': 'csv',
            'original_shape': data.shape
        }
            
        return Dataset(X, y, target_column, metadata)
        
    except Exception as e:
        logger.error(f"Erro ao carregar CSV: {e}")
        raise

def load_excel(
    filepath: str, 
    target_column: str = None,
    sheet_name: Union[str, int] = 0,
    **kwargs
) -> Dataset:
    """
    Carrega um dataset a partir de um arquivo Excel.
    
    Args:
        filepath: Caminho para o arquivo Excel
        target_column: Nome da coluna alvo (opcional)
        sheet_name: Nome ou índice da planilha
        **kwargs: Argumentos adicionais para pd.read_excel
        
    Returns:
        Instância de Dataset
    """
    try:
        # Carregar o Excel
        data = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
        
        # Separar features e alvo, se especificado
        if target_column is not None and target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
        else:
            X = data
            y = None
            
        # Criar metadados
        metadata = {
            'source': filepath,
            'source_type': 'excel',
            'sheet_name': sheet_name,
            'original_shape': data.shape
        }
            
        return Dataset(X, y, target_column, metadata)
        
    except Exception as e:
        logger.error(f"Erro ao carregar Excel: {e}")
        raise

def load_from_dataframe(
    df: pd.DataFrame,
    target_column: str = None,
    metadata: Dict = None
) -> Dataset:
    """
    Cria um Dataset a partir de um pandas DataFrame existente.
    
    Args:
        df: DataFrame com os dados
        target_column: Nome da coluna alvo (opcional)
        metadata: Metadados adicionais (opcional)
        
    Returns:
        Instância de Dataset
    """
    try:
        # Separar features e alvo, se especificado
        if target_column is not None and target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            X = df
            y = None
            
        # Criar metadados básicos
        meta = metadata or {}
        meta.update({
            'source_type': 'dataframe',
            'original_shape': df.shape
        })
            
        return Dataset(X, y, target_column, meta)
        
    except Exception as e:
        logger.error(f"Erro ao criar Dataset a partir do DataFrame: {e}")
        raise