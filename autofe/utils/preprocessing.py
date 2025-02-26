import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy import stats

class DatasetPreprocessor:
    """
    Uma classe abrangente para pré-processamento de datasets com suporte a múltiplos tipos de dados.
    
    Características principais:
    - Detecção automática de tipos de dados
    - Tratamento flexível de valores ausentes
    - Múltiplas estratégias de codificação
    - Normalização adaptativa
    - Análise de multicolinearidade
    - Suporte a diferentes tipos de datasets
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o pré-processador com configurações personalizáveis.
        
        Args:
            config: Dicionário de configurações personalizadas
        """
        # Configurações padrão que podem ser sobrescritas
        self.config = {
            # Limiares de limpeza e processamento
            'missing_threshold': 0.3,  # Porcentagem máxima de valores ausentes
            'low_variance_threshold': 0.01,  # Variância mínima para manter feature
            'correlation_threshold': 0.85,  # Limiar de multicolinearidade
            
            # Estratégias de codificação
            'categorical_encoding': 'onehot',  # Métodos: 'onehot', 'label'
            
            # Estratégias de normalização
            'normalization_method': 'standard',  # Métodos: 'standard', 'minmax'
            
            # Estratégias de imputação
            'numeric_imputation': 'median',  # Métodos: 'median', 'mean', 'constant'
            'categorical_imputation': 'mode',  # Métodos: 'mode', 'constant'
        }
        
        # Atualizar configurações personalizadas
        if config:
            self.config.update(config)
        
        # Armazenar metadados do processamento
        self.metadata = {
            'original_columns': None,
            'dropped_columns': [],
            'encoded_columns': {},
            'correlation_analysis': [],
            'normalization_details': {}
        }
    
    def detect_column_type(self, column: pd.Series) -> str:
        """
        Detecta o tipo semântico de uma coluna.
        
        Args:
            column: Série do pandas para análise
            
        Returns:
            Tipo da coluna ('numeric', 'categorical', 'datetime', 'text')
        """
        # Verificar tipo de dados do pandas
        if pd.api.types.is_numeric_dtype(column):
            return 'numeric'
        
        if pd.api.types.is_datetime64_any_dtype(column):
            return 'datetime'
        
        # Para objetos/strings, fazer análise mais detalhada
        if column.dtype == 'object':
            # Verificar se é data
            try:
                pd.to_datetime(column)
                return 'datetime'
            except:
                pass
            
            # Verificar se é texto ou categórica
            unique_ratio = len(column.unique()) / len(column)
            avg_length = column.astype(str).str.len().mean()
            
            if unique_ratio < 0.05 or len(column.unique()) < 10:
                return 'categorical'
            elif avg_length > 10:
                return 'text'
            else:
                return 'categorical'
        
        return 'unknown'
    
    def clean_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove colunas irrelevantes ou com muitos valores ausentes.
        
        Args:
            data: DataFrame original
            
        Returns:
            DataFrame limpo
        """
        data_cleaned = data.copy()
        
        # Registrar colunas originais
        self.metadata['original_columns'] = list(data.columns)
        
        # Remover colunas de índice
        columns_to_drop = [col for col in data.columns if 'Unnamed:' in str(col)]
        
        # Identificar colunas com muitos valores ausentes
        missing_ratio = data.isnull().mean()
        columns_to_drop.extend(
            missing_ratio[missing_ratio > self.config['missing_threshold']].index.tolist()
        )
        
        # Remover colunas de baixa variância para numéricas
        numeric_columns = data.select_dtypes(include=['number']).columns
        low_variance_columns = [
            col for col in numeric_columns 
            if data[col].var() < self.config['low_variance_threshold']
        ]
        columns_to_drop.extend(low_variance_columns)
        
        # Remover colunas únicas
        single_value_columns = [col for col in data.columns if data[col].nunique() <= 1]
        columns_to_drop.extend(single_value_columns)
        
        # Remover duplicatas e colunas
        columns_to_drop = list(set(columns_to_drop))
        
        # Registrar colunas removidas
        self.metadata['dropped_columns'] = columns_to_drop
        
        # Dropar colunas
        return data_cleaned.drop(columns=columns_to_drop)
    
    def extract_datetime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extrai features temporais de colunas de data.
        
        Args:
            data: DataFrame em processamento
            
        Returns:
            DataFrame com features temporais adicionadas
        """
        data_processed = data.copy()
        
        # Identificar colunas de data
        datetime_columns = [
            col for col in data.columns 
            if self.detect_column_type(data[col]) == 'datetime'
        ]
        
        for col in datetime_columns:
            try:
                # Converter para datetime
                date_series = pd.to_datetime(data[col])
                
                # Extrair features
                data_processed[f'{col}_year'] = date_series.dt.year
                data_processed[f'{col}_month'] = date_series.dt.month
                data_processed[f'{col}_day'] = date_series.dt.day
                data_processed[f'{col}_dayofweek'] = date_series.dt.dayofweek
                data_processed[f'{col}_quarter'] = date_series.dt.quarter
                
                # Adicionar features cíclicas para mês e dia da semana
                data_processed[f'{col}_month_sin'] = np.sin(2 * np.pi * date_series.dt.month / 12)
                data_processed[f'{col}_month_cos'] = np.cos(2 * np.pi * date_series.dt.month / 12)
                data_processed[f'{col}_dayofweek_sin'] = np.sin(2 * np.pi * date_series.dt.dayofweek / 7)
                data_processed[f'{col}_dayofweek_cos'] = np.cos(2 * np.pi * date_series.dt.dayofweek / 7)
                
                # Remover coluna original de data
                data_processed = data_processed.drop(columns=[col])
            except Exception as e:
                print(f"Erro ao processar coluna de data {col}: {e}")
        
        return data_processed
    
    def encode_categorical_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Codifica variáveis categóricas.
        
        Args:
            data: DataFrame em processamento
            
        Returns:
            DataFrame com variáveis categóricas codificadas
        """
        data_encoded = data.copy()
        
        # Identificar colunas categóricas
        categorical_columns = [
            col for col in data.columns 
            if self.detect_column_type(data[col]) == 'categorical'
        ]
        
        # Dicionário para armazenar informações de codificação
        encoding_info = {}
        
        for col in categorical_columns:
            try:
                if self.config['categorical_encoding'] == 'onehot':
                    # One-hot encoding
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    encoded = encoder.fit_transform(data[[col]])
                    encoded_cols = [f'{col}_{cat}' for cat in encoder.categories_[0]]
                    
                    # Adicionar colunas codificadas
                    for i, new_col in enumerate(encoded_cols):
                        data_encoded[new_col] = encoded[:, i]
                    
                    # Remover coluna original
                    data_encoded = data_encoded.drop(columns=[col])
                    
                    # Armazenar informações de codificação
                    encoding_info[col] = {
                        'method': 'onehot',
                        'categories': encoder.categories_[0].tolist()
                    }
                
                else:  # Label encoding
                    encoder = LabelEncoder()
                    data_encoded[col] = encoder.fit_transform(data[col].fillna('Unknown'))
                    
                    # Armazenar informações de codificação
                    encoding_info[col] = {
                        'method': 'label',
                        'classes': encoder.classes_.tolist()
                    }
            
            except Exception as e:
                print(f"Erro ao codificar coluna categórica {col}: {e}")
        
        # Armazenar metadados de codificação
        self.metadata['encoded_columns'] = encoding_info
        
        return data_encoded
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Trata valores ausentes em diferentes tipos de colunas.
        
        Args:
            data: DataFrame com valores ausentes
            
        Returns:
            DataFrame com valores ausentes tratados
        """
        data_imputed = data.copy()
        
        # Identificar tipos de colunas
        numeric_columns = data.select_dtypes(include=['number']).columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        
        # Imputação para colunas numéricas
        if self.config['numeric_imputation'] == 'median':
            numeric_imputer = SimpleImputer(strategy='median')
        elif self.config['numeric_imputation'] == 'mean':
            numeric_imputer = SimpleImputer(strategy='mean')
        else:
            numeric_imputer = SimpleImputer(strategy='constant', fill_value=0)
        
        # Aplicar imputação para numéricas
        if len(numeric_columns) > 0:
            data_imputed[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])
        
        # Imputação para colunas categóricas
        if self.config['categorical_imputation'] == 'mode':
            categorical_imputer = SimpleImputer(strategy='most_frequent')
        else:
            categorical_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        
        # Aplicar imputação para categóricas
        if len(categorical_columns) > 0:
            data_imputed[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])
        
        return data_imputed
    
    def check_multicollinearity(self, data: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """
        Identifica features altamente correlacionadas.
        
        Args:
            data: DataFrame processado
            
        Returns:
            Lista de tuplas com features correlacionadas
        """
        # Selecionar apenas colunas numéricas
        numeric_columns = data.select_dtypes(include=['number']).columns
        correlation_matrix = data[numeric_columns].corr().abs()
        
        # Encontrar pares de features correlacionadas
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                corr_value = correlation_matrix.iloc[i, j]
                if corr_value > self.config['correlation_threshold']:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i], 
                        correlation_matrix.columns[j], 
                        corr_value
                    ))
        
        # Armazenar resultados da análise de correlação
        self.metadata['correlation_analysis'] = high_corr_pairs
        
        return high_corr_pairs
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza dados numéricos.
        
        Args:
            data: DataFrame processado
            
        Returns:
            DataFrame normalizado
        """
        data_normalized = data.copy()
        numeric_columns = data.select_dtypes(include=['number']).columns
        
        # Escolher método de normalização
        if self.config['normalization_method'] == 'standard':
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(data[numeric_columns])
            
            # Armazenar detalhes da normalização
            self.metadata['normalization_details'] = {
                'method': 'standard',
                'mean': scaler.mean_,
                'scale': scaler.scale_
            }
        
        elif self.config['normalization_method'] == 'minmax':
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data[numeric_columns])
            
            # Armazenar detalhes da normalização
            self.metadata['normalization_details'] = {
                'method': 'minmax',
                'min': scaler.data_min_,
                'max': scaler.data_max_
            }
        
        # Substituir colunas normalizadas
        data_normalized[numeric_columns] = normalized_data
        
        return data_normalized
    
    def preprocess(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Pipeline completo de pré-processamento.
        
        Args:
            data: DataFrame original
            
        Returns:
            Dicionário com dados processados e metadados
        """
        # Limpar colunas
        data_cleaned = self.clean_columns(data)
        
        # Extrair features de data
        data_datetime = self.extract_datetime_features(data_cleaned)
        
        # Tratar valores ausentes
        data_imputed = self.handle_missing_values(data_datetime)
        
        # Codificar variáveis categóricas
        data_encoded = self.encode_categorical_variables(data_imputed)
        
        # Verificar multicolinearidade
        multicollinearity_results = self.check_multicollinearity(data_encoded)
        
        # Normalizar dados
        data_normalized = self.normalize_data(data_encoded)
        
        return {
            'processed_data': data_normalized,
            'metadata': self.metadata,
            'multicollinearity': multicollinearity_results
        }

def process_dataset(
    data: pd.DataFrame, 
    target_column: Optional[str] = None,
    custom_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Processa um dataset com configurações flexíveis.
    
    Args:
        data: DataFrame original
        target_column: Coluna alvo (opcional)
        custom_config: Configurações personalizadas de pré-processamento
        
    Returns:
        Dicionário com dados processados, metadados e análises
    """
    # Criar instância do pré-processador com configurações personalizadas
    preprocessor = DatasetPreprocessor(custom_config)
    
    # Processamento principal
    preprocessing_result = preprocessor.preprocess(data)
    
    # Informações adicionais sobre o dataset
    dataset_info = {
        'original_shape': data.shape,
        'processed_shape': preprocessing_result['processed_data'].shape,
        'columns_dropped': preprocessor.metadata['dropped_columns'],
        'encoded_columns': preprocessor.metadata['encoded_columns'],
        'multicollinearity': preprocessing_result['multicollinearity']
    }
    
    # Processamento adicional para alvo, se especificado
    if target_column:
        # Remover coluna alvo do conjunto de features se estiver presente
        if target_column in preprocessing_result['processed_data'].columns:
            target_series = preprocessing_result['processed_data'][target_column]
            X = preprocessing_result['processed_data'].drop(columns=[target_column])
            y = target_series
        else:
            # Se a coluna alvo foi removida ou transformada
            X = preprocessing_result['processed_data']
            y = data[target_column] if target_column in data.columns else None
        
        # Análise descritiva do alvo
        if y is not None:
            target_analysis = {
                'type': 'categorical' if y.dtype == 'object' or y.nunique() < 10 else 'numeric',
                'unique_values': y.nunique(),
                'value_distribution': y.value_counts(normalize=True).to_dict() if y.dtype == 'object' or y.nunique() < 10 else None,
                'descriptive_stats': y.describe().to_dict() if pd.api.types.is_numeric_dtype(y) else None
            }
            dataset_info['target_analysis'] = target_analysis
    else:
        X = preprocessing_result['processed_data']
        y = None
    
    # Análise de outliers
    numeric_columns = X.select_dtypes(include=['number']).columns
    outlier_analysis = {}
    
    for col in numeric_columns:
        # Usar método IQR para detecção de outliers
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = X[(X[col] < lower_bound) | (X[col] > upper_bound)]
        
        outlier_analysis[col] = {
            'outliers_count': len(outliers),
            'outliers_percentage': len(outliers) / len(X) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    dataset_info['outlier_analysis'] = outlier_analysis
    
    # Combinar resultados
    final_result = {
        'processed_data': X,
        'target': y,
        'dataset_info': dataset_info,
        'metadata': preprocessing_result['metadata']
    }
    
    return final_result

def generate_preprocessing_report(preprocessing_result: Dict[str, Any]) -> str:
    """
    Gera um relatório detalhado do pré-processamento.
    
    Args:
        preprocessing_result: Resultado do pré-processamento
        
    Returns:
        String com relatório formatado
    """
    report = "# Relatório de Pré-Processamento de Dataset\n\n"
    
    # Informações do dataset
    info = preprocessing_result['dataset_info']
    report += "## Visão Geral do Dataset\n"
    report += f"- Dimensões originais: {info['original_shape']}\n"
    report += f"- Dimensões processadas: {info['processed_shape']}\n"
    report += f"- Colunas removidas: {len(info['columns_dropped'])} ({', '.join(info['columns_dropped'])})\n\n"
    
    # Colunas codificadas
    report += "## Codificação de Variáveis Categóricas\n"
    for col, encoding_info in info['encoded_columns'].items():
        report += f"### {col}\n"
        report += f"- Método: {encoding_info['method']}\n"
        report += f"- Categorias: {encoding_info['categories'] if 'categories' in encoding_info else 'N/A'}\n\n"
    
    # Multicolinearidade
    report += "## Análise de Multicolinearidade\n"
    if info['multicollinearity']:
        report += "Pares de features altamente correlacionadas:\n"
        for f1, f2, corr in info['multicollinearity']:
            report += f"- {f1} e {f2}: correlação de {corr:.2f}\n"
    else:
        report += "Nenhuma correlação significativa encontrada.\n\n"
    
    # Análise de Outliers
    report += "## Análise de Outliers\n"
    for col, outlier_info in info['outlier_analysis'].items():
        report += f"### {col}\n"
        report += f"- Número de outliers: {outlier_info['outliers_count']}\n"
        report += f"- Porcentagem de outliers: {outlier_info['outliers_percentage']:.2f}%\n"
        report += f"- Limite inferior: {outlier_info['lower_bound']:.2f}\n"
        report += f"- Limite superior: {outlier_info['upper_bound']:.2f}\n\n"
    
    # Análise do Alvo (se disponível)
    if 'target_analysis' in info:
        report += "## Análise da Variável Alvo\n"
        target_analysis = info['target_analysis']
        report += f"- Tipo: {target_analysis['type']}\n"
        report += f"- Valores únicos: {target_analysis['unique_values']}\n"
        
        if target_analysis['type'] == 'categorical':
            report += "### Distribuição de Valores\n"
            for val, prop in target_analysis['value_distribution'].items():
                report += f"- {val}: {prop*100:.2f}%\n"
        else:
            report += "### Estatísticas Descritivas\n"
            for stat, value in target_analysis['descriptive_stats'].items():
                report += f"- {stat}: {value:.2f}\n"
    
    return report

# Exemplo de uso
def example_usage():
    """
    Demonstra o uso do pré-processador de dataset.
    """
    # Carregar dados (substituir com seu próprio dataset)
    import pandas as pd
    data = pd.read_csv('seu_dataset.csv')
    
    # Configurações personalizadas (opcional)
    custom_config = {
        'missing_threshold': 0.2,  # Modificar limite de valores ausentes
        'categorical_encoding': 'label',  # Usar label encoding
        'normalization_method': 'minmax'  # Normalização min-max
    }
    
    # Processar dataset
    resultado = process_dataset(
        data, 
        target_column='target',  # Especificar coluna alvo
        custom_config=custom_config
    )
    
    # Gerar relatório
    relatorio = generate_preprocessing_report(resultado)
    print(relatorio)
    
    # Acessar dados processados
    X = resultado['processed_data']
    y = resultado['target']
    
    # Análise adicional ou treinamento de modelo
    # ...

if __name__ == '__main__':
    example_usage()