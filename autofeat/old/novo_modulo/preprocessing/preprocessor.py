import pandas as pd
import numpy as np
import logging
import networkx as nx
from typing import List, Dict, Callable, Optional
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from scipy import stats
import joblib
import os

class PreProcessor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = {
            'missing_values_strategy': 'median',
            'outlier_method': 'iqr',  # Opções: 'zscore', 'iqr', 'isolation_forest'
            'categorical_strategy': 'onehot',
            'scaling': 'standard',
            'dimensionality_reduction': None,
            'feature_selection': 'variance',
            'generate_features': True,
            'verbosity': 1,
            'min_pca_components': 10,
            'correlation_threshold': 0.95
        }
        if config:
            self.config.update(config)
        
        self.preprocessor = None
        self.column_types = {}
        self.fitted = False
        self.feature_names = []
        self.target_col = None
        
        self._setup_logging()
        self.logger.info("PreProcessor inicializado com sucesso.")

    def _setup_logging(self):
        self.logger = logging.getLogger("AutoFE.PreProcessor")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel({0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(self.config['verbosity'], logging.INFO))

    def _identify_column_types(self, df: pd.DataFrame) -> Dict:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return {'numeric': numeric_cols, 'categorical': categorical_cols}
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            self.logger.warning("DataFrame vazio antes da remoção de outliers. Pulando esta etapa.")
            return df
            
        # Seleciona apenas colunas numéricas para tratamento de outliers
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            return df
            
        method = self.config['outlier_method']
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(numeric_df, nan_policy='omit'))
            mask = (z_scores < 3).all(axis=1)
            filtered_df = df[mask]
        elif method == 'iqr':
            Q1 = numeric_df.quantile(0.25)
            Q3 = numeric_df.quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
            filtered_df = df[mask]
        elif method == 'isolation_forest':
            clf = IsolationForest(contamination=0.05, random_state=42)
            outliers = clf.fit_predict(numeric_df)
            filtered_df = df[outliers == 1]
        else:
            return df  # Caso o método não seja reconhecido, retorna o DataFrame original

        if filtered_df.empty:
            self.logger.warning("Todas as amostras foram removidas na remoção de outliers! Retornando DataFrame original.")
            return df

        return filtered_df

    def _remove_high_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            return df
            
        try:
            corr_matrix = numeric_df.corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.config['correlation_threshold'])]
            
            if to_drop:
                self.logger.info(f"Removendo {len(to_drop)} colunas altamente correlacionadas: {to_drop[:5]}...")
                return df.drop(columns=to_drop, errors='ignore')
            return df
        except Exception as e:
            self.logger.warning(f"Erro ao calcular correlações: {e}. Retornando DataFrame original.")
            return df
    
    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config['generate_features']:
            return df

        num_data = df.select_dtypes(include=['number'])
        if num_data.empty:
            self.logger.warning("Nenhuma feature numérica encontrada. Pulando geração de features polinomiais.")
            return df

        try:
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            new_features = poly.fit_transform(num_data)
            
            # Gera nomes de features mais descritivos
            feature_names = poly.get_feature_names_out(num_data.columns)
            
            # Remove colunas originais dos nomes das features transformadas
            poly_feature_names = [name for name in feature_names 
                                 if ' ' in name]  # Features interativas contêm espaço
            
            # Cria DataFrame apenas com as novas features
            df_poly = pd.DataFrame(
                new_features[:, len(num_data.columns):],
                columns=poly_feature_names,
                index=df.index
            )
            
            self.logger.info(f"Geradas {len(poly_feature_names)} novas features polinomiais")
            return pd.concat([df, df_poly], axis=1)
        except Exception as e:
            self.logger.error(f"Erro ao gerar features polinomiais: {e}")
            return df
    
    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> 'PreProcessor':
        if df.empty:
            raise ValueError("Não é possível ajustar com um DataFrame vazio")
            
        self.target_col = target_col
        df_proc = df.copy()
        
        # Remover coluna alvo se presente
        if target_col and target_col in df_proc.columns:
            df_proc = df_proc.drop(columns=[target_col])
            self.logger.info(f"Coluna alvo '{target_col}' removida para processamento")
        
        # Aplicar transformações
        df_proc = self._remove_outliers(df_proc)
        df_proc = self._generate_features(df_proc)
        df_proc = self._remove_high_correlation(df_proc)

        if df_proc.empty:
            self.logger.error("Erro: O DataFrame está vazio após pré-processamento. Ajuste as configurações.")
            raise ValueError("O DataFrame está vazio após as transformações.")

        self.column_types = self._identify_column_types(df_proc)
        self.logger.info(f"Colunas identificadas: {len(self.column_types['numeric'])} numéricas, {len(self.column_types['categorical'])} categóricas")

        # Configurar pipeline de transformação
        transformers = self._build_transformers()
        self.preprocessor = ColumnTransformer(transformers, remainder='passthrough')

        # Configurar redução de dimensionalidade, se aplicável
        self._setup_dimensionality_reduction(df_proc)
        
        try:
            self.preprocessor.fit(df_proc)
            self.feature_names = df_proc.columns.tolist()
            self.fitted = True
            self.logger.info(f"Preprocessador ajustado com sucesso com {len(self.feature_names)} features")
            return self
        except Exception as e:
            self.logger.error(f"Erro ao ajustar o preprocessador: {e}")
            raise

    def _build_transformers(self) -> List:
        """Constrói os transformadores para colunas numéricas e categóricas"""
        # Configurar imputer
        if self.config['missing_values_strategy'] == 'knn':
            num_imputer = KNNImputer()
        else:
            num_imputer = SimpleImputer(strategy=self.config['missing_values_strategy'])
        
        # Configurar scaler
        scalers = {
            'standard': StandardScaler(), 
            'minmax': MinMaxScaler(), 
            'robust': RobustScaler()
        }
        scaler = scalers.get(self.config['scaling'], 'passthrough')

        # Pipeline para features numéricas
        numeric_transformer = Pipeline([
            ('imputer', num_imputer),
            ('scaler', scaler)
        ])

        # Pipeline para features categóricas
        categorical_encoder = (
            OneHotEncoder(handle_unknown='ignore', sparse_output=False) 
            if self.config['categorical_strategy'] == 'onehot' 
            else OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        )
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', categorical_encoder)
        ])

        # Montar transformers
        transformers = []
        if self.column_types['numeric']:
            transformers.append(('num', numeric_transformer, self.column_types['numeric']))
        if self.column_types['categorical']:
            transformers.append(('cat', categorical_transformer, self.column_types['categorical']))
            
        return transformers

    def _setup_dimensionality_reduction(self, df: pd.DataFrame) -> None:
        """Configura redução de dimensionalidade e seleção de features"""
        pipeline_steps = [('preprocessor', self.preprocessor)]
        
        # Adicionar PCA se configurado
        if self.config['dimensionality_reduction'] == 'pca':
            n_components = min(self.config['min_pca_components'], df.shape[1])
            if n_components > 1:
                pipeline_steps.append(('pca', PCA(n_components=n_components)))
                self.logger.info(f"PCA configurado com {n_components} componentes")
            else:
                self.logger.warning("Número de features insuficiente para aplicar PCA. PCA será ignorado.")

        # Adicionar seleção de features se configurado
        if self.config['feature_selection'] == 'variance':
            pipeline_steps.append(('feature_selection', VarianceThreshold(threshold=0.01)))
            
        # Construir pipeline final
        if len(pipeline_steps) > 1:
            self.preprocessor = Pipeline(pipeline_steps)

    def transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("O preprocessador precisa ser ajustado antes de transformar dados. Use .fit() primeiro.")

        df_proc = df.copy()
        target_data = None
        
        # Separar target se presente
        if target_col and target_col in df_proc.columns:
            target_data = df_proc[target_col].copy()
            df_proc = df_proc.drop(columns=[target_col])
        
        # Aplicar transformações
        df_proc = self._remove_outliers(df_proc)
        df_proc = self._generate_features(df_proc)
        df_proc = self._remove_high_correlation(df_proc)

        # Verificar e ajustar colunas para compatibilidade com o modelo de preprocessamento
        self._check_columns_compatibility(df_proc)
        
        # Aplicar transformação
        try:
            df_transformed = self.preprocessor.transform(df_proc)
            
            # Determinar nomes das colunas
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                feature_names = self.preprocessor.get_feature_names_out()
            else:
                feature_names = [f"feature_{i}" for i in range(df_transformed.shape[1])]

            # Criar DataFrame com os dados transformados
            result_df = pd.DataFrame(
                df_transformed, 
                index=df_proc.index, 
                columns=feature_names
            )
            
            # Adicionar coluna target se existir
            if target_data is not None:
                result_df[target_col] = target_data.loc[result_df.index]
                
            return result_df
            
        except Exception as e:
            self.logger.error(f"Erro na transformação dos dados: {e}")
            raise

    def _check_columns_compatibility(self, df: pd.DataFrame) -> None:
        """Verifica e ajusta as colunas para compatibilidade com o modelo ajustado"""
        # Verificar colunas ausentes
        missing_cols = set(self.feature_names) - set(df.columns)
        if missing_cols:
            self.logger.warning(f"Colunas ausentes na transformação: {missing_cols}. Adicionando com zeros.")
            for col in missing_cols:
                df[col] = 0
                
        # Manter apenas colunas conhecidas pelo modelo
        extra_cols = set(df.columns) - set(self.feature_names)
        if extra_cols:
            self.logger.warning(f"Colunas extras ignoradas: {extra_cols}")
            
    def save(self, filepath: str) -> None:
        if not self.fitted:
            raise ValueError("Não é possível salvar um preprocessador não ajustado.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        self.logger.info(f"Preprocessador salvo em {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PreProcessor':
        preprocessor = joblib.load(filepath)
        preprocessor.logger.info(f"Preprocessador carregado de {filepath}")
        return preprocessor
    
    

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutoFE.Explorer")


class TransformationTree:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph.add_node("root", data=None)
        logger.info("TransformationTree inicializada.")
    
    def add_transformation(self, parent: str, name: str, data, score: float = 0.0):
        """Adiciona uma transformação à árvore."""
        self.graph.add_node(name, data=data, score=score)
        self.graph.add_edge(parent, name)
        feature_diff = data.shape[1] - self.graph.nodes[parent]['data'].shape[1] if self.graph.nodes[parent]['data'] is not None else 0
        logger.info(f"Transformação '{name}' adicionada com score {score}. Dimensão do conjunto: {data.shape}. Alteração nas features: {feature_diff}")
    
    def get_best_transformations(self, heuristic: Callable[[Dict], float]) -> List[str]:
        """Retorna as melhores transformações baseadas em uma heurística."""
        scored_nodes = {node: heuristic(self.graph.nodes[node]['data']) for node in self.graph.nodes if node != "root"}
        best_transformations = sorted(scored_nodes, key=scored_nodes.get, reverse=True)
        logger.info(f"Melhores transformações ordenadas: {best_transformations}")
        return best_transformations

class HeuristicSearch:
    def __init__(self, heuristic: Callable[[pd.DataFrame], float]):
        self.heuristic = heuristic
    
    def search(self, tree: TransformationTree) -> str:
        """Executa uma busca heurística na árvore de transformações."""
        best_nodes = tree.get_best_transformations(self.heuristic)
        best_node = best_nodes[0] if best_nodes else None
        logger.info(f"Melhor transformação encontrada: {best_node}")
        return best_node
    
    @staticmethod
    def custom_heuristic(df: pd.DataFrame) -> float:
        """Heurística baseada na matriz de correlação e diversidade categórica."""
        correlation_penalty = 0
        categorical_diversity_score = 0
        
        # Penaliza alta correlação entre features
        if df.shape[1] > 1:
            correlation_matrix = df.corr().abs()
            high_corr = (correlation_matrix > 0.95).sum().sum() - df.shape[1]  # Remove a diagonal
            correlation_penalty = high_corr / (df.shape[1] ** 2)  # Normaliza penalização
        
        # Avalia diversidade de variáveis categóricas
        categorical_features = df.select_dtypes(include=['object', 'category'])
        if not categorical_features.empty:
            unique_counts = categorical_features.nunique()
            categorical_diversity_score = unique_counts.mean() / max(1, unique_counts.max())  # Normaliza entre 0 e 1
        
        # Score final: penaliza alta correlação e recompensa diversidade categórica
        final_score = -correlation_penalty + categorical_diversity_score
        return final_score

class Explorer:
    def __init__(self, heuristic: Callable[[pd.DataFrame], float] = None, target_col: Optional[str] = None):
        self.tree = TransformationTree()
        self.search = HeuristicSearch(heuristic or HeuristicSearch.custom_heuristic)
        self.target_col = target_col
    
    def add_transformation(self, parent: str, name: str, data, score: float = 0.0):
        """Adiciona uma transformação com uma pontuação atribuída."""
        self.tree.add_transformation(parent, name, data, score)
    
    def find_best_transformation(self) -> str:
        """Retorna a melhor transformação com base na busca heurística."""
        return self.search.search(self.tree)
    
    def analyze_transformations(self, df):
        """Testa diferentes transformações e escolhe a melhor para o PreProcessor."""
        logger.info("Iniciando análise de transformações.")
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df, columns=[f"feature_{i}" for i in range(df.shape[1])])
        
        base_node = "root"
        configurations = [
            {"missing_values_strategy": "mean"},
            {"missing_values_strategy": "median"},
            {"missing_values_strategy": "most_frequent"},
            {"outlier_method": "iqr"},
            {"outlier_method": "zscore"},
            {"outlier_method": "isolation_forest"},
            {"categorical_strategy": "onehot"},
            {"categorical_strategy": "ordinal"},
            {"scaling": "standard"},
            {"scaling": "minmax"},
            {"scaling": "robust"},
            {"dimensionality_reduction": "pca"},
            {"feature_selection": "variance"},
            {"generate_features": True},
            {"generate_features": False}
        ]
        
        for config in configurations:
            name = "_".join([f"{key}-{value}" for key, value in config.items()])
            logger.info(f"Testando transformação: {name}. Dimensão original: {df.shape}")
            
            if df.empty:
                logger.warning(f"O DataFrame está vazio após remoção de outliers. Pulando transformação: {name}")
                continue
            
            transformed_data = PreProcessor(config).fit(df, target_col=self.target_col if self.target_col else None).transform(df, target_col=self.target_col if self.target_col else None)
            
            if transformed_data.empty:
                logger.warning(f"A transformação {name} resultou em um DataFrame vazio. Pulando.")
                continue
            
            score = self.search.heuristic(transformed_data)
            self.add_transformation(base_node, name, transformed_data, score)
        
        best_transformation = self.find_best_transformation()
        logger.info(f"Melhor transformação final: {best_transformation}")
        return self.tree.graph.nodes[best_transformation]['data'] if best_transformation else df


def create_preprocessor(config: Optional[Dict] = None) -> PreProcessor:
    return PreProcessor(config)
