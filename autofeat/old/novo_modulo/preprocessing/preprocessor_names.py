import pandas as pd
import numpy as np
import logging
import networkx as nx
from typing import List, Dict, Callable
from typing import Dict, Optional
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
        method = self.config['outlier_method']
        if df.empty:
            self.logger.warning("DataFrame vazio antes da remoção de outliers. Pulando esta etapa.")
            return df  # Retorna o DataFrame sem modificação

        if method == 'zscore':
            filtered_df = df[(np.abs(stats.zscore(df.select_dtypes(include=['number']))) < 3).all(axis=1)]
        elif method == 'iqr':
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            filtered_df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
            
        elif method == 'isolation_forest':
            clf = IsolationForest(contamination=0.05, random_state=42)
            outliers = clf.fit_predict(df.select_dtypes(include=['number']))
            filtered_df = df[outliers == 1]
        else:
            return df  # Caso o método não seja reconhecido, retorna o DataFrame original

        if filtered_df.empty:
            self.logger.warning("Todas as amostras foram removidas na remoção de outliers! Retornando DataFrame original.")
            return df  # Retorna o DataFrame original caso a remoção tenha eliminado tudo

        return filtered_df

    def _remove_high_correlation(self, df):
        corr_matrix = df.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.config['correlation_threshold'])]
        return df.drop(columns=to_drop, errors='ignore')
    
    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config['generate_features']:
            return df

        num_data = df.select_dtypes(include=['number'])

        if num_data.empty:
            self.logger.warning("Nenhuma feature numérica encontrada. Pulando geração de features polinomiais.")
            return df  # Retorna o DataFrame sem alteração

        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        try:
            new_features = poly.fit_transform(num_data)
            new_feature_names = [f"feature_{i}" for i in range(new_features.shape[1])]
            df_poly = pd.DataFrame(new_features, columns=new_feature_names, index=df.index)
            return pd.concat([df, df_poly], axis=1)
        except Exception as e:
            self.logger.error(f"Erro ao gerar features polinomiais: {e}")
            return df  # Retorna o DataFrame sem alterações se houver erro

    
    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> 'PreProcessor':
        if target_col and target_col in df.columns:
            df = df.drop(columns=[target_col])
        else:
            self.logger.warning(f"Coluna '{target_col}' não encontrada no DataFrame. Nenhuma remoção foi feita.")
        
        df = self._remove_outliers(df)
        df = self._generate_features(df)
        df = self._remove_high_correlation(df)

        if df.empty:
            self.logger.error("Erro: O DataFrame está vazio após pré-processamento. Ajuste as configurações de transformação.")
            raise ValueError("O DataFrame está vazio após as transformações.")

        self.column_types = self._identify_column_types(df)

        imputer = KNNImputer() if self.config['missing_values_strategy'] == 'knn' else SimpleImputer(strategy=self.config['missing_values_strategy'])
        
        scalers = {'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'robust': RobustScaler()}
        scaler = scalers.get(self.config['scaling'], 'passthrough')

        numeric_transformer = Pipeline([
            ('imputer', imputer),
            ('scaler', scaler)
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False) if self.config['categorical_strategy'] == 'onehot' else OrdinalEncoder())
        ])

        transformers = []
        if self.column_types['numeric']:
            transformers.append(('num', numeric_transformer, self.column_types['numeric']))
        if self.column_types['categorical']:
            transformers.append(('cat', categorical_transformer, self.column_types['categorical']))
        
        self.preprocessor = ColumnTransformer(transformers, remainder='passthrough')

        # Ajuste no PCA para evitar erro de número de componentes
        if self.config['dimensionality_reduction'] == 'pca':
            n_components = min(10, df.shape[1])  # Garante que n_components não seja maior que o número de colunas
            if n_components > 1:
                pca = PCA(n_components=n_components)
                self.preprocessor = Pipeline([
                    ('preprocessor', self.preprocessor),
                    ('pca', pca)
                ])
            else:
                self.logger.warning("Número de features insuficiente para aplicar PCA. PCA será ignorado.")

        if self.config['feature_selection'] == 'variance':
            self.preprocessor = Pipeline([
                ('preprocessor', self.preprocessor),
                ('feature_selection', VarianceThreshold(threshold=0.01))
            ])
        
        try:
            self.preprocessor.fit(df)
            self.feature_names = df.columns[df.columns.isin(df.columns)].tolist()  # Apenas colunas sobreviventes  # Armazena os nomes das colunas pós-fit
        except ValueError as e:
            self.logger.error(f"Erro ao ajustar o preprocessador: {e}")
            raise

        self.fitted = True
        return self


    def transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("O preprocessador precisa ser ajustado antes de transformar dados. Use .fit() primeiro.")

        if target_col and target_col in df.columns:
            target_data = df[target_col]
            df = df.drop(columns=[target_col])
        else:
            target_data = None

        df = self._remove_outliers(df)
        df = self._generate_features(df)
        df = self._remove_high_correlation(df)

        missing_cols = set(self.feature_names) - set(df.columns)
        if missing_cols:
            self.logger.warning(f"Colunas ausentes na transformação: {missing_cols}. Ajustando DataFrame.")
            for col in missing_cols:
                df[col] = 0

        df = df.loc[:, df.columns.isin(self.feature_names)]
        
        # **Correção aqui: capturando as colunas resultantes da transformação**
        df_transformed = self.preprocessor.transform(df)
        
        # Se PCA ou feature selection reduziram as colunas, pegar os nomes dinamicamente
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            feature_names = self.preprocessor.get_feature_names_out()
        else:
            feature_names = [f"feature_{i}" for i in range(df_transformed.shape[1])]

        df_transformed = pd.DataFrame(df_transformed, index=df.index, columns=feature_names)

        if target_data is not None:
            target_data = target_data.loc[df_transformed.index]  # Filtrar target para manter só as amostras existentes
            df_transformed[target_col] = target_data.values

        return df_transformed

    
    def save(self, filepath: str) -> None:
        if not self.fitted:
            raise ValueError("Não é possível salvar um preprocessador não ajustado.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'PreProcessor':
        return joblib.load(filepath)
    
    

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
