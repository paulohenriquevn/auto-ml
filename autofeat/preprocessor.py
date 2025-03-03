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
        """
        Remove outliers do DataFrame usando o método especificado.
        Aplica a detecção e remoção de outliers apenas em colunas numéricas.
        """
        method = self.config['outlier_method']
        if df.empty:
            self.logger.warning("DataFrame vazio antes da remoção de outliers. Pulando esta etapa.")
            return df

        # Seleciona apenas colunas numéricas para detecção de outliers
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            self.logger.warning("Nenhuma coluna numérica encontrada para detecção de outliers. Pulando esta etapa.")
            return df
        
        if method == 'zscore':
            # Aplica Z-score apenas em colunas numéricas
            z_scores = np.abs(stats.zscore(numeric_df))
            keep_mask = (z_scores < 3).all(axis=1)
            filtered_df = df[keep_mask]
        elif method == 'iqr':
            # Aplica IQR apenas em colunas numéricas
            Q1 = numeric_df.quantile(0.25)
            Q3 = numeric_df.quantile(0.75)
            IQR = Q3 - Q1
            
            # Cria uma máscara para valores dentro do intervalo aceitável
            keep_mask = ~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
            filtered_df = df[keep_mask]
        elif method == 'isolation_forest':
            # Aplica Isolation Forest com contaminação mais baixa (pode ser ajustada)
            clf = IsolationForest(contamination=0.01, random_state=42, n_estimators=100)
            outliers = clf.fit_predict(numeric_df)
            filtered_df = df[outliers == 1]
        elif method == 'lof':
            # Novo método: Local Outlier Factor para melhor detecção em datasets não balanceados
            from sklearn.neighbors import LocalOutlierFactor
            clf = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
            outliers = clf.fit_predict(numeric_df)
            filtered_df = df[outliers == 1]  # 1 são inliers, -1 são outliers
        else:
            return df  # Caso o método não seja reconhecido, retorna o DataFrame original

        if filtered_df.empty:
            self.logger.warning("Todas as amostras foram removidas na remoção de outliers! Retornando DataFrame original.")
            return df  # Retorna o DataFrame original caso a remoção tenha eliminado tudo

        self.logger.info(f"Foram removidos {len(df) - len(filtered_df)} outliers ({(len(df) - len(filtered_df)) / len(df) * 100:.2f}% dos dados)")
        return filtered_df

    def _remove_high_correlation(self, df):
        # Se não houver pelo menos 2 colunas, não há como calcular correlação
        if df.shape[1] < 2:
            return df

        # Seleciona apenas colunas numéricas para cálculo de correlação
        numeric_df = df.select_dtypes(include=['number'])
        
        # Se não houver colunas numéricas suficientes, retorna o DataFrame original
        if numeric_df.shape[1] < 2:
            self.logger.info("Menos de 2 colunas numéricas disponíveis. Pulando verificação de correlação.")
            return df
        
        # Calcula a matriz de correlação apenas para colunas numéricas
        corr_matrix = numeric_df.corr().abs()
        
        # Obtém o triângulo superior da matriz
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Identifica colunas para remover
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.config['correlation_threshold'])]
        
        if to_drop:
            self.logger.info(f"Removendo {len(to_drop)} features altamente correlacionadas: {to_drop}")
            return df.drop(columns=to_drop, errors='ignore')
        else:
            self.logger.info("Nenhuma feature com alta correlação encontrada.")
            return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trata valores ausentes no DataFrame usando a estratégia configurada.
        """
        if df.empty:
            return df
            
        # Verifica se existem valores ausentes
        if not df.isna().any().any():
            self.logger.info("Nenhum valor ausente encontrado. Pulando tratamento.")
            return df
            
        strategy = self.config['missing_values_strategy']
        self.logger.info(f"Aplicando estratégia '{strategy}' para tratamento de valores ausentes")
        
        # Separar colunas numéricas e categóricas
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Cria uma cópia para não modificar o DataFrame original
        result_df = df.copy()
        
        if strategy == 'knn':
            # KNN Imputer para colunas numéricas
            if numeric_cols:
                # Salvar valores não numéricos temporariamente
                temp_cat = df[categorical_cols] if categorical_cols else None
                
                # Imputa valores ausentes nas colunas numéricas usando KNNImputer
                imputer = KNNImputer(n_neighbors=5, weights='distance')
                numeric_data = df[numeric_cols]
                numeric_data_filled = pd.DataFrame(
                    imputer.fit_transform(numeric_data),
                    columns=numeric_cols,
                    index=df.index
                )
                
                # Atualiza as colunas numéricas no resultado
                for col in numeric_cols:
                    result_df[col] = numeric_data_filled[col]
                
            # Para colunas categóricas, usamos a moda
            for col in categorical_cols:
                result_df[col] = result_df[col].fillna(result_df[col].mode()[0] if not result_df[col].mode().empty else "MISSING")
                
        elif strategy == 'iterative':
            # Novo método: Imputação iterativa (mais sofisticado)
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            
            if numeric_cols:
                imputer = IterativeImputer(max_iter=10, random_state=42)
                numeric_data = df[numeric_cols]
                numeric_data_filled = pd.DataFrame(
                    imputer.fit_transform(numeric_data),
                    columns=numeric_cols,
                    index=df.index
                )
                
                for col in numeric_cols:
                    result_df[col] = numeric_data_filled[col]
            
            # Para colunas categóricas, usamos a moda
            for col in categorical_cols:
                result_df[col] = result_df[col].fillna(result_df[col].mode()[0] if not result_df[col].mode().empty else "MISSING")
        
        else:
            # Métodos padrão: mean, median, most_frequent
            for col in numeric_cols:
                if strategy == 'mean':
                    result_df[col] = result_df[col].fillna(result_df[col].mean())
                elif strategy == 'median':
                    result_df[col] = result_df[col].fillna(result_df[col].median())
                elif strategy == 'most_frequent':
                    result_df[col] = result_df[col].fillna(result_df[col].mode()[0] if not result_df[col].mode().empty else 0)
                else:
                    result_df[col] = result_df[col].fillna(0)  # Padrão: preenche com zero
            
            # Para colunas categóricas, sempre usamos a moda
            for col in categorical_cols:
                result_df[col] = result_df[col].fillna(result_df[col].mode()[0] if not result_df[col].mode().empty else "MISSING")
        
        # Verifica e reporta se ainda existem valores ausentes
        missing_after = result_df.isna().sum().sum()
        if missing_after > 0:
            self.logger.warning(f"Ainda existem {missing_after} valores ausentes após imputação!")
        
        return result_df
    
    def _generate_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Gera automaticamente novas features baseadas nas existentes.
        Inclui interações polinomiais, agregações e features temporais.
        
        Args:
            df: DataFrame original
            target_col: Nome da coluna alvo (opcional)
                
        Returns:
            DataFrame com novas features adicionadas
        """
        if not self.config['generate_features']:
            return df

        # Cria uma cópia para não modificar o DataFrame original
        result_df = df.copy()
        
        # Seleciona apenas dados numéricos
        num_data = df.select_dtypes(include=['number'])

        if num_data.empty:
            self.logger.warning("Nenhuma feature numérica encontrada. Pulando geração de features.")
            return df
        
        # Trata valores ausentes antes da geração
        num_data_filled = num_data.copy()
        for col in num_data_filled.columns:
            if num_data_filled[col].isna().any():
                num_data_filled[col] = num_data_filled[col].fillna(num_data_filled[col].median())
        
        # 1. Estatísticas básicas para todas as colunas numéricas
        try:
            # Limita o número de colunas para evitar explosão combinatória
            max_cols = 10
            if num_data_filled.shape[1] > max_cols:
                self.logger.info(f"Muitas colunas numéricas ({num_data_filled.shape[1]}). Limitando a {max_cols} para geração de features.")
                # Seleciona as colunas com maior variância
                variances = num_data_filled.var()
                top_cols = variances.nlargest(max_cols).index.tolist()
                num_data_filled = num_data_filled[top_cols]
            
            # 2. Features de interação polinomial
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            poly_features = poly.fit_transform(num_data_filled)
            
            # Obtém nomes para as novas features
            if hasattr(poly, 'get_feature_names_out'):
                poly_feature_names = poly.get_feature_names_out(num_data_filled.columns)
            else:
                poly_feature_names = [f"poly_feature_{i}" for i in range(poly_features.shape[1])]
            
            # Cria um DataFrame com as features polinomiais
            df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=num_data_filled.index)
            
            # Remove colunas originais que já existem no DataFrame original
            df_poly = df_poly.iloc[:, num_data_filled.shape[1]:]
            
            # Renomeia para evitar colisões
            df_poly.columns = [f"interaction_{col}" for col in df_poly.columns]
            
            # 3. Adiciona features estatísticas avançadas
            for col in num_data_filled.columns:
                # Implementa estatísticas robustas
                result_df[f'{col}_zscore'] = stats.zscore(num_data_filled[col], nan_policy='omit')  # Z-score
                result_df[f'{col}_log'] = np.log1p(num_data_filled[col] - num_data_filled[col].min() + 1) if (num_data_filled[col] >= 0).all() else np.log1p(np.abs(num_data_filled[col]))  # Log transform (com ajuste para valores negativos)
                result_df[f'{col}_sqrt'] = np.sqrt(num_data_filled[col] - num_data_filled[col].min() + 1e-8)  # Raiz quadrada (com ajuste para valores negativos)
            
            # 4. Detecção de ID ou timestamp para features agregadas
            # Verifica se há colunas que podem ser usadas como identificadores
            id_columns = [col for col in df.columns if 'id' in col.lower() or 'user' in col.lower() or 'customer' in col.lower()]
            
            # Adiciona agregações por ID (se houver IDs identificados)
            if id_columns and len(df) > 10:  # Evita criar agregações para datasets muito pequenos
                for id_col in id_columns[:1]:  # Limita a primeira coluna ID encontrada
                    if id_col in df.columns:
                        self.logger.info(f"Gerando agregações por coluna ID: {id_col}")
                        
                        # Grupo por ID
                        for num_col in num_data_filled.columns[:5]:  # Limita a 5 colunas numéricas para agregação
                            # Agregações por ID
                            aggs = df.groupby(id_col)[num_col].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
                            
                            # Renomear colunas agregadas
                            aggs.columns = [id_col] + [f'{num_col}_by_{id_col}_{agg}' for agg in ['mean', 'std', 'min', 'max', 'count']]
                            
                            # Mesclar agregações com o DataFrame original
                            result_df = result_df.merge(aggs, on=id_col, how='left')
                            
                            # Calcular diferença da média do grupo
                            result_df[f'{num_col}_diff_from_mean'] = result_df[num_col] - result_df[f'{num_col}_by_{id_col}_mean']
            
            # 5. Detecção e processamento de características temporais
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'day' in col.lower()]
            
            for date_col in date_columns:
                if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    self.logger.info(f"Gerando features temporais para coluna: {date_col}")
                    
                    # Extrai componentes de data para criar features temporais
                    result_df[f'{date_col}_year'] = df[date_col].dt.year
                    result_df[f'{date_col}_month'] = df[date_col].dt.month
                    result_df[f'{date_col}_day'] = df[date_col].dt.day
                    result_df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
                    result_df[f'{date_col}_is_weekend'] = df[date_col].dt.dayofweek >= 5
                    result_df[f'{date_col}_hour'] = df[date_col].dt.hour if hasattr(df[date_col].dt, 'hour') else -1
            
            # 6. Criação de features específicas para classificação e regressão
            if target_col is not None and target_col in df.columns:
                # Se for classificação (verifica se o target é categórico)
                if df[target_col].dtype == 'object' or df[target_col].dtype == 'category' or len(df[target_col].unique()) < 10:
                    # Target encoding - média da variável alvo por categoria
                    for cat_col in df.select_dtypes(include=['object', 'category']).columns:
                        if cat_col != target_col and df[cat_col].nunique() < 50:  # Limita a colunas com poucas categorias
                            target_means = df.groupby(cat_col)[target_col].mean().to_dict()
                            result_df[f'{cat_col}_target_mean'] = df[cat_col].map(target_means)
                
                # Se for regressão (verifica se o target é numérico)
                elif df[target_col].dtype in ['int64', 'float64']:
                    # Não implementamos features específicas de regressão por enquanto
                    pass
            
            # Combina todas as novas features
            result_df = pd.concat([result_df, df_poly], axis=1)
            
            # Limita o número máximo de novas features para evitar explosão de dimensionalidade
            if result_df.shape[1] > df.shape[1] * 2:
                self.logger.warning(f"Muitas features geradas ({result_df.shape[1]}). Limitando para evitar dimensionalidade excessiva.")
                # Manter features originais + até 20 novas features com maior correlação com o target
                original_cols = df.columns.tolist()
                new_cols = [c for c in result_df.columns if c not in original_cols]
                
                if target_col is not None and target_col in result_df.columns and len(new_cols) > 20:
                    # Calcula correlação com o target
                    correlations = []
                    for col in new_cols:
                        if pd.api.types.is_numeric_dtype(result_df[col]):
                            corr = result_df[col].corr(result_df[target_col]) if not result_df[col].isna().all() else 0
                            correlations.append((col, abs(corr)))
                    
                    # Ordena por correlação e seleciona as top 20
                    top_features = sorted(correlations, key=lambda x: x[1], reverse=True)[:20]
                    top_feature_names = [f[0] for f in top_features]
                    
                    # Mantém colunas originais + target + top features
                    result_df = result_df[original_cols + top_feature_names]
                    self.logger.info(f"Reduzido para {len(result_df.columns)} features após seleção por correlação.")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar features avançadas: {e}")
            return df  # Retorna o DataFrame sem alterações se houver erro
    
    def _select_best_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Seleciona as melhores features com base em diferentes técnicas.
        
        Args:
            df: DataFrame com todas as features
            target_col: Nome da coluna alvo (opcional)
                
        Returns:
            DataFrame com as features selecionadas
        """
        if not self.config['feature_selection'] or df.empty:
            return df
        
        # Verifica se há features suficientes para seleção
        if df.shape[1] <= 3:  # Muito poucas colunas, não vale a pena selecionar
            return df
        
        # Separar features e target
        X = df.drop(columns=[target_col]) if target_col and target_col in df.columns else df
        y = df[target_col] if target_col and target_col in df.columns else None
        
        # Seleciona apenas colunas numéricas para seleção de features
        numeric_cols = X.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            self.logger.warning("Nenhuma coluna numérica encontrada para seleção de features.")
            return df
        
        # Se não temos target ou ele não está no DataFrame, usamos apenas VarianceThreshold
        if y is None or target_col is None:
            self.logger.info("Aplicando seleção de features baseada em variância (sem target)")
            
            # Remove features com baixa variância
            selector = VarianceThreshold(threshold=0.01)
            X_numeric = X[numeric_cols]
            
            try:
                # Preenche valores ausentes antes de calcular a variância
                X_numeric_filled = X_numeric.fillna(X_numeric.median())
                selected = selector.fit_transform(X_numeric_filled)
                
                # Obtém índices das features selecionadas
                selected_indices = selector.get_support(indices=True)
                selected_features = X_numeric.columns[selected_indices].tolist()
                
                # Mantém as colunas não numéricas e adiciona as numéricas selecionadas
                non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
                final_columns = non_numeric_cols + selected_features
                
                self.logger.info(f"Selecionadas {len(selected_features)} de {len(numeric_cols)} features numéricas baseado em variância")
                
                # Retorna DataFrame apenas com as colunas selecionadas + target (se existir)
                result_df = df[final_columns]
                if target_col and target_col in df.columns:
                    result_df[target_col] = df[target_col]
                    
                return result_df
                
            except Exception as e:
                self.logger.error(f"Erro na seleção por variância: {e}")
                return df
        
        # Caso tenha alvo, podemos usar seleção baseada na variável target
        feature_selection_method = self.config['feature_selection']
        self.logger.info(f"Aplicando seleção de features com método: {feature_selection_method}")
        
        try:
            # Preenche valores ausentes
            X_numeric = X[numeric_cols].fillna(X[numeric_cols].median())
            
            # Implementa diferentes métodos de seleção
            if feature_selection_method == 'mutual_info':
                # Verifica se é classificação ou regressão
                if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
                    # Regressão
                    from sklearn.feature_selection import mutual_info_regression
                    mi_scores = mutual_info_regression(X_numeric, y)
                else:
                    # Classificação
                    from sklearn.feature_selection import mutual_info_classif
                    mi_scores = mutual_info_classif(X_numeric, y)
                    
                # Seleciona top features
                mi_df = pd.DataFrame({'feature': numeric_cols, 'mi_score': mi_scores})
                mi_df = mi_df.sort_values('mi_score', ascending=False)
                
                # Mantém features com score acima de um limiar ou top N features
                top_n = min(20, len(numeric_cols))
                top_features = mi_df.head(top_n)['feature'].tolist()
                
                self.logger.info(f"Selecionadas {len(top_features)} features usando informação mútua.")
                
            elif feature_selection_method == 'model_based':
                from sklearn.feature_selection import SelectFromModel
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                
                # Escolhe modelo baseado no tipo de target
                if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
                    # Regressão
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                else:
                    # Classificação
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                
                # Usa SelectFromModel para escolher features
                selector = SelectFromModel(model, threshold='mean')
                selector.fit(X_numeric, y)
                selected_indices = selector.get_support(indices=True)
                top_features = X_numeric.columns[selected_indices].tolist()
                
                self.logger.info(f"Selecionadas {len(top_features)} features usando seleção baseada em modelo.")
                
            elif feature_selection_method == 'rfe':
                from sklearn.feature_selection import RFE
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                
                # Escolhe modelo baseado no tipo de target
                if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
                    # Regressão
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                else:
                    # Classificação
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                    
                # Número de features a selecionar
                n_features_to_select = min(20, len(numeric_cols))
                
                # Seleciona features recursivamente
                rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
                rfe.fit(X_numeric, y)
                selected_indices = np.where(rfe.support_)[0]
                top_features = X_numeric.columns[selected_indices].tolist()
                
                self.logger.info(f"Selecionadas {len(top_features)} features usando eliminação recursiva.")
                
            else:  # default 'variance'
                selector = VarianceThreshold(threshold=0.01)
                selector.fit(X_numeric)
                selected_indices = selector.get_support(indices=True)
                top_features = X_numeric.columns[selected_indices].tolist()
                
                self.logger.info(f"Selecionadas {len(top_features)} features usando threshold de variância.")
            
            # Mantém as colunas não numéricas e adiciona as numéricas selecionadas
            non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
            final_columns = non_numeric_cols + top_features
            
            # Retorna DataFrame apenas com as colunas selecionadas + target
            result_df = df[final_columns]
            if target_col and target_col in df.columns:
                result_df[target_col] = df[target_col]
                
            return result_df
            
        except Exception as e:
            self.logger.error(f"Erro na seleção de features: {e}")
            return df  # Retorna o DataFrame sem alterações se houver erro
    
    def _apply_dimensionality_reduction(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Aplica técnicas de redução de dimensionalidade adaptativas.
        
        Args:
            df: DataFrame com todas as features
            target_col: Nome da coluna alvo (opcional)
                
        Returns:
            DataFrame com dimensionalidade reduzida
        """
        if not self.config['dimensionality_reduction'] or df.empty:
            return df
        
        # Seleciona apenas colunas numéricas para redução de dimensionalidade
        X = df.drop(columns=[target_col]) if target_col and target_col in df.columns else df
        numeric_cols = X.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) <= 2:  # Muito poucas colunas, não vale a pena reduzir
            self.logger.info("Muito poucas colunas numéricas. Pulando redução de dimensionalidade.")
            return df
        
        # Prepara os dados
        X_numeric = X[numeric_cols]
        X_numeric_filled = X_numeric.fillna(X_numeric.median())
        
        # Salva informações não numéricas
        non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
        non_numeric_data = X[non_numeric_cols] if non_numeric_cols else None
        
        # Método de redução
        method = self.config['dimensionality_reduction']
        self.logger.info(f"Aplicando redução de dimensionalidade com método: {method}")
        
        try:
            if method == 'pca':
                # PCA adaptativo - define número de componentes com base na variância explicada
                from sklearn.decomposition import PCA
                
                # Fazer uma PCA preliminar para análise de variância explicada
                pca_preliminary = PCA()
                pca_preliminary.fit(X_numeric_filled)
                
                # Calcula variância explicada cumulativa
                cumulative_variance = np.cumsum(pca_preliminary.explained_variance_ratio_)
                
                # Determina número de componentes para explicar 95% da variância
                n_components = np.argmax(cumulative_variance >= 0.95) + 1
                
                # Garante que temos pelo menos 2 componentes e no máximo o número original de features
                n_components = max(2, min(n_components, len(numeric_cols) - 1))
                
                self.logger.info(f"PCA adaptativo: usando {n_components} componentes para explicar 95% da variância.")
                
                # Aplica PCA com o número determinado de componentes
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_numeric_filled)
                
                # Cria DataFrame com os componentes principais
                pca_df = pd.DataFrame(
                    X_pca, 
                    columns=[f'pca_component_{i+1}' for i in range(n_components)],
                    index=X_numeric.index
                )
                
                # Combina com dados não numéricos, se existirem
                if non_numeric_data is not None:
                    result_df = pd.concat([pca_df, non_numeric_data], axis=1)
                else:
                    result_df = pca_df
                    
            elif method == 'tsne':
                # t-SNE para visualização
                from sklearn.manifold import TSNE
                
                # t-SNE é melhor para visualização, então limitamos a 2-3 componentes
                n_components = min(3, len(numeric_cols) - 1)
                tsne = TSNE(n_components=n_components, random_state=42)
                X_tsne = tsne.fit_transform(X_numeric_filled)
                
                # Cria DataFrame com as componentes t-SNE
                tsne_df = pd.DataFrame(
                    X_tsne, 
                    columns=[f'tsne_component_{i+1}' for i in range(n_components)],
                    index=X_numeric.index
                )
                
                # Combina com dados não numéricos, se existirem
                if non_numeric_data is not None:
                    result_df = pd.concat([tsne_df, non_numeric_data], axis=1)
                else:
                    result_df = tsne_df
                    
            elif method == 'umap':
                # UMAP (se disponível) - melhor preservação de estrutura
                try:
                    import umap
                    
                    # UMAP com configuração adaptativa
                    n_components = min(5, len(numeric_cols) - 1)
                    reducer = umap.UMAP(n_components=n_components, random_state=42)
                    X_umap = reducer.fit_transform(X_numeric_filled)
                    
                    # Cria DataFrame com as componentes UMAP
                    umap_df = pd.DataFrame(
                        X_umap, 
                        columns=[f'umap_component_{i+1}' for i in range(n_components)],
                        index=X_numeric.index
                    )
                    
                    # Combina com dados não numéricos, se existirem
                    if non_numeric_data is not None:
                        result_df = pd.concat([umap_df, non_numeric_data], axis=1)
                    else:
                        result_df = umap_df
                except ImportError:
                    self.logger.warning("Biblioteca UMAP não disponível. Usando PCA como fallback.")
                    # Fallback para PCA se UMAP não estiver disponível
                    pca = PCA(n_components=min(5, len(numeric_cols) - 1))
                    X_pca = pca.fit_transform(X_numeric_filled)
                    
                    # Cria DataFrame com os componentes principais
                    pca_df = pd.DataFrame(
                        X_pca, 
                        columns=[f'pca_component_{i+1}' for i in range(min(5, len(numeric_cols) - 1))],
                        index=X_numeric.index
                    )
                    
                    # Combina com dados não numéricos, se existirem
                    if non_numeric_data is not None:
                        result_df = pd.concat([pca_df, non_numeric_data], axis=1)
                    else:
                        result_df = pca_df
            else:
                # Método não reconhecido, retorna o DataFrame original
                self.logger.warning(f"Método de redução de dimensionalidade '{method}' não reconhecido.")
                return df
                
            # Adiciona o target, se existir
            if target_col and target_col in df.columns:
                result_df[target_col] = df[target_col]
                
            # Reporta resultados
            self.logger.info(f"Redução de dimensionalidade: de {len(numeric_cols)} para {len(result_df.columns) - len(non_numeric_cols) - (1 if target_col in df.columns else 0)} features numéricas")
            return result_df
        except Exception as e:
            self.logger.error(f"Erro na redução de dimensionalidade: {e}")
            return df  # Retorna o DataFrame sem alterações se houver erro
    
    def _balance_dataset(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Aplica técnicas de balanceamento para lidar com classes desbalanceadas.
        
        Args:
            df: DataFrame com os dados
            target_col: Nome da coluna alvo (obrigatório)
                
        Returns:
            DataFrame balanceado
        """
        if not self.config.get('balance_classes', False) or target_col is None or target_col not in df.columns:
            return df
        
        # Verifica se estamos tratando de um problema de classificação
        y = df[target_col]
        if not (pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y) or y.nunique() <= 10):
            self.logger.info("Target não parece ser categórico. Pulando balanceamento de classes.")
            return df
        
        # Obtém a contagem de classes
        class_counts = y.value_counts()
        self.logger.info(f"Distribuição de classes original: {class_counts.to_dict()}")
        
        # Verifica se há desbalanceamento significativo
        min_class_ratio = class_counts.min() / class_counts.max()
        if min_class_ratio >= 0.2:  # Se a classe minoritária for pelo menos 20% da majoritária
            self.logger.info(f"Dataset relativamente balanceado (ratio: {min_class_ratio:.2f}). Pulando balanceamento.")
            return df
        
        # Método de balanceamento
        balance_method = self.config.get('balance_method', 'smote')
        self.logger.info(f"Aplicando balanceamento de classes com método: {balance_method}")
        
        # Separar features e target
        X = df.drop(columns=[target_col])
        
        try:
            # Verifica se há colunas categóricas
            cat_cols = X.select_dtypes(include=['object', 'category']).columns
            
            if balance_method == 'smote':
                try:
                    # Codifica temporariamente variáveis categóricas para usar SMOTE
                    X_encoded = X.copy()
                    encoders = {}
                    
                    for col in cat_cols:
                        from sklearn.preprocessing import OrdinalEncoder
                        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                        X_encoded[col] = encoder.fit_transform(X_encoded[[col]])
                        encoders[col] = encoder
                    
                    # Importa e aplica SMOTE
                    from imblearn.over_sampling import SMOTE
                    smote = SMOTE(random_state=42, k_neighbors=min(5, class_counts.min() - 1))
                    X_resampled, y_resampled = smote.fit_resample(X_encoded, y)
                    
                    # Restaura codificação original das variáveis categóricas
                    for col in cat_cols:
                        X_resampled[col] = encoders[col].inverse_transform(X_resampled[[col]])
                    
                except (ImportError, ValueError) as e:
                    self.logger.warning(f"Erro ao aplicar SMOTE: {e}. Usando RandomOverSampler como fallback.")
                    # Fallback para RandomOverSampler se SMOTE falhar
                    from imblearn.over_sampling import RandomOverSampler
                    ros = RandomOverSampler(random_state=42)
                    X_resampled, y_resampled = ros.fit_resample(X, y)
            
            elif balance_method == 'adasyn':
                try:
                    # Codifica temporariamente variáveis categóricas para usar ADASYN
                    X_encoded = X.copy()
                    encoders = {}
                    
                    for col in cat_cols:
                        from sklearn.preprocessing import OrdinalEncoder
                        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                        X_encoded[col] = encoder.fit_transform(X_encoded[[col]])
                        encoders[col] = encoder
                    
                    # Importa e aplica ADASYN
                    from imblearn.over_sampling import ADASYN
                    adasyn = ADASYN(random_state=42, n_neighbors=min(5, class_counts.min() - 1))
                    X_resampled, y_resampled = adasyn.fit_resample(X_encoded, y)
                    
                    # Restaura codificação original das variáveis categóricas
                    for col in cat_cols:
                        X_resampled[col] = encoders[col].inverse_transform(X_resampled[[col]])
                        
                except (ImportError, ValueError) as e:
                    self.logger.warning(f"Erro ao aplicar ADASYN: {e}. Usando RandomOverSampler como fallback.")
                    # Fallback para RandomOverSampler
                    from imblearn.over_sampling import RandomOverSampler
                    ros = RandomOverSampler(random_state=42)
                    X_resampled, y_resampled = ros.fit_resample(X, y)
            
            elif balance_method == 'nearmiss':
                try:
                    # Importa e aplica NearMiss
                    from imblearn.under_sampling import NearMiss
                    nearmiss = NearMiss(version=3)
                    X_resampled, y_resampled = nearmiss.fit_resample(X, y)
                    
                except (ImportError, ValueError) as e:
                    self.logger.warning(f"Erro ao aplicar NearMiss: {e}. Usando RandomUnderSampler como fallback.")
                    # Fallback para RandomUnderSampler
                    from imblearn.under_sampling import RandomUnderSampler
                    rus = RandomUnderSampler(random_state=42)
                    X_resampled, y_resampled = rus.fit_resample(X, y)
            
            elif balance_method == 'combined':
                try:
                    # Combina over e under sampling
                    from imblearn.combine import SMOTETomek
                    smotetomek = SMOTETomek(random_state=42)
                    X_resampled, y_resampled = smotetomek.fit_resample(X, y)
                    
                except (ImportError, ValueError) as e:
                    self.logger.warning(f"Erro ao aplicar SMOTETomek: {e}. Usando balanceamento simples como fallback.")
                    # Fallback para balanceamento simples
                    from imblearn.over_sampling import RandomOverSampler
                    ros = RandomOverSampler(random_state=42)
                    X_resampled, y_resampled = ros.fit_resample(X, y)
            
            else:
                # Método não reconhecido, retorna o DataFrame original
                self.logger.warning(f"Método de balanceamento '{balance_method}' não reconhecido.")
                return df
            
            # Criar DataFrame balanceado
            result_df = pd.DataFrame(X_resampled, columns=X.columns)
            result_df[target_col] = y_resampled
            
            # Reporta resultados
            new_class_counts = result_df[target_col].value_counts()
            self.logger.info(f"Distribuição de classes após balanceamento: {new_class_counts.to_dict()}")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Erro no balanceamento de classes: {e}")
            return df  # Retorna o DataFrame sem alterações se houver erro
    
    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> 'PreProcessor':
        """
        Ajusta o preprocessador aos dados.
        
        Args:
            df: DataFrame com os dados
            target_col: Nome da coluna alvo (opcional)
                
        Returns:
            Self para permitir encadeamento
        """
        self.logger.info(f"Iniciando ajuste do preprocessador. Dimensões do DataFrame: {df.shape}")
        
        # Salva a coluna target para uso posterior
        self.target_col = target_col
        
        # Cria uma cópia para não modificar o DataFrame original
        df_copy = df.copy()
        
        # Separar target (se existir)
        target_data = None
        if target_col and target_col in df_copy.columns:
            target_data = df_copy[target_col].copy()
            df_copy = df_copy.drop(columns=[target_col])
        
        # 1. Limpa os dados
        # Trata valores ausentes
        df_copy = self._handle_missing_values(df_copy)
        
        # Remove outliers
        if self.config.get('outlier_method'):
            df_copy = self._remove_outliers(df_copy)
        
        # 2. Identifica tipos de colunas
        self.column_types = self._identify_column_types(df_copy)
        self.logger.info(f"Tipos de colunas identificados: {len(self.column_types['numeric'])} numéricas, {len(self.column_types['categorical'])} categóricas")
        
        # 3. Remove features altamente correlacionadas
        if self.config.get('remove_high_correlation', True):
            df_copy = self._remove_high_correlation(df_copy)
        
        # 4. Gera novas features
        if self.config.get('generate_features', True):
            df_copy = self._generate_features(df_copy, target_col)
        
        # 5. Seleção de features
        if self.config.get('feature_selection'):
            # Reúne target e features para seleção baseada no target
            if target_data is not None:
                temp_df = df_copy.copy()
                temp_df[target_col] = target_data
                df_copy = self._select_best_features(temp_df, target_col)
                
                # Remove o target novamente
                if target_col in df_copy.columns:
                    df_copy = df_copy.drop(columns=[target_col])
            else:
                df_copy = self._select_best_features(df_copy)
        
        # 6. Atualiza lista de tipos de colunas após transformações
        self.column_types = self._identify_column_types(df_copy)
        
        # 7. Cria pipeline de transformação
        
        # Pipeline para features numéricas
        numeric_steps = []
        
        # Imputação de valores ausentes
        if self.config.get('missing_values_strategy') == 'knn':
            numeric_steps.append(('imputer', KNNImputer(n_neighbors=5)))
        elif self.config.get('missing_values_strategy') == 'iterative':
            try:
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer
                numeric_steps.append(('imputer', IterativeImputer(max_iter=10, random_state=42)))
            except ImportError:
                numeric_steps.append(('imputer', SimpleImputer(strategy='median')))
        else:
            numeric_steps.append(('imputer', SimpleImputer(strategy=self.config.get('missing_values_strategy', 'median'))))
        
        # Scaling
        scaling = self.config.get('scaling', 'standard')
        if scaling == 'standard':
            numeric_steps.append(('scaler', StandardScaler()))
        elif scaling == 'minmax':
            numeric_steps.append(('scaler', MinMaxScaler()))
        elif scaling == 'robust':
            numeric_steps.append(('scaler', RobustScaler()))
        
        # Pipeline para features categóricas
        categorical_steps = []
        
        # Imputação para categóricos
        categorical_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        
        # Encoding
        categorical_strategy = self.config.get('categorical_strategy', 'onehot')
        if categorical_strategy == 'onehot':
            categorical_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
        else:  # ordinal
            categorical_steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
        
        # Cria transformadores para cada tipo de coluna
        transformers = []
        
        if self.column_types['numeric']:
            num_pipeline = Pipeline(steps=numeric_steps)
            transformers.append(('num', num_pipeline, self.column_types['numeric']))
        
        if self.column_types['categorical']:
            cat_pipeline = Pipeline(steps=categorical_steps)
            transformers.append(('cat', cat_pipeline, self.column_types['categorical']))
        
        # Cria ColumnTransformer
        self.preprocessor = ColumnTransformer(transformers, remainder='passthrough')
        
        # 8. Redução de dimensionalidade
        if self.config.get('dimensionality_reduction'):
            if self.config['dimensionality_reduction'] == 'pca':
                # Determina número adequado de componentes
                n_samples, n_features = df_copy.shape
                max_components = min(n_samples, n_features, self.config.get('max_pca_components', 10))
                
                if max_components >= 2:
                    pca = PCA(n_components=max_components)
                    self.preprocessor = Pipeline([
                        ('preprocessor', self.preprocessor),
                        ('pca', pca)
                    ])
                else:
                    self.logger.warning("Poucas features/amostras para PCA. Ignorando redução de dimensionalidade.")
        
        # 9. Ajusta o preprocessador aos dados
        try:
            self.preprocessor.fit(df_copy)
            
            # Armazena nomes das features pós-processamento
            self.feature_names = df_copy.columns.tolist()
            
            # Tenta obter nomes das features transformadas, se disponível
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                self.transformed_feature_names = self.preprocessor.get_feature_names_out()
            else:
                # Caso contrário, cria nomes genéricos
                self.transformed_feature_names = [f"feature_{i}" for i in range(len(self.feature_names))]
            
            self.fitted = True
            self.logger.info(f"Preprocessador ajustado com sucesso. Features de entrada: {len(self.feature_names)}, Features transformadas: {len(self.transformed_feature_names)}")
            
            return self
        except Exception as e:
            self.logger.error(f"Erro ao ajustar o preprocessador: {e}")
            raise

    def transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Aplica as transformações aprendidas a um conjunto de dados.
        
        Args:
            df: DataFrame a ser transformado
            target_col: Nome da coluna alvo (opcional)
                
        Returns:
            DataFrame transformado
        """
        if not self.fitted:
            raise ValueError("O preprocessador precisa ser ajustado antes de transformar dados. Use .fit() primeiro.")

        self.logger.info(f"Transformando dados. Dimensões do DataFrame de entrada: {df.shape}")
        
        # Cria uma cópia para evitar modificar o DataFrame original
        df_copy = df.copy()
        
        # Salva a coluna alvo separadamente, se existir
        target_data = None
        if target_col and target_col in df_copy.columns:
            target_data = df_copy[target_col].copy()
            df_copy = df_copy.drop(columns=[target_col])
        
        # 1. Limpa os dados
        # Trata valores ausentes
        df_copy = self._handle_missing_values(df_copy)
        
        # Remove outliers (se configurado)
        if self.config.get('outlier_method') and not self.config.get('skip_outlier_in_transform', False):
            df_copy = self._remove_outliers(df_copy)
        
        # 2. Gera novas features (se configurado)
        if self.config.get('generate_features', True) and not self.config.get('skip_feature_generation_in_transform', False):
            df_copy = self._generate_features(df_copy, target_col)
        
        # 3. Verifica features faltantes
        missing_cols = set(self.feature_names) - set(df_copy.columns)
        if missing_cols:
            self.logger.warning(f"Colunas ausentes na transformação: {missing_cols}. Adicionando com valores padrão.")
            for col in missing_cols:
                df_copy[col] = 0
        
        # 4. Remove colunas extras
        extra_cols = set(df_copy.columns) - set(self.feature_names)
        if extra_cols:
            self.logger.warning(f"Colunas extras encontradas: {extra_cols}. Removendo.")
            df_copy = df_copy.drop(columns=list(extra_cols), errors='ignore')
        
        # 5. Garante a mesma ordem das colunas usada no fit
        df_copy = df_copy[self.feature_names]
        
        # 6. Aplica o preprocessador
        try:
            df_transformed = self.preprocessor.transform(df_copy)
            
            # Converte para DataFrame
            if hasattr(self, 'transformed_feature_names'):
                result_df = pd.DataFrame(df_transformed, index=df_copy.index, columns=self.transformed_feature_names)
            else:
                result_df = pd.DataFrame(df_transformed, index=df_copy.index, columns=[f"feature_{i}" for i in range(df_transformed.shape[1])])
            
            # Adiciona a coluna alvo, se existir
            if target_data is not None:
                # Filtra target_data para manter apenas índices presentes no result_df
                common_indices = result_df.index.intersection(target_data.index)
                
                if len(common_indices) < len(result_df):
                    self.logger.warning(f"Alguns índices foram perdidos durante a transformação: {len(target_data)} → {len(common_indices)}")
                
                # Adiciona a coluna alvo usando apenas os índices comuns
                result_df = result_df.loc[common_indices]
                result_df[target_col] = target_data.loc[common_indices]
            
            self.logger.info(f"Transformação concluída. Dimensões do DataFrame resultante: {result_df.shape}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Erro ao aplicar transformações: {e}")
            raise
    
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
    def __init__(self, heuristic: Callable[[pd.DataFrame, Optional[str]], float]):
        self.heuristic = heuristic
    
    def search(self, tree: TransformationTree, target_col: Optional[str] = None) -> str:
        """Executa uma busca heurística na árvore de transformações."""
        best_nodes = tree.get_best_transformations(lambda data: self.heuristic(data, target_col))
        best_node = best_nodes[0] if best_nodes else None
        logger.info(f"Melhor transformação encontrada: {best_node}")
        return best_node
    
    @staticmethod
    def advanced_heuristic(df: pd.DataFrame, target_col: Optional[str] = None) -> float:
        """
        Heurística avançada que considera múltiplos fatores:
        - Correlação entre features
        - Diversidade categórica
        - Relação com o target (se disponível)
        - Balanceamento de classes (para classificação)
        """
        if df.empty:
            return float('-inf')  # Penalização máxima para DataFrames vazios
        
        # Inicializa componentes do score
        correlation_penalty = 0
        categorical_diversity_score = 0
        target_relation_score = 0
        class_balance_score = 0
        feature_count_penalty = 0
        missing_values_penalty = 0
        
        # 1. Penaliza alta correlação entre features
        numeric_features = df.select_dtypes(include=['number'])
        
        if numeric_features.shape[1] > 1:
            try:
                # Preenche valores ausentes para calcular correlação
                numeric_features_filled = numeric_features.fillna(numeric_features.median())
                correlation_matrix = numeric_features_filled.corr().abs()
                
                # Remove a diagonal
                np.fill_diagonal(correlation_matrix.values, 0)
                
                # Calcula a média das correlações altas (> 0.7)
                high_corr_mask = correlation_matrix > 0.7
                if high_corr_mask.sum().sum() > 0:
                    correlation_penalty = high_corr_mask.sum().sum() / (numeric_features.shape[1] ** 2)
                
            except Exception as e:
                logger.warning(f"Erro ao calcular penalidade de correlação: {e}")
        
        # 2. Avalia diversidade de variáveis categóricas
        categorical_features = df.select_dtypes(include=['object', 'category'])
        if not categorical_features.empty:
            try:
                unique_counts = categorical_features.nunique()
                categorical_diversity_score = unique_counts.mean() / max(1, unique_counts.max())  # Normaliza entre 0 e 1
                
                # Penaliza categorias com muitos valores únicos (potencial high cardinality)
                high_cardinality_penalty = sum(unique_counts > 100) / max(1, len(unique_counts))
                categorical_diversity_score -= high_cardinality_penalty * 0.5
                
            except Exception as e:
                logger.warning(f"Erro ao calcular score de diversidade categórica: {e}")
        
        # 3. Avalia relação com o target (se disponível)
        if target_col and target_col in df.columns:
            y = df[target_col]
            X = df.drop(columns=[target_col])
            
            # Calcula diferentes métricas dependendo do tipo de target
            if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:  # Regressão
                try:
                    # Para regressão, calcula correlação com o target
                    numeric_X = X.select_dtypes(include=['number'])
                    if not numeric_X.empty:
                        # Preenche valores ausentes para calcular correlação
                        numeric_X_filled = numeric_X.fillna(numeric_X.median())
                        y_filled = y.fillna(y.median())
                        
                        correlations = [abs(np.corrcoef(numeric_X_filled[col], y_filled)[0, 1]) 
                                        for col in numeric_X_filled.columns]
                        target_relation_score = sum(abs(c) > 0.1 for c in correlations) / len(correlations)
                except Exception as e:
                    logger.warning(f"Erro ao calcular score de relação com target (regressão): {e}")
                    
            else:  # Classificação
                try:
                    # Para classificação, avalia balanceamento de classes
                    class_counts = y.value_counts(normalize=True)
                    # Penaliza classes muito desbalanceadas
                    class_balance_score = 1 - (class_counts.max() - class_counts.min())
                    
                    # Calcula entropia da distribuição de classes (mais alto = melhor balanceamento)
                    from scipy.stats import entropy
                    class_balance_score = entropy(class_counts) / np.log(len(class_counts))
                    
                    # Avalia poder preditivo das features (ratio-based)
                    categorical_X = X.select_dtypes(include=['object', 'category'])
                    if not categorical_X.empty:
                        target_relation_scores = []
                        for col in categorical_X.columns[:5]:  # Limita a 5 colunas para eficiência
                            # Calcula target encoding e verifica variância
                            target_means = df.groupby(col)[target_col].mean()
                            target_relation_scores.append(target_means.var())
                        if target_relation_scores:
                            target_relation_score = sum(s > 0.01 for s in target_relation_scores) / len(target_relation_scores)
                except Exception as e:
                    logger.warning(f"Erro ao calcular score de relação com target (classificação): {e}")
        
        # 4. Penaliza conjuntos com muitas features (para evitar overfitting)
        feature_count = df.shape[1]
        feature_count_penalty = max(0, (feature_count - 20) / 100)  # Começa a penalizar acima de 20 features
        
        # 5. Penaliza valores ausentes
        missing_ratio = df.isna().mean().mean()
        missing_values_penalty = missing_ratio * 0.5  # Penaliza até 0.5 pontos por valores ausentes
        
        # Combina todos os scores com pesos
        final_score = (
            -0.3 * correlation_penalty +            # Penaliza correlação alta
            0.2 * categorical_diversity_score +     # Recompensa diversidade categórica
            0.3 * target_relation_score +           # Recompensa relação com target
            0.2 * class_balance_score -             # Recompensa balanceamento de classes
            0.1 * feature_count_penalty -           # Penaliza muitas features
            0.1 * missing_values_penalty            # Penaliza valores ausentes
        )
        
        logger.debug(f"Scores da heurística: corr_penalty={correlation_penalty:.3f}, "
                   f"cat_diversity={categorical_diversity_score:.3f}, "
                   f"target_relation={target_relation_score:.3f}, "
                   f"class_balance={class_balance_score:.3f}, "
                   f"final_score={final_score:.3f}")
        
        return final_score

class Explorer:
    def __init__(self, heuristic: Callable[[pd.DataFrame], float] = None, target_col: Optional[str] = None):
        self.tree = TransformationTree()
        self.search = HeuristicSearch(heuristic or HeuristicSearch.advanced_heuristic)
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
