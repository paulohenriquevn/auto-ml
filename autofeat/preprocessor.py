import os
import joblib
import logging
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from typing import Dict, Optional, Union
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures


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
            'correlation_threshold': 0.95,
            'balance_classes': False,
            'balance_method': 'smote',  # 'smote', 'adasyn', 'random_over', 'random_under', 'tomek', 'nearmiss', 'smoteenn', 'smotetomek'
            'sampling_strategy': 'auto',  # 'auto', float (proporção minoritária/majoritária), ou dicionário
            'use_sample_weights': False   # Se deve usar pesos de amostra em vez de balanceamento
        }
        if config:
            self.config.update(config)
        
        self.preprocessor = None
        self.column_types = {}
        self.fitted = False
        self.feature_names = []
        self.target_col = None
        self.sample_weights = None
        
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
        Preserva exemplos da classe minoritária em problemas de classificação desbalanceados.
        """
        method = self.config['outlier_method']
        if method is None or df.empty:
            self.logger.info("Remoção de outliers desativada ou DataFrame vazio. Pulando esta etapa.")
            return df

        # Verificar se temos um problema de classificação desbalanceado
        preserve_minority = False
        minority_indices = []
        
        if self.target_col and self.target_col in df.columns:
            y = df[self.target_col]
            # Verificar se parece ser classificação (categórico ou poucos valores únicos)
            is_classification = pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y) or y.nunique() <= 10
            
            if is_classification and y.nunique() >= 2:
                # Calcular balanceamento das classes
                class_counts = y.value_counts()
                min_class_ratio = class_counts.min() / class_counts.max()
                
                # Se fortemente desbalanceado (menos de 10% da classe majoritária)
                if min_class_ratio < 0.1:
                    preserve_minority = True
                    minority_class = class_counts.idxmin()
                    minority_indices = df[df[self.target_col] == minority_class].index
                    self.logger.info(f"Detectado problema de classificação desbalanceado. "
                                    f"Preservando {len(minority_indices)} exemplos da classe minoritária.")

        # Seleciona apenas colunas numéricas para detecção de outliers
        numeric_df = df.select_dtypes(include=['number']).columns.tolist()
        
        # Remove a coluna target da análise de outliers se for numérica
        if self.target_col in numeric_df:
            numeric_df.remove(self.target_col)
        
        if not numeric_df:
            self.logger.warning("Nenhuma coluna numérica encontrada para detecção de outliers. Pulando esta etapa.")
            return df
        
        # Usa apenas as colunas numéricas para detecção
        numeric_data = df[numeric_df]
        
        # Configuração adaptativa para limites de detecção
        # Mais conservador para datasets menores ou com alta porcentagem de valores atípicos
        samples_threshold = 10000
        conservative = df.shape[0] < samples_threshold
        
        if method == 'zscore':
            # Z-score adaptativo (limite mais alto para datasets pequenos)
            zscore_threshold = 4.0 if conservative else 3.0
            
            # Preenche valores ausentes para cálculo do z-score
            numeric_filled = numeric_data.fillna(numeric_data.median())
            
            # Calcula z-scores
            z_scores = np.abs(stats.zscore(numeric_filled, nan_policy='omit'))
            
            # Identifica inliers
            keep_mask = (z_scores < zscore_threshold).all(axis=1)
            
        elif method == 'iqr':
            # Calcula quartis
            Q1 = numeric_data.quantile(0.25)
            Q3 = numeric_data.quantile(0.75)
            IQR = Q3 - Q1
            
            # Limites mais conservadores para datasets pequenos
            iqr_factor = 3.0 if conservative else 1.5
            
            # Cria máscara para valores dentro do intervalo aceitável
            keep_mask = ~((numeric_data < (Q1 - iqr_factor * IQR)) | 
                        (numeric_data > (Q3 + iqr_factor * IQR))).any(axis=1)
            
        elif method == 'isolation_forest':
            # Ajusta contaminação com base no tamanho do dataset
            # Menor contaminação para datasets pequenos
            contamination = 0.01 if conservative else 0.05
            
            # Preenche valores ausentes
            numeric_filled = numeric_data.fillna(numeric_data.median())
            
            # Aplica isolation forest
            clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
            outliers = clf.fit_predict(numeric_filled)
            
            # -1 são outliers, 1 são inliers
            keep_mask = outliers == 1
            
        else:
            self.logger.warning(f"Método de detecção de outliers '{method}' não reconhecido. Mantendo todos os dados.")
            return df
        
        # Se preservando classes minoritárias, inclui seus índices
        if preserve_minority and len(minority_indices) > 0:
            keep_mask = keep_mask | df.index.isin(minority_indices)
        
        # Aplica a máscara ao DataFrame
        filtered_df = df[keep_mask]
        
        # Verifica se não removeu quase todos os dados
        if len(filtered_df) < len(df) * 0.5:
            self.logger.warning(f"Mais de 50% dos dados seriam removidos como outliers. "
                                f"Usando limites mais conservadores.")
            # Ajusta para uma abordagem mais conservadora
            return self._remove_outliers_conservative(df, preserve_minority, minority_indices)
        
        # Verifica se não removeu todos os dados ou quase todos
        if filtered_df.empty or len(filtered_df) < 10:
            self.logger.warning("Remoção de outliers removeria todos ou quase todos os dados. "
                                "Retornando DataFrame original.")
            return df
        
        # Se preservando classes minoritárias, verifica se todas foram mantidas
        if preserve_minority:
            preserved_count = sum(filtered_df.index.isin(minority_indices))
            if preserved_count < len(minority_indices):
                self.logger.warning(f"Alguns exemplos da classe minoritária foram perdidos "
                                f"({preserved_count}/{len(minority_indices)} preservados). "
                                f"Ajustando limites para preservar todos.")
                # Inclui todos os exemplos da classe minoritária
                missing_indices = [idx for idx in minority_indices if idx not in filtered_df.index]
                filtered_df = pd.concat([filtered_df, df.loc[missing_indices]])
        
        self.logger.info(f"Foram removidos {len(df) - len(filtered_df)} outliers "
                        f"({(len(df) - len(filtered_df)) / len(df) * 100:.2f}% dos dados)")
        return filtered_df

    def _remove_outliers_conservative(self, df: pd.DataFrame, preserve_minority: bool = False, 
                                    minority_indices: list = None) -> pd.DataFrame:
        """
        Versão mais conservadora da remoção de outliers, para quando a remoção padrão
        for muito agressiva.
        """
        # Seleciona apenas colunas numéricas
        numeric_df = df.select_dtypes(include=['number']).columns.tolist()
        if self.target_col in numeric_df:
            numeric_df.remove(self.target_col)
        
        numeric_data = df[numeric_df]
        
        # Usa limites muito mais conservadores
        Q1 = numeric_data.quantile(0.01)  # 1º percentil em vez de 25º
        Q3 = numeric_data.quantile(0.99)  # 99º percentil em vez de 75º
        IQR = Q3 - Q1
        
        # Limites 5x mais permissivos
        keep_mask = ~((numeric_data < (Q1 - 5.0 * IQR)) | 
                    (numeric_data > (Q3 + 5.0 * IQR))).any(axis=1)
        
        # Preserva classe minoritária
        if preserve_minority and minority_indices:
            keep_mask = keep_mask | df.index.isin(minority_indices)
        
        # Aplica a máscara
        filtered_df = df[keep_mask]
        
        # Se ainda remover muitos dados, usa uma abordagem ainda mais conservadora
        if len(filtered_df) < len(df) * 0.8:
            self.logger.warning("Mesmo com limites conservadores, muitos dados seriam removidos. "
                            "Removendo apenas outliers extremos.")
            
            # Detecta apenas outliers muito extremos (> 10 desvios padrão)
            numeric_filled = numeric_data.fillna(numeric_data.median())
            z_scores = np.abs(stats.zscore(numeric_filled, nan_policy='omit'))
            extreme_mask = (z_scores < 10.0).all(axis=1)
            
            # Preserva classe minoritária
            if preserve_minority and minority_indices:
                extreme_mask = extreme_mask | df.index.isin(minority_indices)
                
            filtered_df = df[extreme_mask]
        
        self.logger.info(f"Com abordagem conservadora, foram removidos {len(df) - len(filtered_df)} outliers "
                        f"({(len(df) - len(filtered_df)) / len(df) * 100:.2f}% dos dados)")
        
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
    
    def _compute_sample_weights(y: pd.Series) -> np.ndarray:
        """
        Calcula pesos de amostra para lidar com classes desbalanceadas.
        
        Args:
            y: Série com a variável alvo
            
        Returns:
            Array com pesos para cada amostra
        """
        try:
            # Calcula pesos de classe para balancear
            classes = np.unique(y)
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
            
            # Mapeia classes para seus pesos
            class_weight_dict = {c: w for c, w in zip(classes, weights)}
            
            # Atribui peso para cada amostra
            sample_weights = np.array([class_weight_dict[c] for c in y])
            
            return sample_weights
        
        except Exception as e:
            logging.error(f"Erro ao calcular pesos de amostra: {e}")
            # Retorna pesos iguais como fallback
            return np.ones(len(y))
    
    def _balance_dataset(self, df: pd.DataFrame, target_col: str, 
                    balance_method: str = 'smote', 
                    sampling_strategy: Union[str, float, Dict] = 'auto',
                    random_state: int = 42) -> pd.DataFrame:
        """
        Equilibra um dataset usando técnicas avançadas para lidar com classes desbalanceadas.
        
        Args:
            df: DataFrame com os dados
            target_col: Nome da coluna alvo
            balance_method: Método de balanceamento ('smote', 'adasyn', 'random_over',
                        'random_under', 'tomek', 'nearmiss', 'smoteenn', 'smotetomek',
                        'borderline_smote', 'svm_smote', 'kmeans_smote')
            sampling_strategy: Estratégia de amostragem ('auto', float, ou dict)
            random_state: Semente aleatória para reprodutibilidade
        
        Returns:
            DataFrame balanceado
        """
        if target_col not in df.columns:
            raise ValueError(f"Coluna target '{target_col}' não encontrada no DataFrame")
        
        # Verifica se estamos lidando com um problema de classificação
        y = df[target_col]
        if not (pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y) or y.nunique() <= 10):
            raise ValueError("Balanceamento só é aplicável para problemas de classificação")

        # Extrair features e target
        X = df.drop(columns=[target_col])
        
        # Verifica desbalanceamento
        class_counts = y.value_counts()
        min_class_ratio = class_counts.min() / class_counts.max()
        
        if min_class_ratio >= 0.8:
            # Dataset já está relativamente balanceado (proporção mínima de 80%)
            self.logger.info("Dataset já está razoavelmente balanceado. Mantendo dados originais.")
            return df
        
        self.logger.info(f"Aplicando método de balanceamento: {balance_method}")
        self.logger.info(f"Distribuição original de classes: {class_counts.to_dict()}")
        
        # Identificar colunas categóricas para codificação temporária
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        X_encoded = X.copy()
        
        # Codificar temporariamente variáveis categóricas
        if len(cat_cols) > 0:
            from sklearn.preprocessing import OrdinalEncoder
            encoders = {}
            for col in cat_cols:
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                X_encoded[col] = encoder.fit_transform(X_encoded[[col]])
                encoders[col] = encoder
        
        try:
            # Import técnicas de balanceamento
            if balance_method in ['smote', 'borderline_smote', 'svm_smote']:
                from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
                if balance_method == 'smote':
                    sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
                elif balance_method == 'borderline_smote':
                    sampler = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=random_state, kind='borderline-1')
                elif balance_method == 'svm_smote':
                    sampler = SVMSMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
            
            elif balance_method == 'kmeans_smote':
                try:
                    from imblearn.over_sampling import KMeansSMOTE
                    sampler = KMeansSMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
                except ImportError:
                    self.logger.warning("KMeansSMOTE não disponível. Usando SMOTE padrão.")
                    from imblearn.over_sampling import SMOTE
                    sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
            
            elif balance_method == 'adasyn':
                from imblearn.over_sampling import ADASYN
                sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
            
            elif balance_method == 'random_over':
                from imblearn.over_sampling import RandomOverSampler
                sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
            
            elif balance_method == 'random_under':
                from imblearn.under_sampling import RandomUnderSampler
                sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
            
            elif balance_method == 'tomek':
                from imblearn.under_sampling import TomekLinks
                sampler = TomekLinks(sampling_strategy=sampling_strategy)
            
            elif balance_method == 'nearmiss':
                from imblearn.under_sampling import NearMiss
                sampler = NearMiss(sampling_strategy=sampling_strategy, version=3)
            
            elif balance_method == 'smoteenn':
                from imblearn.combine import SMOTEENN
                sampler = SMOTEENN(sampling_strategy=sampling_strategy, random_state=random_state)
            
            elif balance_method == 'smotetomek':
                from imblearn.combine import SMOTETomek
                sampler = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state)
            
            # Abordagem híbrida:
            elif balance_method == 'hybrid_mix':
                # Combina oversampling moderado e undersampling seletivo
                from imblearn.over_sampling import SMOTE
                from imblearn.under_sampling import NearMiss
                
                # Primeiro aplicamos SMOTE para aumentar a classe minoritária mas não completamente
                smote_ratio = min(0.5, class_counts.min() / class_counts.max() * 5)  # Aumenta até 5x ou 50% da majoritária
                smote_sampler = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
                X_temp, y_temp = smote_sampler.fit_resample(X_encoded, y)
                
                # Depois aplicamos NearMiss para reduzir a classe majoritária de forma seletiva
                nearmiss_sampler = NearMiss(sampling_strategy=0.7)  # Remove 30% da classe majoritária
                X_resampled, y_resampled = nearmiss_sampler.fit_resample(X_temp, y_temp)
                
                # Não precisamos chamar fit_resample abaixo
                already_resampled = True
            
            elif balance_method == 'self_paced':
                # Implementação de balanceamento auto-adaptativo
                # Aumenta classes minoritárias proporcionalmente ao seu tamanho
                from imblearn.over_sampling import SMOTE, RandomOverSampler
                
                # Identifica todas as classes e calcula uma estratégia personalizada
                classes = np.unique(y)
                max_count = class_counts.max()
                
                # Define estratégia - classes menores recebem mais oversampling
                strategy = {}
                for cls in classes:
                    count = class_counts[cls]
                    if count < max_count * 0.5:  # Para classes realmente minoritárias
                        # Ajuste quadrático - classes muito pequenas crescem mais
                        ratio = 1 - (count / max_count) ** 0.5  # 0 = majoritária, próximo de 1 = muito minoritária
                        target_count = int(count + (max_count * ratio * 0.8))  # No máximo 80% da majoritária
                        strategy[cls] = min(target_count, max_count - 1)  # Nunca ultrapassa a majoritária
                
                # Se apenas uma classe é minoritária, simplifica a abordagem
                if len(strategy) == 1:
                    strategy = list(strategy.values())[0] / max_count
                    sampler = SMOTE(sampling_strategy=strategy, random_state=random_state)
                else:
                    # Usa o dicionário completo para múltiplas classes minoritárias
                    sampler = SMOTE(sampling_strategy=strategy, random_state=random_state)
            
            else:
                raise ValueError(f"Método de balanceamento '{balance_method}' não reconhecido")
            
            # Aplicar técnica de balanceamento
            if not locals().get('already_resampled', False):
                X_resampled, y_resampled = sampler.fit_resample(X_encoded, y)
            
            # Restaurar codificação original das variáveis categóricas
            if len(cat_cols) > 0:
                for col in cat_cols:
                    X_resampled[col] = encoders[col].inverse_transform(X_resampled[[col]])
            
            # Criar DataFrame balanceado
            balanced_df = X_resampled.copy()
            balanced_df[target_col] = y_resampled
            
            # Relatar resultados
            new_class_counts = balanced_df[target_col].value_counts()
            self.logger.info(f"Distribuição de classes após balanceamento: {new_class_counts.to_dict()}")
            self.logger.info(f"Balanceamento alterou o tamanho do dataset: {len(df)} → {len(balanced_df)}")
            
            return balanced_df
        
        except ImportError as e:
            self.logger.warning(f"Erro ao importar bibliotecas para balanceamento: {e}. Instalando imbalanced-learn...")
            try:
                import pip
                pip.main(['install', 'imbalanced-learn'])
                self.logger.info("imbalanced-learn instalado. Tente executar novamente.")
            except:
                self.logger.error("Falha ao instalar imbalanced-learn. Mantendo dados originais.")
            return df
        
        except Exception as e:
            self.logger.error(f"Erro ao aplicar balanceamento: {e}. Mantendo dados originais.")
            return df
        
    def get_sample_weights(self) -> Optional[np.ndarray]:
        """Retorna os pesos das amostras para classes desbalanceadas."""
        return self.sample_weights

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
        
        # 6. Balancear classes se configurado
        if self.config.get('balance_classes', False) and target_col and target_col in df_copy.columns:
            # Se houver target, adicione-o de volta ao df_copy para balanceamento
            if target_data is not None:
                df_copy[target_col] = target_data
            
            # Balancear classes
            df_copy = self._balance_dataset(
                df_copy, 
                target_col=target_col,
                balance_method=self.config.get('balance_method', 'smote'),
                sampling_strategy=self.config.get('sampling_strategy', 'auto'),
                random_state=42
            )
            
            # Remover target novamente
            if target_col in df_copy.columns:
                target_data = df_copy[target_col].copy()
                df_copy = df_copy.drop(columns=[target_col])
                
        # 7 Calcular pesos de amostra se necessário
        if self.config.get('use_sample_weights', False) and target_data is not None:
            self.sample_weights = self._compute_sample_weights(target_data)
    
        # 8. Atualiza lista de tipos de colunas após transformações
        self.column_types = self._identify_column_types(df_copy)
        
        # 8. Cria pipeline de transformação
        
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
        
        # 9. Redução de dimensionalidade
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
        
        # 10. Ajusta o preprocessador aos dados
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
        Versão melhorada com verificações de segurança para preservar exemplos.
        
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
        
        # Salva classes e contagem original para verificação
        target_data = None
        original_class_counts = None
        if target_col and target_col in df_copy.columns:
            target_data = df_copy[target_col].copy()
            # Verifica se é classificação
            is_classification = pd.api.types.is_categorical_dtype(target_data) or pd.api.types.is_object_dtype(target_data) or target_data.nunique() <= 10
            
            if is_classification:
                original_class_counts = target_data.value_counts()
                self.logger.info(f"Distribuição original das classes:\n{original_class_counts}")
                
            # Remove target do DataFrame
            df_copy = df_copy.drop(columns=[target_col])
        
        # 1. Trata valores ausentes
        df_copy = self._handle_missing_values(df_copy)
        
        # 2. Remove outliers (se configurado)
        if self.config.get('outlier_method') and not self.config.get('skip_outlier_in_transform', False):
            # Adiciona temporariamente o target para preservação durante remoção de outliers
            if target_data is not None:
                df_copy[target_col] = target_data
                
            df_copy = self._remove_outliers(df_copy)
            
            # Remove o target novamente se foi adicionado
            if target_data is not None:
                target_data = df_copy[target_col].copy()  # Atualiza target_data para refletir as remoções
                df_copy = df_copy.drop(columns=[target_col])
        
        # 3. Gera novas features (se configurado)
        if self.config.get('generate_features', True) and not self.config.get('skip_feature_generation_in_transform', False):
            # Adiciona temporariamente o target para geração de features informativas
            if target_data is not None:
                df_copy[target_col] = target_data
                
            df_copy = self._generate_features(df_copy, target_col)
            
            # Remove o target novamente se foi adicionado
            if target_data is not None:
                df_copy = df_copy.drop(columns=[target_col], errors='ignore')
        
        # 4. Verifica features faltantes
        missing_cols = set(self.feature_names) - set(df_copy.columns)
        if missing_cols:
            self.logger.warning(f"Colunas ausentes na transformação: {missing_cols}. Adicionando com valores padrão.")
            for col in missing_cols:
                df_copy[col] = 0
        
        # 5. Remove colunas extras
        extra_cols = set(df_copy.columns) - set(self.feature_names)
        if extra_cols:
            self.logger.warning(f"Colunas extras encontradas: {extra_cols}. Removendo.")
            df_copy = df_copy.drop(columns=list(extra_cols), errors='ignore')
        
        # 6. Garante a mesma ordem das colunas usada no fit
        df_copy = df_copy[self.feature_names]
        
        # 7. Aplica o preprocessador
        try:
            df_transformed = self.preprocessor.transform(df_copy)
            
            # Converte para DataFrame
            if hasattr(self, 'transformed_feature_names'):
                result_df = pd.DataFrame(df_transformed, index=df_copy.index, columns=self.transformed_feature_names)
            else:
                result_df = pd.DataFrame(df_transformed, index=df_copy.index, columns=[f"feature_{i}" for i in range(df_transformed.shape[1])])
            
            # 8. Adiciona a coluna alvo, se existir
            if target_data is not None:
                # Filtra target_data para manter apenas índices presentes no result_df
                common_indices = result_df.index.intersection(target_data.index)
                
                if len(common_indices) < len(result_df):
                    self.logger.warning(f"Alguns índices foram perdidos durante a transformação: {len(target_data)} → {len(common_indices)}")
                
                # Adiciona a coluna alvo usando apenas os índices comuns
                result_df = result_df.loc[common_indices]
                result_df[target_col] = target_data.loc[common_indices]
                
                # Verifica se mantivemos exemplos de todas as classes
                if original_class_counts is not None:
                    final_class_counts = result_df[target_col].value_counts()
                    self.logger.info(f"Distribuição final das classes:\n{final_class_counts}")
                    
                    # Verifica se perdemos alguma classe
                    for cls in original_class_counts.index:
                        if cls not in final_class_counts:
                            self.logger.warning(f"ATENÇÃO: A classe {cls} foi completamente removida durante o processamento!")
                        elif final_class_counts[cls] < 5 and original_class_counts[cls] >= 5:
                            self.logger.warning(f"ATENÇÃO: A classe {cls} tem menos de 5 exemplos após processamento!")
            
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
  

def create_preprocessor(config: Optional[Dict] = None) -> PreProcessor:
    return PreProcessor(config)
