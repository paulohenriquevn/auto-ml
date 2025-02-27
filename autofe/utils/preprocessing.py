"""
Módulo PreData para análise preliminar e otimização de datasets.
Fornece métricas e recomendações de pré-processamento sem comprometer a performance do sistema.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns

logger = logging.getLogger(__name__)

class PreProcessing:
    """
    Realiza análise preliminar de dados e fornece recomendações para pré-processamento,
    levando em consideração a performance do dataset original.
    """
    
    def __init__(self, max_analysis_time: float = 5.0, max_rows_analysis: int = 10000):
        """
        Inicializa o módulo PreData.
        
        Args:
            max_analysis_time: Tempo máximo (em segundos) para análise
            max_rows_analysis: Número máximo de linhas a considerar para análises detalhadas
        """
        self.max_analysis_time = max_analysis_time
        self.max_rows_analysis = max_rows_analysis
        self.analysis_results = {}
        self.transformation_recommendations = []
        self.performance_insights = {}
    
    def analyze_dataset(
        self, 
        data: pd.DataFrame, 
        target: Optional[Union[str, pd.Series]] = None,
        dataset_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analisa o dataset e fornece recomendações de pré-processamento.
        
        Args:
            data: DataFrame com os dados a serem analisados
            target: Nome da coluna alvo ou Series com valores alvo (opcional)
            dataset_type: Tipo de dataset ('tabular_classification', 'tabular_regression', etc.)
            
        Returns:
            Dicionário com resultados da análise e recomendações
        """
        start_time = time.time()
        self.analysis_results = {}
        
        # Informações básicas do dataset
        self._analyze_basic_info(data)
        
        # Analisar qualidade dos dados (valores ausentes, cardinalidade, etc.)
        self._analyze_data_quality(data)
        
        # Analisar colunas específicas mais detalhadamente
        analysis_tasks = [
            (self._analyze_numeric_features, [data]),
            (self._analyze_categorical_features, [data]),
            (self._analyze_datetime_features, [data]),
            (self._analyze_correlations, [data])
        ]
        
        # Se há target, analisar relação com features
        if target is not None:
            y = target if isinstance(target, pd.Series) else data[target]
            analysis_tasks.append((self._analyze_target_relationships, [data, y, dataset_type]))
        
        # Executar análises em paralelo com limite de tempo
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(func, *args) for func, args in analysis_tasks]
            
            # Esperar até o tempo máximo ou até todas as análises terminarem
            remaining_time = self.max_analysis_time - (time.time() - start_time)
            if remaining_time > 0:
                for future in futures:
                    try:
                        future.result(timeout=remaining_time / len(futures))
                    except TimeoutError:
                        pass  # Ignorar análises que demoraram demais
        
        # Gerar recomendações com base nas análises
        self._generate_recommendations(data, target, dataset_type)
        
        # Estimar impacto na performance
        self._estimate_performance_impact()
        
        # Compile todos os resultados
        final_results = {
            'execution_time': time.time() - start_time,
            'dataset_info': self.analysis_results.get('basic_info', {}),
            'data_quality': self.analysis_results.get('data_quality', {}),
            'recommendations': self.transformation_recommendations,
            'performance_insights': self.performance_insights
        }
        
        return final_results
    
    def apply_recommended_transformations(
        self, 
        data: pd.DataFrame,
        selected_recommendations: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Aplica as transformações recomendadas ao dataset.
        
        Args:
            data: DataFrame com os dados originais
            selected_recommendations: Lista de IDs das recomendações a aplicar (None = todas)
            
        Returns:
            DataFrame com as transformações aplicadas
        """
        transformed_data = data.copy()
        
        # Se nenhuma recomendação específica foi selecionada, aplicar todas
        all_recs = {rec['id']: rec for rec in self.transformation_recommendations}
        to_apply = selected_recommendations or list(all_recs.keys())
        
        # Ordenar recomendações por prioridade
        recommendations_to_apply = [
            all_recs[rec_id] for rec_id in to_apply 
            if rec_id in all_recs
        ]
        recommendations_to_apply.sort(key=lambda x: x.get('priority', 999))
        
        # Aplicar cada transformação recomendada
        applied_transformations = []
        for rec in recommendations_to_apply:
            try:
                transform_func = getattr(self, f"_apply_{rec['id']}", None)
                if transform_func:
                    transformed_data = transform_func(transformed_data)
                    applied_transformations.append(rec['id'])
            except Exception as e:
                logger.warning(f"Erro ao aplicar transformação {rec['id']}: {str(e)}")
        
        # Atualizar resultados
        self.performance_insights['applied_transformations'] = applied_transformations
        self.performance_insights['rows_before'] = len(data)
        self.performance_insights['columns_before'] = len(data.columns)
        self.performance_insights['rows_after'] = len(transformed_data)
        self.performance_insights['columns_after'] = len(transformed_data.columns)
        
        return transformed_data
    
    def get_report(self, include_visualizations: bool = True) -> Dict[str, Any]:
        """
        Gera um relatório detalhado com os resultados da análise e visualizações.
        
        Args:
            include_visualizations: Se deve incluir visualizações no relatório
            
        Returns:
            Dicionário com o relatório completo
        """
        report = {
            'dataset_summary': self._generate_dataset_summary(),
            'data_quality_summary': self._generate_quality_summary(),
            'recommendations_summary': self._format_recommendations(),
            'performance_impact': self._format_performance_insights()
        }
        
        if include_visualizations:
            report['visualizations'] = self._generate_visualizations()
        
        return report
    
    # Métodos privados para análise
    
    def _analyze_basic_info(self, data: pd.DataFrame) -> None:
        """Analisa informações básicas do dataset."""
        self.analysis_results['basic_info'] = {
            'n_rows': len(data),
            'n_columns': len(data.columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024),
            'column_types': {
                'numeric': len(data.select_dtypes(include=['number']).columns),
                'categorical': len(data.select_dtypes(include=['object', 'category']).columns),
                'datetime': len(data.select_dtypes(include=['datetime']).columns),
                'boolean': len(data.select_dtypes(include=['bool']).columns)
            }
        }
    
    def _analyze_data_quality(self, data: pd.DataFrame) -> None:
        """Analisa a qualidade dos dados (valores ausentes, duplicados, etc.)."""
        # Análise de valores ausentes
        missing_counts = data.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        # Análise de duplicatas
        sample_size = min(len(data), self.max_rows_analysis)
        sample_data = data.sample(sample_size) if len(data) > sample_size else data
        duplicate_rows = sample_data.duplicated().sum()
        duplicate_percentage = (duplicate_rows / sample_size) * 100
        
        # Cardinalidade de colunas categóricas
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        high_cardinality_cols = []
        
        for col in categorical_cols:
            n_unique = data[col].nunique()
            if n_unique > 20 and n_unique / len(data) < 0.5:
                high_cardinality_cols.append({
                    'column': col,
                    'unique_values': n_unique,
                    'percentage': (n_unique / len(data)) * 100
                })
        
        # Colunas com baixa variância
        numeric_cols = data.select_dtypes(include=['number']).columns
        low_variance_cols = []
        
        for col in numeric_cols:
            if len(data[col].unique()) <= 1:
                low_variance_cols.append(col)
        
        # Compilar resultados
        self.analysis_results['data_quality'] = {
            'missing_values': {
                'total_missing': missing_counts.sum(),
                'missing_percentage': (missing_counts.sum() / (len(data) * len(data.columns))) * 100,
                'columns_with_missing': [
                    {'column': col, 'count': count, 'percentage': (count / len(data)) * 100}
                    for col, count in missing_cols.items()
                ]
            },
            'duplicates': {
                'duplicate_rows': duplicate_rows,
                'duplicate_percentage': duplicate_percentage,
                'sample_based': len(data) > sample_size
            },
            'high_cardinality': high_cardinality_cols,
            'low_variance_columns': low_variance_cols
        }
    
    def _analyze_numeric_features(self, data: pd.DataFrame) -> None:
        """Analisa features numéricas (estatísticas, distribuição, outliers)."""
        numeric_cols = data.select_dtypes(include=['number']).columns
        
        if not len(numeric_cols):
            return
        
        numeric_stats = {}
        potential_outliers = {}
        
        for col in numeric_cols:
            # Estatísticas básicas
            stats = data[col].describe()
            
            # Detectar outliers (IQR method)
            q1 = stats['25%']
            q3 = stats['75%']
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            outliers_percentage = (outliers_count / len(data)) * 100
            
            if outliers_percentage > 5:
                potential_outliers[col] = {
                    'count': outliers_count,
                    'percentage': outliers_percentage,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
            
            # Calcular assimetria (skewness)
            try:
                skewness = data[col].skew()
                skewed = abs(skewness) > 1
            except:
                skewness = None
                skewed = False
            
            numeric_stats[col] = {
                'min': stats['min'],
                'max': stats['max'],
                'mean': stats['mean'],
                'median': stats['50%'],
                'std': stats['std'],
                'skewness': skewness,
                'highly_skewed': skewed
            }
        
        self.analysis_results['numeric_features'] = {
            'stats': numeric_stats,
            'potential_outliers': potential_outliers
        }
    
    def _analyze_categorical_features(self, data: pd.DataFrame) -> None:
        """Analisa features categóricas (frequências, distribuição)."""
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        if not len(categorical_cols):
            return
        
        categorical_stats = {}
        imbalanced_categories = {}
        
        for col in categorical_cols:
            # Contar valores e calcular percentagens
            value_counts = data[col].value_counts(normalize=True)
            
            # Verificar desbalanceamento (se uma categoria representa mais de 90%)
            if value_counts.iloc[0] > 0.9:
                imbalanced_categories[col] = {
                    'dominant_value': value_counts.index[0],
                    'dominant_percentage': value_counts.iloc[0] * 100
                }
            
            categorical_stats[col] = {
                'unique_values': len(value_counts),
                'top_value': value_counts.index[0],
                'top_percentage': value_counts.iloc[0] * 100,
                'top_categories': value_counts.head(5).to_dict()
            }
        
        self.analysis_results['categorical_features'] = {
            'stats': categorical_stats,
            'imbalanced_categories': imbalanced_categories
        }
    
    def _analyze_datetime_features(self, data: pd.DataFrame) -> None:
        """Analisa features de data/hora (intervalo, frequência, gaps)."""
        datetime_cols = data.select_dtypes(include=['datetime']).columns
        
        # Também verificar colunas que podem ser datas mas estão como strings
        for col in data.select_dtypes(include=['object']).columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'dia' in col.lower():
                try:
                    pd.to_datetime(data[col].iloc[0:100])  # Testar com amostra
                    datetime_cols = datetime_cols.append(pd.Index([col]))
                except:
                    pass
        
        if not len(datetime_cols):
            return
        
        datetime_stats = {}
        
        for col in datetime_cols:
            # Converter para datetime se não for
            if not pd.api.types.is_datetime64_any_dtype(data[col]):
                try:
                    date_series = pd.to_datetime(data[col])
                except:
                    continue
            else:
                date_series = data[col]
            
            # Estatísticas da série temporal
            try:
                min_date = date_series.min()
                max_date = date_series.max()
                time_span = max_date - min_date
                time_span_days = time_span.total_seconds() / (24 * 3600)
                
                # Verificar frequência
                date_diff = date_series.sort_values().diff().dropna()
                
                if len(date_diff):
                    most_common_diff = date_diff.value_counts().index[0]
                    most_common_diff_seconds = most_common_diff.total_seconds()
                    
                    # Determinar frequência aproximada
                    if most_common_diff_seconds < 60:
                        freq = 'Por segundo'
                    elif most_common_diff_seconds < 3600:
                        freq = 'Por minuto'
                    elif most_common_diff_seconds < 24 * 3600:
                        freq = 'Por hora'
                    elif most_common_diff_seconds < 7 * 24 * 3600:
                        freq = 'Diária'
                    elif most_common_diff_seconds < 31 * 24 * 3600:
                        freq = 'Semanal'
                    elif most_common_diff_seconds < 365 * 24 * 3600:
                        freq = 'Mensal'
                    else:
                        freq = 'Anual'
                else:
                    freq = 'Desconhecida'
                
                datetime_stats[col] = {
                    'min_date': min_date,
                    'max_date': max_date,
                    'time_span_days': time_span_days,
                    'approximate_frequency': freq
                }
            except Exception as e:
                logger.warning(f"Erro ao analisar coluna de data {col}: {str(e)}")
        
        self.analysis_results['datetime_features'] = datetime_stats
    
    def _analyze_correlations(self, data: pd.DataFrame) -> None:
        """Analisa correlações entre features numéricas."""
        numeric_cols = data.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 2:
            return
        
        # Selecionar amostra para correlação se dataset for grande
        sample_size = min(len(data), self.max_rows_analysis)
        sample_data = data[numeric_cols].sample(sample_size) if len(data) > sample_size else data[numeric_cols]
        
        try:
            # Calcular matriz de correlação
            corr_matrix = sample_data.corr()
            
            # Encontrar pares com alta correlação
            correlation_threshold = 0.9
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_value = abs(corr_matrix.iloc[i, j])
                    
                    if corr_value > correlation_threshold:
                        high_corr_pairs.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': corr_value
                        })
            
            # Grupo de features altamente correlacionadas
            if high_corr_pairs:
                correlated_groups = self._group_correlated_features(high_corr_pairs)
            else:
                correlated_groups = []
            
            self.analysis_results['correlations'] = {
                'high_correlation_pairs': high_corr_pairs,
                'correlated_groups': correlated_groups,
                'sample_based': len(data) > sample_size
            }
        except Exception as e:
            logger.warning(f"Erro ao calcular correlações: {str(e)}")
    
    def _analyze_target_relationships(
        self, 
        data: pd.DataFrame,
        target: pd.Series,
        dataset_type: Optional[str] = None
    ) -> None:
        """Analisa relações entre features e a variável alvo."""
        if dataset_type not in ['tabular_classification', 'tabular_regression', None]:
            return
        
        is_classification = dataset_type == 'tabular_classification'
        
        # Para classificação, verificar balanceamento das classes
        if is_classification:
            value_counts = target.value_counts(normalize=True)
            is_imbalanced = value_counts.iloc[0] > 0.75  # 75% em uma classe = desbalanceado
            
            self.analysis_results['target_analysis'] = {
                'type': 'classification',
                'n_classes': len(value_counts),
                'class_distribution': value_counts.to_dict(),
                'is_imbalanced': is_imbalanced
            }
        
        # Para regressão, analisar distribuição
        else:
            if pd.api.types.is_numeric_dtype(target):
                stats = target.describe()
                try:
                    skewness = target.skew()
                    is_skewed = abs(skewness) > 1
                except:
                    skewness = None
                    is_skewed = False
                
                self.analysis_results['target_analysis'] = {
                    'type': 'regression',
                    'min': stats['min'],
                    'max': stats['max'],
                    'mean': stats['mean'],
                    'median': stats['50%'],
                    'std': stats['std'],
                    'skewness': skewness,
                    'is_skewed': is_skewed
                }
        
        # Analisar relação com features numéricas (amostra se dataset for grande)
        sample_size = min(len(data), self.max_rows_analysis)
        
        if sample_size < len(data):
            sample_indices = np.random.choice(len(data), sample_size, replace=False)
            sample_data = data.iloc[sample_indices]
            sample_target = target.iloc[sample_indices]
        else:
            sample_data = data
            sample_target = target
        
        numeric_cols = sample_data.select_dtypes(include=['number']).columns
        feature_importance = {}
        
        # Para features numéricas, calcular correlação ou razão de informação
        for col in numeric_cols:
            if pd.api.types.is_numeric_dtype(sample_target):
                # Para target numérico, usar correlação
                corr = sample_data[col].corr(sample_target)
                feature_importance[col] = abs(corr)
            else:
                # Para target categórico, usar razão de entropia
                from sklearn.feature_selection import mutual_info_classif
                try:
                    mi = mutual_info_classif(
                        sample_data[col].values.reshape(-1, 1),
                        sample_target,
                        random_state=42
                    )[0]
                    feature_importance[col] = mi
                except:
                    feature_importance[col] = 0
        
        # Ordenar por importância
        sorted_importance = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Guardar as top 10 features mais importantes
        top_features = sorted_importance[:10]
        
        self.analysis_results['feature_importance'] = {
            'top_features': [
                {'feature': f, 'importance': i} for f, i in top_features
            ],
            'sample_based': len(data) > sample_size
        }
    
    def _group_correlated_features(self, correlation_pairs: List[Dict[str, Any]]) -> List[List[str]]:
        """Agrupa features correlacionadas em clusters."""
        # Construir grafo não direcionado de correlações
        graph = {}
        
        for pair in correlation_pairs:
            col1, col2 = pair['column1'], pair['column2']
            
            if col1 not in graph:
                graph[col1] = []
            if col2 not in graph:
                graph[col2] = []
            
            graph[col1].append(col2)
            graph[col2].append(col1)
        
        # Encontrar componentes conectados (grupos correlacionados)
        visited = set()
        groups = []
        
        for node in graph:
            if node not in visited:
                group = []
                queue = [node]
                visited.add(node)
                
                while queue:
                    current = queue.pop(0)
                    group.append(current)
                    
                    for neighbor in graph[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                groups.append(group)
        
        return groups
    
    def _generate_recommendations(
        self, 
        data: pd.DataFrame,
        target: Optional[Union[str, pd.Series]] = None,
        dataset_type: Optional[str] = None
    ) -> None:
        """Gera recomendações baseadas nas análises realizadas."""
        recommendations = []
        
        # === Recomendações para valores ausentes ===
        missing_info = self.analysis_results.get('data_quality', {}).get('missing_values', {})
        if missing_info.get('total_missing', 0) > 0:
            missing_cols = missing_info.get('columns_with_missing', [])
            
            # Recomendar remoção de colunas com muitos valores ausentes
            high_missing_cols = [
                col['column'] for col in missing_cols 
                if col['percentage'] > 50
            ]
            
            if high_missing_cols:
                recommendations.append({
                    'id': 'remove_high_missing_columns',
                    'title': 'Remover colunas com muitos valores ausentes',
                    'description': f"As seguintes colunas têm mais de 50% de valores ausentes: {', '.join(high_missing_cols)}",
                    'impact': 'Médio',
                    'effort': 'Baixo',
                    'priority': 1
                })
            
            # Recomendar preenchimento de valores ausentes para outras colunas
            medium_missing_cols = [
                col['column'] for col in missing_cols 
                if 5 <= col['percentage'] <= 50
            ]
            
            if medium_missing_cols:
                recommendations.append({
                    'id': 'fill_missing_values',
                    'title': 'Preencher valores ausentes',
                    'description': f"As seguintes colunas têm entre 5% e 50% de valores ausentes: {', '.join(medium_missing_cols)}",
                    'impact': 'Alto',
                    'effort': 'Médio',
                    'priority': 2
                })
        
        # === Recomendações para outliers ===
        outliers_info = self.analysis_results.get('numeric_features', {}).get('potential_outliers', {})
        if outliers_info:
            outlier_cols = list(outliers_info.keys())
            recommendations.append({
                'id': 'handle_outliers',
                'title': 'Tratar outliers em variáveis numéricas',
                'description': f"As seguintes colunas contêm possíveis outliers: {', '.join(outlier_cols)}",
                'impact': 'Médio',
                'effort': 'Médio',
                'priority': 3
            })
        
        # === Recomendações para redução de dimensionalidade ===
        corr_info = self.analysis_results.get('correlations', {})
        correlated_groups = corr_info.get('correlated_groups', [])
        if correlated_groups:
            recommendations.append({
                'id': 'remove_correlated_features',
                'title': 'Remover features altamente correlacionadas',
                'description': f"Encontrados {len(correlated_groups)} grupos de features altamente correlacionadas",
                'impact': 'Alto',
                'effort': 'Baixo',
                'priority': 1
            })
        
        # === Recomendações para variáveis categóricas ===
        high_cardinality = self.analysis_results.get('data_quality', {}).get('high_cardinality', [])
        if high_cardinality:
            high_card_cols = [item['column'] for item in high_cardinality]
            recommendations.append({
                'id': 'reduce_cardinality',
                'title': 'Reduzir cardinalidade de variáveis categóricas',
                'description': f"As seguintes colunas têm alta cardinalidade: {', '.join(high_card_cols)}",
                'impact': 'Alto',
                'effort': 'Médio',
                'priority': 2
            })
        
        # === Recomendações para transformação de datas ===
        datetime_info = self.analysis_results.get('datetime_features', {})
        if datetime_info:
            datetime_cols = list(datetime_info.keys())
            recommendations.append({
                'id': 'extract_datetime_features',
                'title': 'Extrair componentes de data/hora',
                'description': f"As seguintes colunas contêm datas que podem ser decompostas: {', '.join(datetime_cols)}",
                'impact': 'Alto',
                'effort': 'Baixo',
                'priority': 1
            })
        
        # === Recomendações para variáveis numéricas assimétricas ===
        numeric_info = self.analysis_results.get('numeric_features', {}).get('stats', {})
        skewed_cols = [
            col for col, stats in numeric_info.items() 
            if stats.get('highly_skewed', False)
        ]
        
        if skewed_cols:
            recommendations.append({
                'id': 'transform_skewed_features',
                'title': 'Transformar variáveis numéricas assimétricas',
                'description': f"As seguintes colunas têm distribuição altamente assimétrica: {', '.join(skewed_cols)}",
                'impact': 'Médio',
                'effort': 'Baixo',
                'priority': 2
            })
        
        # === Recomendações para balanceamento de classes (classificação) ===
        target_info = self.analysis_results.get('target_analysis', {})
        if target_info.get('type') == 'classification' and target_info.get('is_imbalanced', False):
            recommendations.append({
                'id': 'balance_classes',
                'title': 'Balancear classes do alvo',
                'description': "O dataset apresenta classes desbalanceadas",
                'impact': 'Alto',
                'effort': 'Médio',
                'priority': 2
            })
        
        # === Recomendações para normalização do alvo (regressão) ===
        if target_info.get('type') == 'regression' and target_info.get('is_skewed', False):
            recommendations.append({
                'id': 'transform_target',
                'title': 'Transformar variável alvo',
                'description': "A variável alvo tem distribuição assimétrica",
                'impact': 'Alto',
                'effort': 'Baixo',
                'priority': 1
            })
        
        # === Recomendação para features de baixa variância ===
        low_variance = self.analysis_results.get('data_quality', {}).get('low_variance_columns', [])
        if low_variance:
            recommendations.append({
                'id': 'remove_low_variance',
                'title': 'Remover features com baixa variância',
                'description': f"As seguintes colunas têm variância muito baixa: {', '.join(low_variance)}",
                'impact': 'Baixo',
                'effort': 'Baixo',
                'priority': 3
            })
        
        # === Recomendações específicas para o tipo de dataset ===
        if dataset_type == 'time_series':
            recommendations.append({
                'id': 'create_lag_features',
                'title': 'Criar features de lag (defasagem)',
                'description': "Para séries temporais, é útil criar features defasadas",
                'impact': 'Alto',
                'effort': 'Médio',
                'priority': 1
            })
        
        self.transformation_recommendations = recommendations
    
    def _estimate_performance_impact(self) -> None:
        """Estima o impacto das recomendações na performance do dataset."""
        # Inicializar insights de performance
        self.performance_insights = {
            'estimated_memory_reduction': 0.0,
            'estimated_rows_reduction': 0.0,
            'estimated_columns_reduction': 0.0,
            'expected_model_improvement': 'Baixo'
        }
        
        # Estimar redução de memória
        basic_info = self.analysis_results.get('basic_info', {})
        original_memory = basic_info.get('memory_usage_mb', 0)
        
        # Estimar redução de memória com base nas recomendações
        memory_reduction = 0.0
        rows_reduction = 0.0
        columns_reduction = 0
        
        # Estimar com base em cada recomendação
        for rec in self.transformation_recommendations:
            if rec['id'] == 'remove_high_missing_columns':
                # Estimar com base nas colunas a serem removidas
                missing_cols = self.analysis_results.get('data_quality', {}).get('missing_values', {}).get('columns_with_missing', [])
                high_missing_cols = [col for col in missing_cols if col.get('percentage', 0) > 50]
                columns_reduction += len(high_missing_cols)
                
                # Estimar redução de memória
                if len(high_missing_cols) > 0 and 'n_columns' in basic_info:
                    memory_reduction += original_memory * (len(high_missing_cols) / basic_info['n_columns'])
            
            elif rec['id'] == 'remove_correlated_features':
                # Estimar com base nas colunas correlacionadas
                correlated_groups = self.analysis_results.get('correlations', {}).get('correlated_groups', [])
                redundant_columns = sum(len(group) - 1 for group in correlated_groups)
                columns_reduction += redundant_columns
                
                # Estimar redução de memória
                if redundant_columns > 0 and 'n_columns' in basic_info:
                    memory_reduction += original_memory * (redundant_columns / basic_info['n_columns'])
            
            elif rec['id'] == 'handle_outliers':
                # Estimar melhoria no modelo
                if self.performance_insights['expected_model_improvement'] == 'Baixo':
                    self.performance_insights['expected_model_improvement'] = 'Médio'
        
        # Atualizar insights
        self.performance_insights['estimated_memory_reduction'] = min(memory_reduction, original_memory * 0.5)  # Cap em 50%
        self.performance_insights['estimated_memory_reduction_percentage'] = (memory_reduction / original_memory * 100) if original_memory > 0 else 0
        self.performance_insights['estimated_columns_reduction'] = columns_reduction
        
        # Estimar melhoria no modelo com base no número total de recomendações
        high_impact_recs = sum(1 for rec in self.transformation_recommendations if rec['impact'] == 'Alto')
        
        if high_impact_recs >= 3:
            self.performance_insights['expected_model_improvement'] = 'Alto'
        elif high_impact_recs >= 1:
            self.performance_insights['expected_model_improvement'] = 'Médio'
    
    # Métodos para aplicar transformações
    
    def _apply_remove_high_missing_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove colunas com alta porcentagem de valores ausentes."""
        missing_info = self.analysis_results.get('data_quality', {}).get('missing_values', {})
        missing_cols = missing_info.get('columns_with_missing', [])
        
        high_missing_cols = [
            col['column'] for col in missing_cols 
            if col['percentage'] > 50
        ]
        
        if high_missing_cols:
            return data.drop(columns=high_missing_cols)
        
        return data
    
    def _apply_fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preenche valores ausentes em colunas com percentagem moderada de missings."""
        df = data.copy()
        missing_info = self.analysis_results.get('data_quality', {}).get('missing_values', {})
        missing_cols = missing_info.get('columns_with_missing', [])
        
        medium_missing_cols = [
            col['column'] for col in missing_cols 
            if 5 <= col['percentage'] <= 50
        ]
        
        for col in medium_missing_cols:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Para numéricas, usar mediana
                    df[col] = df[col].fillna(df[col].median())
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    # Para datas, usar mais frequente
                    most_common = df[col].mode().iloc[0] if not df[col].mode().empty else None
                    if most_common is not None:
                        df[col] = df[col].fillna(most_common)
                else:
                    # Para categóricas, usar mais frequente
                    most_common = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(most_common)
        
        return df
    
    def _apply_remove_correlated_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove features altamente correlacionadas, mantendo uma de cada grupo."""
        corr_info = self.analysis_results.get('correlations', {})
        correlated_groups = corr_info.get('correlated_groups', [])
        
        if not correlated_groups:
            return data
        
        # Para cada grupo, manter apenas a primeira feature
        columns_to_drop = []
        
        for group in correlated_groups:
            if len(group) > 1:
                columns_to_drop.extend(group[1:])
        
        if columns_to_drop:
            return data.drop(columns=columns_to_drop)
        
        return data
    
    def _apply_extract_datetime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extrai componentes de features de data/hora."""
        datetime_info = self.analysis_results.get('datetime_features', {})
        if not datetime_info:
            return data
        
        df = data.copy()
        
        for col in datetime_info:
            if col in df.columns:
                # Converter para datetime se não for
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        continue
                
                # Extrair componentes
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df[f'{col}_quarter'] = df[col].dt.quarter
                
                # Se tiver horas, extrair componentes de hora
                if (df[col].dt.hour != 0).any():
                    df[f'{col}_hour'] = df[col].dt.hour
                    df[f'{col}_minute'] = df[col].dt.minute
                
                # Opcionalmente remover coluna original
                # df = df.drop(columns=[col])
        
        return df
    
    def _apply_transform_skewed_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aplica transformações em variáveis numéricas assimétricas."""
        numeric_info = self.analysis_results.get('numeric_features', {}).get('stats', {})
        skewed_cols = [
            col for col, stats in numeric_info.items() 
            if stats.get('highly_skewed', False)
        ]
        
        if not skewed_cols:
            return data
        
        df = data.copy()
        
        for col in skewed_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Verificar se a assimetria é positiva ou negativa
                skewness = numeric_info[col].get('skewness', 0)
                
                if skewness > 0:  # Assimetria positiva
                    # Aplicar log ou raiz quadrada
                    min_val = df[col].min()
                    offset = 1.0 if min_val >= 0 else abs(min_val) + 1.0
                    df[f'{col}_log'] = np.log(df[col] + offset)
                else:  # Assimetria negativa
                    # Aplicar exponencial ou quadrado
                    df[f'{col}_squared'] = df[col] ** 2
        
        return df
    
    def _apply_reduce_cardinality(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reduz a cardinalidade de variáveis categóricas com muitos níveis."""
        high_cardinality = self.analysis_results.get('data_quality', {}).get('high_cardinality', [])
        if not high_cardinality:
            return data
        
        df = data.copy()
        
        for item in high_cardinality:
            col = item['column']
            if col in df.columns:
                # Manter as 10 categorias mais frequentes, agrupar o resto como "Outro"
                top_categories = df[col].value_counts().nlargest(10).index.tolist()
                df[col] = df[col].apply(lambda x: x if x in top_categories else 'Outro')
        
        return df
    
    # Métodos para geração de relatórios
    
    def _generate_dataset_summary(self) -> str:
        """Gera um resumo textual do dataset."""
        basic_info = self.analysis_results.get('basic_info', {})
        
        summary = f"""
        ## Resumo do Dataset
        
        - **Registros**: {basic_info.get('n_rows', 0):,}
        - **Colunas**: {basic_info.get('n_columns', 0)}
        - **Uso de Memória**: {basic_info.get('memory_usage_mb', 0):.2f} MB
        
        ### Tipos de Dados
        - Numéricas: {basic_info.get('column_types', {}).get('numeric', 0)}
        - Categóricas: {basic_info.get('column_types', {}).get('categorical', 0)}
        - Data/Hora: {basic_info.get('column_types', {}).get('datetime', 0)}
        - Booleanas: {basic_info.get('column_types', {}).get('boolean', 0)}
        """
        
        # Adicionar informações sobre o alvo se disponível
        target_info = self.analysis_results.get('target_analysis', {})
        if target_info:
            summary += "\n\n### Informações do Alvo\n"
            
            if target_info.get('type') == 'classification':
                summary += f"- **Tipo**: Classificação\n"
                summary += f"- **Número de Classes**: {target_info.get('n_classes', 0)}\n"
                
                # Distribuição de classes
                summary += "- **Distribuição de Classes**:\n"
                for cls, prop in target_info.get('class_distribution', {}).items():
                    summary += f"  - {cls}: {prop*100:.1f}%\n"
                
                if target_info.get('is_imbalanced', False):
                    summary += "- **Observação**: Classes desbalanceadas detectadas\n"
            
            elif target_info.get('type') == 'regression':
                summary += f"- **Tipo**: Regressão\n"
                summary += f"- **Mínimo**: {target_info.get('min', 0):.2f}\n"
                summary += f"- **Máximo**: {target_info.get('max', 0):.2f}\n"
                summary += f"- **Média**: {target_info.get('mean', 0):.2f}\n"
                summary += f"- **Mediana**: {target_info.get('median', 0):.2f}\n"
                
                if target_info.get('is_skewed', False):
                    summary += f"- **Observação**: Distribuição assimétrica detectada (skewness: {target_info.get('skewness', 0):.2f})\n"
        
        return summary
    
    def _generate_quality_summary(self) -> str:
        """Gera um resumo da qualidade dos dados."""
        quality_info = self.analysis_results.get('data_quality', {})
        
        summary = """
        ## Qualidade dos Dados
        """
        
        # Informações sobre valores ausentes
        missing_info = quality_info.get('missing_values', {})
        missing_total = missing_info.get('total_missing', 0)
        missing_pct = missing_info.get('missing_percentage', 0)
        
        summary += f"\n### Valores Ausentes\n"
        summary += f"- **Total**: {missing_total:,} ({missing_pct:.2f}% do dataset)\n"
        
        if missing_info.get('columns_with_missing'):
            summary += "- **Colunas com Mais Valores Ausentes**:\n"
            for col in sorted(missing_info.get('columns_with_missing', []), key=lambda x: x['count'], reverse=True)[:5]:
                summary += f"  - {col['column']}: {col['count']:,} ({col['percentage']:.2f}%)\n"
        
        # Informações sobre duplicatas
        duplicate_info = quality_info.get('duplicates', {})
        duplicate_rows = duplicate_info.get('duplicate_rows', 0)
        duplicate_pct = duplicate_info.get('duplicate_percentage', 0)
        
        summary += f"\n### Linhas Duplicadas\n"
        if duplicate_info.get('sample_based', False):
            summary += f"- **Estimativa**: {duplicate_pct:.2f}% (baseado em amostra)\n"
        else:
            summary += f"- **Total**: {duplicate_rows:,} ({duplicate_pct:.2f}%)\n"
        
        # Problemas de qualidade detectados
        problems = []
        
        if missing_pct > 15:
            problems.append("Alta porcentagem de valores ausentes")
        
        if duplicate_pct > 5:
            problems.append("Muitas linhas duplicadas")
        
        if quality_info.get('high_cardinality'):
            problems.append(f"{len(quality_info.get('high_cardinality'))} colunas com alta cardinalidade")
        
        if quality_info.get('low_variance_columns'):
            problems.append(f"{len(quality_info.get('low_variance_columns'))} colunas com baixa variância")
        
        corr_info = self.analysis_results.get('correlations', {})
        if corr_info.get('high_correlation_pairs'):
            problems.append(f"{len(corr_info.get('high_correlation_pairs'))} pares de colunas altamente correlacionadas")
        
        numeric_info = self.analysis_results.get('numeric_features', {})
        if numeric_info.get('potential_outliers'):
            problems.append(f"{len(numeric_info.get('potential_outliers'))} colunas com possíveis outliers")
        
        if problems:
            summary += f"\n### Problemas Detectados\n"
            for problem in problems:
                summary += f"- {problem}\n"
        
        return summary
    
    def _format_recommendations(self) -> str:
        """Formata as recomendações para apresentação."""
        recs = self.transformation_recommendations
        
        if not recs:
            return "## Recomendações\n\nNenhuma recomendação específica identificada."
        
        summary = """
        ## Recomendações de Pré-processamento
        
        As seguintes transformações são recomendadas para otimizar o dataset:
        """
        
        for i, rec in enumerate(sorted(recs, key=lambda x: x.get('priority', 999)), 1):
            summary += f"\n### {i}. {rec['title']}\n"
            summary += f"- **Descrição**: {rec['description']}\n"
            summary += f"- **Impacto Esperado**: {rec['impact']}\n"
            summary += f"- **Esforço de Implementação**: {rec['effort']}\n"
        
        return summary
    
    def _format_performance_insights(self) -> str:
        """Formata os insights de performance para apresentação."""
        insights = self.performance_insights
        
        summary = """
        ## Impacto na Performance
        
        Aplicando as recomendações sugeridas, espera-se os seguintes ganhos:
        """
        
        # Redução de dimensionalidade
        cols_reduction = insights.get('estimated_columns_reduction', 0)
        if cols_reduction > 0:
            summary += f"\n- Redução de aproximadamente {cols_reduction} colunas\n"
        
        # Redução de memória
        memory_reduction = insights.get('estimated_memory_reduction', 0)
        memory_pct = insights.get('estimated_memory_reduction_percentage', 0)
        if memory_reduction > 0:
            summary += f"- Redução de memória de aproximadamente {memory_reduction:.2f} MB ({memory_pct:.1f}%)\n"
        
        # Melhoria do modelo
        model_improvement = insights.get('expected_model_improvement', 'Baixo')
        summary += f"- Potencial de melhoria do modelo: {model_improvement}\n"
        
        # Informações de aplicação se disponíveis
        if 'applied_transformations' in insights:
            applied = insights.get('applied_transformations', [])
            before_rows = insights.get('rows_before', 0)
            before_cols = insights.get('columns_before', 0)
            after_rows = insights.get('rows_after', 0)
            after_cols = insights.get('columns_after', 0)
            
            summary += f"\n### Transformações Aplicadas\n"
            summary += f"- Transformações: {', '.join(applied)}\n"
            summary += f"- Linhas: {before_rows:,} → {after_rows:,}\n"
            summary += f"- Colunas: {before_cols:,} → {after_cols:,}\n"
        
        return summary
    
    def _generate_visualizations(self) -> Dict[str, str]:
        """Gera visualizações para o relatório."""
        visualizations = {}
        
        # 1. Distribuição de tipos de colunas
        try:
            column_types = self.analysis_results.get('basic_info', {}).get('column_types', {})
            if column_types:
                plt.figure(figsize=(8, 5))
                plt.bar(column_types.keys(), column_types.values(), color='steelblue')
                plt.title('Distribuição de Tipos de Colunas')
                plt.ylabel('Número de Colunas')
                plt.xticks(rotation=45)
                
                # Converter para base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                img_str = base64.b64encode(buffer.read()).decode('utf-8')
                visualizations['column_types'] = img_str
                plt.close()
        except Exception as e:
            logger.warning(f"Erro ao gerar visualização de tipos de colunas: {str(e)}")
        
        # 2. Mapa de calor de correlações
        try:
            corr_info = self.analysis_results.get('correlations', {})
            high_corr_pairs = corr_info.get('high_correlation_pairs', [])
            
            if high_corr_pairs:
                # Criar matriz de correlação a partir dos pares
                corr_cols = set()
                for pair in high_corr_pairs:
                    corr_cols.add(pair['column1'])
                    corr_cols.add(pair['column2'])
                
                corr_cols = list(corr_cols)
                corr_matrix = pd.DataFrame(index=corr_cols, columns=corr_cols)
                
                # Preencher diagonal com 1.0
                for col in corr_cols:
                    corr_matrix.loc[col, col] = 1.0
                
                # Preencher com valores de correlação
                for pair in high_corr_pairs:
                    col1 = pair['column1']
                    col2 = pair['column2']
                    corr_value = pair['correlation']
                    corr_matrix.loc[col1, col2] = corr_value
                    corr_matrix.loc[col2, col1] = corr_value
                
                # Preencher valores NaN com zeros
                corr_matrix = corr_matrix.fillna(0)
                
                # Plotar mapa de calor
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title('Mapa de Correlações entre Features')
                plt.tight_layout()
                
                # Converter para base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                img_str = base64.b64encode(buffer.read()).decode('utf-8')
                visualizations['correlation_heatmap'] = img_str
                plt.close()
        except Exception as e:
            logger.warning(f"Erro ao gerar mapa de calor de correlações: {str(e)}")
        
        # 3. Visualização de valores ausentes
        try:
            missing_info = self.analysis_results.get('data_quality', {}).get('missing_values', {})
            missing_cols = missing_info.get('columns_with_missing', [])
            
            if missing_cols:
                # Ordenar por percentagem descendente
                missing_cols.sort(key=lambda x: x['percentage'], reverse=True)
                
                # Limitar a 15 colunas para melhor visualização
                if len(missing_cols) > 15:
                    missing_cols = missing_cols[:15]
                
                # Extrair dados para o gráfico
                cols = [col['column'] for col in missing_cols]
                percentages = [col['percentage'] for col in missing_cols]
                
                plt.figure(figsize=(10, 6))
                bars = plt.barh(cols, percentages, color='salmon')
                
                # Adicionar rótulos de percentagem
                for i, bar in enumerate(bars):
                    plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                             f"{percentages[i]:.1f}%", va='center')
                
                plt.title('Percentagem de Valores Ausentes por Coluna')
                plt.xlabel('Percentagem')
                plt.ylabel('Coluna')
                plt.xlim(0, max(percentages) * 1.1)  # Espaço para rótulos
                plt.tight_layout()
                
                # Converter para base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                img_str = base64.b64encode(buffer.read()).decode('utf-8')
                visualizations['missing_values'] = img_str
                plt.close()
        except Exception as e:
            logger.warning(f"Erro ao gerar visualização de valores ausentes: {str(e)}")
        
        # 4. Top Features por Importância
        try:
            feature_importance = self.analysis_results.get('feature_importance', {})
            top_features = feature_importance.get('top_features', [])
            
            if top_features:
                # Extrair dados para o gráfico
                features = [f['feature'] for f in top_features]
                importance = [f['importance'] for f in top_features]
                
                plt.figure(figsize=(10, 6))
                bars = plt.barh(features, importance, color='teal')
                
                # Adicionar rótulos de importância
                for i, bar in enumerate(bars):
                    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                             f"{importance[i]:.3f}", va='center')
                
                plt.title('Top Features por Importância')
                plt.xlabel('Importância Relativa')
                plt.ylabel('Feature')
                plt.xlim(0, max(importance) * 1.1)  # Espaço para rótulos
                plt.tight_layout()
                
                # Converter para base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                img_str = base64.b64encode(buffer.read()).decode('utf-8')
                visualizations['feature_importance'] = img_str
                plt.close()
        except Exception as e:
            logger.warning(f"Erro ao gerar visualização de importância de features: {str(e)}")
        
        return visualizations