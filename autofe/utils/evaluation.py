"""
Funções de avaliação para a qualidade de features.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy import stats

logger = logging.getLogger(__name__)


def evaluate_classification_features(
    X: pd.DataFrame,
    y: pd.Series,
    model: Any,
    metric: str = 'auc'
) -> Dict[str, float]:
    """
    Avalia a qualidade de features para um problema de classificação.
    
    Args:
        X: DataFrame com features
        y: Série com alvo
        model: Modelo treinado
        metric: Métrica de avaliação ('auc', 'accuracy', 'f1', etc.)
        
    Returns:
        Dicionário com métricas
    """
    # Verificar se há dados suficientes
    if len(X) == 0 or len(y) == 0:
        logger.warning("Dados insuficientes para avaliação")
        return {'error': 'Dados insuficientes'}
    
    try:
        # Fazer predições
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)
            if y_proba.shape[1] >= 2:  # Múltiplas classes
                y_pred_proba = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba
            else:
                y_pred_proba = y_proba
        else:
            y_pred_proba = None
        
        y_pred = model.predict(X)
        
        # Calcular métricas
        results = {}
        
        # Acurácia
        results['accuracy'] = accuracy_score(y, y_pred)
        
        # F1 Score (para problemas binários ou multiclasse)
        if len(np.unique(y)) == 2:
            results['f1'] = f1_score(y, y_pred, average='binary')
            results['precision'] = precision_score(y, y_pred, average='binary')
            results['recall'] = recall_score(y, y_pred, average='binary')
        else:
            results['f1'] = f1_score(y, y_pred, average='weighted')
            results['precision'] = precision_score(y, y_pred, average='weighted')
            results['recall'] = recall_score(y, y_pred, average='weighted')
        
        # AUC (apenas para classificação binária)
        if len(np.unique(y)) == 2 and y_pred_proba is not None:
            try:
                results['auc'] = roc_auc_score(y, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba)
            except:
                results['auc'] = 0.5  # Valor neutro
        
        return results
    
    except Exception as e:
        logger.error(f"Erro ao avaliar features de classificação: {str(e)}")
        return {'error': str(e)}


def evaluate_regression_features(
    X: pd.DataFrame,
    y: pd.Series,
    model: Any,
    metric: str = 'rmse'
) -> Dict[str, float]:
    """
    Avalia a qualidade de features para um problema de regressão.
    
    Args:
        X: DataFrame com features
        y: Série com alvo
        model: Modelo treinado
        metric: Métrica de avaliação ('rmse', 'mae', 'r2', etc.)
        
    Returns:
        Dicionário com métricas
    """
    # Verificar se há dados suficientes
    if len(X) == 0 or len(y) == 0:
        logger.warning("Dados insuficientes para avaliação")
        return {'error': 'Dados insuficientes'}
    
    try:
        # Fazer predições
        y_pred = model.predict(X)
        
        # Calcular métricas
        results = {}
        
        # RMSE
        results['rmse'] = np.sqrt(mean_squared_error(y, y_pred))
        
        # MAE
        results['mae'] = mean_absolute_error(y, y_pred)
        
        # R²
        results['r2'] = r2_score(y, y_pred)
        
        # MAPE (se não houver zeros)
        if not (y == 0).any():
            mape = np.mean(np.abs((y - y_pred) / y)) * 100
            results['mape'] = mape
        
        return results
    
    except Exception as e:
        logger.error(f"Erro ao avaliar features de regressão: {str(e)}")
        return {'error': str(e)}


def calculate_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    is_classification: bool = True
) -> pd.DataFrame:
    """
    Calcula a importância das features usando informação mútua.
    
    Args:
        X: DataFrame com features
        y: Série com alvo
        is_classification: Se é um problema de classificação
        
    Returns:
        DataFrame com features e suas importâncias
    """
    # Verificar se há dados suficientes
    if len(X) == 0 or len(y) == 0:
        logger.warning("Dados insuficientes para calcular importância")
        return pd.DataFrame(columns=['feature', 'importance'])
    
    try:
        # Remover valores ausentes
        X_clean = X.copy()
        for col in X_clean.columns:
            if X_clean[col].isnull().any():
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        
        # Calcular informação mútua
        if is_classification:
            importances = mutual_info_classif(X_clean, y, random_state=42)
        else:
            importances = mutual_info_regression(X_clean, y, random_state=42)
        
        # Criar DataFrame com resultados
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    except Exception as e:
        logger.error(f"Erro ao calcular importância das features: {str(e)}")
        return pd.DataFrame(columns=['feature', 'importance'])


def calculate_correlation_matrix(
    X: pd.DataFrame,
    method: str = 'pearson'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula a matriz de correlação entre features.
    
    Args:
        X: DataFrame com features
        method: Método de correlação ('pearson', 'spearman', 'kendall')
        
    Returns:
        Tuple com (matriz de correlação, matriz de p-valores)
    """
    # Verificar se há dados suficientes
    if len(X) == 0 or len(X.columns) < 2:
        logger.warning("Dados insuficientes para calcular correlação")
        return pd.DataFrame(), pd.DataFrame()
    
    # Selecionar apenas colunas numéricas
    X_numeric = X.select_dtypes(include=['number'])
    
    if X_numeric.shape[1] < 2:
        logger.warning("Menos de 2 colunas numéricas para calcular correlação")
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        # Calcular matriz de correlação
        corr_matrix = X_numeric.corr(method=method)
        
        # Calcular matriz de p-valores
        p_matrix = pd.DataFrame(np.ones((len(X_numeric.columns), len(X_numeric.columns))),
                              index=X_numeric.columns, columns=X_numeric.columns)
        
        for i, col1 in enumerate(X_numeric.columns):
            for j, col2 in enumerate(X_numeric.columns):
                if i > j:  # Calcular apenas o triângulo inferior
                    if method == 'pearson':
                        corr, p = stats.pearsonr(X_numeric[col1].dropna(), X_numeric[col2].dropna())
                    elif method == 'spearman':
                        corr, p = stats.spearmanr(X_numeric[col1].dropna(), X_numeric[col2].dropna())
                    elif method == 'kendall':
                        corr, p = stats.kendalltau(X_numeric[col1].dropna(), X_numeric[col2].dropna())
                    
                    p_matrix.loc[col1, col2] = p
                    p_matrix.loc[col2, col1] = p
        
        return corr_matrix, p_matrix
    
    except Exception as e:
        logger.error(f"Erro ao calcular matriz de correlação: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()


def identify_redundant_features(
    X: pd.DataFrame,
    correlation_threshold: float = 0.95,
    p_value_threshold: float = 0.05
) -> List[Tuple[str, str, float]]:
    """
    Identifica pares de features redundantes.
    
    Args:
        X: DataFrame com features
        correlation_threshold: Limiar de correlação
        p_value_threshold: Limiar de p-valor
        
    Returns:
        Lista de tuplas (feature1, feature2, correlação)
    """
    # Calcular matriz de correlação
    corr_matrix, p_matrix = calculate_correlation_matrix(X)
    
    if corr_matrix.empty:
        return []
    
    # Identificar pares com alta correlação
    redundant_pairs = []
    
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Apenas o triângulo superior
                corr = abs(corr_matrix.loc[col1, col2])
                p_val = p_matrix.loc[col1, col2]
                
                if corr >= correlation_threshold and p_val <= p_value_threshold:
                    redundant_pairs.append((col1, col2, corr))
    
    # Ordenar por correlação (decrescente)
    redundant_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return redundant_pairs


def assess_feature_quality(
    X: pd.DataFrame,
    y: pd.Series,
    is_classification: bool = True
) -> pd.DataFrame:
    """
    Avalia a qualidade geral das features.
    
    Args:
        X: DataFrame com features
        y: Série com alvo
        is_classification: Se é um problema de classificação
        
    Returns:
        DataFrame com métricas de qualidade
    """
    # Inicializar DataFrame de resultados
    results = []
    
    # Para cada feature, calcular métricas
    for col in X.columns:
        feature = X[col]
        
        # Estatísticas básicas
        n_missing = feature.isnull().sum()
        missing_ratio = n_missing / len(feature)
        
        if pd.api.types.is_numeric_dtype(feature):
            # Estatísticas para features numéricas
            try:
                skewness = feature.skew()
                kurtosis = feature.kurtosis()
                mean = feature.mean()
                std = feature.std()
                min_val = feature.min()
                max_val = feature.max()
                
                # Correlação com o alvo (para y numérico)
                if pd.api.types.is_numeric_dtype(y):
                    # Remover NaNs para cálculo da correlação
                    valid_mask = ~(feature.isna() | y.isna())
                    if valid_mask.sum() > 1:
                        correlation = feature[valid_mask].corr(y[valid_mask])
                    else:
                        correlation = np.nan
                else:
                    correlation = np.nan
                
                # Informação mútua com o alvo
                if is_classification:
                    # Remover NaNs
                    valid_mask = ~feature.isna()
                    if valid_mask.sum() > 0:
                        mi = mutual_info_classif(
                            feature[valid_mask].values.reshape(-1, 1),
                            y[valid_mask],
                            random_state=42
                        )[0]
                    else:
                        mi = 0
                else:
                    # Remover NaNs
                    valid_mask = ~feature.isna()
                    if valid_mask.sum() > 0:
                        mi = mutual_info_regression(
                            feature[valid_mask].values.reshape(-1, 1),
                            y[valid_mask],
                            random_state=42
                        )[0]
                    else:
                        mi = 0
                
                # Adicionar aos resultados
                results.append({
                    'feature': col,
                    'type': 'numeric',
                    'n_missing': n_missing,
                    'missing_ratio': missing_ratio,
                    'mean': mean,
                    'std': std,
                    'min': min_val,
                    'max': max_val,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'correlation': correlation,
                    'mutual_info': mi
                })
            except Exception as e:
                logger.warning(f"Erro ao calcular estatísticas para {col}: {str(e)}")
                # Adicionar estatísticas básicas
                results.append({
                    'feature': col,
                    'type': 'numeric',
                    'n_missing': n_missing,
                    'missing_ratio': missing_ratio,
                    'error': str(e)
                })
        else:
            # Estatísticas para features categóricas ou outras
            n_unique = feature.nunique()
            
            # Adicionar aos resultados
            results.append({
                'feature': col,
                'type': 'categorical',
                'n_missing': n_missing,
                'missing_ratio': missing_ratio,
                'n_unique': n_unique
            })
    
    # Converter para DataFrame
    quality_df = pd.DataFrame(results)
    
    return quality_df


def evaluate_transformation_impact(
    X_original: pd.DataFrame,
    X_transformed: pd.DataFrame,
    y: pd.Series,
    transformation_info: Dict[str, Any],
    is_classification: bool = True
) -> Dict[str, Any]:
    """
    Avalia o impacto de uma transformação.
    
    Args:
        X_original: DataFrame original
        X_transformed: DataFrame transformado
        y: Série com alvo
        transformation_info: Informações da transformação
        is_classification: Se é um problema de classificação
        
    Returns:
        Dicionário com métricas de impacto
    """
    # Obter informações da transformação
    transformation_type = transformation_info.get('transformation_type', 'unknown')
    feature_name = transformation_info.get('name', 'unknown')
    
    # Calcular importância da feature antes e depois
    importance_before = calculate_feature_importance(X_original, y, is_classification)
    importance_after = calculate_feature_importance(X_transformed, y, is_classification)
    
    # Verificar se a feature está entre as importantes
    if feature_name in importance_after['feature'].values:
        new_importance = importance_after.loc[importance_after['feature'] == feature_name, 'importance'].values[0]
    else:
        new_importance = 0
    
    # Calcular redundância
    redundant_before = identify_redundant_features(X_original)
    redundant_after = identify_redundant_features(X_transformed)
    
    # Verificar se a nova feature é redundante com alguma existente
    is_redundant = False
    redundancy_score = 0.0
    redundant_with = []
    
    for pair in redundant_after:
        if feature_name in pair:
            is_redundant = True
            other_feature = pair[0] if pair[1] == feature_name else pair[1]
            redundant_with.append((other_feature, pair[2]))
            redundancy_score = max(redundancy_score, pair[2])
    
    # Compilar resultados
    impact = {
        'feature_name': feature_name,
        'transformation_type': transformation_type,
        'importance': new_importance,
        'is_redundant': is_redundant,
        'redundancy_score': redundancy_score,
        'redundant_with': redundant_with
    }
    
    return impact


def assess_dataset_quality(
    data: pd.DataFrame,
    target_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Avalia a qualidade geral do dataset.
    
    Args:
        data: DataFrame com os dados
        target_column: Nome da coluna alvo (opcional)
        
    Returns:
        Dicionário com métricas de qualidade do dataset
    """
    # Inicializar resultado
    quality = {
        'n_rows': len(data),
        'n_columns': len(data.columns),
        'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024),
        'column_types': {},
        'missing_values': {},
        'warnings': []
    }
    
    # Calcular percentual por tipo de dados
    n_numeric = len(data.select_dtypes(include=['number']).columns)
    n_categorical = len(data.select_dtypes(include=['object', 'category']).columns)
    n_datetime = len(data.select_dtypes(include=['datetime']).columns)
    n_bool = len(data.select_dtypes(include=['bool']).columns)
    
    quality['column_types'] = {
        'numeric': n_numeric,
        'categorical': n_categorical,
        'datetime': n_datetime,
        'boolean': n_bool
    }
    
    # Calcular valores ausentes globais
    total_cells = data.shape[0] * data.shape[1]
    total_missing = data.isnull().sum().sum()
    
    quality['missing_values'] = {
        'total_missing': total_missing,
        'missing_percentage': (total_missing / total_cells) * 100 if total_cells > 0 else 0,
        'columns_with_missing': [col for col in data.columns if data[col].isnull().any()]
    }
    
    # Verificar outliers em colunas numéricas
    numeric_columns = data.select_dtypes(include=['number']).columns
    outlier_columns = []
    
    for col in numeric_columns:
        # Calcular IQR
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        
        # Limites para outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Contar outliers
        n_outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        
        # Se mais de 5% são outliers, adicionar à lista
        if n_outliers > 0.05 * len(data):
            outlier_columns.append(col)
    
    quality['outliers'] = {
        'columns_with_outliers': outlier_columns
    }
    
    # Se houver coluna alvo, analisar desequilíbrio (para classificação)
    if target_column and target_column in data.columns:
        target = data[target_column]
        
        # Verificar se é classificação
        if not pd.api.types.is_numeric_dtype(target) or target.nunique() < 10:
            # Calcular distribuição
            value_counts = target.value_counts(normalize=True)
            
            # Verificar desequilíbrio
            if len(value_counts) >= 2:
                min_class_pct = value_counts.min() * 100
                max_class_pct = value_counts.max() * 100
                
                quality['target'] = {
                    'type': 'classification',
                    'n_classes': len(value_counts),
                    'class_distribution': value_counts.to_dict(),
                    'min_class_percentage': min_class_pct,
                    'max_class_percentage': max_class_pct
                }
                
                # Adicionar aviso se muito desequilibrado
                if min_class_pct < 10:
                    quality['warnings'].append(
                        f"Desequilíbrio severo da classe alvo: classe minoritária representa apenas {min_class_pct:.1f}%"
                    )
            else:
                quality['target'] = {
                    'type': 'classification',
                    'n_classes': len(value_counts),
                    'class_distribution': value_counts.to_dict()
                }
        else:
            # Regressão
            quality['target'] = {
                'type': 'regression',
                'mean': target.mean(),
                'std': target.std(),
                'min': target.min(),
                'max': target.max(),
                'skewness': target.skew()
            }
            
            # Adicionar aviso se muito assimétrico
            if abs(target.skew()) > 1:
                quality['warnings'].append(
                    f"Assimetria elevada na variável alvo: skewness = {target.skew():.2f}"
                )
    
    # Adicionar avisos gerais
    if quality['missing_values']['missing_percentage'] > 15:
        quality['warnings'].append(
            f"Alta porcentagem de valores ausentes: {quality['missing_values']['missing_percentage']:.1f}%"
        )
    
    if len(outlier_columns) > 0:
        quality['warnings'].append(
            f"Outliers detectados em {len(outlier_columns)} colunas"
        )
    
    if len(data) < 100:
        quality['warnings'].append(
            f"Tamanho da amostra pequeno: apenas {len(data)} registros"
        )
    
    return quality
