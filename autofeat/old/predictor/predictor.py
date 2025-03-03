# predictor/predictor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
import logging
import sys
import os
import json
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error, r2_score
import pickle

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data_types import (
    DataType, ProblemType, DatasetInfo, ColumnInfo, 
    TransformationType, TransformationInfo, TransformationResult
)
from autofeat.explorer.explorer import (
    MathematicalTransformer, 
    TemporalTransformer,
    CategoricalTransformer,
    TextTransformer,
    InteractionTransformer,
    GroupingTransformer
)

logger = logging.getLogger("AutoFE.Predictor")

class Predictor:
    """
    Módulo responsável pelo meta-aprendizado que recomenda transformações.
    
    Este módulo aprende quais transformações são mais eficazes para diferentes
    tipos de problemas e conjuntos de dados, e as aplica automaticamente.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o módulo Predictor.
        
        Args:
            config (dict, optional): Configurações para o Predictor.
        """
        self.config = config or {}
        self.use_meta_learning = self.config.get('use_meta_learning', True)
        self.meta_db_path = self.config.get('meta_db_path', 'meta_learning.json')
        self.evaluation_metric = self.config.get('evaluation_metric', 'auto')
        
        # Transformadores
        self.transformers = {
            'mathematical': MathematicalTransformer(),
            'temporal': TemporalTransformer(),
            'categorical': CategoricalTransformer(),
            'text': TextTransformer(),
            'interaction': InteractionTransformer(),
            'grouping': GroupingTransformer()
        }
        
        # Carrega o banco de dados de meta-aprendizado
        self.meta_db = self._load_meta_db()
        
        logger.info("Módulo Predictor inicializado")
    
    def process(self, df: pd.DataFrame, target_column: str, 
               problem_type: ProblemType, time_column: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Processa o DataFrame aplicando transformações baseadas em meta-aprendizado.
        
        Args:
            df (pd.DataFrame): DataFrame a ser processado.
            target_column (str): Nome da coluna alvo.
            problem_type (ProblemType): Tipo do problema.
            time_column (str, optional): Nome da coluna temporal.
            
        Returns:
            tuple: DataFrame processado e relatório do processamento.
        """
        # Inicializa o relatório
        report = {
            "original_shape": df.shape,
            "problem_type": problem_type.name,
            "transformations_applied": [],
            "meta_learning_used": self.use_meta_learning
        }
        
        # Obtém informações do dataset
        dataset_info = DatasetInfo.from_dataframe(df, target_column, time_column, problem_type)
        
        # Se meta-aprendizado estiver desativado, retorna o dataset original
        if not self.use_meta_learning:
            logger.info("Meta-aprendizado desativado. Pulando etapa de recomendação.")
            report["reason"] = "Meta-aprendizado desativado nas configurações"
            return df, report
        
        # Calcula características do dataset para matching no meta-aprendizado
        dataset_fingerprint = self._calculate_dataset_fingerprint(df, dataset_info)
        report["dataset_fingerprint"] = dataset_fingerprint
        
        # Encontra transformações recomendadas com base no meta-aprendizado
        recommended_transformations = self._get_recommended_transformations(
            dataset_fingerprint, problem_type
        )
        
        report["recommended_transformations"] = len(recommended_transformations)
        
        if not recommended_transformations:
            logger.info("Nenhuma transformação recomendada pelo meta-aprendizado.")
            report["reason"] = "Nenhuma transformação recomendada"
            return df, report
        
        # Avalia o desempenho baseline
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        baseline_score = self._evaluate_performance(X_train, X_val, y_train, y_val, problem_type)
        
        report["baseline_score"] = baseline_score
        
        # Aplica as transformações recomendadas
        df_transformed = df.copy()
        applied_transformations = []
        
        for transformation_info in recommended_transformations:
            column_name = transformation_info["column"]
            transformation_type = TransformationType[transformation_info["transformation"]]
            
            # Pula se a coluna não existe
            if column_name not in df_transformed.columns:
                continue
            
            # Aplica a transformação
            transformation_result = self._apply_transformation(
                df_transformed, column_name, transformation_type, dataset_info
            )
            
            if transformation_result:
                # Avalia se a transformação melhorou o desempenho
                X_new = transformation_result.transformed_data.drop(columns=[target_column])
                X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
                    X_new, transformation_result.transformed_data[target_column], 
                    test_size=0.2, random_state=42
                )
                
                new_score = self._evaluate_performance(
                    X_train_new, X_val_new, y_train_new, y_val_new, problem_type
                )
                
                # Se melhorou, mantém a transformação
                if new_score > baseline_score:
                    df_transformed = transformation_result.transformed_data
                    baseline_score = new_score
                    
                    # Registra a transformação
                    applied_transformations.append({
                        "column": column_name,
                        "transformation": transformation_type.name,
                        "improvement": new_score - baseline_score,
                        "created_columns": transformation_result.created_columns
                    })
                    
                    logger.info(f"Transformação aplicada: {transformation_type.name} em {column_name}")
                    logger.info(f"Novas colunas: {transformation_result.created_columns}")
                    logger.info(f"Score: {new_score} (Melhoria: {new_score - baseline_score})")
        
        report["transformations_applied"] = applied_transformations
        report["final_score"] = baseline_score
        report["final_shape"] = df_transformed.shape
        report["columns_added"] = df_transformed.shape[1] - df.shape[1]
        
        # Atualiza o banco de dados de meta-aprendizado com os resultados
        self._update_meta_db(dataset_fingerprint, problem_type, applied_transformations)
        
        logger.info(f"Transformações por meta-aprendizado aplicadas. Score final: {baseline_score}")
        
        return df_transformed, report
    
    def _calculate_dataset_fingerprint(self, df: pd.DataFrame, 
                                     dataset_info: DatasetInfo) -> Dict[str, Any]:
        """
        Calcula características do dataset para matching no meta-aprendizado.
        
        Args:
            df (pd.DataFrame): DataFrame a ser analisado.
            dataset_info (DatasetInfo): Informações sobre o dataset.
            
        Returns:
            dict: Características distintivas do dataset.
        """
        fingerprint = {
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "problem_type": dataset_info.problem_type.name if dataset_info.problem_type else None,
            "column_types": {},
            "column_stats": {},
            "missing_values_pct": (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            "target_name": dataset_info.target_column
        }
        
        # Contagem de tipos de colunas
        column_type_counts = {dt.name: 0 for dt in DataType}
        
        for col_info in dataset_info.columns:
            column_type_counts[col_info.data_type.name] += 1
            
            # Adiciona estatísticas básicas para cada coluna
            if col_info.data_type == DataType.NUMERIC:
                fingerprint["column_stats"][col_info.name] = {
                    "missing_pct": col_info.missing_percentage,
                    "unique_ratio": col_info.num_unique_values / len(df) if len(df) > 0 else 0,
                    "mean": float(col_info.mean) if col_info.mean is not None else None,
                    "std": float(col_info.std) if col_info.std is not None else None
                }
            elif col_info.data_type == DataType.CATEGORICAL:
                fingerprint["column_stats"][col_info.name] = {
                    "missing_pct": col_info.missing_percentage,
                    "num_categories": col_info.num_unique_values,
                    "unique_ratio": col_info.num_unique_values / len(df) if len(df) > 0 else 0
                }
        
        fingerprint["column_types"] = column_type_counts
        
        # Calcula correlações com o target
        if dataset_info.target_column:
            target = df[dataset_info.target_column]
            
            # Para target numérico, calcula correlação numérica
            if pd.api.types.is_numeric_dtype(target):
                correlations = {}
                
                for col in df.columns:
                    if col != dataset_info.target_column and pd.api.types.is_numeric_dtype(df[col]):
                        try:
                            corr = df[col].corr(target)
                            if not pd.isna(corr):
                                correlations[col] = corr
                        except:
                            pass
                
                fingerprint["target_correlations"] = correlations
        
        return fingerprint
    
    def _get_recommended_transformations(self, dataset_fingerprint: Dict[str, Any], 
                                       problem_type: ProblemType) -> List[Dict[str, Any]]:
        """
        Recomenda transformações com base em meta-aprendizado.
        
        Args:
            dataset_fingerprint (dict): Características do dataset.
            problem_type (ProblemType): Tipo do problema.
            
        Returns:
            list: Lista de transformações recomendadas.
        """
        if not self.meta_db:
            logger.warning("Banco de dados de meta-aprendizado vazio. Usando transformações padrão.")
            return self._get_default_transformations(dataset_fingerprint, problem_type)
        
        # Filtra o banco de dados para o mesmo tipo de problema
        problem_type_str = problem_type.name
        matching_entries = [
            entry for entry in self.meta_db 
            if entry["problem_type"] == problem_type_str
        ]
        
        if not matching_entries:
            logger.warning(f"Nenhuma entrada no meta-DB para o problema tipo {problem_type_str}.")
            return self._get_default_transformations(dataset_fingerprint, problem_type)
        
        # Pontuação para cada transformação baseada em sua eficácia em datasets similares
        transformation_scores = {}
        
        for entry in matching_entries:
            # Calcula similaridade com o dataset atual
            similarity = self._calculate_similarity(dataset_fingerprint, entry["dataset_fingerprint"])
            
            # Peso baseado na similaridade
            weight = max(0.01, similarity)
            
            for transformation in entry["transformations"]:
                # A chave é a combinação de coluna e transformação
                key = f"{transformation['column']}_{transformation['transformation']}"
                
                # Score ponderado pela similaridade e melhoria
                improvement = transformation.get("improvement", 0.1)  # Valor padrão se não disponível
                score = weight * improvement
                
                if key in transformation_scores:
                    transformation_scores[key]["score"] += score
                    transformation_scores[key]["count"] += 1
                else:
                    transformation_scores[key] = {
                        "score": score,
                        "count": 1,
                        "column": transformation["column"],
                        "transformation": transformation["transformation"]
                    }
        
        # Normaliza os scores e seleciona as melhores transformações
        recommended = []
        
        for key, info in transformation_scores.items():
            # Média dos scores
            avg_score = info["score"] / info["count"]
            
            if avg_score > 0:
                recommended.append({
                    "column": info["column"],
                    "transformation": info["transformation"],
                    "score": avg_score
                })
        
        # Ordena por score e retorna as melhores
        recommended = sorted(recommended, key=lambda x: x["score"], reverse=True)
        
        # Limita o número de transformações
        max_transformations = min(20, len(recommended))
        
        logger.info(f"Meta-aprendizado recomendou {max_transformations} transformações.")
        
        return recommended[:max_transformations]
    
    def _get_default_transformations(self, dataset_fingerprint: Dict[str, Any],
                                   problem_type: ProblemType) -> List[Dict[str, Any]]:
        """
        Retorna transformações padrão quando não há informações suficientes no meta-DB.
        
        Args:
            dataset_fingerprint (dict): Características do dataset.
            problem_type (ProblemType): Tipo do problema.
            
        Returns:
            list: Lista de transformações padrão.
        """
        default_transformations = []
        
        # Identifica colunas numéricas, categóricas e temporais do fingerprint
        numeric_columns = []
        categorical_columns = []
        datetime_columns = []
        
        for col_name, stats in dataset_fingerprint.get("column_stats", {}).items():
            # Pula a coluna alvo
            if col_name == dataset_fingerprint.get("target_name"):
                continue
                
            # Tenta determinar o tipo com base nas estatísticas
            if "mean" in stats and "std" in stats:
                numeric_columns.append(col_name)
            elif "num_categories" in stats:
                categorical_columns.append(col_name)
        
        # Transformações padrão por tipo de problema
        if problem_type == ProblemType.CLASSIFICATION:
            # Para classificação, normalmente transformações categóricas e numéricas simples são úteis
            
            # Para numéricas, frequentemente standardization e binning são úteis
            for col in numeric_columns[:5]:  # Limita a 5 colunas
                default_transformations.append({
                    "column": col,
                    "transformation": "STANDARDIZE",
                    "score": 0.8
                })
                
                default_transformations.append({
                    "column": col,
                    "transformation": "BIN",
                    "score": 0.7
                })
            
            # Para categóricas, one-hot encoding e target encoding são úteis
            for col in categorical_columns[:5]:  # Limita a 5 colunas
                default_transformations.append({
                    "column": col,
                    "transformation": "ONE_HOT_ENCODE",
                    "score": 0.9
                })
                
                default_transformations.append({
                    "column": col,
                    "transformation": "TARGET_ENCODE",
                    "score": 0.8
                })
                
        elif problem_type == ProblemType.REGRESSION:
            # Para regressão, transformações matemáticas são frequentemente úteis
            
            for col in numeric_columns[:5]:  # Limita a 5 colunas
                default_transformations.append({
                    "column": col,
                    "transformation": "LOG",
                    "score": 0.7
                })
                
                default_transformations.append({
                    "column": col,
                    "transformation": "SQUARE_ROOT",
                    "score": 0.6
                })
                
                default_transformations.append({
                    "column": col,
                    "transformation": "STANDARDIZE",
                    "score": 0.8
                })
            
            # Para categóricas em regressão, target encoding é geralmente melhor
            for col in categorical_columns[:5]:
                default_transformations.append({
                    "column": col,
                    "transformation": "TARGET_ENCODE",
                    "score": 0.9
                })
                
        elif problem_type == ProblemType.TIME_SERIES:
            # Para séries temporais, transformações temporais e defasagens são cruciais
            
            for col in datetime_columns[:3]:
                default_transformations.append({
                    "column": col,
                    "transformation": "EXTRACT_MONTH",
                    "score": 0.9
                })
                
                default_transformations.append({
                    "column": col,
                    "transformation": "EXTRACT_WEEKDAY",
                    "score": 0.8
                })
                
                default_transformations.append({
                    "column": col,
                    "transformation": "LAG",
                    "score": 1.0
                })
                
                default_transformations.append({
                    "column": col,
                    "transformation": "ROLLING_MEAN",
                    "score": 0.9
                })
            
            # Para numéricas em séries temporais, geralmente queremos médias móveis
            for col in numeric_columns[:5]:
                default_transformations.append({
                    "column": col,
                    "transformation": "ROLLING_MEAN",
                    "score": 0.8
                })
                
                default_transformations.append({
                    "column": col,
                    "transformation": "ROLLING_STD",
                    "score": 0.7
                })
                
        elif problem_type == ProblemType.TEXT:
            # Para problemas de texto, transformações de texto são essenciais
            
            # Encontra colunas de texto
            text_columns = [col for col, stats in dataset_fingerprint.get("column_stats", {}).items()
                          if col != dataset_fingerprint.get("target_name") and 
                          "mean" not in stats and "num_categories" not in stats]
            
            for col in text_columns[:3]:
                default_transformations.append({
                    "column": col,
                    "transformation": "TF_IDF",
                    "score": 0.9
                })
                
                default_transformations.append({
                    "column": col,
                    "transformation": "COUNT_VECTORIZE",
                    "score": 0.8
                })
                
                default_transformations.append({
                    "column": col,
                    "transformation": "WORD_EMBEDDING",
                    "score": 0.7
                })
        
        # Adiciona algumas transformações de interação para numericas
        if len(numeric_columns) >= 2:
            for col in numeric_columns[:3]:
                default_transformations.append({
                    "column": col,
                    "transformation": "MULTIPLY",
                    "score": 0.6
                })
                
                default_transformations.append({
                    "column": col,
                    "transformation": "ADD",
                    "score": 0.5
                })
        
        logger.info(f"Geradas {len(default_transformations)} transformações padrão para problema do tipo {problem_type.name}")
        
        return default_transformations
    
    def _calculate_similarity(self, fingerprint1: Dict[str, Any], 
                            fingerprint2: Dict[str, Any]) -> float:
        """
        Calcula a similaridade entre dois datasets baseado em seus fingerprints.
        
        Args:
            fingerprint1 (dict): Características do primeiro dataset.
            fingerprint2 (dict): Características do segundo dataset.
            
        Returns:
            float: Score de similaridade entre 0 e 1.
        """
        # Inicializa score
        similarity_score = 0.0
        total_weight = 0.0
        
        # Similaridade no tamanho do dataset (número de linhas)
        rows1 = fingerprint1.get("num_rows", 0)
        rows2 = fingerprint2.get("num_rows", 0)
        
        if rows1 > 0 and rows2 > 0:
            # Normaliza diferença de tamanho
            size_ratio = min(rows1, rows2) / max(rows1, rows2)
            similarity_score += size_ratio * 0.1
            total_weight += 0.1
        
        # Similaridade na estrutura (número de colunas)
        cols1 = fingerprint1.get("num_cols", 0)
        cols2 = fingerprint2.get("num_cols", 0)
        
        if cols1 > 0 and cols2 > 0:
            # Normaliza diferença de número de colunas
            cols_ratio = min(cols1, cols2) / max(cols1, cols2)
            similarity_score += cols_ratio * 0.1
            total_weight += 0.1
        
        # Similaridade nos tipos de colunas
        types1 = fingerprint1.get("column_types", {})
        types2 = fingerprint2.get("column_types", {})
        
        if types1 and types2:
            # Calcula similaridade de distribuição de tipos
            type_similarity = 0.0
            type_weight = 0.0
            
            for dtype in DataType:
                dtype_name = dtype.name
                count1 = types1.get(dtype_name, 0)
                count2 = types2.get(dtype_name, 0)
                
                # Se ambos os datasets têm esse tipo
                if count1 > 0 or count2 > 0:
                    # Normaliza diferença na proporção desse tipo
                    ratio = min(count1, count2) / max(count1, count2) if max(count1, count2) > 0 else 0
                    
                    # Peso proporcional à presença do tipo
                    weight = (count1 + count2) / (cols1 + cols2) if (cols1 + cols2) > 0 else 0
                    
                    type_similarity += ratio * weight
                    type_weight += weight
            
            if type_weight > 0:
                similarity_score += (type_similarity / type_weight) * 0.4
                total_weight += 0.4
        
        # Similaridade na porcentagem de valores ausentes
        missing1 = fingerprint1.get("missing_values_pct", 0)
        missing2 = fingerprint2.get("missing_values_pct", 0)
        
        # Calcular similaridade na proporção de valores ausentes
        missing_diff = abs(missing1 - missing2)
        missing_similarity = max(0, 1 - (missing_diff / 100))  # 0 a 1
        
        similarity_score += missing_similarity * 0.1
        total_weight += 0.1
        
        # Similaridade nas correlações com target (se disponíveis)
        corr1 = fingerprint1.get("target_correlations", {})
        corr2 = fingerprint2.get("target_correlations", {})
        
        if corr1 and corr2:
            # Encontrar colunas em comum
            common_cols = set(corr1.keys()) & set(corr2.keys())
            
            if common_cols:
                corr_similarity = 0.0
                
                for col in common_cols:
                    # Similaridade da correlação para cada coluna
                    corr_diff = abs(corr1[col] - corr2[col])
                    col_similarity = max(0, 1 - corr_diff)  # 0 a 1
                    corr_similarity += col_similarity
                
                # Média da similaridade de correlação
                corr_similarity /= len(common_cols)
                
                similarity_score += corr_similarity * 0.3
                total_weight += 0.3
        
        # Normaliza o score final
        if total_weight > 0:
            final_similarity = similarity_score / total_weight
        else:
            final_similarity = 0.0
        
        return final_similarity
    
    def _apply_transformation(self, df: pd.DataFrame, column_name: str, 
                            transformation_type: TransformationType, 
                            dataset_info: DatasetInfo) -> Optional[TransformationResult]:
        """
        Aplica uma transformação específica a uma coluna.
        
        Args:
            df (pd.DataFrame): DataFrame a ser transformado.
            column_name (str): Nome da coluna a ser transformada.
            transformation_type (TransformationType): Tipo de transformação a aplicar.
            dataset_info (DatasetInfo): Informações do dataset.
            
        Returns:
            TransformationResult or None: Resultado da transformação ou None se falhou.
        """
        # Determina a categoria do transformador com base no tipo
        transformer_category = self._get_transformer_category(transformation_type)
        
        if transformer_category in self.transformers:
            transformer = self.transformers[transformer_category]
            
            try:
                # Tenta aplicar a transformação
                result = transformer.transform(df, column_name, transformation_type, dataset_info)
                return result
            except Exception as e:
                logger.warning(f"Erro ao aplicar transformação {transformation_type.name} " 
                             f"em {column_name}: {str(e)}")
                return None
        else:
            logger.warning(f"Categoria de transformador '{transformer_category}' não encontrada")
            return None
    
    def _get_transformer_category(self, transformation_type: TransformationType) -> str:
        """
        Determina a categoria do transformador com base no tipo de transformação.
        
        Args:
            transformation_type (TransformationType): Tipo de transformação.
            
        Returns:
            str: Categoria do transformador.
        """
        if transformation_type in [
            TransformationType.LOG, TransformationType.SQUARE_ROOT,
            TransformationType.SQUARE, TransformationType.STANDARDIZE,
            TransformationType.NORMALIZE, TransformationType.RECIPROCAL,
            TransformationType.BIN, TransformationType.POLYNOMIAL
        ]:
            return 'mathematical'
            
        elif transformation_type in [
            TransformationType.EXTRACT_DAY, TransformationType.EXTRACT_MONTH,
            TransformationType.EXTRACT_YEAR, TransformationType.EXTRACT_WEEKDAY,
            TransformationType.EXTRACT_HOUR, TransformationType.LAG,
            TransformationType.ROLLING_MEAN, TransformationType.ROLLING_STD
        ]:
            return 'temporal'
            
        elif transformation_type in [
            TransformationType.ONE_HOT_ENCODE, TransformationType.LABEL_ENCODE,
            TransformationType.TARGET_ENCODE, TransformationType.FREQUENCY_ENCODE
        ]:
            return 'categorical'
            
        elif transformation_type in [
            TransformationType.TF_IDF, TransformationType.COUNT_VECTORIZE,
            TransformationType.WORD_EMBEDDING
        ]:
            return 'text'
            
        elif transformation_type in [
            TransformationType.MULTIPLY, TransformationType.DIVIDE,
            TransformationType.ADD, TransformationType.SUBTRACT
        ]:
            return 'interaction'
            
        elif transformation_type in [
            TransformationType.GROUP_MEAN, TransformationType.GROUP_MEDIAN,
            TransformationType.GROUP_MAX, TransformationType.GROUP_MIN,
            TransformationType.GROUP_COUNT
        ]:
            return 'grouping'
            
        # Fallback para transformador matemático
        return 'mathematical'
    
    def _evaluate_performance(self, X_train: pd.DataFrame, X_val: pd.DataFrame,
                           y_train: pd.Series, y_val: pd.Series,
                           problem_type: ProblemType) -> float:
        """
        Avalia o desempenho do modelo com as features atuais.
        
        Args:
            X_train (pd.DataFrame): Features de treinamento.
            X_val (pd.DataFrame): Features de validação.
            y_train (pd.Series): Valores alvo de treinamento.
            y_val (pd.Series): Valores alvo de validação.
            problem_type (ProblemType): Tipo do problema.
            
        Returns:
            float: Pontuação de desempenho.
        """
        # Seleciona o modelo apropriado com base no tipo de problema
        if problem_type == ProblemType.CLASSIFICATION:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            
            # Treina o modelo
            try:
                model.fit(X_train, y_train)
                
                # Avalia usando F1-score
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average='weighted')
                
                return score
            except Exception as e:
                logger.warning(f"Erro na avaliação de classificação: {str(e)}")
                return 0.0
                
        elif problem_type == ProblemType.REGRESSION:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            # Treina o modelo
            try:
                model.fit(X_train, y_train)
                
                # Avalia usando R²
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                
                return max(0, score)  # R² pode ser negativo, mas queremos um score não-negativo
            except Exception as e:
                logger.warning(f"Erro na avaliação de regressão: {str(e)}")
                return 0.0
                
        elif problem_type == ProblemType.TIME_SERIES:
            # Para séries temporais, usamos regressão como aproximação
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            try:
                model.fit(X_train, y_train)
                
                # Avalia usando R²
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                
                return max(0, score)
            except Exception as e:
                logger.warning(f"Erro na avaliação de série temporal: {str(e)}")
                return 0.0
                
        elif problem_type == ProblemType.TEXT:
            # Para problemas de texto, usamos classificação como aproximação
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            
            try:
                model.fit(X_train, y_train)
                
                # Avalia usando F1-score
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average='weighted')
                
                return score
            except Exception as e:
                logger.warning(f"Erro na avaliação de texto: {str(e)}")
                return 0.0
        
        # Caso não reconhecido
        return 0.0
    
    def _load_meta_db(self) -> List[Dict[str, Any]]:
        """
        Carrega o banco de dados de meta-aprendizado.
        
        Returns:
            list: Lista de entradas do banco de dados de meta-aprendizado.
        """
        if os.path.exists(self.meta_db_path):
            try:
                with open(self.meta_db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Erro ao carregar banco de dados de meta-aprendizado: {str(e)}")
        
        # Se não existe ou falhou, retorna uma lista vazia
        return []
    
    def _save_meta_db(self) -> None:
        """
        Salva o banco de dados de meta-aprendizado.
        """
        try:
            # Cria o diretório se não existir
            os.makedirs(os.path.dirname(os.path.abspath(self.meta_db_path)), exist_ok=True)
            
            with open(self.meta_db_path, 'w') as f:
                json.dump(self.meta_db, f, indent=2)
                
            logger.info(f"Banco de dados de meta-aprendizado salvo em {self.meta_db_path}")
        except Exception as e:
            logger.warning(f"Erro ao salvar banco de dados de meta-aprendizado: {str(e)}")
    
    def _update_meta_db(self, dataset_fingerprint: Dict[str, Any], 
                      problem_type: ProblemType,
                      transformations: List[Dict[str, Any]]) -> None:
        """
        Atualiza o banco de dados de meta-aprendizado com novos resultados.
        
        Args:
            dataset_fingerprint (dict): Características do dataset.
            problem_type (ProblemType): Tipo do problema.
            transformations (list): Lista de transformações aplicadas e seus resultados.
        """
        if not transformations:
            return
            
        # Cria uma nova entrada no meta-DB
        new_entry = {
            "dataset_fingerprint": dataset_fingerprint,
            "problem_type": problem_type.name,
            "transformations": transformations,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Adiciona ao banco de dados
        self.meta_db.append(new_entry)
        
        # Limita o tamanho do meta-DB (mantém as entradas mais recentes)
        max_entries = 1000
        if len(self.meta_db) > max_entries:
            self.meta_db = self.meta_db[-max_entries:]
            
        # Salva o meta-DB atualizado
        self._save_meta_db()
        
        logger.info("Banco de dados de meta-aprendizado atualizado com novos resultados")