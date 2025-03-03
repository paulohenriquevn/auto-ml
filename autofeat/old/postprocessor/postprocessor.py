# postprocessor/postprocessor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
import logging
import sys
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, RFE
from sklearn.model_selection import cross_val_score
import json

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data_types import (
    DataType, ProblemType, DatasetInfo, ColumnInfo, 
    TransformationType, TransformationInfo
)

logger = logging.getLogger("AutoFE.PosProcessor")

class PosProcessor:
    """
    Módulo responsável pela seleção final de features e geração de relatórios.
    
    Este módulo realiza as seguintes operações:
    - Seleção de features mais importantes
    - Remoção de features redundantes
    - Geração de relatórios e estatísticas
    - Pontuação da qualidade do dataset
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o módulo PosProcessor.
        
        Args:
            config (dict, optional): Configurações para o PosProcessor.
        """
        self.config = config or {}
        self.feature_selection = self.config.get('feature_selection', True)
        self.min_importance_threshold = self.config.get('min_importance_threshold', 0.05)
        self.report_detail_level = self.config.get('report_detail_level', 'detailed')
        
        logger.info("Módulo PosProcessor inicializado")
    
    def process(self, df: pd.DataFrame, target_column: str, 
               problem_type: ProblemType, time_column: Optional[str] = None,
               module_reports: Optional[List[Dict[str, Any]]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Processa o DataFrame final aplicando seleção de features e gerando relatórios.
        
        Args:
            df (pd.DataFrame): DataFrame a ser processado.
            target_column (str): Nome da coluna alvo.
            problem_type (ProblemType): Tipo do problema.
            time_column (str, optional): Nome da coluna temporal.
            module_reports (list, optional): Relatórios dos módulos anteriores.
            
        Returns:
            tuple: DataFrame processado e relatório do processamento.
        """
        # Inicializa o relatório
        report = {
            "original_shape": df.shape,
            "problem_type": problem_type.name,
            "dataset_score": 0.0
        }
        
        # Obtém informações do dataset
        dataset_info = DatasetInfo.from_dataframe(df, target_column, time_column, problem_type)
        
        # Realiza a seleção de features (se habilitada)
        df_processed = df.copy()
        feature_importances = {}
        
        if self.feature_selection:
            df_processed, feature_importances, selection_report = self._select_features(
                df, target_column, problem_type
            )
            
            report["feature_selection"] = selection_report
        
        # Adiciona estatísticas do dataset
        report["dataset_statistics"] = self._calculate_dataset_statistics(df_processed, dataset_info)
        
        # Gera o relatório final
        summary_report = self._generate_summary_report(
            df_processed, dataset_info, feature_importances, module_reports
        )
        
        report["summary"] = summary_report
        
        # Calcula a pontuação do dataset
        report["dataset_score"] = self._calculate_dataset_score(
            df_processed, dataset_info, feature_importances, module_reports
        )
        
        # Adiciona informações finais
        report["final_shape"] = df_processed.shape
        report["num_features"] = df_processed.shape[1] - 1  # Exclui a coluna alvo
        
        logger.info(f"Pós-processamento concluído. Score final do dataset: {report['dataset_score']}/10")
        
        return df_processed, report
    
    def _select_features(self, df: pd.DataFrame, target_column: str, 
                       problem_type: ProblemType) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
        """
        Seleciona as features mais importantes do dataset.
        
        Args:
            df (pd.DataFrame): DataFrame a ser processado.
            target_column (str): Nome da coluna alvo.
            problem_type (ProblemType): Tipo do problema.
            
        Returns:
            tuple: DataFrame com features selecionadas, importâncias e relatório.
        """
        # Inicializa o relatório de seleção
        selection_report = {
            "before_selection": df.shape,
            "methods_applied": []
        }
        
        # Prepara os dados
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Verifica se há colunas suficientes para seleção
        if X.shape[1] <= 2:
            logger.info("Número insuficiente de features para seleção. Pulando etapa.")
            return df, {}, {"reason": "Número insuficiente de features", "features_kept": list(X.columns)}
        
        # Inicializa as colunas a manter
        columns_to_keep = set(X.columns)
        original_columns = len(columns_to_keep)
        
        # Passo 1: Remove features com variância próxima de zero
        try:
            var_threshold = VarianceThreshold(threshold=0.01)  # limiar de 1% de variância
            X_var = X.select_dtypes(include=np.number)  # Apenas colunas numéricas
            
            if X_var.shape[1] > 0:
                var_threshold.fit(X_var)
                
                # Obtém as colunas que passaram no teste
                var_support = var_threshold.get_support()
                var_kept_columns = X_var.columns[var_support].tolist()
                
                # Atualiza as colunas a manter
                removed_var = set(X_var.columns) - set(var_kept_columns)
                columns_to_keep -= removed_var
                
                selection_report["methods_applied"].append({
                    "method": "variance_threshold",
                    "n_removed": len(removed_var),
                    "removed_features": list(removed_var)
                })
                
                logger.info(f"Seleção por variância: {len(removed_var)} features removidas")
        except Exception as e:
            logger.warning(f"Erro na seleção por variância: {str(e)}")
        
        # Se o número de colunas restantes for muito baixo, volte ao conjunto original
        if len(columns_to_keep) < 2:
            columns_to_keep = set(X.columns)
        
        # Passo 2: Seleção baseada em modelo (Random Forest)
        try:
            # Seleciona apenas as colunas mantidas até agora
            X_selected = X[list(columns_to_keep)]
            
            # Usa diferentes modelos com base no tipo de problema
            if problem_type in [ProblemType.CLASSIFICATION, ProblemType.TEXT]:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:  # REGRESSION ou TIME_SERIES
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            # Treina o modelo
            model.fit(X_selected, y)
            
            # Calcula importâncias e seleciona features
            feature_importances = {col: imp for col, imp in zip(X_selected.columns, model.feature_importances_)}
            
            # Ordena por importância
            sorted_importances = sorted(
                feature_importances.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Seleciona features acima do threshold de importância
            important_features = [col for col, imp in sorted_importances 
                                if imp >= self.min_importance_threshold]
            
            # Garante um número mínimo de features (pelo menos 5 ou todas se houver menos)
            min_features = min(5, len(sorted_importances))
            top_features = [col for col, _ in sorted_importances[:min_features]]
            
            # Combina as listas
            model_kept_columns = list(set(important_features + top_features))
            
            # Atualiza as colunas a manter
            removed_model = set(X_selected.columns) - set(model_kept_columns)
            columns_to_keep -= removed_model
            
            selection_report["methods_applied"].append({
                "method": "model_based_selection",
                "n_removed": len(removed_model),
                "removed_features": list(removed_model),
                "importance_threshold": self.min_importance_threshold
            })
            
            logger.info(f"Seleção baseada em modelo: {len(removed_model)} features removidas")
        except Exception as e:
            logger.warning(f"Erro na seleção baseada em modelo: {str(e)}")
            # Em caso de erro, mantém todas as features
            feature_importances = {col: 1.0 / len(X.columns) for col in X.columns}
        
        # Passo 3: Remove colunas altamente correlacionadas
        try:
            # Seleciona apenas as colunas mantidas até agora
            X_selected = X[list(columns_to_keep)]
            
            # Calcula a matriz de correlação (apenas para numéricas)
            X_numeric = X_selected.select_dtypes(include=np.number)
            
            if X_numeric.shape[1] > 1:  # Precisa de pelo menos 2 colunas
                corr_matrix = X_numeric.corr().abs()
                
                # Identifica pares de features altamente correlacionadas
                high_corr_features = set()
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > 0.95:  # Threshold alto para correlação
                            col_i = corr_matrix.columns[i]
                            col_j = corr_matrix.columns[j]
                            
                            # Remove a feature com menor importância
                            if feature_importances.get(col_i, 0) < feature_importances.get(col_j, 0):
                                high_corr_features.add(col_i)
                            else:
                                high_corr_features.add(col_j)
                
                # Atualiza as colunas a manter
                columns_to_keep -= high_corr_features
                
                selection_report["methods_applied"].append({
                    "method": "correlation_removal",
                    "n_removed": len(high_corr_features),
                    "removed_features": list(high_corr_features),
                    "correlation_threshold": 0.95
                })
                
                logger.info(f"Remoção por correlação: {len(high_corr_features)} features removidas")
        except Exception as e:
            logger.warning(f"Erro na remoção por correlação: {str(e)}")
        
        # Seleciona apenas as colunas mantidas até agora
        kept_columns = list(columns_to_keep)
        
        # Garante um número mínimo de features
        if len(kept_columns) < 2:
            logger.warning("Muito poucas features selecionadas. Revertendo para o conjunto original.")
            kept_columns = list(X.columns)
        
        # Adiciona a coluna alvo
        kept_columns.append(target_column)
        
        # Cria o DataFrame final
        df_selected = df[kept_columns].copy()
        
        # Atualiza o relatório
        selection_report["after_selection"] = df_selected.shape
        selection_report["features_kept"] = len(kept_columns) - 1  # Exclui a coluna alvo
        selection_report["features_removed"] = original_columns - (len(kept_columns) - 1)
        
        logger.info(f"Seleção de features concluída: {selection_report['features_removed']} features removidas")
        
        # Retorna apenas as importâncias das features mantidas
        filtered_importances = {col: imp for col, imp in feature_importances.items() if col in kept_columns}
        
        return df_selected, filtered_importances, selection_report
    
    def _calculate_dataset_statistics(self, df: pd.DataFrame, 
                                    dataset_info: DatasetInfo) -> Dict[str, Any]:
        """
        Calcula estatísticas do dataset para o relatório.
        
        Args:
            df (pd.DataFrame): DataFrame processado.
            dataset_info (DatasetInfo): Informações sobre o dataset.
            
        Returns:
            dict: Estatísticas do dataset.
        """
        statistics = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "num_numeric_features": 0,
            "num_categorical_features": 0,
            "num_datetime_features": 0,
            "num_text_features": 0,
            "missing_values_pct": (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            "column_types": {}
        }
        
        # Contadores de tipos de colunas
        for col_info in dataset_info.columns:
            col_type = col_info.data_type.name
            
            if col_info.data_type == DataType.NUMERIC:
                statistics["num_numeric_features"] += 1
            elif col_info.data_type == DataType.CATEGORICAL:
                statistics["num_categorical_features"] += 1
            elif col_info.data_type == DataType.DATETIME:
                statistics["num_datetime_features"] += 1
            elif col_info.data_type == DataType.TEXT:
                statistics["num_text_features"] += 1
                
            statistics["column_types"][col_info.name] = col_type
            
            # Para colunas numéricas, adiciona estatísticas descritivas
            if col_info.data_type == DataType.NUMERIC and col_info.name in df.columns:
                statistics[f"{col_info.name}_stats"] = {
                    "mean": float(df[col_info.name].mean()) if not df[col_info.name].isna().all() else None,
                    "std": float(df[col_info.name].std()) if not df[col_info.name].isna().all() else None,
                    "min": float(df[col_info.name].min()) if not df[col_info.name].isna().all() else None,
                    "25%": float(df[col_info.name].quantile(0.25)) if not df[col_info.name].isna().all() else None,
                    "50%": float(df[col_info.name].quantile(0.5)) if not df[col_info.name].isna().all() else None,
                    "75%": float(df[col_info.name].quantile(0.75)) if not df[col_info.name].isna().all() else None,
                    "max": float(df[col_info.name].max()) if not df[col_info.name].isna().all() else None
                }
        
        return statistics
    
    def _generate_summary_report(self, df: pd.DataFrame, dataset_info: DatasetInfo,
                               feature_importances: Dict[str, float],
                               module_reports: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Gera um relatório resumido do processo de engenharia de features.
        
        Args:
            df (pd.DataFrame): DataFrame processado.
            dataset_info (DatasetInfo): Informações sobre o dataset.
            feature_importances (dict): Importância das features.
            module_reports (list, optional): Relatórios dos módulos anteriores.
            
        Returns:
            dict: Relatório resumido.
        """
        summary = {
            "data_shape": {
                "rows": len(df),
                "columns": len(df.columns)
            },
            "problem_type": dataset_info.problem_type.name if dataset_info.problem_type else "Unknown",
            "target_column": dataset_info.target_column,
            "transformations_summary": {},
            "top_features": []
        }
        
        # Agrega informações dos relatórios dos módulos
        if module_reports:
            # Extrai informações do relatório de pré-processamento
            preprocess_report = module_reports[0] if len(module_reports) > 0 else None
            if preprocess_report:
                summary["preprocessing"] = {
                    "columns_modified": preprocess_report.get("columns_modified", []),
                    "missing_values_handled": any("missing_values" in step for step in preprocess_report.get("steps", [])),
                    "outliers_handled": any("outliers" in step for step in preprocess_report.get("steps", [])),
                    "duplicates_removed": any("duplicates_removed" in step for step in preprocess_report.get("steps", []))
                }
            
            # Extrai informações do relatório do Explorer
            explorer_report = module_reports[1] if len(module_reports) > 1 else None
            if explorer_report:
                transformations = explorer_report.get("transformations_applied", [])
                
                # Conta os tipos de transformações aplicadas
                transformation_counts = {}
                for t in transformations:
                    t_type = t.get("type", "Unknown")
                    transformation_counts[t_type] = transformation_counts.get(t_type, 0) + 1
                
                summary["transformations_summary"]["counts"] = transformation_counts
                summary["transformations_summary"]["total"] = len(transformations)
                
                # Adiciona informação sobre melhoria
                if "improvement" in explorer_report:
                    summary["performance_improvement"] = {
                        "baseline": explorer_report.get("baseline_score", 0),
                        "after_transformations": explorer_report.get("final_score", 0),
                        "improvement": explorer_report.get("improvement", 0)
                    }
            
            # Extrai informações do relatório do Predictor
            predictor_report = module_reports[2] if len(module_reports) > 2 else None
            if predictor_report:
                meta_transformations = predictor_report.get("transformations_applied", [])
                
                summary["meta_learning"] = {
                    "used": predictor_report.get("meta_learning_used", False),
                    "transformations_applied": len(meta_transformations),
                    "fingerprint_match": predictor_report.get("dataset_fingerprint", {}).get("problem_type", "Unknown")
                }
        
        # Adiciona as top features por importância
        if feature_importances:
            # Ordena por importância
            sorted_importances = sorted(
                feature_importances.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Pega as 10 mais importantes (ou todas se houver menos)
            top_n = min(10, len(sorted_importances))
            top_features = []
            
            for i, (feature, importance) in enumerate(sorted_importances[:top_n]):
                top_features.append({
                    "rank": i + 1,
                    "feature": feature,
                    "importance": importance
                })
            
            summary["top_features"] = top_features
        
        return summary
    
    def _calculate_dataset_score(self, df: pd.DataFrame, dataset_info: DatasetInfo,
                               feature_importances: Dict[str, float],
                               module_reports: Optional[List[Dict[str, Any]]] = None) -> float:
        """
        Calcula uma pontuação para a qualidade do dataset após a engenharia de features.
        
        Args:
            df (pd.DataFrame): DataFrame processado.
            dataset_info (DatasetInfo): Informações sobre o dataset.
            feature_importances (dict): Importância das features.
            module_reports (list, optional): Relatórios dos módulos anteriores.
            
        Returns:
            float: Pontuação do dataset (0 a 10).
        """
        # Base: todos os datasets começam com 5 pontos
        score = 5.0
        
        # 1. Avalia a quantidade de dados
        rows = len(df)
        if rows > 10000:
            score += 1.0
        elif rows > 1000:
            score += 0.5
        elif rows < 100:
            score -= 1.0
        
        # 2. Avalia o balanceamento de classes para problemas de classificação
        if dataset_info.problem_type == ProblemType.CLASSIFICATION and dataset_info.target_column:
            target = df[dataset_info.target_column]
            class_counts = target.value_counts(normalize=True)
            
            # Verifica se alguma classe tem muito poucos exemplos
            if class_counts.min() < 0.1:  # Se a menor classe tem menos de 10%
                score -= 0.5
            if class_counts.min() < 0.05:  # Se a menor classe tem menos de 5%
                score -= 0.5
        
        # 3. Avalia a informatividade das features (baseado no Explorer/Predictor)
        if module_reports and len(module_reports) > 1:
            explorer_report = module_reports[1]
            if "improvement" in explorer_report:
                improvement = explorer_report.get("improvement", 0)
                
                # Bonifica melhorias significativas
                if improvement > 0.2:
                    score += 1.5
                elif improvement > 0.1:
                    score += 1.0
                elif improvement > 0.05:
                    score += 0.5
        
        # 4. Avalia a quantidade de features informativas
        if feature_importances:
            # Conta features com importância significativa
            significant_features = sum(1 for imp in feature_importances.values() if imp > 0.05)
            
            if significant_features > 10:
                score += 1.0
            elif significant_features > 5:
                score += 0.5
            elif significant_features < 2:
                score -= 0.5
        
        # 5. Penaliza valores ausentes
        missing_pct = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_pct > 20:
            score -= 1.0
        elif missing_pct > 10:
            score -= 0.5
        elif missing_pct < 1:
            score += 0.5
        
        # 6. Bonifica transformações aplicadas com sucesso
        if module_reports:
            # Conta transformações aplicadas
            total_transformations = 0
            
            if len(module_reports) > 1 and "transformations_applied" in module_reports[1]:
                total_transformations += len(module_reports[1]["transformations_applied"])
                
            if len(module_reports) > 2 and "transformations_applied" in module_reports[2]:
                total_transformations += len(module_reports[2]["transformations_applied"])
            
            if total_transformations > 15:
                score += 1.0
            elif total_transformations > 8:
                score += 0.5
        
        # Garantir que o score esteja entre 0 e 10
        score = max(0, min(10, score))
        
        return round(score, 1)