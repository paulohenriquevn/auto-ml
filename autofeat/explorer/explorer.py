# explorer/explorer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
import logging
import sys
import os
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
import itertools

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data_types import (
    DataType, ProblemType, DatasetInfo, ColumnInfo, 
    TransformationType, TransformationInfo, TransformationResult
)
from explorer.transformers import (
    MathematicalTransformer, 
    TemporalTransformer,
    CategoricalTransformer,
    TextTransformer,
    InteractionTransformer,
    GroupingTransformer
)

logger = logging.getLogger("AutoFE.Explorer")

class Explorer:
    """
    Módulo responsável por explorar e gerar novas features.
    
    Este módulo implementa a "Árvore de Transformações" que navega pelo espaço
    de possíveis transformações e gera novas features a partir das existentes.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o módulo Explorer.
        
        Args:
            config (dict, optional): Configurações para o Explorer.
        """
        self.config = config or {}
        self.max_depth = self.config.get('max_depth', 3)
        self.min_feature_importance = self.config.get('min_feature_importance', 0.01)
        self.max_features = self.config.get('max_features', 100)
        self.transformations_applied = []
        
        # Inicializa os transformadores
        self.transformers = {
            'mathematical': MathematicalTransformer(),
            'temporal': TemporalTransformer(),
            'categorical': CategoricalTransformer(),
            'text': TextTransformer(),
            'interaction': InteractionTransformer(),
            'grouping': GroupingTransformer()
        }
        
        logger.info("Módulo Explorer inicializado")
    
    def process(self, df: pd.DataFrame, target_column: str, 
               problem_type: ProblemType, time_column: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Processa o DataFrame gerando novas features através da árvore de transformações.
        
        Args:
            df (pd.DataFrame): DataFrame a ser processado.
            target_column (str): Nome da coluna alvo.
            problem_type (ProblemType): Tipo do problema.
            time_column (str, optional): Nome da coluna temporal.
            
        Returns:
            tuple: DataFrame com novas features e relatório do processamento.
        """
        # Reinicializa a lista de transformações aplicadas
        self.transformations_applied = []
        
        # Inicializa o relatório
        report = {
            "original_shape": df.shape,
            "problem_type": problem_type.name,
            "transformations_applied": [],
            "feature_importances": {}
        }
        
        # Obtém informações do dataset
        dataset_info = DatasetInfo.from_dataframe(df, target_column, time_column, problem_type)
        
        # Separa os dados em treinamento e validação
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split para métrica de baseline
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Avalia o desempenho baseline do modelo sem novas features
        baseline_score = self._evaluate_performance(X_train, X_val, y_train, y_val, problem_type)
        report["baseline_score"] = baseline_score
        
        logger.info(f"Score baseline: {baseline_score}")
        
        # Inicia a exploração da árvore de transformações
        df_transformed = df.copy()
        
        # Obtém as features de maior importância no conjunto original
        feature_importances = self._calculate_feature_importance(
            X_train, y_train, problem_type
        )
        
        # Transforma o dataframe original aplicando as transformações
        df_transformed, exploration_report = self._explore_transformations(
            df_transformed, dataset_info, feature_importances
        )
        
        # Atualiza o relatório com as transformações aplicadas
        report["transformations_applied"] = [t.get_report() for t in self.transformations_applied]
        
        # Avalia o desempenho final
        X_transformed = df_transformed.drop(columns=[target_column])
        
        # Final split para avaliar desempenho
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X_transformed, df_transformed[target_column], test_size=0.2, random_state=42
        )
        
        final_score = self._evaluate_performance(
            X_train_final, X_val_final, y_train_final, y_val_final, problem_type
        )
        
        report["final_score"] = final_score
        report["improvement"] = final_score - baseline_score
        report["final_shape"] = df_transformed.shape
        report["new_features_count"] = df_transformed.shape[1] - df.shape[1]
        
        # Calcula importância final das features
        final_importances = self._calculate_feature_importance(
            X_train_final, y_train_final, problem_type
        )
        
        # Adiciona importância final ao relatório
        report["feature_importances"] = final_importances
        
        logger.info(f"Exploração concluída. Score final: {final_score} (Melhoria: {final_score - baseline_score})")
        
        return df_transformed, report
    
    def _explore_transformations(self, df: pd.DataFrame, dataset_info: DatasetInfo, 
                               feature_importances: Dict[str, float]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Explora possíveis transformações usando uma abordagem de árvore.
        
        Args:
            df (pd.DataFrame): DataFrame a ser transformado.
            dataset_info (DatasetInfo): Informações sobre o dataset.
            feature_importances (dict): Importância inicial das features.
            
        Returns:
            tuple: DataFrame transformado e relatório da exploração.
        """
        report = {
            "iterations": 0,
            "transformations_tested": 0,
            "transformations_applied": 0
        }
        
        # Lista para acompanhar as features mais importantes a cada iteração
        important_features = sorted(
            feature_importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:min(10, len(feature_importances))]
        
        current_df = df.copy()
        target_column = dataset_info.target_column
        
        # Features já testadas
        tested_transformations = set()
        
        # Número máximo de features para o resultado final
        max_features = min(self.max_features, df.shape[1] * 3)
        
        # Executa a exploração por iterações
        for depth in range(self.max_depth):
            logger.info(f"Iniciando nível {depth+1} da árvore de transformações")
            iterations_at_depth = 0
            
            # Para cada coluna importante, tenta aplicar transformações
            for feature_name, importance in important_features:
                
                # Pula se a feature não existe mais (pode ter sido transformada)
                if feature_name not in current_df.columns:
                    continue
                    
                # Obtém informações da coluna
                col_info = next((col for col in dataset_info.columns if col.name == feature_name), None)
                if not col_info:
                    # Tenta obter informações atualizadas se a coluna foi adicionada em iterações anteriores
                    col_series = current_df[feature_name]
                    col_info = ColumnInfo.from_series(col_series)
                
                # Lista de possíveis transformações para esta coluna
                potential_transformations = self._get_potential_transformations(col_info, dataset_info)
                
                for transformation_type in potential_transformations:
                    # Gera uma chave única para esta transformação
                    transformation_key = f"{feature_name}_{transformation_type.name}"
                    
                    # Pula se já foi testada
                    if transformation_key in tested_transformations:
                        continue
                        
                    tested_transformations.add(transformation_key)
                    report["transformations_tested"] += 1
                    
                    # Aplica a transformação
                    transformation_result = self._apply_transformation(
                        current_df, feature_name, transformation_type, dataset_info
                    )
                    
                    if transformation_result:
                        # Avalia a qualidade da transformação
                        new_df = transformation_result.transformed_data
                        
                        # Se o score melhorou, mantém a transformação
                        if transformation_result.performance_score > 0:
                            current_df = new_df
                            
                            # Registra a transformação
                            self.transformations_applied.append(
                                TransformationInfo(
                                    transformation_type=transformation_type,
                                    params={},  # Parâmetros específicos seriam adicionados aqui
                                    input_columns=[feature_name],
                                    output_columns=transformation_result.created_columns,
                                    score=transformation_result.performance_score
                                )
                            )
                            
                            report["transformations_applied"] += 1
                            
                            logger.info(f"Transformação aplicada: {transformation_type.name} em {feature_name}")
                            logger.info(f"Novas colunas: {transformation_result.created_columns}")
                            logger.info(f"Score: {transformation_result.performance_score}")
                            
                            # Atualiza a lista de features importantes
                            if transformation_result.feature_importance:
                                important_features = sorted(
                                    transformation_result.feature_importance.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True
                                )[:min(10, len(transformation_result.feature_importance))]
                    
                    iterations_at_depth += 1
                    
                    # Verifica se já atingiu o número máximo de features
                    if current_df.shape[1] >= max_features:
                        logger.info(f"Número máximo de features atingido: {max_features}")
                        return current_df, report
            
            report["iterations"] += iterations_at_depth
            
            # Se não houve melhorias neste nível, para a exploração
            if iterations_at_depth == 0:
                logger.info(f"Sem melhorias no nível {depth+1}. Parando exploração.")
                break
                
            logger.info(f"Concluído nível {depth+1}. Realizadas {iterations_at_depth} iterações.")
        
        return current_df, report
    
    def _get_potential_transformations(self, col_info: ColumnInfo, 
                                     dataset_info: DatasetInfo) -> List[TransformationType]:
        """
        Determina quais transformações podem ser aplicadas a uma coluna.
        
        Args:
            col_info (ColumnInfo): Informações da coluna.
            dataset_info (DatasetInfo): Informações do dataset.
            
        Returns:
            list: Lista de transformações potenciais.
        """
        potential_transformations = []
        
        # Com base no tipo de dados, sugere transformações apropriadas
        if col_info.data_type == DataType.NUMERIC:
            potential_transformations.extend([
                TransformationType.LOG,
                TransformationType.SQUARE_ROOT,
                TransformationType.SQUARE,
                TransformationType.STANDARDIZE,
                TransformationType.NORMALIZE,
                TransformationType.BIN
            ])
            
            # Se houver muitos valores únicos, sugere polinomiais
            if col_info.num_unique_values and col_info.num_unique_values > 10:
                potential_transformations.append(TransformationType.POLYNOMIAL)
                
        elif col_info.data_type == DataType.CATEGORICAL:
            potential_transformations.extend([
                TransformationType.ONE_HOT_ENCODE,
                TransformationType.LABEL_ENCODE,
                TransformationType.FREQUENCY_ENCODE
            ])
            
            # Se houver uma coluna alvo, sugere target encoding
            if dataset_info.problem_type in [ProblemType.CLASSIFICATION, ProblemType.REGRESSION]:
                potential_transformations.append(TransformationType.TARGET_ENCODE)
                
        elif col_info.data_type == DataType.DATETIME:
            potential_transformations.extend([
                TransformationType.EXTRACT_DAY,
                TransformationType.EXTRACT_MONTH,
                TransformationType.EXTRACT_YEAR,
                TransformationType.EXTRACT_WEEKDAY,
                TransformationType.EXTRACT_HOUR
            ])
            
            # Se for um problema de série temporal, sugere transformações de lag
            if dataset_info.problem_type == ProblemType.TIME_SERIES:
                potential_transformations.extend([
                    TransformationType.LAG,
                    TransformationType.ROLLING_MEAN,
                    TransformationType.ROLLING_STD
                ])
                
        elif col_info.data_type == DataType.TEXT:
            potential_transformations.extend([
                TransformationType.TF_IDF,
                TransformationType.COUNT_VECTORIZE
            ])
            
            # Embedding pode ser muito caro computacionalmente, então é mais restrito
            if dataset_info.problem_type == ProblemType.TEXT:
                potential_transformations.append(TransformationType.WORD_EMBEDDING)
        
        # Se a coluna não for alvo nem temporal, considerar interações com outras colunas
        if not col_info.is_target and not col_info.is_time_column and col_info.data_type == DataType.NUMERIC:
            potential_transformations.extend([
                TransformationType.MULTIPLY,
                TransformationType.DIVIDE,
                TransformationType.ADD,
                TransformationType.SUBTRACT
            ])
        
        return potential_transformations
    
    def _apply_transformation(self, df: pd.DataFrame, feature_name: str, 
                        transformation_type: TransformationType, 
                        dataset_info: DatasetInfo) -> Optional[TransformationResult]:
        """
        Aplica uma transformação específica a uma feature.
        
        Args:
            df (pd.DataFrame): DataFrame a ser transformado.
            feature_name (str): Nome da feature a ser transformada.
            transformation_type (TransformationType): Tipo de transformação a aplicar.
            dataset_info (DatasetInfo): Informações do dataset.
            
        Returns:
            TransformationResult or None: Resultado da transformação ou None se falhou.
        """
        try:
            X = df.drop(columns=[dataset_info.target_column])
            y = df[dataset_info.target_column]
            
            # Aplica a transformação com base no tipo
            transformer_category = self._get_transformer_category(transformation_type)
            transformer = self.transformers[transformer_category]
            
            # Se estamos transformando uma coluna de data em um problema de série temporal,
            # priorizamos transformações temporais específicas
            if (pd.api.types.is_datetime64_dtype(df[feature_name]) and 
                dataset_info.problem_type == ProblemType.TIME_SERIES and
                transformer_category == 'temporal'):
                
                # Força continuar com a transformação
                result = transformer.transform(df, feature_name, transformation_type, dataset_info)
                
                if result and len(result.created_columns) > 0:
                    # Para transformações de data, atribuímos automaticamente um score positivo
                    # já que as colunas de data são cruciais para séries temporais
                    result.performance_score = 0.1
                    
                    # Calcula importância das features no novo conjunto (omitindo colunas de data)
                    X_new = result.transformed_data.drop(columns=[dataset_info.target_column])
                    X_processed = X_new.copy()
                    
                    datetime_cols = [col for col in X_new.columns if pd.api.types.is_datetime64_dtype(X_new[col])]
                    if datetime_cols:
                        X_processed = X_processed.drop(columns=datetime_cols)
                    
                    # Treina um modelo para obter importâncias de features
                    if dataset_info.problem_type == ProblemType.TIME_SERIES:
                        model = RandomForestRegressor(n_estimators=50, random_state=42)
                        try:
                            model.fit(X_processed, result.transformed_data[dataset_info.target_column])
                            
                            # Mapeia importâncias para features
                            feature_importance = {}
                            for i, col_name in enumerate(X_processed.columns):
                                feature_importance[col_name] = model.feature_importances_[i]
                                
                            # Para colunas de data, atribuímos uma importância padrão
                            for col in datetime_cols:
                                feature_importance[col] = 0.05
                                
                            result.feature_importance = feature_importance
                        except:
                            # Se ocorrer erro, atribui importâncias iguais
                            result.feature_importance = {col: 1.0/len(X_new.columns) for col in X_new.columns}
                    
                    return result
            
            # Para outras transformações, segue o fluxo normal
            result = transformer.transform(df, feature_name, transformation_type, dataset_info)
            
            if result and len(result.created_columns) > 0:
                # Avalia a qualidade da transformação
                df_transformed = result.transformed_data
                
                # Gera X e y transformados
                X_new = df_transformed.drop(columns=[dataset_info.target_column])
                
                # Split de treino e validação
                X_train, X_val, y_train, y_val = train_test_split(
                    X_new, y, test_size=0.2, random_state=42
                )
                
                # Calcula o score antes e depois
                score_before = self._evaluate_performance(
                    X_train[X.columns], X_val[X.columns], y_train, y_val, dataset_info.problem_type
                )
                
                score_after = self._evaluate_performance(
                    X_train, X_val, y_train, y_val, dataset_info.problem_type
                )
                
                # Calcula melhoria relativa
                relative_improvement = (score_after - score_before) / max(0.001, abs(score_before))
                
                # Define um limite mínimo de melhoria (1%)
                if relative_improvement > 0.01:
                    # Calcula importância das features no novo conjunto
                    feature_importance = self._calculate_feature_importance(
                        X_train, y_train, dataset_info.problem_type
                    )
                    
                    result.performance_score = relative_improvement
                    result.feature_importance = feature_importance
                    
                    return result
                else:
                    logger.debug(f"Transformação {transformation_type.name} em {feature_name} " 
                                f"não trouxe melhoria significativa: {relative_improvement:.4f}")
            
            return None
            
        except Exception as e:
            logger.warning(f"Erro ao aplicar transformação {transformation_type.name} " 
                        f"em {feature_name}: {str(e)}")
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
        # Pré-processamento para remover colunas de data/hora antes da avaliação
        X_train_processed = X_train.copy()
        X_val_processed = X_val.copy()
        
        # Identifica e remove colunas de datetime para evitar erros de tipo
        datetime_cols = []
        for col in X_train.columns:
            if pd.api.types.is_datetime64_dtype(X_train[col]):
                datetime_cols.append(col)
        
        # Remove colunas de datetime
        if datetime_cols:
            X_train_processed = X_train_processed.drop(columns=datetime_cols)
            X_val_processed = X_val_processed.drop(columns=datetime_cols)
            
            # Se não sobrarem colunas, retorna 0 (indicando que precisamos gerar features)
            if len(X_train_processed.columns) == 0:
                return 0.0
        
        # Seleciona o modelo apropriado com base no tipo de problema
        if problem_type == ProblemType.CLASSIFICATION:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            
            # Treina o modelo
            try:
                model.fit(X_train_processed, y_train)
                
                # Avalia usando F1-score
                y_pred = model.predict(X_val_processed)
                score = f1_score(y_val, y_pred, average='weighted')
                
                return score
            except Exception as e:
                logger.warning(f"Erro na avaliação de classificação: {str(e)}")
                return 0.0
                
        elif problem_type == ProblemType.REGRESSION:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            # Treina o modelo
            try:
                model.fit(X_train_processed, y_train)
                
                # Avalia usando R²
                y_pred = model.predict(X_val_processed)
                score = r2_score(y_val, y_pred)
                
                return max(0, score)  # R² pode ser negativo, mas queremos um score não-negativo
            except Exception as e:
                logger.warning(f"Erro na avaliação de regressão: {str(e)}")
                return 0.0
                
        elif problem_type == ProblemType.TIME_SERIES:
            # Para séries temporais, usamos regressão como aproximação
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            try:
                model.fit(X_train_processed, y_train)
                
                # Avalia usando R²
                y_pred = model.predict(X_val_processed)
                score = r2_score(y_val, y_pred)
                
                return max(0, score)
            except Exception as e:
                logger.warning(f"Erro na avaliação de série temporal: {str(e)}")
                return 0.0
                
        elif problem_type == ProblemType.TEXT:
            # Para problemas de texto, usamos classificação como aproximação
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            
            try:
                model.fit(X_train_processed, y_train)
                
                # Avalia usando F1-score
                y_pred = model.predict(X_val_processed)
                score = f1_score(y_val, y_pred, average='weighted')
                
                return score
            except Exception as e:
                logger.warning(f"Erro na avaliação de texto: {str(e)}")
                return 0.0
        
        # Caso não reconhecido
        return 0.0



    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series, 
                                problem_type: ProblemType) -> Dict[str, float]:
        """
        Calcula a importância das features usando um modelo de árvore.
        
        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Valores alvo.
            problem_type (ProblemType): Tipo do problema.
            
        Returns:
            dict: Mapeamento de nomes de features para seus valores de importância.
        """
        # Pré-processamento para remover colunas de data/hora antes do cálculo
        X_processed = X.copy()
        
        # Identifica e remove colunas de datetime para evitar erros de tipo
        datetime_cols = []
        for col in X.columns:
            if pd.api.types.is_datetime64_dtype(X[col]):
                datetime_cols.append(col)
        
        # Remove colunas de datetime
        if datetime_cols:
            X_processed = X_processed.drop(columns=datetime_cols)
        
        # Seleciona o modelo apropriado com base no tipo de problema
        if problem_type in [ProblemType.CLASSIFICATION, ProblemType.TEXT]:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:  # REGRESSION ou TIME_SERIES
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        try:
            # Treina o modelo
            model.fit(X_processed, y)
            
            # Obtém importâncias das features
            importances = model.feature_importances_
            
            # Mapeia nomes de features para importâncias
            feature_importance = {}
            for i, feature_name in enumerate(X_processed.columns):
                feature_importance[feature_name] = importances[i]
            
            # Para colunas datetime removidas, atribui importância zero
            for col in datetime_cols:
                feature_importance[col] = 0.0
                
            return feature_importance
            
        except Exception as e:
            logger.warning(f"Erro ao calcular importância das features: {str(e)}")
            
            # Retorna um valor padrão em caso de erro
            return {feature: 1.0 / len(X.columns) for feature in X.columns}