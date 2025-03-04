
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
from typing import List, Dict, Callable, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from explorer import TransformationCombiner, TransformationTree


class ErrorDetectionAndRecovery:
    """
    Classe para implementar detecção automática de erros e recuperação
    durante o processamento de dados.
    """
    def __init__(self, logger=None):
        """
        Inicializa o sistema de detecção e recuperação.
        
        Args:
            logger: Logger para registrar mensagens
        """
        import logging
        self.logger = logger or logging.getLogger("AutoFE.ErrorRecovery")
        self.error_history = []
        self.recovery_attempts = {}
        
        self._setup_logging()
        self.logger.info("PreProcessor inicializado com sucesso.")

    def _setup_logging(self):
        self.logger = logging.getLogger("AutoFE.PreProcessor")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def validate_transformation_result(self, df_original, df_transformed, target_col=None):
        """
        Valida o resultado de uma transformação para detectar problemas.
        
        Args:
            df_original: DataFrame original
            df_transformed: DataFrame transformado
            target_col: Nome da coluna alvo (opcional)
            
        Returns:
            Tupla (válido, mensagem, dados_corrigidos)
        """
        # Verifica se o resultado é None
        if df_transformed is None:
            return False, "Resultado da transformação é None", df_original
        
        # Verifica se o DataFrame está vazio
        if df_transformed.empty:
            return False, "DataFrame resultante está vazio", df_original
        
        # Verifica perda significativa de dados
        data_loss_ratio = 1 - (len(df_transformed) / len(df_original))
        if data_loss_ratio > 0.5:  # Perdeu mais de 50% dos dados
            self.self.logger.warning(f"Transformação removeu {data_loss_ratio*100:.1f}% dos dados")
            if data_loss_ratio > 0.9:  # Perda crítica
                return False, f"Perda crítica de dados: {data_loss_ratio*100:.1f}%", df_original
            
        # Verificações específicas para classificação
        if target_col and target_col in df_original.columns and target_col in df_transformed.columns:
            original_counts = df_original[target_col].value_counts()
            transformed_counts = df_transformed[target_col].value_counts()
            
            # Verifica se alguma classe foi completamente perdida
            lost_classes = set(original_counts.index) - set(transformed_counts.index)
            if lost_classes:
                self.self.logger.warning(f"Classes perdidas durante transformação: {lost_classes}")
                return False, f"Classes perdidas: {lost_classes}", self._recover_lost_classes(
                    df_original, df_transformed, target_col, lost_classes
                )
            
            # Verifica se houve perda severa em alguma classe minoritária
            for cls in original_counts.index:
                if cls in transformed_counts:
                    loss_ratio = 1 - (transformed_counts[cls] / original_counts[cls])
                    if loss_ratio > 0.9 and original_counts[cls] < original_counts.max() * 0.2:
                        # Perda severa em classe minoritária
                        self.self.logger.warning(f"Perda severa na classe minoritária {cls}: {loss_ratio*100:.1f}%")
                        return False, f"Perda severa na classe minoritária {cls}", self._recover_lost_classes(
                            df_original, df_transformed, target_col, [cls]
                        )
        
        # Verifica se há valores NaN onde não deveria (colunas que não tinham NaN antes)
        for col in df_original.columns:
            if col in df_transformed.columns:
                had_na_before = df_original[col].isna().any()
                has_na_after = df_transformed[col].isna().any()
                
                if not had_na_before and has_na_after:
                    na_ratio = df_transformed[col].isna().mean()
                    if na_ratio > 0.1:  # Mais de 10% NaN
                        self.self.logger.warning(f"Aumento significativo de valores ausentes na coluna {col}: {na_ratio*100:.1f}%")
                        # Se for grave, tenta corrigir
                        if na_ratio > 0.5:  # Mais de 50% NaN
                            return False, f"Valores ausentes críticos na coluna {col}", self._fix_missing_values(
                                df_original, df_transformed
                            )
        
        # Verifica se há features constantes (variância zero) nas colunas numéricas
        numeric_cols = df_transformed.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df_transformed[col].nunique() <= 1:
                self.self.logger.warning(f"Feature constante após transformação: {col}")
                # Não é um erro crítico, mas vale a pena registrar
        
        # Tudo ok
        return True, "Transformação validada com sucesso", df_transformed
    
    def _recover_lost_classes(self, df_original, df_transformed, target_col, lost_classes):
        """
        Recupera exemplos de classes perdidas do DataFrame original.
        
        Args:
            df_original: DataFrame original
            df_transformed: DataFrame transformado
            target_col: Nome da coluna alvo
            lost_classes: Lista de classes perdidas
            
        Returns:
            DataFrame com exemplos recuperados
        """
        # Extrai exemplos das classes perdidas do DataFrame original
        lost_examples = df_original[df_original[target_col].isin(lost_classes)]
        
        # Limita a um número razoável de exemplos por classe
        max_examples_per_class = 50
        sampled_examples = []
        
        for cls in lost_classes:
            class_examples = lost_examples[lost_examples[target_col] == cls]
            if len(class_examples) > max_examples_per_class:
                # Amostra aleatória se houver muitos exemplos
                sampled_examples.append(class_examples.sample(max_examples_per_class, random_state=42))
            else:
                sampled_examples.append(class_examples)
        
        # Combina exemplos amostrados
        if sampled_examples:
            recovered_examples = pd.concat(sampled_examples)
            self.self.logger.info(f"Recuperados {len(recovered_examples)} exemplos de {len(lost_classes)} classes perdidas")
            
            # Combina com o DataFrame transformado
            recovered_df = pd.concat([df_transformed, recovered_examples])
            
            # Garante que não há duplicatas
            recovered_df = recovered_df.drop_duplicates()
            
            return recovered_df
        else:
            return df_transformed
    
    def _fix_missing_values(self, df_original, df_transformed):
        """
        Tenta corrigir valores ausentes introduzidos pela transformação.
        
        Args:
            df_original: DataFrame original
            df_transformed: DataFrame transformado
            
        Returns:
            DataFrame corrigido
        """
        from sklearn.impute import SimpleImputer
        
        # Identifica colunas com muitos valores ausentes
        problem_cols = []
        for col in df_transformed.columns:
            if col in df_transformed.columns:  # Garante que a coluna existe
                na_ratio = df_transformed[col].isna().mean()
                if na_ratio > 0.1:  # Mais de 10% ausentes
                    problem_cols.append(col)
        
        if not problem_cols:
            return df_transformed
            
        # Cria cópia para modificar
        fixed_df = df_transformed.copy()
        
        # Separa colunas por tipo
        numeric_cols = [c for c in problem_cols if c in df_transformed.select_dtypes(include=['number']).columns]
        cat_cols = [c for c in problem_cols if c in df_transformed.select_dtypes(include=['object', 'category']).columns]
        
        # Aplica imputação para colunas numéricas
        if numeric_cols:
            imputer = SimpleImputer(strategy='median')
            fixed_df[numeric_cols] = imputer.fit_transform(fixed_df[numeric_cols])
        
        # Aplica imputação para colunas categóricas
        if cat_cols:
            imputer = SimpleImputer(strategy='most_frequent')
            fixed_df[cat_cols] = imputer.fit_transform(fixed_df[cat_cols])
        
        self.self.logger.info(f"Corrigidos valores ausentes em {len(problem_cols)} colunas")
        return fixed_df
    
    def safe_transform(self, processor, df, target_col=None, config_override=None):
        """
        Executa uma transformação com detecção e recuperação automática de erros.
        
        Args:
            processor: Instância de PreProcessor
            df: DataFrame a transformar
            target_col: Nome da coluna alvo (opcional)
            config_override: Substituições de configuração (opcional)
            
        Returns:
            DataFrame transformado (possivelmente corrigido)
        """
        import traceback
        
        # Salva a configuração original
        original_config = processor.config.copy() if hasattr(processor, 'config') else {}
        
        # Aplica substituições de configuração, se fornecidas
        if config_override and hasattr(processor, 'config'):
            for key, value in config_override.items():
                processor.config[key] = value
        
        # Tenta a transformação
        try:
            # Se o processador já foi ajustado, apenas transforma
            if hasattr(processor, 'fitted') and processor.fitted:
                result_df = processor.transform(df, target_col=target_col)
            else:
                # Caso contrário, ajusta e transforma
                processor.fit(df, target_col=target_col)
                result_df = processor.transform(df, target_col=target_col)
            
            # Valida o resultado
            valid, message, corrected_df = self.validate_transformation_result(df, result_df, target_col)
            
            if not valid:
                self.self.logger.warning(f"Problema detectado: {message}. Aplicando correção automática.")
                
                # Registra a falha para análise posterior
                self.error_history.append({
                    'error_type': message,
                    'config': processor.config.copy() if hasattr(processor, 'config') else {},
                    'timestamp': pd.Timestamp.now()
                })
                
                return corrected_df
            
            return result_df
            
        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()
            self.self.logger.error(f"Erro durante transformação: {error_msg}")
            
            # Registra a exceção
            self.error_history.append({
                'error_type': 'exception',
                'error_message': error_msg,
                'error_trace': error_trace,
                'config': processor.config.copy() if hasattr(processor, 'config') else {},
                'timestamp': pd.Timestamp.now()
            })
            
            # Tenta recuperação
            return self._recover_from_exception(processor, df, target_col, error_msg)
        finally:
            # Restaura a configuração original
            if hasattr(processor, 'config') and original_config:
                processor.config = original_config
    
    def _recover_from_exception(self, processor, df, target_col, error_msg):
        """
        Tenta se recuperar de uma exceção durante a transformação.
        
        Args:
            processor: Instância de PreProcessor
            df: DataFrame original
            target_col: Nome da coluna alvo
            error_msg: Mensagem de erro
            
        Returns:
            DataFrame recuperado ou original
        """
        # Estratégias de recuperação baseadas no tipo de erro
        if 'memory' in error_msg.lower():
            # Problema de memória - tenta reduzir o tamanho do dataset
            self.self.logger.info("Tentando recuperação de erro de memória")
            if len(df) > 10000:
                # Amostra o dataset para reduzir tamanho
                recovery_size = min(10000, int(len(df) * 0.5))
                
                # Para classificação, usa amostragem estratificada
                if target_col and target_col in df.columns:
                    try:
                        from sklearn.model_selection import train_test_split
                        _, sampled_df = train_test_split(
                            df, test_size=recovery_size/len(df), 
                            stratify=df[target_col], random_state=42
                        )
                    except:
                        # Fallback para amostragem simples
                        sampled_df = df.sample(recovery_size, random_state=42)
                else:
                    sampled_df = df.sample(recovery_size, random_state=42)
                
                self.self.logger.info(f"Recuperação: reduzindo dataset de {len(df)} para {recovery_size} linhas")
                
                # Tenta novamente com dataset menor
                try:
                    processor.config['generate_features'] = False  # Desativa geração de features para economizar memória
                    processor.fit(sampled_df, target_col=target_col)
                    return processor.transform(df, target_col=target_col)
                except:
                    self.self.logger.warning("Falha na recuperação com dataset reduzido")
        
        elif 'matrix' in error_msg.lower() or 'singular' in error_msg.lower():
            # Problema numérico - tenta configuração mais robusta
            self.self.logger.info("Tentando recuperação de erro numérico/matriz")
            
            try:
                # Configura para usar scaling robusto e sem remoção de outliers
                processor.config['scaling'] = 'robust'
                processor.config['outlier_method'] = None
                processor.config['generate_features'] = False
                
                processor.fit(df, target_col=target_col)
                return processor.transform(df, target_col=target_col)
            except:
                self.self.logger.warning("Falha na recuperação com configuração numérica robusta")
        
        elif 'NaN' in error_msg or 'infinite' in error_msg.lower():
            # Problema com valores inválidos - limpa dados
            self.self.logger.info("Tentando recuperação de valores inválidos/NaN")
            
            try:
                # Cria cópia limpa - remove linhas com valores inválidos
                clean_df = df.copy()
                numeric_cols = clean_df.select_dtypes(include=['number']).columns
                
                # Remove linhas com NaN ou infinito
                clean_df = clean_df.dropna(subset=numeric_cols)
                clean_df = clean_df.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_cols)
                
                if len(clean_df) < len(df) * 0.1:  # Perdeu mais de 90% dos dados
                    self.self.logger.warning("Muitos dados seriam perdidos na limpeza, abortando essa estratégia")
                else:
                    self.self.logger.info(f"Removidas {len(df) - len(clean_df)} linhas com valores inválidos")
                    
                    # Configuração conservadora
                    processor.config['outlier_method'] = None
                    processor.fit(clean_df, target_col=target_col)
                    return processor.transform(df, target_col=target_col)
            except:
                self.self.logger.warning("Falha na recuperação com limpeza de valores inválidos")
        
        # Como último recurso, retorna uma configuração mínima
        self.self.logger.info("Tentando recuperação com configuração mínima")
        try:
            minimal_config = {
                'missing_values_strategy': 'median',
                'outlier_method': None,
                'categorical_strategy': 'onehot',
                'scaling': 'standard',
                'generate_features': False,
                'balance_classes': False
            }
            
            for key, value in minimal_config.items():
                if hasattr(processor, 'config'):
                    processor.config[key] = value
            
            processor.fit(df, target_col=target_col)
            return processor.transform(df, target_col=target_col)
        except:
            self.self.logger.error("Todas as tentativas de recuperação falharam. Retornando DataFrame original.")
            
            # Se tudo falhar, retorna o DataFrame original
            return df
    
    def analyze_errors(self):
        """
        Analisa os erros registrados para identificar padrões e tendências.
        
        Returns:
            Dicionário com análise de erros
        """
        if not self.error_history:
            return {"message": "Nenhum erro registrado"}
        
        # Contagem de tipos de erro
        error_types = {}
        for error in self.error_history:
            error_type = error.get('error_type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Configurações problemáticas
        problematic_configs = {}
        for error in self.error_history:
            config = error.get('config', {})
            for key, value in config.items():
                if key not in problematic_configs:
                    problematic_configs[key] = {}
                
                value_str = str(value)
                problematic_configs[key][value_str] = problematic_configs[key].get(value_str, 0) + 1
        
        # Ordena tipos de erro por frequência
        error_types = dict(sorted(error_types.items(), key=lambda x: x[1], reverse=True))
        
        # Analisa problemas mais comuns
        common_issues = []
        if 'exception' in error_types and error_types['exception'] > 0:
            exception_msgs = {}
            for error in self.error_history:
                if error.get('error_type') == 'exception':
                    msg = error.get('error_message', '')
                    # Pega apenas a primeira linha ou os primeiros 100 caracteres
                    short_msg = msg.split('\n')[0][:100]
                    exception_msgs[short_msg] = exception_msgs.get(short_msg, 0) + 1
            
            # Ordena exceções por frequência
            top_exceptions = dict(sorted(exception_msgs.items(), key=lambda x: x[1], reverse=True)[:5])
            common_issues.append({
                'issue_type': 'common_exceptions',
                'data': top_exceptions
            })
        
        # Analisa configurações problemáticas
        for key, values in problematic_configs.items():
            if len(values) > 0:
                sorted_values = dict(sorted(values.items(), key=lambda x: x[1], reverse=True)[:3])
                if sum(sorted_values.values()) > 2:  # Pelo menos 3 ocorrências
                    common_issues.append({
                        'issue_type': f'problematic_config_{key}',
                        'data': sorted_values
                    })
        
        return {
            'total_errors': len(self.error_history),
            'error_types': error_types,
            'problematic_configs': problematic_configs,
            'common_issues': common_issues,
            'first_error_time': min([e.get('timestamp', pd.Timestamp.now()) for e in self.error_history]),
            'last_error_time': max([e.get('timestamp', pd.Timestamp.now()) for e in self.error_history])
        }

# Integração com PreProcessor
def safe_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
    """
    Versão segura do método transform com recuperação automática de erros.
    
    Args:
        df: DataFrame a ser transformado
        target_col: Nome da coluna alvo (opcional)
            
    Returns:
        DataFrame transformado
    """
    # Inicializa o sistema de recuperação se não existir
    if not hasattr(self, 'error_recovery'):
        self.error_recovery = ErrorDetectionAndRecovery(self.logger)
    
    # Usa o sistema de recuperação para transformação segura
    return self.error_recovery.safe_transform(self, df, target_col)

# Integração com Explorer
def safe_explore(self, df: pd.DataFrame) -> TransformationTree:
    """
    Versão segura do método explore com recuperação automática de erros.
    
    Args:
        df: DataFrame a explorar
        
    Returns:
        Árvore de transformações
    """
    # Inicializa o sistema de recuperação se não existir
    if not hasattr(self, 'error_recovery'):
        self.error_recovery = ErrorDetectionAndRecovery(logger)
    
    self.logger.info(f"Iniciando exploração segura para DataFrame de dimensões {df.shape}")
    
    # Analisa o dataset para configurações adaptativas
    try:
        self._dataset_analysis = self._analyze_dataset(df, self.target_col)
        self.logger.info(f"Análise do dataset: tipo={self._dataset_analysis.get('problem_type', 'unknown')}, "
                   f"desbalanceado={self._dataset_analysis.get('is_imbalanced', False)}")
        
        # Reconstrói base_configs com configurações adaptativas
        self.base_configs = self._get_default_configs()
    except Exception as e:
        self.logger.warning(f"Erro na análise do dataset: {e}. Usando configurações padrão.")
    
    # Atualiza o combinador com novas configurações
    self.combiner = TransformationCombiner(
        base_transformations=self.base_configs,
        max_depth=self.max_depth,
        beam_width=self.beam_width,
        parallel=self.parallel,
        n_jobs=self.n_jobs
    )
    
    try:
        # Cria um perfil do dataset
        dataset_profile = self.meta_learner.profile_dataset(df, self.target_col)
        
        # Obtém recomendações com base em experiências anteriores
        recommendations = self.meta_learner.recommend_transformations(
            df, self.target_col, n_recommendations=5
        )
        
        # Adiciona recomendações ao combinador
        for config in recommendations:
            self.combiner.add_base_transformation(config)
    except Exception as e:
        self.logger.warning(f"Erro ao gerar recomendações: {e}. Usando apenas configurações base.")
        recommendations = []
    
    # Inclui configuração fail-safe para garantir pelo menos uma transformação bem-sucedida
    fail_safe_config = {
        'missing_values_strategy': 'median',
        'outlier_method': None,
        'categorical_strategy': 'onehot',
        'scaling': 'standard',
        'generate_features': False,
        'balance_classes': False
    }
    self.combiner.add_base_transformation(fail_safe_config)
    
    try:
        # Executa busca em feixe com tratamento de erros
        tree = self.combiner.beam_search(
            df=df,
            target_col=self.target_col,
            evaluator=self.evaluator,
            recommendations=recommendations
        )
        
        # Se a árvore estiver vazia ou inválida, cria um nó failsafe
        if len(tree.nodes) <= 1:  # Apenas o nó raiz
            self.logger.warning("Árvore de transformações vazia. Aplicando transformação failsafe.")
            
            # Cria preprocessador failsafe
            from preprocessor import PreProcessor
            failsafe_processor = PreProcessor(fail_safe_config)
            
            # Aplica transformação segura
            result_df = self.error_recovery.safe_transform(failsafe_processor, df, self.target_col)
            
            # Avalia
            metrics = self.evaluator.evaluate_transformation(result_df)
            score = self.evaluator.compute_overall_score(metrics)
            
            # Adiciona à árvore
            tree.add_transformation(
                parent="root",
                name="failsafe_transformation",
                config=fail_safe_config,
                data=result_df,
                score=score,
                metrics=metrics
            )
    except Exception as e:
        self.logger.error(f"Erro na busca em feixe: {e}. Criando árvore básica.")
        
        # Cria árvore básica
        tree = TransformationTree()
        tree.nodes["root"].data = df
        
        # Adiciona transformação failsafe
        from preprocessor import PreProcessor
        failsafe_processor = PreProcessor(fail_safe_config)
        
        # Aplica transformação segura
        result_df = self.error_recovery.safe_transform(failsafe_processor, df, self.target_col)
        
        # Avalia
        metrics = self.evaluator.evaluate_transformation(result_df)
        score = self.evaluator.compute_overall_score(metrics)
        
        # Adiciona à árvore
        tree.add_transformation(
            parent="root",
            name="failsafe_transformation",
            config=fail_safe_config,
            data=result_df,
            score=score,
            metrics=metrics
        )
    
    # Armazena resultado
    self.exploration_result = tree
    
    # Registra resultados no meta-learner
    try:
        best_nodes = tree.get_best_nodes(limit=3)
        for node_id in best_nodes:
            node = tree.nodes[node_id]
            self.meta_learner.record_result(
                dataset_profile=dataset_profile if 'dataset_profile' in locals() else {},
                config=node.config,
                score=node.score,
                metrics=node.metrics
            )
        
        # Salva a base de experiências
        if self.experience_db:
            self.meta_learner.save()
    except Exception as e:
        self.logger.warning(f"Erro ao registrar resultados no meta-learner: {e}")
    
    # Analisa os erros encontrados
    if hasattr(self.error_recovery, 'error_history') and self.error_recovery.error_history:
        error_analysis = self.error_recovery.analyze_errors()
        self.logger.info(f"Análise de erros: {len(self.error_recovery.error_history)} erros encontrados durante a exploração")
    
    return tree