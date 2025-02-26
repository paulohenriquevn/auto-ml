"""
Implementação do meta-aprendizado para o Learner-Predictor.
"""

import os
import json
import pickle
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from datetime import datetime

# Importações internas
from config import LEARNER_PREDICTOR_CONFIG


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder personalizado que lida com tipos numpy."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super(NumpyJSONEncoder, self).default(obj)


class MetaLearner:
    """
    Implementa o sistema de meta-aprendizado que aprende quais transformações são
    mais eficazes com base em experiências anteriores.
    """
    
    def __init__(self):
        """
        Inicializa o sistema de meta-aprendizado.
        """
        self.logger = logging.getLogger(__name__)
        self.history_path = LEARNER_PREDICTOR_CONFIG['history_path']
        self.meta_model = None
        self.dataset_history = []
        
        # Criar diretório para histórico se não existir
        os.makedirs(self.history_path, exist_ok=True)
        
        # Carregar histórico existente
        self._load_history()
    
    def _load_history(self):
        """
        Carrega o histórico de datasets e transformações.
        """
        history_file = os.path.join(self.history_path, 'dataset_history.json')
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    self.dataset_history = json.load(f)
                self.logger.info(f"Histórico carregado com {len(self.dataset_history)} datasets")
            except Exception as e:
                self.logger.warning(f"Erro ao carregar histórico: {str(e)}")
                self.dataset_history = []
        else:
            self.logger.info("Histórico não encontrado, iniciando novo")
            self.dataset_history = []
        
        # Carregar modelo de meta-aprendizado se existir
        model_file = os.path.join(self.history_path, 'meta_model.pkl')
        if os.path.exists(model_file):
            try:
                self.meta_model = joblib.load(model_file)
                self.logger.info("Modelo de meta-aprendizado carregado")
            except Exception as e:
                self.logger.warning(f"Erro ao carregar modelo de meta-aprendizado: {str(e)}")
                self.meta_model = None
    
    def _save_history(self):
        """
        Salva o histórico de datasets e transformações.
        """
        history_file = os.path.join(self.history_path, 'dataset_history.json')
        
        try:
            with open(history_file, 'w') as f:
                # Usar encoder personalizado para lidar com tipos numpy
                json.dump(self.dataset_history, f, indent=2, cls=NumpyJSONEncoder)
            self.logger.info(f"Histórico salvo com {len(self.dataset_history)} datasets")
        except Exception as e:
            self.logger.warning(f"Erro ao salvar histórico: {str(e)}")
    
    def _save_meta_model(self):
        """
        Salva o modelo de meta-aprendizado.
        """
        if self.meta_model is None:
            return
        
        model_file = os.path.join(self.history_path, 'meta_model.pkl')
        
        try:
            joblib.dump(self.meta_model, model_file)
            self.logger.info("Modelo de meta-aprendizado salvo")
        except Exception as e:
            self.logger.warning(f"Erro ao salvar modelo de meta-aprendizado: {str(e)}")
    
    def _convert_to_json_serializable(self, obj):
        """
        Converte objetos para formatos serializáveis em JSON.
        
        Args:
            obj: Objeto a ser convertido
            
        Returns:
            Versão serializável do objeto
        """
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return self._convert_to_json_serializable(obj.tolist())
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (dict,)):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif pd.isna(obj):
            return None
        return obj
    
    def update_history(
        self,
        dataset_type: str,
        data_properties: Dict[str, Any],
        successful_transformations: List[Dict[str, Any]],
        performance_metrics: Dict[str, float]
    ):
        """
        Atualiza o histórico com informações sobre um novo dataset e suas transformações.
        
        Args:
            dataset_type: Tipo de dataset ('tabular_classification', 'tabular_regression', etc.)
            data_properties: Propriedades do dataset (número de features, tipo de dados, etc.)
            successful_transformations: Lista de transformações bem-sucedidas
            performance_metrics: Métricas de desempenho das transformações
        """
        # Criar entrada para o histórico - convertendo para tipos serializáveis
        entry = {
            'timestamp': datetime.now().isoformat(),
            'dataset_type': dataset_type,
            'data_properties': self._convert_to_json_serializable(data_properties),
            'successful_transformations': self._convert_to_json_serializable(successful_transformations),
            'performance_metrics': self._convert_to_json_serializable(performance_metrics)
        }
        
        # Adicionar ao histórico
        self.dataset_history.append(entry)
        
        # Salvar histórico
        self._save_history()
        
        # Treinar modelo de meta-aprendizado
        self._train_meta_model()
    
    def _train_meta_model(self):
        """
        Treina o modelo de meta-aprendizado com base no histórico.
        """
        self.logger.info("Treinando modelo de meta-aprendizado")
        
        # Verificar se há histórico suficiente
        if len(self.dataset_history) < 2:
            self.logger.info("Histórico insuficiente para treinar modelo")
            return
        
        # Preparar dados para treinamento
        X_meta = []  # Features do meta-dataset
        y_meta = []  # Alvos do meta-dataset (eficácia das transformações)
        
        for entry in self.dataset_history:
            dataset_features = self._extract_dataset_features(entry['data_properties'])
            
            for transformation in entry['successful_transformations']:
                # Extrair features da transformação
                transform_features = self._extract_transformation_features(transformation)
                
                # Combinar features do dataset e da transformação
                meta_features = {**dataset_features, **transform_features}
                
                # Adicionar ao conjunto de dados de meta-aprendizado
                X_meta.append(meta_features)
                
                # Adicionar eficácia como alvo
                y_meta.append(transformation['performance_gain'])
        
        # Converter para DataFrame e Series
        X_meta_df = pd.DataFrame(X_meta)
        y_meta_series = pd.Series(y_meta)
        
        # Treinar modelo de meta-aprendizado
        model_type = LEARNER_PREDICTOR_CONFIG['meta_model']
        
        try:
            if model_type == 'random_forest':
                self.meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == 'xgboost':
                try:
                    import xgboost as xgb
                    self.meta_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                except ImportError:
                    self.logger.warning("XGBoost não disponível, usando Random Forest")
                    self.meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == 'neural_network':
                try:
                    from sklearn.neural_network import MLPRegressor
                    self.meta_model = MLPRegressor(
                        hidden_layer_sizes=(100, 50),
                        max_iter=500,
                        random_state=42
                    )
                except ImportError:
                    self.logger.warning("MLPRegressor não disponível, usando Random Forest")
                    self.meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                self.logger.warning(f"Modelo {model_type} não reconhecido, usando Random Forest")
                self.meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Treinar modelo
            self.meta_model.fit(X_meta_df, y_meta_series)
            
            # Salvar modelo
            self._save_meta_model()
            
            self.logger.info("Modelo de meta-aprendizado treinado com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao treinar modelo de meta-aprendizado: {str(e)}")
    
    def _extract_dataset_features(self, data_properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrai features que caracterizam um dataset.
        
        Args:
            data_properties: Propriedades do dataset
            
        Returns:
            Dicionário com features extraídas
        """
        # Extrair características básicas do dataset
        features = {
            'num_samples': data_properties.get('num_samples', 0),
            'num_features': data_properties.get('num_features', 0),
            'num_numeric': data_properties.get('num_numeric', 0),
            'num_categorical': data_properties.get('num_categorical', 0),
            'num_datetime': data_properties.get('num_datetime', 0),
            'num_text': data_properties.get('num_text', 0),
            'missing_ratio': data_properties.get('missing_ratio', 0),
            'target_type': data_properties.get('target_type', 'unknown')
        }
        
        # Adicionar características estatísticas se disponíveis
        if 'feature_stats' in data_properties:
            stats = data_properties['feature_stats']
            features.update({
                'mean_skewness': stats.get('mean_skewness', 0),
                'mean_kurtosis': stats.get('mean_kurtosis', 0),
                'mean_variance': stats.get('mean_variance', 0)
            })
        
        return features
    
    def _extract_transformation_features(self, transformation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrai features que caracterizam uma transformação.
        
        Args:
            transformation: Informações sobre a transformação
            
        Returns:
            Dicionário com features extraídas
        """
        # Codificar o tipo de transformação como one-hot
        transformation_type = transformation['transformation_type']
        
        # Lista de possíveis tipos de transformação
        all_types = [
            'log', 'sqrt', 'square', 'cube', 'reciprocal', 
            'sin', 'cos', 'tan', 'sigmoid', 'tanh',
            'standardize', 'normalize', 'min_max_scale',
            'quantile_transform', 'power_transform', 'boxcox',
            'one_hot_encode', 'label_encode', 'target_encode', 
            'count_encode', 'frequency_encode', 'mean_encode',
            'extract_year', 'extract_month', 'extract_day', 
            'extract_hour', 'extract_minute', 'extract_second',
            'word_count', 'char_count', 'stop_word_count',
            'lag', 'rolling_mean', 'rolling_std', 'rolling_min', 
            'rolling_max', 'differencing', 'decompose_trend',
            'sum', 'difference', 'product', 'ratio', 'polynomial'
        ]
        
        # Criar features one-hot
        transform_features = {
            f'transform_{t}': 1 if t == transformation_type else 0
            for t in all_types
        }
        
        # Adicionar profundidade da transformação
        if 'depth' in transformation:
            transform_features['transform_depth'] = transformation['depth']
        else:
            transform_features['transform_depth'] = 1
        
        return transform_features
    
    def find_similar_datasets(
        self,
        dataset_type: str,
        data_properties: Dict[str, Any],
        n_similar: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Encontra datasets similares no histórico.
        
        Args:
            dataset_type: Tipo de dataset
            data_properties: Propriedades do dataset
            n_similar: Número de datasets similares a retornar (None = usar configuração)
            
        Returns:
            Lista de datasets similares
        """
        if n_similar is None:
            n_similar = LEARNER_PREDICTOR_CONFIG['n_similar_datasets']
        
        # Se não há histórico, retornar lista vazia
        if not self.dataset_history:
            return []
        
        # Extrair features do dataset atual
        dataset_features = self._extract_dataset_features(data_properties)
        
        # Calcular similaridade com cada dataset no histórico
        similarities = []
        
        for i, entry in enumerate(self.dataset_history):
            # Filtrar por tipo de dataset
            if entry['dataset_type'] != dataset_type:
                continue
            
            # Extrair features do dataset histórico
            hist_features = self._extract_dataset_features(entry['data_properties'])
            
            # Calcular similaridade
            similarity = self._calculate_similarity(dataset_features, hist_features)
            
            similarities.append((i, similarity))
        
        # Ordenar por similaridade (decrescente)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Selecionar os N mais similares
        similar_datasets = [
            self.dataset_history[i]
            for i, _ in similarities[:n_similar]
        ]
        
        return similar_datasets
    
    def _calculate_similarity(
        self,
        features1: Dict[str, Any],
        features2: Dict[str, Any]
    ) -> float:
        """
        Calcula a similaridade entre dois conjuntos de features.
        
        Args:
            features1: Primeiro conjunto de features
            features2: Segundo conjunto de features
            
        Returns:
            Valor numérico representando a similaridade
        """
        # Extrair chaves comuns
        common_keys = set(features1.keys()) & set(features2.keys())
        
        # Se não há chaves comuns, retornar similaridade zero
        if not common_keys:
            return 0.0
        
        # Calcular distância euclidiana normalizada
        squared_diff_sum = 0.0
        
        for key in common_keys:
            # Ignorar features categóricas
            if key == 'target_type':
                # Adicionar 1.0 se forem iguais, 0.0 se diferentes
                squared_diff_sum += 0.0 if features1[key] == features2[key] else 1.0
            else:
                # Calcular diferença normalizada para features numéricas
                try:
                    val1 = float(features1[key])
                    val2 = float(features2[key])
                    
                    # Evitar divisão por zero
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 0:
                        squared_diff_sum += ((val1 - val2) / max_val) ** 2
                    else:
                        squared_diff_sum += 0.0
                except (ValueError, TypeError):
                    # Ignorar features não numéricas
                    pass
        
        # Calcular similaridade como 1 / (1 + distância)
        distance = np.sqrt(squared_diff_sum / len(common_keys))
        similarity = 1.0 / (1.0 + distance)
        
        return similarity
    
    def predict_transformation_effectiveness(
        self,
        dataset_type: str,
        data_properties: Dict[str, Any],
        transformation: Dict[str, Any]
    ) -> float:
        """
        Prediz a eficácia de uma transformação para um dataset específico.
        
        Args:
            dataset_type: Tipo de dataset
            data_properties: Propriedades do dataset
            transformation: Informações sobre a transformação
            
        Returns:
            Valor previsto de ganho de performance
        """
        # Se não há modelo treinado, usar método baseado em similaridade
        if self.meta_model is None:
            return self._predict_by_similarity(dataset_type, data_properties, transformation)
        
        # Extrair features do dataset e da transformação
        dataset_features = self._extract_dataset_features(data_properties)
        transform_features = self._extract_transformation_features(transformation)
        
        # Combinar features
        meta_features = {**dataset_features, **transform_features}
        
        # Converter para DataFrame
        meta_features_df = pd.DataFrame([meta_features])
        
        try:
            # Garantir que o DataFrame tenha todas as colunas esperadas pelo modelo
            expected_columns = self.meta_model.feature_names_in_
            
            # Adicionar colunas faltantes
            for col in expected_columns:
                if col not in meta_features_df.columns:
                    meta_features_df[col] = 0
            
            # Reordenar colunas para corresponder ao modelo
            meta_features_df = meta_features_df[expected_columns]
            
            # Fazer a predição
            predicted_gain = self.meta_model.predict(meta_features_df)[0]
            
            return max(0.0, predicted_gain)  # Garantir que o ganho seja não-negativo
            
        except Exception as e:
            self.logger.warning(f"Erro ao fazer predição com modelo: {str(e)}")
            # Em caso de erro, usar método baseado em similaridade
            return self._predict_by_similarity(dataset_type, data_properties, transformation)
    
    def _predict_by_similarity(
        self,
        dataset_type: str,
        data_properties: Dict[str, Any],
        transformation: Dict[str, Any]
    ) -> float:
        """
        Prediz a eficácia de uma transformação usando datasets similares.
        
        Args:
            dataset_type: Tipo de dataset
            data_properties: Propriedades do dataset
            transformation: Informações sobre a transformação
            
        Returns:
            Valor previsto de ganho de performance
        """
        # Encontrar datasets similares
        similar_datasets = self.find_similar_datasets(dataset_type, data_properties)
        
        if not similar_datasets:
            # Se não há datasets similares, retornar valor padrão
            return 0.01
        
        # Buscar a mesma transformação em datasets similares
        gains = []
        
        for dataset in similar_datasets:
            for trans in dataset['successful_transformations']:
                if trans['transformation_type'] == transformation['transformation_type']:
                    gains.append(trans['performance_gain'])
        
        # Se encontrou a transformação, retornar média dos ganhos
        if gains:
            return np.mean(gains)
        
        # Caso contrário, retornar valor padrão
        return 0.01