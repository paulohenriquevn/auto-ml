"""
Implementação da imagificação de features para o Learner-Predictor.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy import stats
import matplotlib.pyplot as plt
import io
import base64

# Importações internas
from config import LEARNER_PREDICTOR_CONFIG


class FeatureImagification:
    """
    Implementa a técnica de imagificação de features, que transforma
    variáveis em representações visuais para facilitar o meta-aprendizado.
    """
    
    def __init__(self):
        """
        Inicializa o módulo de imagificação de features.
        """
        self.logger = logging.getLogger(__name__)
        self.n_bins = LEARNER_PREDICTOR_CONFIG['imagification_bins']
    
    def imagify_feature(
        self,
        feature: pd.Series,
        target: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Converte uma feature em uma representação matricial (imagem).
        
        Args:
            feature: Série do pandas com a feature
            target: Série do pandas com a variável alvo (opcional)
            
        Returns:
            Array NumPy representando a "imagem" da feature
        """
        # Detectar tipo de feature
        feature_type = self._detect_feature_type(feature)
        
        if feature_type == 'numeric':
            return self._imagify_numeric(feature, target)
        elif feature_type == 'categorical':
            return self._imagify_categorical(feature, target)
        elif feature_type == 'datetime':
            return self._imagify_datetime(feature, target)
        elif feature_type == 'text':
            return self._imagify_text(feature, target)
        else:
            self.logger.warning(f"Tipo de feature não suportado: {feature_type}")
            return np.zeros((self.n_bins, self.n_bins))
    
    def _detect_feature_type(self, feature: pd.Series) -> str:
        """
        Detecta o tipo de uma feature.
        
        Args:
            feature: Série do pandas com a feature
            
        Returns:
            Tipo da feature ('numeric', 'categorical', 'datetime', 'text')
        """
        if pd.api.types.is_numeric_dtype(feature):
            return 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(feature):
            return 'datetime'
        elif feature.dtype == 'object':
            # Se a média do comprimento das strings for maior que 10, considerar texto
            if feature.astype(str).str.len().mean() > 10:
                return 'text'
            else:
                return 'categorical'
        else:
            return 'categorical'
    
    def _imagify_numeric(
        self,
        feature: pd.Series,
        target: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Cria uma representação matricial de uma feature numérica.
        
        Args:
            feature: Série do pandas com a feature numérica
            target: Série do pandas com a variável alvo (opcional)
            
        Returns:
            Array NumPy representando a "imagem" da feature
        """
        # Remover valores ausentes
        feature = feature.dropna()
        
        if len(feature) == 0:
            return np.zeros((self.n_bins, self.n_bins))
        
        # Calcular histograma da feature
        hist, bin_edges = np.histogram(feature, bins=self.n_bins, density=True)
        
        # Normalizar para [0, 1]
        hist_normalized = hist / np.max(hist) if np.max(hist) > 0 else hist
        
        # Criar matriz 2D inicial
        image = np.zeros((self.n_bins, self.n_bins))
        
        # Preencher primeira linha com histograma
        image[0, :] = hist_normalized
        
        # Se temos a variável alvo, adicionar informações adicionais
        if target is not None and len(target) == len(feature):
            # Remover valores ausentes correspondentes no alvo
            valid_indices = ~target.isna()
            valid_feature = feature[valid_indices]
            valid_target = target[valid_indices]
            
            if len(valid_feature) > 0:
                # Verificar se a variável alvo é numérica ou categórica
                if pd.api.types.is_numeric_dtype(valid_target):
                    # Para alvos numéricos, calcular correlação em cada bin
                    for i in range(self.n_bins - 1):
                        bin_mask = (valid_feature >= bin_edges[i]) & (valid_feature < bin_edges[i + 1])
                        if bin_mask.sum() > 1:
                            bin_target = valid_target[bin_mask]
                            # Calcular estatísticas para este bin
                            bin_mean = bin_target.mean()
                            bin_std = bin_target.std()
                            
                            # Normalizar para [0, 1]
                            global_mean = valid_target.mean()
                            global_std = valid_target.std()
                            
                            if global_std > 0:
                                norm_mean = (bin_mean - global_mean) / (2 * global_std) + 0.5
                                norm_mean = min(max(norm_mean, 0), 1)
                            else:
                                norm_mean = 0.5
                            
                            if global_std > 0:
                                norm_std = bin_std / (2 * global_std)
                                norm_std = min(max(norm_std, 0), 1)
                            else:
                                norm_std = 0
                            
                            # Preencher duas linhas adicionais
                            image[1, i] = norm_mean
                            image[2, i] = norm_std
                else:
                    # Para alvos categóricos, calcular distribuição de classes em cada bin
                    unique_classes = valid_target.unique()
                    num_classes = len(unique_classes)
                    
                    for i in range(self.n_bins - 1):
                        bin_mask = (valid_feature >= bin_edges[i]) & (valid_feature < bin_edges[i + 1])
                        if bin_mask.sum() > 0:
                            bin_target = valid_target[bin_mask]
                            
                            # Calcular distribuição de classes
                            class_counts = bin_target.value_counts(normalize=True)
                            
                            # Preencher linhas para cada classe
                            for j, cls in enumerate(unique_classes):
                                if j < self.n_bins - 1:  # Limitar ao número de bins
                                    image[j + 1, i] = class_counts.get(cls, 0)
        
        return image
    
    def _imagify_categorical(
        self,
        feature: pd.Series,
        target: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Cria uma representação matricial de uma feature categórica.
        
        Args:
            feature: Série do pandas com a feature categórica
            target: Série do pandas com a variável alvo (opcional)
            
        Returns:
            Array NumPy representando a "imagem" da feature
        """
        # Remover valores ausentes
        feature = feature.dropna()
        
        if len(feature) == 0:
            return np.zeros((self.n_bins, self.n_bins))
        
        # Obter contagem de categorias
        value_counts = feature.value_counts(normalize=True)
        
        # Limitar ao número de bins
        top_categories = value_counts.index[:self.n_bins].tolist()
        
        # Criar matriz 2D inicial
        image = np.zeros((self.n_bins, self.n_bins))
        
        # Preencher primeira linha com distribuição de categorias
        for i, category in enumerate(top_categories):
            if i < self.n_bins:
                image[0, i] = value_counts.get(category, 0)
        
        # Se temos a variável alvo, adicionar informações adicionais
        if target is not None and len(target) == len(feature):
            # Remover valores ausentes correspondentes no alvo
            valid_indices = ~target.isna()
            valid_feature = feature[valid_indices]
            valid_target = target[valid_indices]
            
            if len(valid_feature) > 0:
                # Verificar se a variável alvo é numérica ou categórica
                if pd.api.types.is_numeric_dtype(valid_target):
                    # Para alvos numéricos, calcular estatísticas para cada categoria
                    for i, category in enumerate(top_categories):
                        if i < self.n_bins:
                            cat_mask = (valid_feature == category)
                            if cat_mask.sum() > 0:
                                cat_target = valid_target[cat_mask]
                                
                                # Calcular estatísticas para esta categoria
                                cat_mean = cat_target.mean()
                                cat_std = cat_target.std()
                                
                                # Normalizar para [0, 1]
                                global_mean = valid_target.mean()
                                global_std = valid_target.std()
                                
                                if global_std > 0:
                                    norm_mean = (cat_mean - global_mean) / (2 * global_std) + 0.5
                                    norm_mean = min(max(norm_mean, 0), 1)
                                else:
                                    norm_mean = 0.5
                                
                                if global_std > 0:
                                    norm_std = cat_std / (2 * global_std)
                                    norm_std = min(max(norm_std, 0), 1)
                                else:
                                    norm_std = 0
                                
                                # Preencher duas linhas adicionais
                                image[1, i] = norm_mean
                                image[2, i] = norm_std
                else:
                    # Para alvos categóricos, calcular distribuição de classes para cada categoria
                    unique_classes = valid_target.unique()
                    num_classes = len(unique_classes)
                    
                    for i, category in enumerate(top_categories):
                        if i < self.n_bins:
                            cat_mask = (valid_feature == category)
                            if cat_mask.sum() > 0:
                                cat_target = valid_target[cat_mask]
                                
                                # Calcular distribuição de classes
                                class_counts = cat_target.value_counts(normalize=True)
                                
                                # Preencher linhas para cada classe
                                for j, cls in enumerate(unique_classes):
                                    if j < self.n_bins - 1:  # Limitar ao número de bins
                                        image[j + 1, i] = class_counts.get(cls, 0)
        
        return image
    
    def _imagify_datetime(
        self,
        feature: pd.Series,
        target: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Cria uma representação matricial de uma feature de data/hora.
        
        Args:
            feature: Série do pandas com a feature de data/hora
            target: Série do pandas com a variável alvo (opcional)
            
        Returns:
            Array NumPy representando a "imagem" da feature
        """
        # Remover valores ausentes
        feature = feature.dropna()
        
        if len(feature) == 0:
            return np.zeros((self.n_bins, self.n_bins))
        
        # Extrair componentes de data/hora
        components = {
            'year': feature.dt.year,
            'month': feature.dt.month,
            'day': feature.dt.day,
            'dayofweek': feature.dt.dayofweek,
            'hour': feature.dt.hour if hasattr(feature.dt, 'hour') else None,
            'minute': feature.dt.minute if hasattr(feature.dt, 'minute') else None,
        }
        
        # Criar matriz 2D inicial
        image = np.zeros((self.n_bins, self.n_bins))
        
        # Preencher linhas com histogramas dos componentes
        row = 0
        for name, component in components.items():
            if component is not None and row < self.n_bins:
                try:
                    hist, _ = np.histogram(component, bins=self.n_bins, density=True)
                    hist_normalized = hist / np.max(hist) if np.max(hist) > 0 else hist
                    image[row, :] = hist_normalized
                    row += 1
                except Exception as e:
                    self.logger.warning(f"Erro ao processar componente {name}: {str(e)}")
        
        # Se temos a variável alvo, adicionar informações adicionais
        if target is not None and len(target) == len(feature) and row < self.n_bins:
            # Remover valores ausentes correspondentes no alvo
            valid_indices = ~target.isna()
            valid_feature = feature[valid_indices]
            valid_target = target[valid_indices]
            
            if len(valid_feature) > 0 and pd.api.types.is_numeric_dtype(valid_target):
                # Calcular média móvel da variável alvo ao longo do tempo
                valid_df = pd.DataFrame({'date': valid_feature, 'target': valid_target})
                valid_df = valid_df.sort_values('date')
                
                # Dividir em intervalos de tempo iguais
                valid_df['time_bin'] = pd.qcut(
                    valid_df['date'].astype(np.int64),
                    self.n_bins,
                    duplicates='drop',
                    labels=False
                )
                
                # Calcular média do alvo em cada intervalo
                time_stats = valid_df.groupby('time_bin')['target'].agg(['mean', 'std']).reset_index()
                
                # Normalizar para [0, 1]
                global_mean = valid_target.mean()
                global_std = valid_target.std()
                
                if global_std > 0:
                    time_stats['norm_mean'] = (time_stats['mean'] - global_mean) / (2 * global_std) + 0.5
                    time_stats['norm_mean'] = time_stats['norm_mean'].clip(0, 1)
                else:
                    time_stats['norm_mean'] = 0.5
                
                if global_std > 0:
                    time_stats['norm_std'] = time_stats['std'] / (2 * global_std)
                    time_stats['norm_std'] = time_stats['norm_std'].clip(0, 1)
                else:
                    time_stats['norm_std'] = 0
                
                # Preencher linhas adicionais
                for i, (_, stats) in enumerate(time_stats.iterrows()):
                    if i < self.n_bins:
                        if row < self.n_bins:
                            image[row, i] = stats['norm_mean']
                        if row + 1 < self.n_bins:
                            image[row + 1, i] = stats['norm_std']
        
        return image
    
    def _imagify_text(
        self,
        feature: pd.Series,
        target: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Cria uma representação matricial de uma feature de texto.
        
        Args:
            feature: Série do pandas com a feature de texto
            target: Série do pandas com a variável alvo (opcional)
            
        Returns:
            Array NumPy representando a "imagem" da feature
        """
        # Remover valores ausentes
        feature = feature.dropna()
        
        if len(feature) == 0:
            return np.zeros((self.n_bins, self.n_bins))
        
        # Extrair estatísticas do texto
        text_stats = pd.DataFrame({
            'length': feature.astype(str).str.len(),
            'word_count': feature.astype(str).str.split().str.len(),
            'upper_ratio': feature.astype(str).apply(
                lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
            ),
            'digit_ratio': feature.astype(str).apply(
                lambda x: sum(1 for c in x if c.isdigit()) / max(len(x), 1)
            ),
            'space_ratio': feature.astype(str).apply(
                lambda x: sum(1 for c in x if c.isspace()) / max(len(x), 1)
            ),
            'punct_ratio': feature.astype(str).apply(
                lambda x: sum(1 for c in x if c in '.,;:!?"-()[]{}') / max(len(x), 1)
            )
        })
        
        # Criar matriz 2D inicial
        image = np.zeros((self.n_bins, self.n_bins))
        
        # Preencher linhas com histogramas das estatísticas
        for i, col in enumerate(text_stats.columns):
            if i < self.n_bins:
                try:
                    hist, _ = np.histogram(text_stats[col], bins=self.n_bins, density=True)
                    hist_normalized = hist / np.max(hist) if np.max(hist) > 0 else hist
                    image[i, :] = hist_normalized
                except Exception as e:
                    self.logger.warning(f"Erro ao processar estatística de texto {col}: {str(e)}")
        
        # Se temos a variável alvo, adicionar informações adicionais
        # A implementação seria similar à de outras features, adaptada para texto
        
        return image
    
    def imagify_dataset(
        self,
        data: pd.DataFrame,
        target: Optional[Union[str, pd.Series]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Cria representações matriciais para todas as features de um dataset.
        
        Args:
            data: DataFrame com os dados
            target: Nome da coluna alvo ou Series com valores alvo (opcional)
            
        Returns:
            Dicionário com nome da feature -> representação matricial
        """
        self.logger.info(f"Criando representações matriciais para {len(data.columns)} features")
        
        # Extrair variável alvo se for string
        if isinstance(target, str) and target in data.columns:
            y = data[target]
            X = data.drop(columns=[target])
        elif isinstance(target, pd.Series):
            y = target
            X = data
        else:
            y = None
            X = data
        
        # Criar representações para cada feature
        imagified_features = {}
        
        for col in X.columns:
            try:
                imagified_features[col] = self.imagify_feature(X[col], y)
            except Exception as e:
                self.logger.warning(f"Erro ao imagificar feature {col}: {str(e)}")
                imagified_features[col] = np.zeros((self.n_bins, self.n_bins))
        
        return imagified_features
    
    def visualize_imagified_feature(
        self,
        feature_name: str,
        feature_image: np.ndarray,
        cmap: str = 'viridis'
    ) -> str:
        """
        Visualiza a representação matricial de uma feature como uma imagem.
        
        Args:
            feature_name: Nome da feature
            feature_image: Representação matricial da feature
            cmap: Mapa de cores para visualização
            
        Returns:
            String HTML com a imagem codificada em base64
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(feature_image, cmap=cmap, aspect='auto')
        plt.colorbar(label='Valor normalizado')
        plt.title(f'Representação de "{feature_name}"')
        plt.xlabel('Bin / Categoria')
        plt.ylabel('Estatística / Componente')
        
        # Salvar figura em buffer de memória
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Codificar em base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # Retornar tag HTML
        return f'<img src="data:image/png;base64,{img_str}" />'
    
    def compare_imagified_features(
        self,
        feature_images: Dict[str, np.ndarray],
        top_n: int = 5,
        cmap: str = 'viridis'
    ) -> str:
        """
        Compara as representações matriciais de múltiplas features.
        
        Args:
            feature_images: Dicionário de nome da feature -> representação matricial
            top_n: Número de features a mostrar
            cmap: Mapa de cores para visualização
            
        Returns:
            String HTML com as imagens codificadas em base64
        """
        feature_names = list(feature_images.keys())
        
        if len(feature_names) > top_n:
            feature_names = feature_names[:top_n]
        
        n_features = len(feature_names)
        
        # Criar figura com subplots
        fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 4))
        
        # Caso tenha apenas uma feature, axes não será um array
        if n_features == 1:
            axes = [axes]
        
        # Plotar cada feature
        for i, name in enumerate(feature_names):
            axes[i].imshow(feature_images[name], cmap=cmap, aspect='auto')
            axes[i].set_title(name)
            axes[i].set_xlabel('Bin / Categoria')
            
            if i == 0:
                axes[i].set_ylabel('Estatística / Componente')
        
        plt.tight_layout()
        
        # Salvar figura em buffer de memória
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Codificar em base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # Retornar tag HTML
        return f'<img src="data:image/png;base64,{img_str}" />'
