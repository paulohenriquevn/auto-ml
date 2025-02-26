"""
Funções de visualização para análise de features e transformações.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import io
import base64
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm

logger = logging.getLogger(__name__)


def fig_to_base64(fig: Figure) -> str:
    """
    Converte uma figura do matplotlib para string base64.
    
    Args:
        fig: Figura do matplotlib
        
    Returns:
        String base64 da imagem
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def plot_feature_distribution(
    data: pd.DataFrame,
    feature: str,
    target: Optional[str] = None,
    bins: int = 30,
    figsize: Tuple[int, int] = (10, 6)
) -> str:
    """
    Plota a distribuição de uma feature, opcionalmente colorida pelo alvo.
    
    Args:
        data: DataFrame com os dados
        feature: Nome da feature
        target: Nome da coluna alvo (opcional)
        bins: Número de bins para o histograma
        figsize: Tamanho da figura
        
    Returns:
        String base64 da imagem
    """
    if feature not in data.columns:
        logger.error(f"Feature {feature} não encontrada nos dados")
        return ""
    
    fig, ax = plt.subplots(figsize=figsize)
    
    try:
        if pd.api.types.is_numeric_dtype(data[feature]):
            # Feature numérica
            if target and target in data.columns and data[target].nunique() <= 10:
                # Colorir por alvo categórico
                for target_val in data[target].unique():
                    subset = data[data[target] == target_val]
                    ax.hist(subset[feature], bins=bins, alpha=0.5, label=f"{target_val}")
                ax.legend()
                ax.set_title(f"Distribuição de {feature} por {target}")
            elif target and target in data.columns and pd.api.types.is_numeric_dtype(data[target]):
                # Scatter plot com alvo numérico
                scatter = ax.scatter(data[feature], data[target], alpha=0.5, c=data[target], cmap='viridis')
                plt.colorbar(scatter, ax=ax, label=target)
                ax.set_xlabel(feature)
                ax.set_ylabel(target)
                ax.set_title(f"Relação entre {feature} e {target}")
            else:
                # Histograma simples
                ax.hist(data[feature].dropna(), bins=bins, alpha=0.7)
                ax.set_title(f"Distribuição de {feature}")
                ax.set_xlabel(feature)
                ax.set_ylabel("Frequência")
                
            # Adicionar estatísticas ao título
            if not target:
                mean = data[feature].mean()
                median = data[feature].median()
                std = data[feature].std()
                ax.set_title(f"Distribuição de {feature}\nMédia: {mean:.2f}, Mediana: {median:.2f}, Desvio Padrão: {std:.2f}")
                
        else:
            # Feature categórica
            value_counts = data[feature].value_counts().sort_values(ascending=False)
            
            # Limitar número de categorias para visualização
            if len(value_counts) > 20:
                top_n = value_counts.head(19)
                others = pd.Series({'Outros': value_counts[19:].sum()})
                value_counts = pd.concat([top_n, others])
            
            # Plotar barras
            ax.bar(value_counts.index, value_counts.values, alpha=0.7)
            ax.set_title(f"Distribuição de {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Contagem")
            
            # Rotacionar rótulos se muitas categorias
            if len(value_counts) > 5:
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
    except Exception as e:
        logger.error(f"Erro ao plotar distribuição: {str(e)}")
        ax.text(0.5, 0.5, f"Erro ao plotar: {str(e)}", ha='center', va='center')
    
    return fig_to_base64(fig)


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8)
) -> str:
    """
    Plota a importância das features.
    
    Args:
        feature_importance: DataFrame com features e importâncias
        top_n: Número de features mais importantes a mostrar
        figsize: Tamanho da figura
        
    Returns:
        String base64 da imagem
    """
    if feature_importance.empty:
        logger.error("DataFrame de importância vazio")
        return ""
    
    # Verificar se tem as colunas necessárias
    if not all(col in feature_importance.columns for col in ['feature', 'importance']):
        logger.error("DataFrame não tem colunas 'feature' e 'importance'")
        return ""
    
    # Ordenar e limitar ao top_n
    df = feature_importance.sort_values('importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    try:
        # Plotar barras
        bars = ax.barh(df['feature'], df['importance'], alpha=0.7)
        
        # Adicionar valores nas barras
        for bar in bars:
            width = bar.get_width()
            ax.text(width * 1.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
                    va='center', fontsize=10)
        
        ax.set_title(f"Top {len(df)} Features por Importância")
        ax.set_xlabel("Importância")
        ax.set_ylabel("Feature")
        
        # Inverter eixo y para mostrar feature mais importante no topo
        ax.invert_yaxis()
        
        plt.tight_layout()
    except Exception as e:
        logger.error(f"Erro ao plotar importância: {str(e)}")
        ax.text(0.5, 0.5, f"Erro ao plotar: {str(e)}", ha='center', va='center')
    
    return fig_to_base64(fig)


def plot_correlation_matrix(
    data: pd.DataFrame,
    features: Optional[List[str]] = None,
    method: str = 'pearson',
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'coolwarm',
    annotate: bool = True
) -> str:
    """
    Plota a matriz de correlação entre features.
    
    Args:
        data: DataFrame com os dados
        features: Lista de features para incluir (opcional)
        method: Método de correlação ('pearson', 'spearman', 'kendall')
        figsize: Tamanho da figura
        cmap: Mapa de cores
        annotate: Se deve adicionar anotações com valores
        
    Returns:
        String base64 da imagem
    """
    # Selecionar apenas colunas numéricas
    numeric_data = data.select_dtypes(include=['number'])
    
    if numeric_data.empty:
        logger.error("Nenhuma coluna numérica nos dados")
        return ""
    
    # Filtrar features se especificadas
    if features:
        valid_features = [f for f in features if f in numeric_data.columns]
        if not valid_features:
            logger.error(f"Nenhuma das features especificadas encontrada nos dados")
            return ""
        numeric_data = numeric_data[valid_features]
    
    # Calcular correlação
    corr_matrix = numeric_data.corr(method=method)
    
    # Plotar matriz de correlação
    fig, ax = plt.subplots(figsize=figsize)
    
    try:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        heatmap = sns.heatmap(corr_matrix, mask=mask, annot=annotate, fmt='.2f',
                              cmap=cmap, vmin=-1, vmax=1, center=0,
                              square=True, linewidths=.5, cbar_kws={"shrink": .5})
        
        ax.set_title(f"Matriz de Correlação ({method.capitalize()})")
        
        # Rotacionar rótulos se muitas features
        if len(corr_matrix) > 10:
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        
        plt.tight_layout()
    except Exception as e:
        logger.error(f"Erro ao plotar matriz de correlação: {str(e)}")
        ax.text(0.5, 0.5, f"Erro ao plotar: {str(e)}", ha='center', va='center')
    
    return fig_to_base64(fig)


def plot_transformation_effect(
    original_data: pd.Series,
    transformed_data: pd.Series,
    transformation_type: str,
    target: Optional[pd.Series] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> str:
    """
    Plota o efeito de uma transformação em uma feature.
    
    Args:
        original_data: Série com dados originais
        transformed_data: Série com dados transformados
        transformation_type: Tipo de transformação aplicada
        target: Série com alvo (opcional)
        figsize: Tamanho da figura
        
    Returns:
        String base64 da imagem
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    try:
        # Plotar distribuição original
        if pd.api.types.is_numeric_dtype(original_data):
            # Histograma para dados numéricos
            axes[0].hist(original_data.dropna(), bins=30, alpha=0.7)
            
            if target is not None and pd.api.types.is_numeric_dtype(target):
                # Adicionar segunda escala para visualizar relação com alvo
                ax2 = axes[0].twinx()
                ax2.scatter(original_data, target, alpha=0.5, color='red', s=15)
                ax2.set_ylabel('Target', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
            
            axes[0].set_title(f"Distribuição Original\nSkewness: {original_data.skew():.2f}")
            axes[0].set_xlabel("Valor")
            axes[0].set_ylabel("Frequência")
        else:
            # Barras para dados categóricos
            value_counts = original_data.value_counts().sort_values(ascending=False)
            if len(value_counts) > 10:
                value_counts = value_counts.head(10)
            axes[0].bar(value_counts.index.astype(str), value_counts.values, alpha=0.7)
            axes[0].set_title("Distribuição Original")
            axes[0].set_xlabel("Valor")
            axes[0].set_ylabel("Contagem")
            plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plotar distribuição transformada
        if pd.api.types.is_numeric_dtype(transformed_data):
            # Histograma para dados numéricos
            axes[1].hist(transformed_data.dropna(), bins=30, alpha=0.7)
            
            if target is not None and pd.api.types.is_numeric_dtype(target):
                # Adicionar segunda escala para visualizar relação com alvo
                ax2 = axes[1].twinx()
                ax2.scatter(transformed_data, target, alpha=0.5, color='red', s=15)
                ax2.set_ylabel('Target', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
            
            axes[1].set_title(f"Após {transformation_type}\nSkewness: {transformed_data.skew():.2f}")
            axes[1].set_xlabel("Valor")
            axes[1].set_ylabel("Frequência")
        else:
            # Barras para dados categóricos
            value_counts = transformed_data.value_counts().sort_values(ascending=False)
            if len(value_counts) > 10:
                value_counts = value_counts.head(10)
            axes[1].bar(value_counts.index.astype(str), value_counts.values, alpha=0.7)
            axes[1].set_title(f"Após {transformation_type}")
            axes[1].set_xlabel("Valor")
            axes[1].set_ylabel("Contagem")
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
    except Exception as e:
        logger.error(f"Erro ao plotar efeito da transformação: {str(e)}")
        axes[0].text(0.5, 0.5, f"Erro ao plotar: {str(e)}", ha='center', va='center')
        axes[1].text(0.5, 0.5, f"Erro ao plotar: {str(e)}", ha='center', va='center')
    
    return fig_to_base64(fig)


def plot_transformation_tree(
    tree_data: Dict[str, Any],
    max_nodes: int = 50,
    figsize: Tuple[int, int] = (15, 10)
) -> str:
    """
    Plota a árvore de transformações.
    
    Args:
        tree_data: Dicionário representando a árvore
        max_nodes: Número máximo de nós a mostrar
        figsize: Tamanho da figura
        
    Returns:
        String base64 da imagem
    """
    # Criar grafo direcionado
    G = nx.DiGraph()
    
    # Função recursiva para adicionar nós
    def add_nodes_to_graph(node_data, parent_id=None, count=0):
        if count > max_nodes:
            return count
        
        node_id = node_data.get('id', f"node_{count}")
        node_name = node_data.get('name', f"Node {count}")
        is_selected = node_data.get('is_selected', False)
        importance = node_data.get('importance', 0)
        
        # Adicionar nó
        G.add_node(node_id, name=node_name, selected=is_selected, importance=importance)
        
        # Adicionar aresta se tiver pai
        if parent_id:
            G.add_edge(parent_id, node_id)
        
        # Processar filhos
        children = node_data.get('children', [])
        for child in children:
            count = add_nodes_to_graph(child, node_id, count + 1)
            if count > max_nodes:
                break
        
        return count
    
    # Processar nós raiz
    nodes_count = 0
    for root_node in tree_data.get('root_nodes', []):
        nodes_count = add_nodes_to_graph(root_node, None, nodes_count)
        if nodes_count > max_nodes:
            break
    
    # Limitar número de nós
    if len(G) > max_nodes:
        logger.warning(f"Árvore muito grande, limitando a {max_nodes} nós")
        # Pegar os primeiros max_nodes nós
        nodes = list(G.nodes())[:max_nodes]
        G = G.subgraph(nodes)
    
    # Plotar grafo
    fig, ax = plt.subplots(figsize=figsize)
    
    try:
        # Definir layout
        pos = nx.spring_layout(G, seed=42)
        
        # Cores baseadas em seleção e importância
        node_colors = []
        for node in G.nodes():
            if G.nodes[node].get('selected', False):
                # Nós selecionados em verde, mais escuro = mais importante
                importance = G.nodes[node].get('importance', 0)
                green_intensity = 0.5 + min(0.5, importance * 5)  # Escala de 0.5 a 1.0
                node_colors.append((0, green_intensity, 0))
            else:
                # Nós não selecionados em azul
                node_colors.append((0.1, 0.1, 0.8))
        
        # Desenhar nós
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
        
        # Desenhar arestas
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True, arrowsize=15)
        
        # Adicionar rótulos
        labels = {node: G.nodes[node]['name'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        ax.set_title("Árvore de Transformações")
        ax.axis('off')
        
        plt.tight_layout()
    except Exception as e:
        logger.error(f"Erro ao plotar árvore de transformações: {str(e)}")
        ax.text(0.5, 0.5, f"Erro ao plotar: {str(e)}", ha='center', va='center')
        ax.axis('on')
    
    return fig_to_base64(fig)


def plot_feature_space(
    data: pd.DataFrame,
    target: Optional[Union[str, pd.Series]] = None,
    method: str = 'pca',
    n_components: int = 2,
    figsize: Tuple[int, int] = (10, 8)
) -> str:
    """
    Plota o espaço de features em 2D usando PCA ou t-SNE.
    
    Args:
        data: DataFrame com features
        target: Nome da coluna alvo ou Series com valores alvo
        method: Método de redução de dimensionalidade ('pca' ou 'tsne')
        n_components: Número de componentes (2 ou 3)
        figsize: Tamanho da figura
        
    Returns:
        String base64 da imagem
    """
    # Selecionar apenas colunas numéricas
    numeric_data = data.select_dtypes(include=['number'])
    
    if numeric_data.empty:
        logger.error("Nenhuma coluna numérica nos dados")
        return ""
    
    # Preparar target
    if isinstance(target, str) and target in data.columns:
        y = data[target]
    elif isinstance(target, pd.Series):
        y = target
    else:
        y = None
    
    # Preencher valores ausentes
    X = numeric_data.fillna(numeric_data.median())
    
    # Verificar se há dados suficientes
    if len(X) < 2:
        logger.error("Dados insuficientes para visualização")
        return ""
    
    # Aplicar redução de dimensionalidade
    if method.lower() == 'pca':
        model = PCA(n_components=n_components, random_state=42)
        try:
            components = model.fit_transform(X)
        except Exception as e:
            logger.error(f"Erro ao aplicar PCA: {str(e)}")
            return ""
        
        explained_variance = model.explained_variance_ratio_
        method_name = "PCA"
    elif method.lower() == 'tsne':
        model = TSNE(n_components=n_components, random_state=42)
        try:
            components = model.fit_transform(X)
        except Exception as e:
            logger.error(f"Erro ao aplicar t-SNE: {str(e)}")
            return ""
        
        explained_variance = None
        method_name = "t-SNE"
    else:
        logger.error(f"Método desconhecido: {method}")
        return ""
    
    # Plotar resultado
    fig = plt.figure(figsize=figsize)
    
    try:
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            
            if y is not None:
                # Colorir por target
                if pd.api.types.is_numeric_dtype(y):
                    scatter = ax.scatter(components[:, 0], components[:, 1], components[:, 2],
                                        c=y, cmap='viridis', alpha=0.7)
                    plt.colorbar(scatter, ax=ax, label='Target')
                else:
                    # Target categórico
                    y_categories = y.astype('category').cat.codes
                    scatter = ax.scatter(components[:, 0], components[:, 1], components[:, 2],
                                        c=y_categories, cmap='tab10', alpha=0.7)
                    
                    # Adicionar legenda
                    legend_handles = []
                    for i, cat in enumerate(y.unique()):
                        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                                        markerfacecolor=plt.cm.tab10(i/10),
                                                        markersize=10, label=cat))
                    
                    ax.legend(handles=legend_handles, title='Target')
            else:
                # Sem target, colorir por densidade
                scatter = ax.scatter(components[:, 0], components[:, 1], components[:, 2],
                                    alpha=0.7)
            
            ax.set_xlabel(f"Componente 1")
            ax.set_ylabel(f"Componente 2")
            ax.set_zlabel(f"Componente 3")
            
        else:  # 2D
            ax = fig.add_subplot(111)
            
            if y is not None:
                # Colorir por target
                if pd.api.types.is_numeric_dtype(y):
                    scatter = ax.scatter(components[:, 0], components[:, 1],
                                        c=y, cmap='viridis', alpha=0.7)
                    plt.colorbar(scatter, ax=ax, label='Target')
                else:
                    # Target categórico
                    y_categories = y.astype('category').cat.codes
                    scatter = ax.scatter(components[:, 0], components[:, 1],
                                        c=y_categories, cmap='tab10', alpha=0.7)
                    
                    # Adicionar legenda
                    legend_handles = []
                    for i, cat in enumerate(y.unique()):
                        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                                        markerfacecolor=plt.cm.tab10(i/10),
                                                        markersize=10, label=cat))
                    
                    ax.legend(handles=legend_handles, title='Target')
            else:
                # Sem target, colorir por densidade
                scatter = ax.scatter(components[:, 0], components[:, 1], alpha=0.7)
            
            ax.set_xlabel(f"Componente 1")
            ax.set_ylabel(f"Componente 2")
        
        # Adicionar título
        if method_name == "PCA" and explained_variance is not None:
            var_text = ", ".join([f"{var:.1%}" for var in explained_variance[:n_components]])
            ax.set_title(f"{method_name} - Variância explicada: {var_text}")
        else:
            ax.set_title(f"Visualização de Features usando {method_name}")
        
        plt.tight_layout()
    except Exception as e:
        logger.error(f"Erro ao plotar espaço de features: {str(e)}")
        if 'ax' in locals():
            ax.text(0.5, 0.5, f"Erro ao plotar: {str(e)}", ha='center', va='center')
            ax.axis('on')
    
    return fig_to_base64(fig)


def plot_feature_pair_grid(
    data: pd.DataFrame,
    features: List[str],
    target: Optional[Union[str, pd.Series]] = None,
    n_features: int = 5,
    figsize: Tuple[int, int] = (12, 10)
) -> str:
    """
    Plota um grid de pares de features com distribuições e targets.
    
    Args:
        data: DataFrame com dados
        features: Lista de features para incluir
        target: Nome da coluna alvo ou Series com valores alvo
        n_features: Número máximo de features a incluir
        figsize: Tamanho da figura
        
    Returns:
        String base64 da imagem
    """
    # Verificar features
    valid_features = [f for f in features if f in data.columns]
    
    if not valid_features:
        logger.error("Nenhuma feature válida especificada")
        return ""
    
    # Limitar número de features
    if len(valid_features) > n_features:
        valid_features = valid_features[:n_features]
    
    # Preparar dados
    plot_data = data[valid_features].copy()
    
    # Adicionar target se especificado
    if isinstance(target, str) and target in data.columns:
        plot_data['target'] = data[target]
        hue = 'target'
    elif isinstance(target, pd.Series):
        plot_data['target'] = target.values
        hue = 'target'
    else:
        hue = None
    
    # Plotar usando seaborn pairgrid
    try:
        fig = plt.figure(figsize=figsize)
        
        # Usar pairplot para criar grid
        g = sns.pairplot(plot_data, hue=hue, diag_kind='kde',
                          plot_kws={'alpha': 0.5}, diag_kws={'alpha': 0.5})
        
        # Ajustar título
        if hue:
            plt.subplots_adjust(top=0.95)
            fig.suptitle(f"Relações entre Features Coloridas por Target", fontsize=14)
        else:
            plt.subplots_adjust(top=0.95)
            fig.suptitle(f"Relações entre Features", fontsize=14)
        
        # Obter figura do seaborn e converter
        return fig_to_base64(g.fig)
    except Exception as e:
        logger.error(f"Erro ao plotar grid de features: {str(e)}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Erro ao plotar grid: {str(e)}", ha='center', va='center')
        return fig_to_base64(fig)


def plot_partial_dependence(
    data: pd.DataFrame,
    feature: str,
    model: Any,
    target: Optional[Union[str, pd.Series]] = None,
    n_points: int = 50,
    figsize: Tuple[int, int] = (10, 6)
) -> str:
    """
    Plota a dependência parcial de uma feature em relação ao alvo.
    
    Args:
        data: DataFrame com dados
        feature: Feature para analisar
        model: Modelo treinado
        target: Nome da coluna alvo ou Series com valores alvo
        n_points: Número de pontos para calcular dependência
        figsize: Tamanho da figura
        
    Returns:
        String base64 da imagem
    """
    # Verificar feature
    if feature not in data.columns:
        logger.error(f"Feature {feature} não encontrada nos dados")
        return ""
    
    # Verificar se a feature é numérica
    if not pd.api.types.is_numeric_dtype(data[feature]):
        logger.error(f"Feature {feature} não é numérica")
        return ""
    
    # Preparar X para predição
    X = data.copy()
    
    # Remover alvo de X se for uma coluna
    if isinstance(target, str) and target in X.columns:
        X = X.drop(columns=[target])
    
    # Verificar funções no modelo
    if not hasattr(model, 'predict'):
        logger.error("Modelo não tem método 'predict'")
        return ""
    
    try:
        # Criar valores para a feature analisada
        feature_min = data[feature].min()
        feature_max = data[feature].max()
        
        grid = np.linspace(feature_min, feature_max, n_points)
        
        # Calcular dependência parcial
        mean_predictions = []
        
        for value in grid:
            # Criar cópia dos dados com feature fixa
            X_mod = X.copy()
            X_mod[feature] = value
            
            # Fazer predição
            predictions = model.predict(X_mod)
            mean_prediction = np.mean(predictions)
            mean_predictions.append(mean_prediction)
        
        # Plotar resultado
        fig, ax = plt.subplots(figsize=figsize)
        
        # Linha de dependência parcial
        ax.plot(grid, mean_predictions, 'b-', linewidth=2)
        
        # Adicionar histograma para distribuição da feature
        ax_twin = ax.twinx()
        ax_twin.hist(data[feature], bins=30, alpha=0.3, color='gray')
        ax_twin.set_ylabel('Frequência', color='gray')
        ax_twin.tick_params(axis='y', labelcolor='gray')
        
        # Configurar gráfico
        ax.set_xlabel(feature)
        ax.set_ylabel('Predição média')
        ax.set_title(f"Dependência Parcial para {feature}")
        
        # Adicionar linhas de grade
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig_to_base64(fig)
        
    except Exception as e:
        logger.error(f"Erro ao plotar dependência parcial: {str(e)}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Erro ao plotar dependência parcial: {str(e)}", ha='center', va='center')
        return fig_to_base64(fig)


def plot_time_series_features(
    data: pd.DataFrame,
    date_column: str,
    features: List[str],
    figsize: Tuple[int, int] = (12, 8)
) -> str:
    """
    Plota features ao longo do tempo para séries temporais.
    
    Args:
        data: DataFrame com dados
        date_column: Nome da coluna de data
        features: Lista de features para plotar
        figsize: Tamanho da figura
        
    Returns:
        String base64 da imagem
    """
    # Verificar colunas
    if date_column not in data.columns:
        logger.error(f"Coluna de data {date_column} não encontrada")
        return ""
    
    valid_features = [f for f in features if f in data.columns]
    
    if not valid_features:
        logger.error("Nenhuma feature válida para plotar")
        return ""
    
    # Converter para datetime se não for
    if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
        try:
            date_series = pd.to_datetime(data[date_column])
        except:
            logger.error(f"Não foi possível converter {date_column} para data")
            return ""
    else:
        date_series = data[date_column]
    
    # Ordenar por data
    sorted_data = data.sort_values(by=date_column)
    
    # Plotar séries
    fig, ax = plt.subplots(figsize=figsize)
    
    try:
        # Plotar cada feature
        for feature in valid_features:
            if pd.api.types.is_numeric_dtype(sorted_data[feature]):
                ax.plot(sorted_data[date_column], sorted_data[feature], label=feature, alpha=0.7)
        
        # Configurar gráfico
        ax.set_xlabel('Data')
        ax.set_ylabel('Valor')
        ax.set_title('Features ao Longo do Tempo')
        ax.legend()
        
        # Rotacionar rótulos de data
        plt.xticks(rotation=45)
        
        # Ajustar layout
        plt.tight_layout()
        return fig_to_base64(fig)
        
    except Exception as e:
        logger.error(f"Erro ao plotar séries temporais: {str(e)}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Erro ao plotar séries temporais: {str(e)}", ha='center', va='center')
        return fig_to_base64(fig)


def create_reports_dashboard(
    data: pd.DataFrame,
    target: Optional[Union[str, pd.Series]] = None,
    top_features: Optional[List[str]] = None,
    transformation_results: Optional[Dict[str, Any]] = None
) -> str:
    """
    Cria um dashboard HTML com relatórios e visualizações.
    
    Args:
        data: DataFrame com dados
        target: Nome da coluna alvo ou Series com valores alvo
        top_features: Lista das features mais importantes
        transformation_results: Resultados de transformações aplicadas
        
    Returns:
        String HTML com o dashboard
    """
    if top_features is None:
        # Selecionar colunas numéricas como padrão
        top_features = data.select_dtypes(include=['number']).columns.tolist()[:5]
    
    # Limitar número de features para não sobrecarregar
    if len(top_features) > 5:
        top_features = top_features[:5]
    
    # Iniciar HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Dashboard de Engenharia de Features</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .section { margin-bottom: 30px; }
            .plot-container { margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; }
            h1 { color: #2c3e50; }
            h2 { color: #3498db; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .full-width { grid-column: span 2; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
        </style>
    </head>
    <body>
        <h1>Dashboard de Engenharia de Features</h1>
    """
    
    # Seção de estatísticas básicas
    html += """
        <div class="section">
            <h2>Estatísticas Básicas do Dataset</h2>
            <div class="grid">
                <div class="plot-container">
    """
    
    # Tabela com estatísticas básicas
    stats_df = pd.DataFrame({
        'Métrica': ['Número de Registros', 'Número de Features', 'Features Numéricas', 'Features Categóricas',
                   'Valores Ausentes (%)'],
        'Valor': [
            len(data),
            len(data.columns),
            len(data.select_dtypes(include=['number']).columns),
            len(data.select_dtypes(exclude=['number']).columns),
            f"{(data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100):.2f}%"
        ]
    })
    
    html += stats_df.to_html(index=False, classes='dataframe')
    html += """
                </div>
    """
    
    # Distribuição do alvo (se disponível)
    if target is not None:
        html += """
                <div class="plot-container">
                    <h3>Distribuição da Variável Alvo</h3>
        """
        
        if isinstance(target, str) and target in data.columns:
            target_series = data[target]
        else:
            target_series = target
        
        try:
            plt.figure(figsize=(8, 6))
            
            if pd.api.types.is_numeric_dtype(target_series):
                plt.hist(target_series.dropna(), bins=30, alpha=0.7)
                plt.title(f"Distribuição de {target if isinstance(target, str) else 'Alvo'}")
                plt.xlabel("Valor")
                plt.ylabel("Frequência")
            else:
                # Gráfico de barras para alvo categórico
                value_counts = target_series.value_counts().sort_values(ascending=False)
                plt.bar(value_counts.index.astype(str), value_counts.values, alpha=0.7)
                plt.title(f"Distribuição de {target if isinstance(target, str) else 'Alvo'}")
                plt.xlabel("Categoria")
                plt.ylabel("Contagem")
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            target_img = fig_to_base64(plt.gcf())
            html += f'<img src="data:image/png;base64,{target_img}" style="width:100%">'
        except Exception as e:
            logger.error(f"Erro ao plotar distribuição do alvo: {str(e)}")
            html += f"<p>Erro ao gerar visualização: {str(e)}</p>"
        
        html += """
                </div>
        """
    
    html += """
            </div>
        </div>
    """
    
    # Seção de distribuição de features
    html += """
        <div class="section">
            <h2>Distribuição das Features Principais</h2>
            <div class="grid">
    """
    
    # Plotar distribuição para cada feature principal
    for feature in top_features:
        if feature in data.columns:
            html += f"""
                <div class="plot-container">
                    <h3>{feature}</h3>
            """
            
            try:
                feature_img = plot_feature_distribution(data, feature, target)
                html += f'<img src="data:image/png;base64,{feature_img}" style="width:100%">'
            except Exception as e:
                logger.error(f"Erro ao plotar distribuição de {feature}: {str(e)}")
                html += f"<p>Erro ao gerar visualização: {str(e)}</p>"
            
            html += """
                </div>
            """
    
    html += """
            </div>
        </div>
    """
    
    # Seção de correlação
    html += """
        <div class="section">
            <h2>Correlação entre Features</h2>
            <div class="plot-container full-width">
    """
    
    try:
        corr_img = plot_correlation_matrix(data, features=top_features)
        html += f'<img src="data:image/png;base64,{corr_img}" style="width:100%">'
    except Exception as e:
        logger.error(f"Erro ao plotar matriz de correlação: {str(e)}")
        html += f"<p>Erro ao gerar visualização de correlação: {str(e)}</p>"
    
    html += """
            </div>
        </div>
    """
    
    # Seção de pares de features
    html += """
        <div class="section">
            <h2>Relações entre Pares de Features</h2>
            <div class="plot-container full-width">
    """
    
    try:
        pairs_img = plot_feature_pair_grid(data, top_features, target)
        html += f'<img src="data:image/png;base64,{pairs_img}" style="width:100%">'
    except Exception as e:
        logger.error(f"Erro ao plotar grid de pares: {str(e)}")
        html += f"<p>Erro ao gerar visualização de pares: {str(e)}</p>"
    
    html += """
            </div>
        </div>
    """
    
    # Seção de transformações (se disponível)
    if transformation_results:
        html += """
            <div class="section">
                <h2>Impacto das Transformações</h2>
                <div class="grid">
        """
        
        # Mostrar exemplos de transformações
        for transform_name, transform_data in transformation_results.items():
            if 'original' in transform_data and 'transformed' in transform_data:
                html += f"""
                    <div class="plot-container full-width">
                        <h3>Transformação: {transform_name}</h3>
                """
                
                try:
                    # Criar Series a partir dos dados
                    original = pd.Series(transform_data['original'])
                    transformed = pd.Series(transform_data['transformed'])
                    
                    # Plotar efeito da transformação
                    transform_img = plot_transformation_effect(
                        original, transformed, transform_name,
                        target=pd.Series(transform_data.get('target', []))
                        if 'target' in transform_data else None
                    )
                    
                    html += f'<img src="data:image/png;base64,{transform_img}" style="width:100%">'
                except Exception as e:
                    logger.error(f"Erro ao plotar transformação {transform_name}: {str(e)}")
                    html += f"<p>Erro ao gerar visualização da transformação: {str(e)}</p>"
                
                html += """
                    </div>
                """
        
        html += """
                </div>
            </div>
        """
    
    # Fechar HTML
    html += """
    </body>
    </html>
    """
    
    return html
