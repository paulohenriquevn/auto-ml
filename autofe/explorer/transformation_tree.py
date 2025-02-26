"""
Implementação da árvore de transformações para o Explorer.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import pandas as pd
import numpy as np
import pickle
import json
import logging
from uuid import uuid4

# Importações internas
from config import EXPLORER_CONFIG, TRANSFORMATIONS


class TransformationNode:
    """
    Representa um nó na árvore de transformações, que corresponde a uma feature.
    """
    
    def __init__(
        self, 
        name: str, 
        parent_id: Optional[str] = None,
        transformation_type: Optional[str] = None,
        transformation_params: Optional[Dict[str, Any]] = None,
        depth: int = 0
    ):
        """
        Inicializa um nó de transformação.
        
        Args:
            name: Nome da feature
            parent_id: ID do nó pai (None para features originais)
            transformation_type: Tipo de transformação aplicada (None para features originais)
            transformation_params: Parâmetros da transformação (None para features originais)
            depth: Profundidade do nó na árvore
        """
        self.id = str(uuid4())
        self.name = name
        self.parent_id = parent_id
        self.transformation_type = transformation_type
        self.transformation_params = transformation_params or {}
        self.depth = depth
        self.children = []
        
        # Métricas de qualidade da feature
        self.importance = 0.0
        self.performance_gain = 0.0
        self.redundancy_score = 0.0
        self.is_selected = False
    
    def add_child(self, child_node: 'TransformationNode'):
        """
        Adiciona um nó filho a este nó.
        
        Args:
            child_node: Nó filho a ser adicionado
        """
        self.children.append(child_node)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte o nó e seus filhos para um dicionário.
        
        Returns:
            Dicionário representando o nó e seus filhos
        """
        return {
            'id': self.id,
            'name': self.name,
            'parent_id': self.parent_id,
            'transformation_type': self.transformation_type,
            'transformation_params': self.transformation_params,
            'depth': self.depth,
            'importance': self.importance,
            'performance_gain': self.performance_gain,
            'redundancy_score': self.redundancy_score,
            'is_selected': self.is_selected,
            'children': [child.to_dict() for child in self.children]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransformationNode':
        """
        Cria um nó a partir de um dicionário.
        
        Args:
            data: Dicionário com os dados do nó
            
        Returns:
            Nó de transformação criado
        """
        node = cls(
            name=data['name'],
            parent_id=data['parent_id'],
            transformation_type=data['transformation_type'],
            transformation_params=data['transformation_params'],
            depth=data['depth']
        )
        node.id = data['id']
        node.importance = data['importance']
        node.performance_gain = data['performance_gain']
        node.redundancy_score = data['redundancy_score']
        node.is_selected = data['is_selected']
        
        for child_data in data['children']:
            child = cls.from_dict(child_data)
            node.add_child(child)
        
        return node


class TransformationTree:
    """
    Árvore hierárquica que representa diferentes sequências de transformações.
    """
    
    def __init__(self):
        """
        Inicializa a árvore de transformações.
        """
        self.logger = logging.getLogger(__name__)
        self.root_nodes = []  # Nós raiz (features originais)
        self.nodes_by_id = {}  # Mapeamento de IDs para nós
        self.nodes_by_name = {}  # Mapeamento de nomes para nós
        self.max_depth = EXPLORER_CONFIG['max_transformation_depth']
    
    def add_root_node(self, name: str) -> TransformationNode:
        """
        Adiciona um nó raiz (feature original) à árvore.
        
        Args:
            name: Nome da feature original
            
        Returns:
            Nó raiz criado
        """
        node = TransformationNode(name=name, depth=0)
        self.root_nodes.append(node)
        self.nodes_by_id[node.id] = node
        self.nodes_by_name[node.name] = node
        return node
    
    def add_transformation_node(
        self,
        parent_id: str,
        name: str,
        transformation_type: str,
        transformation_params: Dict[str, Any]
    ) -> TransformationNode:
        """
        Adiciona um nó de transformação à árvore.
        
        Args:
            parent_id: ID do nó pai
            name: Nome da feature transformada
            transformation_type: Tipo de transformação aplicada
            transformation_params: Parâmetros da transformação
            
        Returns:
            Nó de transformação criado
        """
        parent_node = self.nodes_by_id.get(parent_id)
        if not parent_node:
            raise ValueError(f"Nó pai com ID {parent_id} não encontrado")
        
        if parent_node.depth >= self.max_depth:
            self.logger.warning(
                f"Não é possível adicionar transformação em profundidade {parent_node.depth + 1} "
                f"(máximo: {self.max_depth})"
            )
            return None
        
        # Verificar se o nome já existe
        if name in self.nodes_by_name:
            self.logger.warning(f"Feature com nome '{name}' já existe")
            return self.nodes_by_name[name]
        
        # Criar nó filho
        child_node = TransformationNode(
            name=name,
            parent_id=parent_id,
            transformation_type=transformation_type,
            transformation_params=transformation_params,
            depth=parent_node.depth + 1
        )
        
        # Adicionar à árvore
        parent_node.add_child(child_node)
        self.nodes_by_id[child_node.id] = child_node
        self.nodes_by_name[child_node.name] = child_node
        
        return child_node
    
    def build(
        self, 
        data: pd.DataFrame, 
        recommended_transformations: Optional[Dict[str, List[str]]] = None
    ):
        """
        Constrói a árvore de transformações inicial a partir dos dados.
        
        Args:
            data: DataFrame com os dados
            recommended_transformations: Transformações recomendadas pelo Learner-Predictor
        """
        self.logger.info("Construindo árvore de transformações inicial")
        
        # Limpar árvore existente
        self.root_nodes = []
        self.nodes_by_id = {}
        self.nodes_by_name = {}
        
        # Adicionar nós raiz para cada coluna
        for column in data.columns:
            self.add_root_node(column)
            
            # Detectar tipo de dados da coluna
            col_type = self._detect_column_type(data[column])
            
            # Se há transformações recomendadas, usá-las
            transformations_to_apply = []
            if recommended_transformations and column in recommended_transformations:
                transformations_to_apply = recommended_transformations[column]
            else:
                # Caso contrário, usar transformações padrão para o tipo
                if col_type in TRANSFORMATIONS:
                    transformations_to_apply = TRANSFORMATIONS[col_type]
            
            # Não fazer nada se não houver transformações aplicáveis
            if not transformations_to_apply:
                continue
                
            # Adicionar nós de primeiro nível
            self._add_first_level_transformations(column, col_type, transformations_to_apply)
        
        self.logger.info(f"Árvore de transformações inicializada com {len(self.root_nodes)} features originais")
    
    def _detect_column_type(self, column: pd.Series) -> str:
        """
        Detecta o tipo de dados de uma coluna.
        
        Args:
            column: Série do pandas com os dados da coluna
            
        Returns:
            Tipo de dados ('numeric', 'categorical', 'datetime', 'text')
        """
        if pd.api.types.is_numeric_dtype(column):
            return 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(column):
            return 'datetime'
        elif column.dtype == 'object':
            # Verificar se é texto (mais de 10 caracteres em média)
            if column.str.len().mean() > 10:
                return 'text'
            else:
                return 'categorical'
        else:
            return 'categorical'
    
    def _add_first_level_transformations(
        self, 
        column: str, 
        col_type: str, 
        transformations: List[str]
    ):
        """
        Adiciona transformações de primeiro nível para uma coluna.
        
        Args:
            column: Nome da coluna
            col_type: Tipo de dados da coluna
            transformations: Lista de transformações a aplicar
        """
        parent_id = self.nodes_by_name[column].id
        
        for transformation in transformations:
            # Criar nome para a feature transformada
            new_name = f"{transformation}({column})"
            
            # Adicionar nó de transformação
            self.add_transformation_node(
                parent_id=parent_id,
                name=new_name,
                transformation_type=transformation,
                transformation_params={'column': column}
            )
    
    def get_all_nodes(self) -> List[TransformationNode]:
        """
        Retorna todos os nós da árvore.
        
        Returns:
            Lista com todos os nós
        """
        return list(self.nodes_by_id.values())
    
    def get_selected_nodes(self) -> List[TransformationNode]:
        """
        Retorna todos os nós selecionados para uso final.
        
        Returns:
            Lista com nós selecionados
        """
        return [node for node in self.nodes_by_id.values() if node.is_selected]
    
    def get_node_by_id(self, node_id: str) -> Optional[TransformationNode]:
        """
        Retorna um nó pelo seu ID.
        
        Args:
            node_id: ID do nó
            
        Returns:
            Nó correspondente ou None se não encontrado
        """
        return self.nodes_by_id.get(node_id)
    
    def get_node_by_name(self, name: str) -> Optional[TransformationNode]:
        """
        Retorna um nó pelo seu nome.
        
        Args:
            name: Nome do nó
            
        Returns:
            Nó correspondente ou None se não encontrado
        """
        return self.nodes_by_name.get(name)
    
    def get_node_lineage(self, node_id: str) -> List[TransformationNode]:
        """
        Retorna a linhagem completa de um nó (caminho da raiz até o nó).
        
        Args:
            node_id: ID do nó
            
        Returns:
            Lista com todos os nós no caminho (do nó raiz até o nó especificado)
        """
        node = self.get_node_by_id(node_id)
        if not node:
            return []
        
        lineage = [node]
        current = node
        
        while current.parent_id:
            parent = self.get_node_by_id(current.parent_id)
            if parent:
                lineage.insert(0, parent)
                current = parent
            else:
                break
        
        return lineage
    
    def mark_node_as_selected(self, node_id: str, selected: bool = True):
        """
        Marca um nó como selecionado para uso final.
        
        Args:
            node_id: ID do nó
            selected: Valor de seleção (True/False)
        """
        node = self.get_node_by_id(node_id)
        if node:
            node.is_selected = selected
    
    def prune_redundant_nodes(self, redundancy_threshold: Optional[float] = None):
        """
        Remove nós redundantes da árvore.
        
        Args:
            redundancy_threshold: Limiar de redundância (usa o valor da configuração se None)
        """
        if redundancy_threshold is None:
            redundancy_threshold = EXPLORER_CONFIG['redundancy_threshold']
        
        # Identificar nós a serem removidos (com alta redundância)
        nodes_to_remove = []
        for node in self.get_all_nodes():
            if node.redundancy_score > redundancy_threshold:
                nodes_to_remove.append(node.id)
        
        # Remover nós
        for node_id in nodes_to_remove:
            self._remove_node(node_id)
        
        self.logger.info(f"Removidos {len(nodes_to_remove)} nós redundantes")
    
    def _remove_node(self, node_id: str):
        """
        Remove um nó da árvore.
        
        Args:
            node_id: ID do nó a ser removido
        """
        node = self.get_node_by_id(node_id)
        if not node:
            return
        
        # Remover referências em nós pai
        if node.parent_id:
            parent = self.get_node_by_id(node.parent_id)
            if parent:
                parent.children = [child for child in parent.children if child.id != node_id]
        else:
            # Se for nó raiz, remover da lista de raízes
            self.root_nodes = [root for root in self.root_nodes if root.id != node_id]
        
        # Remover dos dicionários
        if node.id in self.nodes_by_id:
            del self.nodes_by_id[node.id]
        
        if node.name in self.nodes_by_name:
            del self.nodes_by_name[node.name]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte a árvore para um dicionário.
        
        Returns:
            Dicionário representando a árvore
        """
        return {
            'root_nodes': [node.to_dict() for node in self.root_nodes],
            'max_depth': self.max_depth
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransformationTree':
        """
        Cria uma árvore a partir de um dicionário.
        
        Args:
            data: Dicionário com os dados da árvore
            
        Returns:
            Árvore de transformação criada
        """
        tree = cls()
        tree.max_depth = data['max_depth']
        
        # Recriar nós raiz
        for root_data in data['root_nodes']:
            root_node = TransformationNode.from_dict(root_data)
            tree.root_nodes.append(root_node)
            
            # Função recursiva para adicionar todos os nós ao dicionário
            def add_nodes_recursively(node):
                tree.nodes_by_id[node.id] = node
                tree.nodes_by_name[node.name] = node
                for child in node.children:
                    add_nodes_recursively(child)
            
            add_nodes_recursively(root_node)
        
        return tree
    
    def save(self, path: str):
        """
        Salva a árvore em um arquivo.
        
        Args:
            path: Caminho para salvar a árvore
        """
        with open(path, 'wb') as f:
            pickle.dump(self.to_dict(), f)
        
        self.logger.info(f"Árvore de transformações salva em {path}")
    
    def load(self, path: str):
        """
        Carrega a árvore de um arquivo.
        
        Args:
            path: Caminho para carregar a árvore
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        tree = self.from_dict(data)
        self.root_nodes = tree.root_nodes
        self.nodes_by_id = tree.nodes_by_id
        self.nodes_by_name = tree.nodes_by_name
        self.max_depth = tree.max_depth
        
        self.logger.info(f"Árvore de transformações carregada de {path}")
