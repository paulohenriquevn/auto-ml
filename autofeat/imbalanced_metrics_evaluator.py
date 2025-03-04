import numpy as np
import pandas as pd
import logging
from typing import Dict
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
from sklearn.metrics import f1_score, recall_score, precision_score, balanced_accuracy_score
    
    
class ImbalancedMetricsEvaluator:
    """
    Avaliador especializado em métricas para problemas de classificação desbalanceados.
    Pode ser usado em conjunto com o TransformationEvaluator existente.
    """
    def __init__(self, minority_classes=None, pos_label=1):
        """
        Inicializa o avaliador de métricas para problemas desbalanceados.
        
        Args:
            minority_classes: Lista de classes consideradas minoritárias (ou None para detectar automaticamente)
            pos_label: Classe positiva para métricas binárias
        """
        self.minority_classes = minority_classes
        self.pos_label = pos_label
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
    
    def evaluate(self, y_true, y_pred, y_proba=None):
        """
        Avalia o desempenho usando métricas específicas para classes desbalanceadas.
        
        Args:
            y_true: Valores reais
            y_pred: Valores previstos
            y_proba: Probabilidades previstas (opcional)
            
        Returns:
            Dicionário com métricas calculadas
        """
        metrics = {}
        
        # Detecta classes minoritárias se não forem especificadas
        if self.minority_classes is None:
            class_counts = np.bincount(y_true.astype(int)) if hasattr(y_true, 'astype') else np.bincount(y_true)
            min_count = class_counts.min()
            self.minority_classes = [i for i, count in enumerate(class_counts) if count == min_count]
        
        # Calcula a matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        
        # Métricas básicas para cada classe minoritária
        for cls in self.minority_classes:
            # Para classificação binária, ajusta índices
            if len(cm) == 2:
                if cls == 1:  # classe positiva
                    tn, fp, fn, tp = cm.ravel()
                else:  # classe negativa (0)
                    tp, fn, fp, tn = cm.ravel()
            else:
                # Multiclasse: calcula TP, FP, FN para a classe específica
                tp = cm[cls, cls]
                fp = cm[:, cls].sum() - tp
                fn = cm[cls, :].sum() - tp
                tn = cm.sum() - (tp + fp + fn)
            
            # Métricas específicas para a classe
            # Recall (sensibilidade) - quão bem detectamos a classe minoritária
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics[f'recall_class_{cls}'] = recall
            
            # Precision - quão confiáveis são nossas detecções positivas
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics[f'precision_class_{cls}'] = precision
            
            # F1 Score para a classe
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            metrics[f'f1_class_{cls}'] = f1
            
            # Specificity - quão bem detectamos os negativos
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics[f'specificity_class_{cls}'] = specificity
            
            # G-mean - média geométrica entre recall e specificity (importante para desbalanceamento)
            g_mean = np.sqrt(recall * specificity)
            metrics[f'g_mean_class_{cls}'] = g_mean
            
            # Medida balanced - dá igual importância para classes majoritárias e minoritárias
            balanced_metric = (recall + specificity) / 2
            metrics[f'balanced_accuracy_class_{cls}'] = balanced_metric
        
        # Métricas globais para classificação
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Métricas baseadas em probabilidade (se disponíveis)
        if y_proba is not None:
            # Para problemas multiclasse, confirma formato correto de y_proba
            if len(np.array(y_proba).shape) == 2 and np.array(y_proba).shape[1] > 2:
                # Multiclasse: calcula AUPRC para cada classe minoritária
                for i, cls in enumerate(self.minority_classes):
                    # Transforma em problema one-vs-rest
                    y_true_bin = np.array(y_true) == cls
                    if isinstance(y_proba, list):
                        y_proba_cls = np.array(y_proba)[:, cls]
                    else:
                        y_proba_cls = y_proba[:, cls]
                    
                    # Calcula curva precision-recall
                    precision_curve, recall_curve, _ = precision_recall_curve(y_true_bin, y_proba_cls)
                    # AUC da curva PR (importante para desbalanceamento)
                    metrics[f'auprc_class_{cls}'] = auc(recall_curve, precision_curve)
            else:
                # Caso binário
                if isinstance(y_proba, list) and len(np.array(y_proba).shape) == 2:
                    # Extrair probabilidade da classe positiva
                    y_proba_pos = np.array(y_proba)[:, 1]
                else:
                    y_proba_pos = y_proba
                
                # Calcula curva precision-recall
                y_true_bin = np.array(y_true) == self.pos_label
                precision_curve, recall_curve, _ = precision_recall_curve(y_true_bin, y_proba_pos)
                # AUC da curva PR
                metrics['auprc'] = auc(recall_curve, precision_curve)
        
        # Adiciona MCC (Matthews Correlation Coefficient) - métrica robusta para desbalanceamento
        try:
            from sklearn.metrics import matthews_corrcoef
            metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        except:
            pass
            
        return metrics

def evaluate_transformation_imbalanced(self, df: pd.DataFrame, prefix: str = "") -> Dict[str, float]:
    """
    Método para a classe TransformationEvaluator que avalia uma transformação 
    usando métricas especializadas para problemas desbalanceados.
    
    Args:
        df: DataFrame transformado
        prefix: Prefixo para nomes das métricas
        
    Returns:
        Dicionário com métricas calculadas
    """
    # Obtém métricas padrão
    metrics = self.evaluate_transformation(df, prefix)
    
    # Se não houver target ou não for classificação, retorna métricas padrão
    if not self.target_col or self.target_col not in df.columns:
        return metrics
    
    # Verifica se é problema de classificação
    y = df[self.target_col]
    is_classification = (pd.api.types.is_categorical_dtype(y) or 
                        pd.api.types.is_object_dtype(y) or 
                        y.nunique() <= 10)
    
    if not is_classification:
        return metrics
    
    # Verifica desbalanceamento
    class_counts = y.value_counts()
    min_class_ratio = class_counts.min() / class_counts.max()
    
    # Só aplica métricas especializadas se houver desbalanceamento significativo
    if min_class_ratio >= 0.2:  # Não é fortemente desbalanceado
        return metrics
    
    # Detecta classes minoritárias
    minority_classes = class_counts[class_counts == class_counts.min()].index.tolist()
    
    # Configurar avaliador especializado
    if y.nunique() == 2:
        # Classificação binária, definir pos_label corretamente
        pos_label = minority_classes[0]
        imbalanced_evaluator = ImbalancedMetricsEvaluator(minority_classes, pos_label)
    else:
        # Classificação multiclasse
        imbalanced_evaluator = ImbalancedMetricsEvaluator(minority_classes)
    
    # Separar features e target
    X = df.drop(columns=[self.target_col])
    
    # Executar cross-validation com métricas especializadas
    try:
        # Preparar dados
        X_filled = X.copy()
        for col in X_filled.select_dtypes(include=['number']).columns:
            X_filled[col] = X_filled[col].fillna(X_filled[col].median())
        
        for col in X_filled.select_dtypes(include=['object', 'category']).columns:
            X_filled[col] = X_filled[col].fillna(X_filled[col].mode()[0] if len(X_filled[col].mode()) > 0 else 'missing')
        
        # Codifica variáveis categóricas
        X_encoded = pd.get_dummies(X_filled, drop_first=True)
        
        # Treina um modelo simplificado
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        
        # Usa StratifiedKFold para preservar a proporção de classes
        cv = StratifiedKFold(n_splits=min(5, class_counts.min()), shuffle=True, random_state=42)
        
        # Modelo com ajustes para desbalanceamento
        model = RandomForestClassifier(
            n_estimators=50, 
            class_weight='balanced', 
            random_state=42
        )
        
        # Predições e probabilidades através de cross-validation
        y_pred = cross_val_predict(model, X_encoded, y, cv=cv)
        try:
            y_proba = cross_val_predict(model, X_encoded, y, cv=cv, method='predict_proba')
            
            # Calcula métricas especializadas
            imb_metrics = imbalanced_evaluator.evaluate(y, y_pred, y_proba)
        except:
            # Se falhar ao obter probabilidades
            imb_metrics = imbalanced_evaluator.evaluate(y, y_pred)
        
        # Adiciona métricas com prefixo
        for key, value in imb_metrics.items():
            metrics[f'{prefix}imb_{key}'] = value
        
    except Exception as e:
        self.logger.warning(f"Erro ao calcular métricas para classes desbalanceadas: {e}")
    
    return metrics