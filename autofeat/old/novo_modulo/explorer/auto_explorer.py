import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

class AutoExplorer:
    def __init__(self, max_iterations=10, min_improvement=0.01, improvement_threshold=0.001, model=None, metric='r2'):
        """
        AutoExplorer: Seleção automática de features com iterações baseadas em melhoria de métricas.
        
        Args:
            max_iterations (int): Número máximo de iterações.
            min_improvement (float): Melhoria mínima esperada para continuar.
            improvement_threshold (float): Critério de parada baseado na melhoria relativa.
            model (sklearn model, opcional): Modelo de avaliação. Default é LinearRegression.
            metric (str): Métrica para otimização ('r2', 'mse', 'mae', 'rmse').
        """
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        self.improvement_threshold = improvement_threshold
        self.best_features = None
        self.metric = metric
        self.model = model if model else LinearRegression()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Executa o processo iterativo de seleção de features.
        
        Args:
            X (pd.DataFrame): Dados de entrada.
            y (pd.Series): Variável alvo.
        
        Returns:
            pd.DataFrame: Subconjunto de features selecionadas.
        """
        prev_score = -np.inf if self.metric == 'r2' else np.inf
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            print(f"Iteração {iteration}...")
            
            # Garantir que a variável alvo tenha um nome válido
            if not isinstance(y.name, str):
                y.name = "target"
            
            # Ajustar dinamicamente o número de features
            num_features = min(max(10, X.shape[1] // 2), 50)
            selector = SelectKBest(score_func=mutual_info_regression, k=num_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()

            # Treinar modelo com validação cruzada
            scores = cross_val_score(self.model, X_selected, y, cv=5, scoring=self.metric)
            score = np.mean(scores)
            print(f"Métrica ({self.metric}): {score:.4f}")

            # Critério de parada
            if self.metric == 'r2':
                improvement = (score - prev_score) / (abs(prev_score) + 1e-10)
                if improvement < self.improvement_threshold:
                    print("Melhoria insuficiente. Parando iteração.")
                    break
            else:
                improvement = (prev_score - score) / (prev_score + 1e-10)
                if improvement < self.improvement_threshold:
                    print("Melhoria insuficiente. Parando iteração.")
                    break

            prev_score = score
            self.best_features = selected_features

        print(f"Melhores features selecionadas: {self.best_features}")
        return X[self.best_features]
