# Tutorial Passo a Passo: Como Usar o AutoFE

Este tutorial prático irá guiá-lo através do processo de utilização do AutoFE para automatizar a engenharia de features em seus projetos de machine learning, mesmo se você não tiver experiência prévia em ciência de dados.

## Passo 1: Preparação do Ambiente

Primeiro, vamos configurar seu ambiente:

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/auto-fe.git
cd auto-fe

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows, use: venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

## Passo 2: Importando o AutoFE e Seus Dados

Vamos começar importando as bibliotecas necessárias e carregando seus dados:

```python
import pandas as pd
import numpy as np
from autofe.preprocessor import PreProcessor, Explorer, create_preprocessor

# Carregue seus dados (substitua 'seus_dados.csv' pelo caminho do seu arquivo)
dados = pd.read_csv('seus_dados.csv')

# Vamos dar uma olhada nos primeiros registros
print(dados.head())

# Verificar informações básicas sobre os dados
print(dados.info())
print(dados.describe())
```

## Passo 3: Utilizando o PreProcessor com Configurações Padrão

O modo mais simples de começar é utilizar o PreProcessor com suas configurações padrão:

```python
# Defina qual é sua coluna alvo (a variável que você quer prever)
coluna_alvo = 'sua_coluna_alvo'  # Substitua pelo nome da sua coluna alvo

# Crie um preprocessador com configurações padrão
preprocessador = PreProcessor()

# Ajuste o preprocessador aos seus dados
preprocessador.fit(dados, target_col=coluna_alvo)

# Transforme seus dados
dados_transformados = preprocessador.transform(dados, target_col=coluna_alvo)

# Veja como ficaram os dados transformados
print(dados_transformados.head())
print(f"Dimensão original: {dados.shape}, Dimensão após transformação: {dados_transformados.shape}")
```

## Passo 4: Personalizando as Configurações do PreProcessor

Agora vamos personalizar o PreProcessor de acordo com suas necessidades:

```python
# Defina suas configurações personalizadas
configuracao = {
    'missing_values_strategy': 'median',  # Estratégia para valores ausentes (mean, median, most_frequent, knn)
    'outlier_method': 'iqr',              # Método para detecção de outliers (zscore, iqr, isolation_forest)
    'categorical_strategy': 'onehot',     # Codificação de variáveis categóricas (onehot, ordinal)
    'scaling': 'standard',                # Normalização de variáveis numéricas (standard, minmax, robust)
    'generate_features': True,            # Gerar automaticamente novas features
    'verbosity': 1                        # Nível de detalhamento dos logs (0, 1, 2)
}

# Crie um novo preprocessador com suas configurações
preprocessador_personalizado = PreProcessor(configuracao)

# Ajuste e transforme seus dados
preprocessador_personalizado.fit(dados, target_col=coluna_alvo)
dados_transformados_personalizados = preprocessador_personalizado.transform(dados, target_col=coluna_alvo)

# Compare os resultados
print(f"Número de features após transformação padrão: {dados_transformados.shape[1]}")
print(f"Número de features após transformação personalizada: {dados_transformados_personalizados.shape[1]}")
```

## Passo 5: Exploração Automática de Transformações

O Explorer pode testar diversas configurações automaticamente e encontrar a melhor para seus dados:

```python
# Crie uma instância do Explorer
explorador = Explorer(target_col=coluna_alvo)

# Analise diferentes transformações e encontre a melhor
# Atenção: Este processo pode demorar um pouco dependendo do tamanho dos seus dados
dados_otimizados = explorador.analyze_transformations(dados)

# Veja o resultado da exploração automática
print(f"Dimensão após exploração automática: {dados_otimizados.shape}")
```

## Passo 6: Salvando e Carregando seu Preprocessador

Para usar seu preprocessador em dados futuros ou em produção, você pode salvá-lo:

```python
# Salve o preprocessador personalizado
preprocessador_personalizado.save('meu_preprocessador.joblib')

# Para carregar o preprocessador no futuro (em outro script ou sessão)
preprocessador_carregado = PreProcessor.load('meu_preprocessador.joblib')

# Você pode usar o preprocessador carregado para transformar novos dados
novos_dados = pd.read_csv('novos_dados.csv')
novos_dados_transformados = preprocessador_carregado.transform(novos_dados, target_col=coluna_alvo)
```

## Passo 7: Preparando Dados para Machine Learning

Agora que você transformou seus dados, pode usá-los para treinar um modelo:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # ou outro modelo de sua escolha
from sklearn.metrics import accuracy_score  # ou outra métrica adequada ao seu problema

# Separe as features (X) e o alvo (y)
X = dados_transformados.drop(columns=[coluna_alvo])
y = dados_transformados[coluna_alvo]

# Divida em conjuntos de treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

# Treine um modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_treino, y_treino)

# Avalie o modelo
previsoes = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, previsoes)
print(f"Acurácia do modelo: {acuracia:.4f}")
```

## Passo 8: Comparando Resultados Com e Sem Engenharia de Features

Para avaliar o impacto da engenharia de features:

```python
# Prepare os dados originais (somente com tratamento básico)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Identifique colunas numéricas e categóricas
colunas_numericas = dados.select_dtypes(include=['number']).columns.tolist()
colunas_categoricas = dados.select_dtypes(include=['object', 'category']).columns.tolist()

# Retire a coluna alvo das listas
if coluna_alvo in colunas_numericas:
    colunas_numericas.remove(coluna_alvo)
if coluna_alvo in colunas_categoricas:
    colunas_categoricas.remove(coluna_alvo)

# Crie um preprocessador básico
preprocessador_basico = ColumnTransformer([
    ('num', StandardScaler(), colunas_numericas),
    ('cat', OneHotEncoder(handle_unknown='ignore'), colunas_categoricas)
])

# Prepare X e y
X_original = dados.drop(columns=[coluna_alvo])
y_original = dados[coluna_alvo]

# Transforme os dados
X_original_transformado = preprocessador_basico.fit_transform(X_original)

# Divida em treino e teste
X_treino_orig, X_teste_orig, y_treino_orig, y_teste_orig = train_test_split(
    X_original_transformado, y_original, test_size=0.3, random_state=42
)

# Treine um modelo com os dados originais
modelo_original = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_original.fit(X_treino_orig, y_treino_orig)

# Avalie o modelo
previsoes_orig = modelo_original.predict(X_teste_orig)
acuracia_orig = accuracy_score(y_teste_orig, previsoes_orig)
print(f"Acurácia sem engenharia de features avançada: {acuracia_orig:.4f}")
print(f"Acurácia com engenharia de features via AutoFE: {acuracia:.4f}")
print(f"Melhoria: {(acuracia - acuracia_orig) * 100:.2f}%")
```

## Exemplo Completo para um Dataset de Classificação

Vamos ver um exemplo completo usando o famoso dataset Iris:

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from autofe.preprocessor import PreProcessor, Explorer

# Carregar o dataset Iris
iris = load_iris()
dados = pd.DataFrame(data=iris.data, columns=iris.feature_names)
dados['target'] = iris.target

# Definir a coluna alvo
coluna_alvo = 'target'

# 1. Usar PreProcessor com configurações padrão
preprocessador = PreProcessor()
preprocessador.fit(dados, target_col=coluna_alvo)
dados_transformados = preprocessador.transform(dados, target_col=coluna_alvo)

# 2. Treinar um modelo com os dados transformados
X = dados_transformados.drop(columns=[coluna_alvo])
y = dados_transformados[coluna_alvo]

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, previsoes)
print(f"Acurácia com PreProcessor padrão: {acuracia:.4f}")

# 3. Usar Explorer para encontrar a melhor transformação
explorador = Explorer(target_col=coluna_alvo)
dados_otimizados = explorador.analyze_transformations(dados)

# 4. Treinar um modelo com os dados otimizados pelo Explorer
X_otim = dados_otimizados.drop(columns=[coluna_alvo])
y_otim = dados_otimizados[coluna_alvo]

X_treino_otim, X_teste_otim, y_treino_otim, y_teste_otim = train_test_split(
    X_otim, y_otim, test_size=0.3, random_state=42
)

modelo_otim = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_otim.fit(X_treino_otim, y_treino_otim)

previsoes_otim = modelo_otim.predict(X_teste_otim)
acuracia_otim = accuracy_score(y_teste_otim, previsoes_otim)
print(f"Acurácia com Explorer: {acuracia_otim:.4f}")
```

## Dicas Para Usuários Iniciantes

1. **Comece com Dados Limpos**: Mesmo com automação, é bom verificar seus dados antes (valores faltantes, erros de digitação, etc.)

2. **Entenda Suas Colunas**: Identifique quais colunas são numéricas, categóricas e qual é sua coluna alvo

3. **Experimente Gradualmente**: Comece com as configurações padrão e vá ajustando uma configuração por vez

4. **Monitore o Tamanho dos Dados**: Após transformações, verifique se o número de linhas e colunas faz sentido

5. **Guarde Suas Métricas**: Compare o desempenho de modelos antes e depois de usar o AutoFE

## Conclusão

Parabéns! Você agora sabe como utilizar o AutoFE para automatizar a engenharia de features em seus projetos de machine learning. Esta ferramenta pode ajudar significativamente a melhorar o desempenho de seus modelos, mesmo se você não tiver experiência prévia em ciência de dados.

Experimente diferentes configurações e veja como elas afetam o desempenho final do seu modelo. Com o tempo, você desenvolverá intuição sobre quais configurações funcionam melhor para diferentes tipos de dados e problemas.
