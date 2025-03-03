# Glossário AutoFE - Termos e Conceitos

Este glossário explica os principais termos e conceitos utilizados no AutoFE, tornando mais fácil para usuários sem conhecimento técnico compreenderem o sistema.

## A

### Algoritmo
Conjunto de instruções ou regras definidas para resolver um problema específico ou executar uma tarefa. No contexto do AutoFE, vários algoritmos são utilizados para transformar e melhorar os dados.

### Automação
Processo de realizar tarefas automaticamente com mínima intervenção humana. O AutoFE automatiza o processo de engenharia de features, que normalmente exigiria conhecimento especializado.

## C

### Classificação
Tipo de problema de machine learning onde o objetivo é prever uma categoria ou classe (como "sim/não", "spam/não spam"). O AutoFE suporta automação para problemas de classificação.

### Codificação
Processo de transformar dados categóricos (como cores, nomes de cidades) em formatos numéricos que os algoritmos de machine learning possam entender. O AutoFE oferece diferentes estratégias de codificação, como "onehot" e "ordinal".

### Correlação
Medida de relação entre duas variáveis. Uma alta correlação significa que quando uma variável muda, a outra tende a mudar de maneira semelhante. O AutoFE identifica e trata features altamente correlacionadas para evitar redundância.

## D

### DataFrame
Estrutura de dados bidimensional (similar a uma tabela) usada pela biblioteca pandas do Python. O AutoFE trabalha com DataFrames para processar e transformar dados.

### Dados Categóricos
Dados que representam categorias ou grupos, como cores, tipos de produto, cidades. Estes dados precisam ser codificados para uso em modelos de machine learning.

### Dados Numéricos
Dados que são representados por números e podem ser divididos em contínuos (podem assumir qualquer valor dentro de um intervalo) e discretos (números inteiros ou contáveis).

## E

### Engenharia de Features
Processo de selecionar, transformar e criar novas variáveis (features) para melhorar o desempenho de modelos de machine learning. Este é o principal foco do AutoFE.

### Explorer
Módulo do AutoFE responsável por testar diferentes transformações e identificar as mais eficazes para cada conjunto de dados específico.

## F

### Feature
Característica ou atributo no conjunto de dados que é usado para fazer previsões. Por exemplo, em um modelo que prevê preços de casas, features podem incluir número de quartos, área, localização, etc.

### Feature Selection (Seleção de Features)
Processo de selecionar um subconjunto das features mais relevantes para o modelo, eliminando aquelas que não contribuem significativamente para o resultado.

## H

### Heurística
Método prático que ajuda a encontrar soluções satisfatórias, embora não necessariamente ótimas, para problemas complexos. O AutoFE usa heurísticas para avaliar a qualidade das transformações.

## M

### Machine Learning
Subcampo da inteligência artificial que permite que sistemas aprendam e melhorem a partir de dados sem serem explicitamente programados para cada tarefa.

### Meta-aprendizado
Processo em que um sistema aprende a partir de experiências anteriores para melhorar seu desempenho futuro. No AutoFE, o meta-aprendizado é usado para recomendar transformações eficazes.

## N

### Normalização
Processo de ajustar valores medidos em diferentes escalas para uma escala comum. O AutoFE oferece diferentes métodos de normalização como 'standard', 'minmax' e 'robust'.

## O

### Outlier
Valor que se desvia significativamente dos outros valores observados. Outliers podem distorcer análises e modelos. O AutoFE oferece diferentes métodos para detectar e tratar outliers.

## P

### PCA (Análise de Componentes Principais)
Técnica para reduzir a dimensionalidade dos dados, preservando a maior parte da informação original. O AutoFE pode aplicar PCA como parte da engenharia de features.

### Pipeline
Sequência de etapas de processamento de dados que são executadas em ordem. O AutoFE cria pipelines para automatizar todo o fluxo de transformação de dados.

### PreProcessor
Módulo principal do AutoFE responsável pela limpeza e preparação inicial dos dados, incluindo tratamento de valores ausentes, normalização e codificação.

## R

### Regressão
Tipo de problema de machine learning onde o objetivo é prever um valor numérico contínuo (como preço, temperatura). O AutoFE suporta automação para problemas de regressão.

## S

### Scaler (Escalador)
Componente que transforma variáveis numéricas para uma escala comum. Diferentes tipos de scalers incluem StandardScaler, MinMaxScaler e RobustScaler.

### Série Temporal
Dados coletados ou registrados em pontos consecutivos no tempo. O AutoFE suporta problemas de séries temporais com horizonte personalizado.

## T

### Target (Alvo)
Variável que o modelo de machine learning está tentando prever. Também conhecida como variável dependente ou resposta.

### TransformationTree
Estrutura de dados utilizada pelo AutoFE para manter um registro organizado das diferentes transformações testadas e seus resultados.

## V

### Valor Ausente
Dado faltante em um conjunto de dados. O AutoFE implementa diferentes estratégias para lidar com valores ausentes, como substituí-los pela média, mediana ou valor mais frequente.

### Variável Categórica
Tipo de variável que pode assumir um de um conjunto limitado de valores (categorias). Exemplos incluem cor dos olhos, marca de carro, tipo sanguíneo.

### Variável Numérica
Tipo de variável que representa quantidades e pode ser medida. Exemplos incluem altura, peso, temperatura, preço.
