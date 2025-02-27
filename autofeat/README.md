# AutoFE: Sistema de Automação de Engenharia de Features

AutoFE é um sistema modular para automação do processo de engenharia de features em projetos de machine learning. Projetado para usuários sem conhecimento prévio em ciência de dados, o sistema facilita a transformação e melhoria de conjuntos de dados para obter melhores resultados em modelos preditivos.

## Características Principais

- 🔄 **Automação completa** da engenharia de features
- 🌲 **Exploração estruturada** através de árvore de transformações
- 🧠 **Meta-aprendizado** para recomendar transformações eficazes
- 📊 **Suporte para múltiplos tipos de dados**:
  - Classificação com dados tabulares
  - Regressão com dados tabulares
  - Problemas com variáveis de texto
  - Séries temporais com horizonte personalizado
- 🔍 **Avaliação inteligente** de transformações e features
- 🚀 **Otimização** para maximizar a qualidade dos modelos
- 📝 **Relatório detalhado** com todas as alterações feitas no dataset e pontuação de qualidade

## Estrutura do Sistema

O sistema é composto por quatro módulos principais:

1. **PreProcessor**: Limpeza e preparação inicial dos dados
2. **Explorer**: Exploração do espaço de possíveis transformações
3. **Predictor**: Meta-aprendizado para recomendar transformações eficazes
4. **PosProcessor**: Seleção final de features e geração de relatórios

## Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/auto-fe.git
cd auto-fe

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows, use: venv\Scripts\activate

# Instale as dependências
pip install -r requirements.