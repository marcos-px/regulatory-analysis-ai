# Análise de Textos Regulatórios com Graph Neural Networks (GNN)

A API de Análise de Textos Regulatórios é um sistema sofisticado que analisa e rastreia mudanças em documentos regulatórios utilizando processamento de linguagem natural avançado e análise baseada em grafos, com ênfase especial em Graph Neural Networks (GNN).

O sistema combina múltiplas tecnologias de inteligência artificial, incluindo embeddings de texto, análise de similaridade e extração de frases-chave para fornecer insights abrangentes sobre mudanças regulatórias. Utiliza serviços de IA do Azure e capacidades de processamento de grafos para entregar análises robustas, incluindo detecção de mudanças numéricas, rastreamento de entidades e análise de diferenças textuais.

## Estrutura do Repositório
```
.
├── requirements.txt          # Dependências Python, incluindo bibliotecas de IA/ML e componentes Azure SDK
└── src/                     # Diretório de código-fonte
    ├── analyzer/            # Módulo principal de análise
    │   └── regulatory_analysis.py    # Implementação principal da análise regulatória
    └── app.py              # Ponto de entrada da aplicação FastAPI e definição da API
```

## Instruções de Uso
### Pré-requisitos
- Python 3.7 ou superior
- Assinatura Azure com os seguintes serviços configurados:
  - Azure OpenAI
  - Azure Language Service
  - Azure Cosmos DB
  - Azure Blob Storage
- Variáveis de ambiente configuradas para serviços Azure:
  - AZURE_LANGUAGE_ENDPOINT
  - AZURE_LANGUAGE_KEY
  - AZURE_STORAGE_CONNECTION_STRING
  - AZURE_OPENAI_ENDPOINT
  - AZURE_OPENAI_KEY
  - AZURE_OPENAI_DEPLOYMENT

### Instalação
1. Clone o repositório:
```bash
git clone <url-do-repositorio>
cd <nome-do-repositorio>
```

2. Crie e ative um ambiente virtual:
```bash
# Linux/MacOS
python -m venv regulatory-analysis-env
source regulatory-analysis-env/bin/activate

# Windows
python -m venv regulatory-analysis-env
regulatory-analysis-env\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Baixe o modelo de linguagem português para spaCy:
```bash
python -m spacy download pt_core_news_lg
```

### Início Rápido
1. Inicie o servidor da API:
```bash
cd src
python app.py
```

2. A API estará disponível em `http://localhost:8000`

3. Teste a saúde da API:
```bash
curl http://localhost:8000/
```

### Exemplos Detalhados
A classe RegulatoryChangeAnalyzer fornece métodos-chave:

1. Analisando mudanças de texto:
```python
from analyzer.regulatory_analysis import RegulatoryChangeAnalyzer

analyzer = RegulatoryChangeAnalyzer()
changes = analyzer.extract_key_changes(old_text, new_text)
```

2. Adicionando regulações ao grafo de conhecimento com GNN:
```python
analyzer.add_to_knowledge_graph_gnn(
    regulation_id="REG001",
    date="2024-02-20",
    text="Conteúdo do texto regulatório",
    previous_regulation_id="REG000"
)
```

### Solução de Problemas
Problemas comuns e soluções:

1. Problemas de Conexão com Serviços Azure
- Problema: Incapaz de conectar aos serviços Azure
- Solução: Verifique se as variáveis de ambiente estão configuradas corretamente

2. Erro de Carregamento do Modelo Spacy
- Problema: "Modelo 'pt_core_news_lg' não encontrado"
- Solução: Instale o modelo manualmente
```bash
python -m spacy download pt_core_news_lg
```

3. Problemas de Memória com Documentos Grandes
- Problema: Memória insuficiente ao processar textos longos
- Solução: Habilite divisão de texto
  - Divida documentos grandes em fragmentos menores (máximo 512 tokens)
  - Processe fragmentos sequencialmente

## Fluxo de Dados com Graph Neural Networks (GNN)
O sistema processa textos regulatórios através de múltiplos estágios de análise, com ênfase especial nas Graph Neural Networks para capturar relações complexas entre regulações.

```ascii
[Texto Bruto] -> [Pré-processamento] -> [Geração de Embeddings] -> [Análise com GNN] -> [Grafo de Conhecimento]
     |              |                    |                        |                  |
     v              v                    v                        v                  v
  Texto de Entrada -> Texto Limpo -> Representação Vetorial -> Detecção de Mudanças -> Armazenamento em Grafo
```

Interações de componentes-chave:
1. Pré-processamento remove ruídos e normaliza a entrada
2. Azure OpenAI gera embeddings de texto para análise de similaridade
3. Serviço de Linguagem Azure extrai frases-chave e entidades
4. Detecção de mudanças compara versões usando múltiplos algoritmos
5. Graph Neural Networks analisam padrões e relacionamentos entre regulações
6. Resultados são armazenados no Cosmos DB e Blob Storage
7. Endpoints FastAPI fornecem acesso aos resultados de análise

## Tecnologias GNN Utilizadas
- Graph Convolutional Network (GCN)
- GraphSAGE
- Graph Attention Network (GAT)

Cada variante de GNN permite capturar diferentes aspectos das relações entre regulações, proporcionando insights mais profundos sobre a evolução dos textos regulatórios.

---

By Marcos Teixeira