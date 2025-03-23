Vou compartilhar o conteúdo do README.md deste projeto. Pelo que vejo nos arquivos, este é um sistema de análise de textos regulatórios que usa IA para detectar e prever mudanças em regulamentações.

```markdown
# Regulatory Text Analysis API: Intelligent Analysis of Regulatory Changes

The Regulatory Text Analysis API is a sophisticated system that analyzes and tracks changes in regulatory texts using advanced natural language processing and graph-based analytics. It provides automated analysis of regulatory documents, extracts key changes, and builds a knowledge graph to track the evolution of regulations over time.

The system combines multiple AI technologies including text embeddings, similarity analysis, and key phrase extraction to provide comprehensive insights into regulatory changes. It leverages Azure's AI services and graph database capabilities to deliver robust analysis capabilities including numerical change detection, entity tracking, and text difference analysis. The API is built with FastAPI for high performance and easy integration, making it ideal for compliance teams, legal analysts, and regulatory technology solutions.

## Repository Structure
```
.
├── requirements.txt          # Python dependencies including AI/ML libraries and Azure SDK components
└── src/                     # Source code directory
    ├── analyzer/            # Core analysis module directory
    │   └── regulatory_analysis.py    # Main regulatory analysis implementation
    └── app.py              # FastAPI application entry point and API definition
```

## Usage Instructions
### Prerequisites
- Python 3.7 or higher
- Azure subscription with the following services configured:
  - Azure OpenAI
  - Azure Language Service
  - Azure Cosmos DB with Gremlin API
  - Azure Blob Storage
- Environment variables set for Azure services:
  - AZURE_LANGUAGE_ENDPOINT
  - AZURE_LANGUAGE_KEY
  - AZURE_STORAGE_CONNECTION_STRING
  - COSMOS_GREMLIN_ENDPOINT
  - COSMOS_GREMLIN_KEY
  - COSMOS_GREMLIN_DATABASE
  - COSMOS_GREMLIN_COLLECTION
  - AZURE_OPENAI_ENDPOINT
  - AZURE_OPENAI_KEY
  - AZURE_OPENAI_DEPLOYMENT

### Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
# On Linux/MacOS
python -m venv regulatory-analysis-env
source regulatory-analysis-env/bin/activate

# On Windows
python -m venv regulatory-analysis-env
regulatory-analysis-env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the Portuguese language model for spaCy:
```bash
python -m spacy download pt_core_news_lg
```

### Quick Start
1. Start the API server:
```bash
cd src
python app.py
```

2. The API will be available at `http://localhost:8000`

3. Test the API health:
```bash
curl http://localhost:8000/
```

### More Detailed Examples
The RegulatoryChangeAnalyzer class provides several key methods:

1. Analyzing text changes:
```python
from analyzer.regulatory_analysis import RegulatoryChangeAnalyzer

analyzer = RegulatoryChangeAnalyzer()
changes = analyzer.extract_key_changes(old_text, new_text)
```

2. Adding regulations to the knowledge graph:
```python
analyzer.add_to_knowledge_graph(
    regulation_id="REG001",
    date="2024-02-20",
    text="Regulation text content",
    previous_regulation_id="REG000"
)
```

### Troubleshooting
Common issues and solutions:

1. Azure Service Connection Issues
- Problem: Unable to connect to Azure services
- Solution: Verify environment variables are set correctly:
```bash
echo $AZURE_LANGUAGE_ENDPOINT
echo $AZURE_OPENAI_ENDPOINT
```

2. Spacy Model Loading Error
- Problem: "Can't find model 'pt_core_news_lg'"
- Solution: Install the model manually:
```bash
python -m spacy download pt_core_news_lg
```

3. Memory Issues with Large Documents
- Problem: Out of memory when processing large texts
- Solution: Enable text chunking:
  - Break down large documents into smaller chunks (max 512 tokens)
  - Process chunks sequentially

## Data Flow
The system processes regulatory texts through multiple stages of analysis, from text preprocessing to knowledge graph integration. The core workflow transforms raw regulatory text into structured insights about changes and relationships between regulations.

```ascii
[Raw Text] -> [Preprocessing] -> [Embedding Generation] -> [Change Analysis] -> [Knowledge Graph]
     |              |                    |                        |                  |
     v              v                    v                        v                  v
  Input Text -> Clean Text -> Vector Representation -> Change Detection -> Graph Storage
```

Key component interactions:
1. Text preprocessing removes noise and normalizes the input
2. Azure OpenAI generates text embeddings for similarity analysis
3. Azure Language Service extracts key phrases and entities
4. Change detection compares versions using multiple algorithms
5. Results are stored in both Cosmos DB (graph) and Blob Storage (embeddings)
6. FastAPI endpoints provide access to analysis results
7. The knowledge graph maintains relationships between regulations
```

Este README fornece uma visão geral completa do sistema, incluindo sua estrutura, como configurá-lo e usá-lo, além de solucionar problemas comuns. Ele destaca as principais funcionalidades e o fluxo de dados do sistema de análise regulatória.