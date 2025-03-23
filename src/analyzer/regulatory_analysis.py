import os
import re
import spacy
import json
import pandas as pd
import numpy as np
import networkx as nx
import torch 
import ast
import requests
import aiohttp
import io

from transformers import AutoTokenizer, AutoModel
from datetime import datetime
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from dotenv import load_dotenv

from azure.ai.textanalytics import TextAnalyticsClient
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.ai.inference import EmbeddingsClient
from azure.cosmos import CosmosClient, PartitionKey, exceptions

from gremlin_python.driver import client
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.strategies import *
from gremlin_python.structure.graph import Graph
from collections import defaultdict
from .ai_provider import AIProvider
from .gnn_processor import GNNProcessor

load_dotenv()

class RegulatoryChangeAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("pt_core_news_lg")
        self.tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-large-portuguese-cased")
        self.model = AutoModel.from_pretrained("neuralmind/bert-large-portuguese-cased")

        self.azure_language_endpoint = os.environ.get("AZURE_LANGUAGE_ENDPOINT")
        self.azure_language_key = os.environ.get("AZURE_LANGUAGE_KEY")
        
        self.azure_storage_connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

        self.gremlin_endpoint = os.environ.get("COSMOS_GREMLIN_ENDPOINT")
        self.gremlin_key = os.environ.get("COSMOS_GREMLIN_KEY")
        self.gremlin_database = os.environ.get("COSMOS_GREMLIN_DATABASE")
        self.gremlin_collection = os.environ.get("COSMOS_GREMLIN_COLLECTION")

        self.cosmos_endpoint = os.environ.get("COSMOS_SQL_ENDPOINT")
        self.cosmos_key = os.environ.get("COSMOS_SQL_KEY")
        self.cosmos_database = os.environ.get("COSMOS_SQL_DATABASE")

        self.azure_ai_endpoint = os.environ.get("AZURE_AI_ENDPOINT")
        self.azure_ai_key = os.environ.get("AZURE_AI_API_KEY")
        self.azure_ai_completion_model = os.environ.get("AZURE_AI_COMPLETION_MODEL")

        self.cosmos_client = None
        self.regulations_container = None
        self.relationships_container = None

        try:
            if self.cosmos_key:
                self.cosmos_client = CosmosClient(self.cosmos_endpoint, credential=self.cosmos_key)
                database = self.cosmos_client.get_database_client(self.cosmos_database)

                try:
                    self.regulations_container = database.get_container_client("regulations")
                    print("Conectado ao container 'regulations'")
                except exceptions.CosmosResourceNotFoundError:
                    self.regulations_container = database.create_container(
                        id='regulations',
                        partition_key=PartitionKey(path="/id")
                    )
                    print("Criando container 'regulations'")
                
                try:
                    self.relationships_container = database.get_container_client("relationships")
                    print("Conectado ao container 'relationships'")
                except exceptions.CosmosResourceNotFoundError:
                    self.relationships_container = database.create_container(
                        id='relationships',
                        partition_key=PartitionKey(path="/id")
                    )
                    print("Criando container 'relationships'")
        except Exception as e:
            print(f"Erro ao conectar ao Cosmos DB: {str(e)}")

        self.azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_key = os.environ.get("AZURE_OPENAI_API_KEY")
        self.azure_openai_embedding_model = os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.azure_openai_completion_model = os.environ.get("AZURE_OPENAI_COMPLETION_MODEL", "gpt-4o")

        self.azure_openai_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", self.azure_openai_embedding_model)

        if not self.azure_openai_key:
            print("AVISO: A chave da API Azure OpenAI para embeddings não está definida")
            self.azure_openai_key = "B6BfKQbW8eevwoLG9VIBQLgTK5wQ6C3v3mnmSBvzQGfi1cS2tkw5JQQJ99BCACHYHv6XJ3w3AAAAACOGn8rp"
            print(f"Usando chave de teste para Azure OpenAI: {self.azure_openai_key[:10]}...")

        if not self.azure_openai_endpoint:
            print("AVISO: O endpoint Azure OpenAI para embeddings não está definido")
            self.azure_openai_endpoint = "https://youtu-m8lv96p4-eastus2.cognitiveservices.azure.com"
            print(f"Usando endpoint de teste para Azure OpenAI: {self.azure_openai_endpoint}")

        if not self.azure_ai_key:
            print("AVISO: A chave da API Azure AI para completions não está definida")
            print("Por favor, defina a variável de ambiente AZURE_AI_API_KEY no arquivo .env")

        self.gnn_processor = GNNProcessor(
            embedding_dim=1536,
            hidden_dim=256,
            output_dim=128
        )

        self.gnn_embeddings = {}

        try:
            self.embeddings_client = EmbeddingsClient(
                endpoint=self.azure_openai_endpoint,
                credential=AzureKeyCredential(self.azure_openai_key),
                deployment_name=self.azure_openai_embedding_model
            )
        except Exception as e:
            print(f"Erro ao inicializar o cliente de embeddings: {str(e)}")

        try:
            self.text_analytics_client = TextAnalyticsClient(
                endpoint=self.azure_language_endpoint,
                credential=AzureKeyCredential(self.azure_language_key)
            )
        except Exception as e:
            print(f"Erro ao inicializar o cliente de análise de texto: {str(e)}")

        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(self.azure_storage_connection_string)
        except Exception as e:
            print(f"Erro ao inicializar o cliente de Blob Storage: {str(e)}")

        self.knowledge_graph = nx.DiGraph()
        self.gremlin_conn = None
        self.g = None
    
    async def init_gremlin_connection(self):
        """Método de inicialização desativado - usando apenas armazenamento local"""
        print("Conexão Gremlin desativada - usando apenas grafo local e Blob Storage")
        return False

    def preprocess_text(self,text):
        doc = self.nlp(text)
        processed_text = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
        return processed_text

    def get_text_embeddings(self, text):
        """Obter embeddings de texto usando a Azure OpenAI API"""
        try:
            endpoint = f"{self.azure_openai_endpoint}/openai/deployments/{self.azure_openai_embedding_model}/embeddings?api-version=2023-05-15"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": self.azure_openai_key
            }
            data = {
                "input": text
            }
            
            print(f"Chamando API embeddings...")
            print(f"Endpoint: {endpoint}")
            
            response = requests.post(endpoint, headers=headers, json=data)
            
            print(f"Status da resposta embeddings: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                return np.array(result['data'][0]['embedding'])
            else:
                error_message = f"Erro na API OpenAI: {response.status_code} - {response.text}"
                print(error_message)
                return self._get_local_embeddings(text)
        except Exception as e:
            print(f"Erro ao obter embeddings: {str(e)}")
            return self._get_local_embeddings(text)

    def _get_local_embeddings(self, text):
        """Gerar embeddings localmente usando o modelo carregado"""
        try:
            encoded_input = self.tokenizer(text, padding=True, truncation=True, 
                                        max_length=512, return_tensors='pt')
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
            embeddings = model_output.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            embeddings_norm = embeddings / np.linalg.norm(embeddings)
            
            return embeddings_norm
        except Exception as e:
            print(f"Erro ao gerar embeddings locais: {str(e)}")
            random_embedding = np.random.rand(1024)
            random_embedding = random_embedding / np.linalg.norm(random_embedding)
            return random_embedding

    async def predict_future_changes_with_gpt(self, num_predictions=1):
        """Prever mudanças futuras usando GNN para análise estrutural e GPT para geração"""
        try:
            if not self.gnn_embeddings:
                await self.enrich_embeddings_with_gnn()
            
            regulations = await self.get_all_regulations()
            sorted_regulations = sorted(regulations, key=lambda x: x.get('date', ''))
            
            if len(sorted_regulations) < 2:
                print("Não há regulações suficientes para análise")
                return []
            
            latest_regulation = sorted_regulations[-1]
            
            structural_insights = self._extract_structural_insights_from_gnn()
            
            regulations_history = []
            for i, reg in enumerate(sorted_regulations):
                regulations_history.append(f"Regulação {i+1} ({reg['date']}): {reg['text']}")
            
            changes_history = []
            for i in range(1, len(sorted_regulations)):
                prev_reg = sorted_regulations[i-1]
                curr_reg = sorted_regulations[i]
                changes = self.extract_key_changes(prev_reg['text'], curr_reg['text'])
                
                changes_summary = []
                if 'numerical_changes' in changes and changes['numerical_changes']:
                    for pair in changes['numerical_changes']:
                        if isinstance(pair, (list, tuple)) and len(pair) == 2:
                            changes_summary.append(f"Alteração numérica: {pair[0]} → {pair[1]}")
                
                if 'text_diff_blocks' in changes:
                    for block in changes['text_diff_blocks']:
                        if block.get('opcode') == 'replace':
                            changes_summary.append(f"Substituição: '{block.get('old_text', '')}' → '{block.get('new_text', '')}'")
                        elif block.get('opcode') == 'insert':
                            changes_summary.append(f"Adição: '{block.get('new_text', '')}'")
                        elif block.get('opcode') == 'delete':
                            changes_summary.append(f"Remoção: '{block.get('old_text', '')}'")
                
                changes_history.append({
                    "from": prev_reg['date'],
                    "to": curr_reg['date'],
                    "changes": changes_summary
                })
            
            prompt = self._build_prediction_prompt_for_gpt_with_gnn(
                regulations_history=regulations_history,
                changes_history=changes_history,
                latest_regulation=latest_regulation,
                structural_insights=structural_insights
            )
            
            predictions = await self._get_gpt_predictions(prompt)
            return predictions
                
        except Exception as e:
            print(f"Erro ao gerar previsões com GNN e GPT: {str(e)}")
            return await self.predict_future_changes(num_predictions)

    def _extract_structural_insights_from_gnn(self):

        insights = []
        
        if not self.gnn_embeddings:
            return insights
        
        node_ids = list(self.gnn_embeddings.keys())
        embeddings = np.array([self.gnn_embeddings[node_id] for node_id in node_ids])
        
        from sklearn.cluster import KMeans
        n_clusters = min(3, len(node_ids))
        if len(node_ids) >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            cluster_nodes = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_nodes:
                    cluster_nodes[cluster_id] = []
                cluster_nodes[cluster_id].append(node_ids[i])
            
            for cluster_id, nodes in cluster_nodes.items():
                cluster_texts = []
                dates = []
                for node in nodes:
                    node_data = self.knowledge_graph.nodes[node]
                    if 'text' in node_data:
                        cluster_texts.append(node_data['text'])
                    if 'date' in node_data:
                        dates.append(node_data['date'])
                
                common_phrases = set()
                for node in nodes:
                    node_data = self.knowledge_graph.nodes[node]
                    if 'key_phrases' in node_data:
                        if not common_phrases:
                            common_phrases = set(node_data['key_phrases'])
                        else:
                            common_phrases &= set(node_data['key_phrases'])
                
                insight = {
                    "cluster_id": int(cluster_id),
                    "node_count": len(nodes),
                    "dates": dates,
                    "common_phrases": list(common_phrases)[:5]
                }
                insights.append(insight)
        
        if len(node_ids) >= 2:
            nodes_with_dates = []
            for node in node_ids:
                node_data = self.knowledge_graph.nodes[node]
                if 'date' in node_data:
                    nodes_with_dates.append((node, node_data['date']))
            
            sorted_nodes = sorted(nodes_with_dates, key=lambda x: x[1])
            if len(sorted_nodes) >= 2:
                first_node, _ = sorted_nodes[0]
                last_node, _ = sorted_nodes[-1]
                
                first_emb = self.gnn_embeddings[first_node]
                last_emb = self.gnn_embeddings[last_node]
                
                direction = last_emb - first_emb
                
                projections = {}
                origin = first_emb
                direction_norm = np.linalg.norm(direction)
                
                if direction_norm > 0:
                    unit_direction = direction / direction_norm
                    
                    for node in node_ids:
                        emb = self.gnn_embeddings[node]
                        node_vector = emb - origin
                        projection = np.dot(node_vector, unit_direction)
                        projections[node] = projection
                    
                    sorted_by_projection = sorted(projections.items(), key=lambda x: x[1])
                    
                    direction_insight = {
                        "type": "evolution_direction",
                        "description": "Evolução direcional das regulações",
                        "progression": [node for node, _ in sorted_by_projection]
                    }
                    
                    insights.append(direction_insight)
        
        return insights

    def _save_embedding_to_blob(self, text, regulation_id):
        """Método auxiliar para salvar embedding no Blob Storage"""
        try:
            container_name = "embeddings"
            try:
                container_client = self.blob_service_client.get_container_client(container_name)
                if not container_client.exists():
                    container_client = self.blob_service_client.create_container(container_name)
            except Exception as e:
                print(f"Aviso: Erro ao verificar/criar container de embeddings: {str(e)}")
                container_client = self.blob_service_client.create_container(container_name)
            
            embedding = self.get_text_embeddings(text)
            
            blob_name = f"{regulation_id}_embedding.npy"
            blob_client = container_client.get_blob_client(blob_name)
            
            embedding_bytes = io.BytesIO()
            np.save(embedding_bytes, embedding)
            embedding_bytes.seek(0)
            
            blob_client.upload_blob(embedding_bytes, overwrite=True)
            print(f"Embedding saved for {regulation_id}")

        except Exception as e:
            print(f"Aviso: Erro ao salvar embedding: {str(e)}")

    def _save_relationship_to_blob(self, source_id, target_id, similarity, changes):
        """Método auxiliar para salvar relacionamento no Blob Storage"""
        try:
            container_name = "relationships"
            try:
                container_client = self.blob_service_client.get_container_client(container_name)
                if not container_client.exists():
                    container_client = self.blob_service_client.create_container(container_name)
            except Exception as e:
                print(f"Aviso: Erro ao verificar/criar container de relacionamentos: {str(e)}")
                container_client = self.blob_service_client.create_container(container_name)
            
            relationship_data = {
                "source": source_id,
                "target": target_id,
                "similarity": float(similarity),
                "changes": changes
            }
            
            blob_name = f"{source_id}_to_{target_id}.json"
            blob_client = container_client.get_blob_client(blob_name)
            
            blob_client.upload_blob(json.dumps(relationship_data), overwrite=True)
            print(f"Relationship saved: {source_id} -> {target_id}")
        except Exception as e:
            print(f"Aviso: Erro ao salvar relacionamento: {str(e)}")
    
    async def enrich_embeddings_with_gnn(self):

        try:
            print("Iniciando enriquecimento de embeddings com GNN...")
            
            await self._ensure_graph_has_embeddings()
            
            graph_data, node_mapping = self.gnn_processor.networkx_to_pytorch_geometric(
                self.knowledge_graph
            )
            
            self.gnn_processor.init_model(model_type='gcn')
            self.gnn_processor.train(graph_data, epochs=50)
            
            self.gnn_embeddings = self.gnn_processor.get_node_embeddings(
                graph_data, node_mapping
            )
            
            print(f"Enriquecimento com GNN concluído. Processados {len(self.gnn_embeddings)} nós.")
            return True
        except Exception as e:
            print(f"Erro ao enriquecer embeddings com GNN: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    async def _ensure_graph_has_embeddings(self):

        for node in self.knowledge_graph.nodes():
            if 'embedding' not in self.knowledge_graph.nodes[node]:
                node_data = self.knowledge_graph.nodes[node]
                if 'text' in node_data:
                    embedding = self.get_text_embeddings(node_data['text'])
                    self.knowledge_graph.nodes[node]['embedding'] = embedding
                else:
                    print(f"Aviso: Nó {node} não tem texto para gerar embedding")
                    self.knowledge_graph.nodes[node]['embedding'] = np.zeros(1024)

    def calculate_similarity_with_gnn(self, text1, text2, use_gnn=True):

        embedding1 = self.get_text_embeddings(text1)
        embedding2 = self.get_text_embeddings(text2)
        
        if not use_gnn or not self.gnn_embeddings:
            return cosine_similarity([embedding1], [embedding2])[0][0]
        
        reg_id1 = self._find_regulation_by_text(text1)
        reg_id2 = self._find_regulation_by_text(text2)
        
        if reg_id1 and reg_id2 and reg_id1 in self.gnn_embeddings and reg_id2 in self.gnn_embeddings:
            gnn_emb1 = self.gnn_embeddings[reg_id1]
            gnn_emb2 = self.gnn_embeddings[reg_id2]
            similarity = self.gnn_processor.predict_similarity(gnn_emb1, gnn_emb2)
            return similarity
        
        return cosine_similarity([embedding1], [embedding2])[0][0]

    def _find_regulation_by_text(self, text):
        """
        Encontra o ID de uma regulação pelo seu texto.
        """
        for node, data in self.knowledge_graph.nodes(data=True):
            if data.get('text') == text:
                return node
        return None

    async def _get_comparison_analysis_with_gpt(self, text1, text2, similarity, key_differences):
        """Obter análise detalhada da comparação usando o novo endpoint Azure AI"""
        try:
            differences_text = "\n".join([f"- {diff}" for diff in key_differences])
            
            prompt = f"""Analise duas versões de um texto regulatório e explique as mudanças e seus possíveis impactos:

    TEXTO ORIGINAL:
    {text1}

    TEXTO MODIFICADO:
    {text2}

    SIMILARIDADE CALCULADA: {similarity*100:.1f}%

    DIFERENÇAS IDENTIFICADAS:
    {differences_text}

    Por favor, forneça:
    1. Uma explicação das mudanças principais
    2. O possível propósito ou intenção por trás dessas mudanças
    3. O impacto potencial dessas mudanças
    4. Tendências observáveis que poderiam indicar futuras direções regulatórias

    Formate sua resposta como JSON seguindo exatamente esta estrutura:
    {{
        "main_changes": "Explicação concisa das principais mudanças",
        "purpose": "Análise do propósito ou intenção por trás das mudanças",
        "impact": "Avaliação do impacto potencial dessas mudanças",
        "future_trends": "Indicação de possíveis tendências regulatórias futuras"
    }}
    """

            endpoint = f"{self.azure_ai_endpoint}/models/chat/completions?api-version=2024-05-01-preview"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": self.azure_ai_key,
                "x-ms-model-mesh-model-name": "Phi-4-multimodal-instruct"
            }
            
            data = {
                "messages": [
                    {"role": "system", "content": "Você é um especialista em análise de tendências regulatórias."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.9,
                "max_tokens": 8000,
                "model": "Phi-4-multimodal-instruct"
            }
            
            print(f"Chamando API Azure AI para análise de comparação...")
            print(f"Endpoint: {endpoint}")
            print(f"Modelo: Phi-4-multimodal-instruct")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    status = response.status
                    response_text = await response.text()
                    print(f"Status da resposta: {status}")
                    print(f"Resposta bruta: {response_text[:200]}...")
                    
                    if status == 200:
                        result = json.loads(response_text)
                        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                        
                        try:
                            import re
                            json_match = re.search(r'(\{[\s\S]*\})', content)
                            if json_match:
                                json_str = json_match.group(1)
                                analysis = json.loads(json_str)
                                return analysis
                            else:
                                print("JSON não encontrado na resposta")
                                return {
                                    "main_changes": "Não foi possível analisar as mudanças",
                                    "purpose": "Indeterminado",
                                    "impact": "Indeterminado",
                                    "future_trends": "Indeterminado"
                                }
                        except Exception as e:
                            print(f"Erro ao processar análise de completions: {str(e)}")
                            return None
                    else:
                        print(f"Erro na API de completions: {status} - {response_text}")
                        return None
        except Exception as e:
            import traceback
            print(f"Erro ao obter análise de completions: {str(e)}")
            print(traceback.format_exc())
            return None

    def _build_prediction_prompt_for_gpt_with_gnn(self, regulations_history, changes_history, latest_regulation, structural_insights):
        """Construir prompt para o GPT-4o baseado no histórico, embeddings e insights de GNN"""
        regulations_text = "\n\n".join(regulations_history)
        
        changes_text = ""
        for change in changes_history:
            changes_text += f"\nMudanças de {change['from']} para {change['to']}:\n"
            for item in change['changes']:
                changes_text += f"- {item}\n"
        
        # Extrair insights estruturais do GNN
        insights_text = "\nINSIGHTS ESTRUTURAIS DE GNN:\n"
        
        # Adicionar insights de clusters
        cluster_insights = [i for i in structural_insights if 'cluster_id' in i]
        if cluster_insights:
            insights_text += "\nGrupos de regulações similares identificados:\n"
            for i, insight in enumerate(cluster_insights):
                insights_text += f"Grupo {i+1}: "
                if insight['common_phrases']:
                    insights_text += f"Temas comuns: {', '.join(insight['common_phrases'][:3])}"
                insights_text += f" ({insight['node_count']} regulações)\n"
        
        # Adicionar insights direcionais
        direction_insights = [i for i in structural_insights if i.get('type') == 'evolution_direction']
        if direction_insights:
            insights_text += "\nDireção de evolução regulatória detectada:\n"
            for insight in direction_insights:
                progression = insight['progression']
                if len(progression) >= 2:
                    insights_text += f"Progressão: {' -> '.join(progression[:3])}"
                    if len(progression) > 3:
                        insights_text += f" -> ... -> {progression[-1]}"
                    insights_text += "\n"
        
        number_pattern = r'(\d+)%'
        percentage_values = []
        
        for reg in regulations_history:
            matches = re.findall(number_pattern, reg)
            for match in matches:
                try:
                    percentage_values.append(int(match))
                except ValueError:
                    pass
        
        trend_analysis = ""
        if len(percentage_values) >= 2:
            if all(percentage_values[i] <= percentage_values[i+1] for i in range(len(percentage_values)-1)):
                trend_analysis = "Tendência observada: Valores percentuais CRESCENTES ao longo do tempo."
            elif all(percentage_values[i] >= percentage_values[i+1] for i in range(len(percentage_values)-1)):
                trend_analysis = "Tendência observada: Valores percentuais DECRESCENTES ao longo do tempo."
            else:
                trend_analysis = "Tendência observada: Valores percentuais VARIÁVEIS sem padrão claro."
        
        prompt = f"""Você é um especialista em análise de tendências regulatórias com conhecimento profundo de regulações financeiras. Analise o histórico de textos regulatórios abaixo e preveja as prováveis mudanças futuras.

    HISTÓRICO DE REGULAÇÕES:
    {regulations_text}

    ANÁLISE DE MUDANÇAS:
    {changes_text}

    {trend_analysis}

    {insights_text}

    REGULAÇÃO MAIS RECENTE:
    {latest_regulation['text']}

    Com base nos dados acima, incluindo os insights estruturais gerados por Graph Neural Networks, gere uma previsão estruturada das mudanças mais prováveis que ocorrerão na próxima iteração desta regulação. Considere padrões numéricos, tendências de linguagem, agrupamentos estruturais identificados e a direção de evolução regulatória.

    Formate sua resposta como um JSON seguindo exatamente esta estrutura:
    [
        {{
            "type": "numerical",
            "current_value": "valor atual (ex: 30%)",
            "predicted_value": "valor previsto (ex: 40%)",
            "confidence": valor de confiança entre 0.0 e 1.0,
            "explanation": "Explicação detalhada da previsão, considerando os insights estruturais de GNN"
        }},
        {{
            "type": "textual",
            "current_text": "texto atual",
            "predicted_text": "texto previsto",
            "confidence": valor de confiança entre 0.0 e 1.0,
            "explanation": "Explicação detalhada da previsão, considerando os insights estruturais de GNN"
        }}
    ]

    Gere apenas o JSON válido, sem texto introdutório ou de fechamento.
    """
        return prompt

    async def _get_gpt_predictions(self, prompt):
        """Obter previsões do modelo de chat usando o novo endpoint Azure AI"""
        try:
            endpoint = f"{self.azure_ai_endpoint}/models/chat/completions?api-version=2024-05-01-preview"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": self.azure_ai_key,
                "x-ms-model-mesh-model-name": "Phi-4-multimodal-instruct"
            }
            
            data = {
                "messages": [
                    {"role": "system", "content": "Você é um especialista em análise de tendências regulatórias."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.9,
                "max_tokens": 8000,
                "model": "Phi-4-multimodal-instruct"
            }
            
            print(f"Chamando API Azure AI para completions...")
            print(f"Endpoint: {endpoint}")
            print(f"Modelo: Phi-4-multimodal-instruct")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    status = response.status
                    response_text = await response.text()
                    print(f"Status da resposta: {status}")
                    print(f"Resposta bruta: {response_text[:200]}...")
                    
                    if status == 200:
                        result = json.loads(response_text)
                        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                        
                        try:
                            import re
                            start_idx = content.find('[')
                            end_idx = content.rfind(']') + 1
                            
                            if start_idx >= 0 and end_idx > start_idx:
                                json_str = content[start_idx:end_idx]
                                predictions = json.loads(json_str)
                                return predictions
                            else:
                                print("JSON não encontrado na resposta")
                                return self._generate_fallback_predictions([], {})
                        except json.JSONDecodeError as e:
                            print(f"Erro ao decodificar JSON: {str(e)}")
                            return self._generate_fallback_predictions([], {})
                    else:
                        print(f"Erro na API de completions: {status} - {response_text}")
                        return self._generate_fallback_predictions([], {})
        except Exception as e:
            import traceback
            print(f"Erro ao chamar API de completions: {str(e)}")
            print(traceback.format_exc())
            return self._generate_fallback_predictions([], {})

    def calculate_similarity(self, text1, text2):
        embedding1 = self.get_text_embeddings(text1)
        embedding2 = self.get_text_embeddings(text2)
        
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return similarity
    
    def extract_key_changes(self, old_text, new_text):
        number_pattern = r'(\d+(?:\.\d+)?(?:\s*%)?)'

        old_numbers = re.findall(number_pattern, old_text)
        new_numbers = re.findall(number_pattern, new_text)

        old_doc = self.nlp(old_text)
        new_doc = self.nlp(new_text)

        old_entities = {ent.text: ent.label_ for ent in old_doc.ents}
        new_entities = {ent.text: ent.label_ for ent in new_doc.ents}

        modified_entities = {}
        for entity in set(old_entities.keys()).union(new_entities.keys()):
            if entity in old_entities and entity in new_entities:
                if old_entities[entity] != new_entities[entity]:
                    modified_entities[entity] = (old_entities[entity], new_entities[entity])
            elif entity in old_entities:
                modified_entities[entity] = (old_entities[entity], None)
            else:
                modified_entities[entity] = (None, new_entities[entity])
        
        matcher = SequenceMatcher(None, old_text, new_text)
        diff_blocks = []

        for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
            if opcode != 'equal':
                diff_blocks.append({
                    'opcode': opcode,
                    'old_text': old_text[i1:i2],
                    'new_text': new_text[j1:j2]
                })
        
        changes = {
            'numerical_changes': list(zip(old_numbers, new_numbers)) if len(old_numbers) == len(new_numbers) else [],
            'entity_changes': modified_entities,
            'text_diff_blocks': diff_blocks
        }
        
        return changes
    
    def analyze_key_phares(self, text):
        try:
            response = self.text_analytics_client.extract_key_phrases(documents=[{
                "id": "1",
                "language": "pt",
                "text": text
            }]
            )
            if not response or len(response) == 0:
                return []

            doc_result = response[0]
            if not doc_result or doc_result.is_error:
                return []

            return doc_result.key_phrases

        except Exception as e:
            print(f"Error analyzing key phrases with Azure Language Service: {e}")
            return []

    def query_gremlin_rest(self, query):
        """
        Executa uma consulta Gremlin usando a API REST do Cosmos DB
        """
        try:
            import requests
            import json
            import base64
            import uuid
            
            hostname = self.gremlin_endpoint.replace('wss://', '').split(':')[0]
            
            url = f"https://{hostname}:443/dbs/{self.gremlin_database}/colls/{self.gremlin_collection}"
            
            auth_header = base64.b64encode(f":{self.gremlin_key}".encode('utf-8')).decode('utf-8')
            
            headers = {
                "Authorization": f"Basic {auth_header}",
                "Accept": "application/json",
                "Content-Type": "application/query+json",
                "x-ms-version": "2017-02-22",
                "x-ms-documentdb-isquery": "true",
                "x-ms-documentdb-query-enablecrosspartition": "true"
            }
            
            request_body = {
                "query": query
            }
            
            print(f"Tentando consulta Gremlin via Cosmos DB REST API")
            print(f"URL: {url}")
            print(f"Consulta: {query}")
            
            response = requests.post(url, json=request_body, headers=headers)
            
            print(f"Status: {response.status_code}")
            print(f"Resposta: {response.text}")
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Erro na consulta Gremlin: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"Erro ao executar consulta Gremlin via REST: {str(e)}")
            return []

    async def add_edge_to_gremlin(self, from_id, to_id, similarity, changes_dict):
        """
        Método auxiliar para adicionar uma aresta no grafo Gremlin
        Usando uma abordagem mais segura para evitar problemas com loops aninhados
        """
        try:
            import requests
            import json
            import base64
            import uuid
            
            query = f"g.V().has('regulation', 'id', '{from_id}')" + \
                    f".addE('changed_to')" + \
                    f".property('similarity', {float(similarity)})" + \
                    f".property('changes', '{json.dumps(changes_dict)}')" + \
                    f".to(g.V().has('regulation', 'id', '{to_id}'))"
                    
            print(f"Executando consulta Gremlin: {query}")
            
            endpoint = self.gremlin_endpoint.replace('wss://', 'https://').replace('/gremlin', '')
            endpoint = f"{endpoint}"
            
            auth = base64.b64encode(f"{self.gremlin_database}/colls/{self.gremlin_collection}:{self.gremlin_key}".encode()).decode()
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Basic {auth}",
                "x-ms-date": datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT"),
                "x-ms-version": "2018-12-31"
            }
            
            payload = {
                "gremlin": query,
                "bindings": {},
                "requestId": str(uuid.uuid4())
            }
            
            response = requests.post(endpoint, json=payload, headers=headers)
            
            if response.status_code == 200 or response.status_code == 201:
                print(f"Aresta adicionada com sucesso: {from_id} -> {to_id}")
                return True
            else:
                print(f"Erro ao adicionar aresta via API REST: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"Erro ao adicionar aresta no Gremlin: {str(e)}")
            return False

    async def get_all_regulations(self):
            """Obter todas as regulações do Cosmos DB e do grafo de conhecimento"""
            regulations = []
            
            if self.regulations_container:
                try:
                    query = "SELECT * FROM c"
                    items = list(self.regulations_container.query_items(
                        query=query,
                        enable_cross_partition_query=True
                    ))
                    
                    for item in items:
                        regulation = {
                            "id": item["id"],
                            "date": item["date"],
                            "text": item["text"],
                            "key_phrases": item.get("key_phrases", [])
                        }
                        
                        self.knowledge_graph.add_node(
                            regulation["id"],
                            date=regulation["date"],
                            text=regulation["text"],
                            key_phrases=regulation["key_phrases"]
                        )
                        
                        regulations.append(regulation)
                        
                    if self.relationships_container:
                        query = "SELECT * FROM c"
                        relationships = list(self.relationships_container.query_items(
                            query=query,
                            enable_cross_partition_query=True
                        ))
                        
                        for rel in relationships:
                            source_id = rel["source"]
                            target_id = rel["target"]
                            
                            if source_id in self.knowledge_graph.nodes and target_id in self.knowledge_graph.nodes:
                                self.knowledge_graph.add_edge(
                                    source_id,
                                    target_id,
                                    similarity=rel["similarity"],
                                    changes=rel.get("changes", {})
                                )
                        
                    print(f"Carregadas {len(regulations)} regulações do Cosmos DB")
                    
                except Exception as e:
                    print(f"Erro ao carregar regulações do Cosmos DB: {str(e)}")
            
            if not regulations:
                for node_id in self.knowledge_graph.nodes:
                    node_data = self.knowledge_graph.nodes[node_id]
                    regulation = {
                        "id": node_id,
                        "date": node_data.get("date"),
                        "text": node_data.get("text"),
                        "key_phrases": node_data.get("key_phrases", [])
                    }
                    regulations.append(regulation)
            
            if not regulations:
                print("Inicializando com regulações de exemplo")
                example_regulations = [
                    {
                        "id": "reg1",
                        "date": "outubro de 2024",
                        "text": "Dispõe sobre a constituição, a administração, o funcionamento e a divulgação das informações dos fundos de investimento. Investidores estrangeiros podem deter até 20% das ações com direito a voto."
                    },
                    {
                        "id": "reg2",
                        "date": "dezembro de 2024",
                        "text": "Dispõe sobre a constituição, a administração, o funcionamento e a divulgação das informações dos fundos de investimento. Investidores estrangeiros podem deter até 30% das ações com direito a voto."
                    },
                    {
                        "id": "reg3",
                        "date": "fevereiro de 2025",
                        "text": "Dispõe sobre a constituição, a administração, o funcionamento e a divulgação das informações dos fundos de investimento. Não há limite para a participação de investidores estrangeiros nas ações com direito a voto."
                    }
                ]
                
                prev_id = None
                for reg in example_regulations:
                    key_phrases = self.analyze_key_phares(reg["text"])
                    
                    self.knowledge_graph.add_node(
                        reg["id"],
                        date=reg["date"],
                        text=reg["text"],
                        key_phrases=key_phrases
                    )
                    
                    if self.regulations_container:
                        try:
                            regulation_item = {
                                "id": reg["id"],
                                "date": reg["date"],
                                "text": reg["text"],
                                "key_phrases": key_phrases
                            }
                            
                            try:
                                self.regulations_container.create_item(body=regulation_item)
                                print(f"Regulation {reg['id']} created in Cosmos DB")
                            except exceptions.CosmosResourceExistsError:
                                self.regulations_container.replace_item(item=reg["id"], body=regulation_item)
                                print(f"Regulation {reg['id']} updated in Cosmos DB")
                            except Exception as e:
                                print(f"Erro ao salvar regulação {reg['id']} no Cosmos DB: {str(e)}")
                        except Exception as e:
                            print(f"Erro ao salvar regulação {reg['id']} no Cosmos DB: {str(e)}")
                    
                    if prev_id:
                        prev_text = self.knowledge_graph.nodes[prev_id]["text"]
                        current_text = reg["text"]
                        
                        similarity = self.calculate_similarity(prev_text, current_text)
                        changes = self.extract_key_changes(prev_text, current_text)
                        
                        self.knowledge_graph.add_edge(
                            prev_id,
                            reg["id"],
                            similarity=similarity,
                            changes=changes
                        )
                        
                        if self.relationships_container:
                            try:
                                relationship_id = f"{prev_id}_to_{reg['id']}"
                                relationship_item = {
                                    "id": relationship_id,
                                    "source": prev_id,
                                    "target": reg["id"],
                                    "similarity": float(similarity),
                                    "changes": changes
                                }
                                
                                try:
                                    self.relationships_container.create_item(body=relationship_item)
                                    print(f"Relationship {relationship_id} created in Cosmos DB")
                                except exceptions.CosmosResourceExistsError:
                                    self.relationships_container.replace_item(item=relationship_id, body=relationship_item)
                                    print(f"Relationship {relationship_id} updated in Cosmos DB")
                                except Exception as e:
                                    print(f"Erro ao salvar relacionamento no Cosmos DB: {str(e)}")
                            except Exception as e:
                                print(f"Erro ao salvar relacionamento no Cosmos DB: {str(e)}")
                    
                    prev_id = reg["id"]
                    
                    reg["key_phrases"] = key_phrases
                    regulations.append(reg)
            
            return regulations

    async def get_knowledge_graph(self):
        """Obter o grafo de conhecimento em formato JSON"""
        try:
            await self.get_all_regulations()
            
            nodes = []
            for node_id in self.knowledge_graph.nodes:
                node_data = self.knowledge_graph.nodes[node_id]
                nodes.append({
                    "id": str(node_id),
                    "date": str(node_data.get("date", "")),
                    "key_phrases": [str(phrase) for phrase in node_data.get("key_phrases", [])]
                })
            
            edges = []
            for source, target, data in self.knowledge_graph.edges(data=True):
                try:
                    similarity = float(data.get("similarity", 0))
                except (ValueError, TypeError):
                    similarity = 0.0
                    
                edge_data = {
                    "source": str(source),
                    "target": str(target),
                    "similarity": similarity,
                    "has_changes": bool("changes" in data)
                }
                edges.append(edge_data)
            
            return {
                "nodes": nodes,
                "edges": edges
            }
        except Exception as e:
            print(f"Erro ao obter grafo de conhecimento: {str(e)}")
            return {
                "nodes": [],
                "edges": []
            }

    async def add_to_knowledge_graph(self, regulation_id, date, text, previous_regulation_id=None):
        """Adiciona uma regulação ao grafo de conhecimento e ao Cosmos DB"""
        try:
            key_phrases = self.analyze_key_phares(text)
            
            self.knowledge_graph.add_node(
                regulation_id, 
                date=date, 
                text=text,
                key_phrases=key_phrases
            )
            
            print(f"Regulation {regulation_id} added to local knowledge graph")
            
            regulation_item = {
                "id": regulation_id,
                "date": date,
                "text": text,
                "key_phrases": key_phrases
            }
            
            try:
                if self.regulations_container:
                    try:
                        existing_item = self.regulations_container.read_item(item=regulation_id, partition_key=regulation_id)
                        self.regulations_container.replace_item(item=regulation_id, body=regulation_item)
                        print(f"Regulation {regulation_id} updated in Cosmos DB")
                    except exceptions.CosmosResourceNotFoundError:
                        self.regulations_container.create_item(body=regulation_item)
                        print(f"Regulation {regulation_id} created in Cosmos DB")
                    except Exception as e:
                        print(f"Erro ao salvar regulação no Cosmos DB: {str(e)}")
            except Exception as e:
                print(f"Erro ao salvar regulação no Cosmos DB: {str(e)}")
            
            if previous_regulation_id:
                if previous_regulation_id in self.knowledge_graph.nodes:
                    try:
                        prev_text = self.knowledge_graph.nodes[previous_regulation_id]['text']
                        similarity = self.calculate_similarity(prev_text, text)
                        changes = self.extract_key_changes(prev_text, text)
                        
                        serializable_changes = self._make_changes_serializable(changes)
                        
                        self.knowledge_graph.add_edge(
                            previous_regulation_id,
                            regulation_id,
                            similarity=similarity,
                            changes=serializable_changes
                        )
                        
                        relationship_id = f"{previous_regulation_id}_to_{regulation_id}"
                        relationship_item = {
                            "id": relationship_id,
                            "source": previous_regulation_id,
                            "target": regulation_id,
                            "similarity": float(similarity),
                            "changes": serializable_changes
                        }
                        
                        try:
                            if self.relationships_container:
                                try:
                                    existing_item = self.relationships_container.read_item(
                                        item=relationship_id, 
                                        partition_key=relationship_id
                                    )
                                    self.relationships_container.replace_item(
                                        item=relationship_id, 
                                        body=relationship_item
                                    )
                                    print(f"Relationship {relationship_id} updated in Cosmos DB")
                                except exceptions.CosmosResourceNotFoundError:
                                    self.relationships_container.create_item(body=relationship_item)
                                    print(f"Relationship {relationship_id} created in Cosmos DB")
                                except Exception as e:
                                    print(f"Erro ao salvar relacionamento no Cosmos DB: {str(e)}")
                        except Exception as e:
                            print(f"Erro ao salvar relacionamento no Cosmos DB: {str(e)}")
                        
                        print(f"Relação adicionada: {previous_regulation_id} -> {regulation_id}")
                    except Exception as e:
                        print(f"Aviso: Erro ao processar similaridade e mudanças: {str(e)}")
                else:
                    print(f"Regulação anterior {previous_regulation_id} não encontrada no grafo")
            
            return True
        except Exception as e:
            print(f"Erro ao adicionar regulação {regulation_id} ao grafo de conhecimento: {str(e)}")
            return False

    def _make_changes_serializable(self, changes):
        """Converte as mudanças para um formato serializável"""
        if not changes:
            return {}
            
        serializable_changes = {}
        
        if "numerical_changes" in changes:
            numerical_changes = changes["numerical_changes"]
            serializable_changes["numerical_changes"] = [
                {"old": str(pair[0]), "new": str(pair[1])} 
                for pair in numerical_changes if isinstance(pair, (list, tuple)) and len(pair) == 2
            ]
        
        if "entity_changes" in changes:
            entity_changes = changes["entity_changes"]
            serializable_changes["entity_changes"] = {
                entity: {"old": str(change[0]), "new": str(change[1])}
                for entity, change in entity_changes.items()
            }
        
        if "text_diff_blocks" in changes:
            text_diff_blocks = changes["text_diff_blocks"]
            serializable_changes["text_diff_blocks"] = [
                {
                    "opcode": block.get("opcode", ""),
                    "old_text": block.get("old_text", ""),
                    "new_text": block.get("new_text", "")
                }
                for block in text_diff_blocks if isinstance(block, dict)
            ]
        
        return serializable_changes

    async def predict_future_changes_with_text(self, text, num_predictions=1):
        """Prever mudanças futuras com base em um texto específico usando IA"""
        try:
            regulations = await self.get_all_regulations()
            sorted_regulations = sorted(regulations, key=lambda x: x.get('date', ''))
            
            regulations_history = []
            for i, reg in enumerate(sorted_regulations):
                regulations_history.append(f"Regulação {i+1} ({reg['date']}): {reg['text']}")
            
            changes_history = []
            for i in range(1, len(sorted_regulations)):
                prev_reg = sorted_regulations[i-1]
                curr_reg = sorted_regulations[i]
                changes = self.extract_key_changes(prev_reg['text'], curr_reg['text'])
                
                changes_summary = []
                if 'numerical_changes' in changes and changes['numerical_changes']:
                    for pair in changes['numerical_changes']:
                        if isinstance(pair, (list, tuple)) and len(pair) == 2:
                            changes_summary.append(f"Alteração numérica: {pair[0]} → {pair[1]}")
                
                if 'text_diff_blocks' in changes:
                    for block in changes['text_diff_blocks']:
                        if block.get('opcode') == 'replace':
                            changes_summary.append(f"Substituição: '{block.get('old_text', '')}' → '{block.get('new_text', '')}'")
                        elif block.get('opcode') == 'insert':
                            changes_summary.append(f"Adição: '{block.get('new_text', '')}'")
                        elif block.get('opcode') == 'delete':
                            changes_summary.append(f"Remoção: '{block.get('old_text', '')}'")
                
                changes_history.append({
                    "from": prev_reg['date'],
                    "to": curr_reg['date'],
                    "changes": changes_summary
                })
            
            key_phrases = self.analyze_key_phares(text)
            
            prompt = f"""Você é um especialista em análise de tendências regulatórias com conhecimento profundo de regulações financeiras. 
            
    HISTÓRICO DE REGULAÇÕES:
    {regulations_history[-3:] if len(regulations_history) > 3 else regulations_history}

    TEXTO ATUAL PARA PREVISÃO:
    {text}

    FRASES-CHAVE IDENTIFICADAS:
    {', '.join(key_phrases) if key_phrases else 'Nenhuma frase-chave identificada'}

    Com base no texto atual e no contexto histórico, gere {num_predictions} previsões específicas e detalhadas sobre como este texto regulatório poderá evoluir no futuro. 
    Sua previsão deve:
    1. Identificar trechos específicos do texto fornecido que provavelmente mudarão
    2. Prever como esses trechos serão modificados (exemplo: percentuais que aumentarão/diminuirão, requisitos que serão flexibilizados/endurecidos, etc.)
    3. Fornecer uma explicação contextual para cada previsão, baseada nas tendências observadas

    Formate cada previsão como um objeto JSON seguindo exatamente esta estrutura:
    {{
        "type": "textual",
        "current_text": "Trecho específico do texto atual que provavelmente mudará",
        "predicted_text": "Como esse trecho provavelmente ficará na próxima versão",
        "confidence": valor de confiança entre 0.0 e 1.0,
        "explanation": "Explicação detalhada que justifica essa previsão"
    }}

    Suas previsões devem ser específicas, baseadas no texto fornecido, e não genéricas. Formate sua resposta como um array JSON contendo {num_predictions} objetos de previsão.
            """
            
            predictions = await self._get_gpt_predictions(prompt)
            return predictions
        except Exception as e:
            import traceback
            print(f"Erro ao gerar previsões com texto específico: {str(e)}")
            print(traceback.format_exc())
            return []

    async def predict_future_changes(self, num_predictions=1):
        """Prever mudanças futuras com base nas tendências históricas usando IA"""
        if len(self.knowledge_graph.nodes) < 2:
            print("Not enough regulations to predict future changes")
            return []
        
        nodes = list(self.knowledge_graph.nodes)
        node_data = [(node, self.knowledge_graph.nodes[node]) for node in nodes]
        sorted_nodes = sorted(node_data, key=lambda x: x[1].get('date', ''))
        
        historical_data = []
        for node_id, data in sorted_nodes:
            historical_data.append({
                "id": node_id,
                "date": data.get('date', ''),
                "text": data.get('text', ''),
                "key_phrases": data.get('key_phrases', [])
            })
        
        numerical_changes = []
        for i in range(1, len(sorted_nodes)):
            prev_node_id, _ = sorted_nodes[i-1]
            curr_node_id, _ = sorted_nodes[i]
            
            if self.knowledge_graph.has_edge(prev_node_id, curr_node_id):
                edge_data = self.knowledge_graph.get_edge_data(prev_node_id, curr_node_id)
                changes = edge_data.get('changes', {})
                
                if isinstance(changes, dict) and 'numerical_changes' in changes:
                    num_changes = changes['numerical_changes']
                    numerical_changes.extend(num_changes)
        
        latest_regulation = historical_data[-1] if historical_data else None
        
        if not latest_regulation:
            return []
        
        prompt = self._build_prediction_prompt(historical_data, numerical_changes, latest_regulation)
        
        try:
            predictions = await self._get_ai_predictions(prompt, numerical_changes, latest_regulation)
            return predictions
        except Exception as e:
            print(f"Erro ao obter previsões da IA: {str(e)}")
            return self._generate_fallback_predictions(numerical_changes, latest_regulation)

    def _build_prediction_prompt(self, historical_data, numerical_changes, latest_regulation):
        """Construir prompt para enviar ao modelo LLM"""
        historical_context = ""
        for i, reg in enumerate(historical_data):
            historical_context += f"Regulação {i+1} ({reg['date']}): {reg['text']}\n\n"
        
        numerical_trends = ""
        if numerical_changes:
            numerical_trends = "Mudanças numéricas observadas:\n"
            for pair in numerical_changes:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    numerical_trends += f"- {pair[0]} -> {pair[1]}\n"
        
        prompt = f"""Você é um especialista em análise de tendências regulatórias. Analise a sequência histórica de regulações e preveja prováveis mudanças futuras.

    CONTEXTO HISTÓRICO:
    {historical_context}

    MUDANÇAS NUMÉRICAS OBSERVADAS:
    {numerical_trends}

    REGULAÇÃO MAIS RECENTE:
    {latest_regulation['text']}

    Com base nos dados acima:
    1. Identifique tendências numéricas (especialmente percentuais e limites) e preveja seus prováveis valores futuros.
    2. Analise a evolução das políticas e preveja mudanças textuais prováveis.
    3. Atribua um nível de confiança (0.0 a 1.0) para cada previsão.

    Formate sua resposta como JSON no seguinte formato:
    [
    {{
        "type": "numerical",
        "current_value": "valor atual (ex: 30%)",
        "predicted_value": "valor previsto (ex: 40%)",
        "confidence": valor de confiança (ex: 0.85)
    }},
    {{
        "type": "textual",
        "current_text": "texto atual",
        "predicted_text": "texto previsto",
        "confidence": valor de confiança
    }}
    ]
    """
        return prompt

    async def _get_ai_predictions(self, prompt, numerical_changes, latest_regulation):
        """Obter previsões do modelo usando o novo endpoint Azure AI"""
        try:
            endpoint = f"{self.azure_ai_endpoint}/models/chat/completions?api-version=2024-05-01-preview"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": self.azure_ai_key,
                "x-ms-model-mesh-model-name": "Phi-4-multimodal-instruct"
            }
            
            data = {
                "messages": [
                    {"role": "system", "content": "Você é um especialista em análise de tendências regulatórias."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.9,
                "max_tokens": 8000,
                "model": "Phi-4-multimodal-instruct"
            }
            
            print(f"Chamando API Azure AI para previsões...")
            print(f"Endpoint: {endpoint}")
            print(f"Modelo: Phi-4-multimodal-instruct")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    status = response.status
                    response_text = await response.text()
                    print(f"Status da resposta: {status}")
                    print(f"Resposta bruta: {response_text[:200]}...")
                    
                    if status == 200:
                        result = json.loads(response_text)
                        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                        
                        try:
                            import re
                            json_match = re.search(r'(\[[\s\S]*\])', content)
                            
                            if json_match:
                                json_str = json_match.group(1)
                                predictions = json.loads(json_str)
                                return predictions
                            else:
                                print("JSON não encontrado na resposta")
                                return self._generate_fallback_predictions(numerical_changes, latest_regulation)
                        except json.JSONDecodeError:
                            print(f"Erro ao decodificar JSON da resposta")
                            return self._generate_fallback_predictions(numerical_changes, latest_regulation)
                    else:
                        print(f"Erro na API de completions: {status} - {response_text}")
                        return self._generate_fallback_predictions(numerical_changes, latest_regulation)
        except Exception as e:
            import traceback
            print(f"Erro ao chamar API de completions: {str(e)}")
            print(traceback.format_exc())
            return self._generate_fallback_predictions(numerical_changes, latest_regulation)

    def _generate_fallback_predictions(self, numerical_changes, latest_regulation):
        """Gerar previsões de fallback baseadas em padrões simples"""
        predictions = []
        
        import re
        latest_text = latest_regulation.get('text', '')
        latest_numbers = re.findall(r'(\d+(?:\.\d+)?(?:\s*%)?)', latest_text)
        
        for number_str in latest_numbers:
            if '%' in number_str:
                try:
                    current_value = float(number_str.replace('%', '').strip())
                    
                    new_value = current_value + 10.0
                    if new_value > 100:
                        new_value = 100.0
                    
                    predictions.append({
                        "type": "numerical",
                        "current_value": f"{current_value}%",
                        "predicted_value": f"{new_value:.1f}%",
                        "confidence": 0.75
                    })
                except ValueError:
                    continue
        
        if not predictions:
            if "limite" in latest_text.lower() or "restrição" in latest_text.lower():
                predictions.append({
                    "type": "textual",
                    "current_text": "Há limites e restrições específicas no texto atual",
                    "predicted_text": "As restrições serão reduzidas ou eliminadas nos próximos textos regulatórios",
                    "confidence": 0.65
                })
            else:
                predictions.append({
                    "type": "numerical",
                    "current_value": "20%",
                    "predicted_value": "40%",
                    "confidence": 0.7
                })
        
        return predictions

    def visualize_knowledge_graph(self):
        """Visualizar o grafo de conhecimento"""
        if len(self.knowledge_graph.nodes) == 0:
            return "Grafo vazio - adicione algumas regulações primeiro."
        
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go
        
        plt.figure(figsize=(12, 8))
        
        pos = nx.spring_layout(self.knowledge_graph)
        
        nx.draw_networkx_nodes(self.knowledge_graph, pos, node_size=700, node_color='lightblue')
        
        edges = self.knowledge_graph.edges(data=True)
        edge_colors = [data.get('similarity', 0.5) for _, _, data in edges]
        
        nx.draw_networkx_edges(self.knowledge_graph, pos, width=2, edge_color=edge_colors, edge_cmap=plt.cm.Blues)
        
        labels = {node: f"{node}\n{self.knowledge_graph.nodes[node]['date']}" for node in self.knowledge_graph.nodes}
        nx.draw_networkx_labels(self.knowledge_graph, pos, labels=labels, font_size=10)
        
        plt.title("Grafo de Conhecimento de Mudanças Regulatórias")
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues)
        sm.set_array([])
        plt.colorbar(sm, label="Similaridade")
        
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig("knowledge_graph.png")
        
        edge_traces = []
        for edge in self.knowledge_graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            similarity = edge[2].get('similarity', 0.5)
            
            edge_trace = go.Scatter(
                x=[x0, x1], y=[y0, y1],
                line=dict(width=2, color=f'rgba(0, 0, 255, {similarity})'),
                hoverinfo='text',
                text=f'Similaridade: {similarity:.2f}',
                mode='lines'
            )
            edge_traces.append(edge_trace)
        
        node_trace = go.Scatter(
            x=[pos[node][0] for node in self.knowledge_graph.nodes()],
            y=[pos[node][1] for node in self.knowledge_graph.nodes()],
            text=[f"ID: {node}<br>Data: {self.knowledge_graph.nodes[node]['date']}" for node in self.knowledge_graph.nodes()],
            mode='markers+text',
            marker=dict(
                size=15,
                color='lightblue',
                line=dict(width=2, color='DarkSlateGrey')
            ),
            textposition="top center"
        )
        
        fig = go.Figure(data=edge_traces + [node_trace],
                layout=go.Layout(
                    title='Grafo de Conhecimento Interativo',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
        
        fig.write_html("knowledge_graph_interactive.html")
        
        return "Visualização gerada com sucesso."
        
    async def close(self):
        """Encerrar conexões com os serviços"""
        if hasattr(self, 'gremlin_conn') and self.gremlin_conn:
            try:
                await self.gremlin_conn.close()
                self.gremlin_conn = None
                self.g = None
            except Exception as e:
                print(f"Erro ao fechar conexão Gremlin: {str(e)}")