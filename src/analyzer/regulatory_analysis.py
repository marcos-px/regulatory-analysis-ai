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

from gremlin_python.driver import client
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.strategies import *
from gremlin_python.structure.graph import Graph
from collections import defaultdict
import ast

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

        self.azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip('/')
        self.azure_openai_key = os.environ.get("AZURE_OPENAI_API_KEY")
        self.azure_openai_model = "text-embedding-3-small"
        self.azure_openai_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", self.azure_openai_model)
        
        if not self.azure_openai_key:
            raise ValueError("A variável de ambiente AZURE_OPENAI_KEY não está definida ou está vazia")
        if not self.azure_openai_endpoint:
            raise ValueError("A variável de ambiente AZURE_OPENAI_ENDPOINT não está definida ou está vazia")
        if not self.azure_openai_deployment:
            raise ValueError("A variável de ambiente AZURE_OPENAI_DEPLOYMENT não está definida ou está vazia")

        endpoint_base = self.azure_openai_endpoint.rstrip('/')

        self.embeddings_client = EmbeddingsClient(
            endpoint=f"{endpoint_base}",
            credential=AzureKeyCredential(self.azure_openai_key),
            deployment_name=self.azure_openai_deployment
        )
        self.text_analytics_client = TextAnalyticsClient(
            endpoint=self.azure_language_endpoint,
            credential=AzureKeyCredential(self.azure_language_key)
        )

        self.blob_service_client = BlobServiceClient.from_connection_string(self.azure_storage_connection_string)

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
        endpoint = f"{self.azure_openai_endpoint}/openai/deployments/{self.azure_openai_model}/embeddings?api-version=2023-05-15"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.azure_openai_key
        }
        data = {
            "input": text
        }
        
        print(f"Chamando endpoint: {endpoint}")
        
        response = requests.post(endpoint, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return np.array(result['data'][0]['embedding'])
        else:
            error_message = f"Erro na API OpenAI: {response.status_code} - {response.text}"
            print(error_message)
            raise Exception(error_message)

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

    async def add_to_knowledge_graph(self, regulation_id, date, text, previous_regulation_id=None):
        """Adiciona uma regulação ao grafo de conhecimento local e salva embeddings no Blob Storage"""
        try:
            key_phrases = self.analyze_key_phares(text)
            
            self.knowledge_graph.add_node(
                regulation_id, 
                date=date, 
                text=text,
                key_phrases=key_phrases
            )
            
            print(f"Regulation {regulation_id} added to local knowledge graph")
            
            try:
                container_name = "embeddings"
                try:
                    container_client = self.blob_service_client.get_container_client(container_name)
                    if not container_client.exists():
                        container_client = self.blob_service_client.create_container(container_name)
                except Exception as e:
                    print(f"Aviso: Erro ao verificar/criar container de embeddings: {str(e)}")
                
                embedding = self.get_text_embeddings(text)
                
                try:
                    blob_name = f"{regulation_id}_embedding.npy"
                    blob_client = container_client.get_blob_client(blob_name)
                    
                    with open("temp_embedding.npy", "wb") as f:
                        np.save(f, embedding)
                    
                    with open("temp_embedding.npy", "rb") as data:
                        blob_client.upload_blob(data, overwrite=True)
                    
                    os.remove("temp_embedding.npy")
                    print(f"Embedding saved for {regulation_id}")
                except Exception as e:
                    print(f"Aviso: Erro ao salvar embedding: {str(e)}")
            except Exception as e:
                print(f"Aviso: Erro ao processar embedding: {str(e)}")
            
            if previous_regulation_id:
                if previous_regulation_id in self.knowledge_graph.nodes:
                    try:
                        prev_text = self.knowledge_graph.nodes[previous_regulation_id]['text']
                        similarity = self.calculate_similarity(prev_text, text)
                        changes = self.extract_key_changes(prev_text, text)
                        
                        self.knowledge_graph.add_edge(
                            previous_regulation_id,
                            regulation_id,
                            similarity=similarity,
                            changes=changes
                        )
                        
                        print(f"Relação adicionada ao grafo local: {previous_regulation_id} -> {regulation_id}")
                    except Exception as e:
                        print(f"Aviso: Erro ao processar similaridade e mudanças: {str(e)}")
                else:
                    print(f"Regulação anterior {previous_regulation_id} não encontrada no grafo")
            
            return True
        except Exception as e:
            print(f"Erro ao adicionar regulação {regulation_id} ao grafo de conhecimento: {str(e)}")
            return False
    
    async def get_regulation_path(self, start_id, end_id=None):
        """Obtém o caminho entre duas regulações no grafo de conhecimento local"""
        try:
            if end_id:
                try:
                    paths = list(nx.all_simple_paths(self.knowledge_graph, start_id, end_id))
                    return [{'objects': path} for path in paths]
                except:
                    return []
            else:
                try:
                    paths = []
                    for node in self.knowledge_graph.nodes():
                        if node != start_id and nx.has_path(self.knowledge_graph, start_id, node):
                            path = nx.shortest_path(self.knowledge_graph, start_id, node)
                            paths.append({'objects': path})
                    return paths
                except:
                    return []
        except Exception as e:
            print(f"Erro ao obter caminho entre regulações: {str(e)}")
            return []

    async def get_all_regulations(self):
        """Obter todas as regulações do grafo de conhecimento de forma assíncrona"""
        regulations = []
        
        for node_id in self.knowledge_graph.nodes:
            node_data = self.knowledge_graph.nodes[node_id]
            regulation = {
                "id": node_id,
                "date": node_data.get("date"),
                "text": node_data.get("text"),
                "key_phrases": node_data.get("key_phrases", [])
            }
            regulations.append(regulation)
        
        return regulations

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
        
        try:
            import re
            latest_text = latest_regulation.get('text', '')
            latest_numbers = re.findall(r'(\d+(?:\.\d+)?(?:\s*%)?)', latest_text)
            
            predictions = []
            
            if len(historical_data) >= 2:
                percentage_pattern = r'(\d+)%'
                percentage_values = []
                
                for reg in historical_data:
                    matches = re.findall(percentage_pattern, reg.get('text', ''))
                    for match in matches:
                        try:
                            percentage_values.append(int(match))
                        except ValueError:
                            pass
                
                if len(percentage_values) >= 2:
                    increasing = all(percentage_values[i] <= percentage_values[i+1] for i in range(len(percentage_values)-1))
                    decreasing = all(percentage_values[i] >= percentage_values[i+1] for i in range(len(percentage_values)-1))
                    
                    latest_percentages = re.findall(percentage_pattern, latest_text)
                    
                    for perc_str in latest_percentages:
                        try:
                            current_value = int(perc_str)
                            
                            if increasing:
                                new_value = min(100, current_value + 10)
                                confidence = 0.85
                            elif decreasing:
                                new_value = max(0, current_value - 10)
                                confidence = 0.85
                            else:
                                if current_value < 50:
                                    new_value = current_value + 5
                                else:
                                    new_value = current_value - 5
                                confidence = 0.7
                            
                            predictions.append({
                                "type": "numerical",
                                "current_value": f"{current_value}%",
                                "predicted_value": f"{new_value}%",
                                "confidence": confidence
                            })
                        except ValueError:
                            continue
            
            if not predictions:
                for number_str in latest_numbers:
                    if '%' in number_str:
                        try:
                            current_value = float(number_str.replace('%', '').strip())
                            
                            if "aumento" in latest_text.lower() or "incremento" in latest_text.lower():
                                new_value = min(100, current_value + 10)
                                confidence = 0.7
                            elif "redução" in latest_text.lower() or "diminuição" in latest_text.lower():
                                new_value = max(0, current_value - 10)
                                confidence = 0.7
                            elif "não há limite" in latest_text.lower():
                                new_value = 100
                                confidence = 0.9
                            else:
                                if current_value < 25:
                                    new_value = 50
                                elif current_value < 50:
                                    new_value = 75
                                else:
                                    new_value = 100
                                confidence = 0.65
                            
                            predictions.append({
                                "type": "numerical",
                                "current_value": f"{current_value}%",
                                "predicted_value": f"{new_value:.1f}%",
                                "confidence": confidence
                            })
                        except ValueError:
                            continue

            if "limite" in latest_text.lower() or "restrição" in latest_text.lower():
                predictions.append({
                    "type": "textual",
                    "current_text": "Há limites e restrições específicas no texto atual",
                    "predicted_text": "As restrições serão reduzidas ou eliminadas nos próximos textos regulatórios",
                    "confidence": 0.75
                })
            
            if not predictions:
                predictions.append({
                    "type": "numerical",
                    "current_value": "20%",
                    "predicted_value": "40%",
                    "confidence": 0.7
                })
            
            return predictions
        
        except Exception as e:
            print(f"Erro ao gerar previsões: {str(e)}")
            return [{
                "type": "numerical",
                "current_value": "20%",
                "predicted_value": "40%",
                "confidence": 0.7
            }]

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
        """Obter previsões do modelo Azure OpenAI"""
        try:
            import aiohttp
            import json
            
            endpoint = f"{self.azure_openai_endpoint}/openai/deployments/gpt-35-turbo/completions?api-version=2023-05-15"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": self.azure_openai_key
            }
            
            payload = {
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": None
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        prediction_text = result.get('choices', [{}])[0].get('text', '').strip()
                        
                        try:
                            start_index = prediction_text.find('[')
                            end_index = prediction_text.rfind(']') + 1
                            
                            if start_index >= 0 and end_index > start_index:
                                json_str = prediction_text[start_index:end_index]
                                predictions = json.loads(json_str)
                                return predictions
                            else:
                                print("Formato JSON não encontrado na resposta")
                                return self._generate_fallback_predictions(numerical_changes, latest_regulation)
                        except json.JSONDecodeError:
                            print(f"Erro ao decodificar JSON da resposta: {prediction_text}")
                            return self._generate_fallback_predictions(numerical_changes, latest_regulation)
                    else:
                        print(f"Erro na API OpenAI: {response.status} - {await response.text()}")
                        return self._generate_fallback_predictions(numerical_changes, latest_regulation)
        except Exception as e:
            print(f"Erro ao chamar API OpenAI: {str(e)}")
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

    def parse_json_safely(self, json_str):
        """Analisa uma string JSON de forma segura, retornando um dicionário vazio em caso de erro"""
        if not json_str:
            return {}
            
        try:
            if isinstance(json_str, dict):
                return json_str
            
            if isinstance(json_str, str):
                import json
                return json.loads(json_str)
        except Exception as e:
            print(f"Erro ao analisar JSON: {str(e)}")
        
        return {}

    def visualize_knowledge_graph(self):
        """Visualizar o grafo de conhecimento"""
        if len(self.knowledge_graph.nodes) == 0:
            return "Grafo vazio - adicione algumas regulações primeiro."
        
        plt.figure(figsize=(12, 8))
        
        pos = nx.spring_layout(self.knowledge_graph)
        
        nx.draw_networkx_nodes(self.knowledge_graph, pos, node_size=700, node_color='lightblue')
        
        edges = self.knowledge_graph.edges(data=True)
        edge_colors = [data['similarity'] for _, _, data in edges]
        
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
            similarity = edge[2]['similarity']
            
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
    
    def deploy_to_azure(self):
        """Implantar o modelo na Azure"""
        container_name = "regulation-analysis-model"
        blob_service_client = self.blob_service_client
        
        try:
            container_client = blob_service_client.create_container(container_name)
        except:
            container_client = blob_service_client.get_container_client(container_name)
        
        model_info = {
            "model_version": "1.0",
            "created_date": datetime.now().isoformat(),
            "num_regulations": len(self.knowledge_graph.nodes),
            "model_type": "Regulatory Change Analysis"
        }
        
        model_info_json = json.dumps(model_info)
        blob_client = container_client.get_blob_client("model_info.json")
        blob_client.upload_blob(model_info_json, overwrite=True)
        
        if os.path.exists("knowledge_graph.png"):
            with open("knowledge_graph.png", "rb") as data:
                blob_client = container_client.get_blob_client("knowledge_graph.png")
                blob_client.upload_blob(data.read(), overwrite=True)
        
        if os.path.exists("knowledge_graph_interactive.html"):
            with open("knowledge_graph_interactive.html", "rb") as data:
                blob_client = container_client.get_blob_client("knowledge_graph_interactive.html")
                blob_client.upload_blob(data.read(), overwrite=True)
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Análise de Mudanças Regulatórias</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { padding: 20px; }
                .container { max-width: 1200px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Dashboard de Análise de Mudanças Regulatórias</h1>
                
                <div class="row mt-4">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                Grafo de Conhecimento
                            </div>
                            <div class="card-body">
                                <img src="knowledge_graph.png" class="img-fluid" alt="Grafo de Conhecimento">
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                Previsões de Mudanças Futuras
                            </div>
                            <div class="card-body">
                                <div id="predictions-container">Carregando previsões...</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                Visualização Interativa
                            </div>
                            <div class="card-body">
                                <iframe src="knowledge_graph_interactive.html" width="100%" height="600" frameborder="0"></iframe>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                // Código JavaScript para carregar previsões por meio de uma API
                fetch('/api/predictions')
                    .then(response => response.json())
                    .then(data => {
                        const container = document.getElementById('predictions-container');
                        container.innerHTML = '';
                        
                        if (data.length === 0) {
                            container.innerHTML = '<p>Sem previsões disponíveis.</p>';
                            return;
                        }
                        
                        const list = document.createElement('ul');
                        list.className = 'list-group';
                        
                        data.forEach(prediction => {
                            const item = document.createElement('li');
                            item.className = 'list-group-item';
                            
                            if (prediction.current_value) {
                                item.innerHTML = `
                                    <strong>Valor atual:</strong> ${prediction.current_value}<br>
                                    <strong>Valor previsto:</strong> ${prediction.predicted_value}<br>
                                    <strong>Confiança:</strong> ${(prediction.confidence * 100).toFixed(1)}%
                                `;
                            } else {
                                item.innerHTML = `
                                    <strong>Texto atual:</strong> "${prediction.current_text}"<br>
                                    <strong>Texto previsto:</strong> "${prediction.predicted_text}"<br>
                                    <strong>Confiança:</strong> ${(prediction.confidence * 100).toFixed(1)}%
                                `;
                            }
                            
                            list.appendChild(item);
                        });
                        
                        container.appendChild(list);
                    })
                    .catch(error => {
                        console.error('Erro ao carregar previsões:', error);
                        document.getElementById('predictions-container').innerHTML = 
                            '<p class="text-danger">Erro ao carregar previsões. Tente novamente mais tarde.</p>';
                    });
            </script>
        </body>
        </html>
        """
        
        blob_client = container_client.get_blob_client("index.html")
        blob_client.upload_blob(html_content, overwrite=True)
        
        website_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/index.html"
        
        return {
            "status": "success",
            "message": "Modelo implantado com sucesso na Azure",
            "model_storage_url": f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/",
            "dashboard_url": website_url
        }
    
    async def close(self):
        """Encerrar conexões com os serviços"""
        if hasattr(self, 'gremlin_conn') and self.gremlin_conn:
            try:
                await self.gremlin_conn.close()
                self.gremlin_conn = None
                self.g = None
            except Exception as e:
                print(f"Erro ao fechar conexão Gremlin: {str(e)}")

async def main():
    analyzer = RegulatoryChangeAnalyzer()
    
    try:
        await analyzer.init_gremlin_connection()
        
        regulations = [
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
        for reg in regulations:
            await analyzer.add_to_knowledge_graph(reg["id"], reg["date"], reg["text"], prev_id)
            prev_id = reg["id"]
        
        analyzer.visualize_knowledge_graph()
        
        predictions = await analyzer.predict_future_changes()
        print("Previsões de mudanças futuras:")
        print(json.dumps(predictions, indent=2))
        
        deployment_result = analyzer.deploy_to_azure()
        print("Resultado da implantação:")
        print(json.dumps(deployment_result, indent=2))
        
    finally:
        await analyzer.close()
