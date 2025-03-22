import os
import re
import spacy
import json
import pandas as pd
import numpy as np
import networkx as nx
import torch 
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

from azure.ai.textanalytics import TextAnalyticsClient
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient

from gremlin_python.driver import client
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.strategies import *
from gremlin_python.structure.graph import Graph


class RegulatoryChangeAnalyser:
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

        self.azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_key = os.environ.get("AZURE_OPENAI_KEY")
        self.azure_openai_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        
        self.openai_client = AzureOpenAI(
            api_key=self.azure_openai_key,
            api_version="2024-02-01",
            azure_endpoint=self.azure_openai_endpoint
        )

        self.text_analytics_client = TextAnalyticsClient(
            endpoint=self.azure_language_endpoint,
            credential=AzureKeyCredential(self.azure_language_key)
        )

        self.blob_service_client = BlobServiceClient.from_connection_string(self.azure_storage_connection_string)

        self.gremlin_client = client.Client(
            self.gremlin_endpoint,
            'g',
            username=f"/dbs/{self.gremlin_database}/colls/{self.gremlin_collection}",
            password=self.gremlin_key,
            message_serializer=client.serializer.GraphSONSerializersV2d0()
        )
        self.knowledge_graph = nx.DiGraph()
    
    def preprocess_text(self,text):
        doc = self.nlp(text)
        processed_text = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
        return processed_text

    def get_text_embeddings(self, text):
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                deployment_id = self.azure_openai_deployment,
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error getting text embeddings with Azure OpenAI: {e}")
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=Tru, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

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

        modified_entities = []
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
            response = self.text_analytics_client.extract_key_phrases(documents={
                "id": "1",
                "language": "pt",
                "text": text
            })
            if not response or len(response) == 0:
                return []

            doc_result = response[0]
            if not doc_result or doc_result.is_error:
                return []

            return doc_result.key_phrases

        except Exception as e:
            print(f"Error analyzing key phrases with Azure Language Service: {e}")
            return []

    
    def add_to_knowledge_graph(self, regulation_id, date, text, previous_regulation_id=None):

        key_phrases = self.analyze_key_phares(text)
        embedding = self.get_text_embeddings(text)

        self.knowledge_graph.add_node(
            regulation_id, 
            date=date, 
            text=text,
            key_phrases=key_phrases
            )

        query = "g.addV('regulation')" + \
                f".property('id', '{regulation_id}')" + \
                f".property('pk', '{regulation_id}')" + \
                f".property('date', '{date}')" + \
                f".property('text', '{text}')" + \
                f".property('key_phrases', '{json.dumps(key_phrases)}')"

        try:
            self.gremlin_client.submit(query).all().result()
            print(f"Regulation {regulation_id} added to knowledge graph")
        except Exception as e:
            print(f"Error adding regulation to knowledge graph vertice:{str(e)}")

        container_name = "embeddings"
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
        except:
            container_client = self.blob_service_client.create_container(container_name)

        blob_name = f"{regulation_id}_embedding.npy"
        blob_client = container_client.get_blob_client(blob_name)

        with open("temp_embedding.npy", "wb") as f:
            np.save(f, embedding)
        
        with open("temp_embedding.npy", "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        os.remove("temp_embedding.npy")

        if previous_regulation_id:
            prev_text = self.knowledge_graph.nodes[previous_regulation_id]['text']
            similarity = self.calculate_similarity(prev_text, text)

            changes = self.extract_key_changes(prev_text, text)

            self.knowledge_graph.add_edge(
                previous_regulation_id,
                regulation_id,
                similarity=similarity,
                changes=changes
            )

        serializable_changes = {
            'numerical_changes': str(changes.get('numerical_changes', [])),
            'text_blocks': str(changes.get('text_diff_blocks', [])),
        }

        edge_query = f"g.V().has('regulation', 'id', '{previous_regulation_id}')" + \
                    f".addE('changed_to')" + \
                    f".property('similarity', {similarity})" + \
                    f".property('changes', '{json.dumps(serializable_changes)}')" + \
                    f".to(g.V().has('regulation', 'id', '{regulation_id}'))"

        try:
            self.gremlin_client.submit(edge_query).all().result()
            print(f"Adicionada relação: {previous_regulation_id} -> {regulation_id}")
        except Exception as e:
            print(f"Erro ao adicionar aresta: {str(e)}")

    def get_regulation_path(self, start_id, end_id=None):
        if end_id:
            query = f"g.V().has('regulation', 'id', '{start_id}')" + \
                        f".repeat(out('changed_to').simplePath())" + \
                        f".until(has('id', '{end_id}'))" + \
                        f".path().by('id')"
        else:
            query = f"g.V().has('regulation', 'id', '{start_id}')" + \
                    f".repeat(out('changed_to').simplePath())" + \
                    f".emit().path().by('id')"
        
        try:
            results = self.gremlin_client.submit(query).all().result()
            return results
        except Exception as e:
            print(f"Error getting regulation path-way: {str(e)}")
            return None
    
    def predict_future_changes(self, num_predictions=1):

        if len(self.knowledge_graph.nodes) < 2:
            print("Not enough regulations to predict future changes")
            return "Not enough regulations to predict future changes"
        
        query = "g.E().hasLabel('changed_to')" + \
        ".project('from', 'to', 'similarity', 'changes')" + \
        ".by(outV().values('id'))" + \
        ".by(inV().values('id'))" + \
        ".by('similarity')" + \
        ".by('changes')"

        try:
            changes_over_time = self.gremlin_client.submit(query).all().result()
        except Exception as e:
            print(f"Erro ao obter mudanças históricas: {str(e)}")
            changes_over_time = []


