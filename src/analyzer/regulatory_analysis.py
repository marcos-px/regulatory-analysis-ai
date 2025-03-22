import os
import re
import spacy
import json
# import pandas as pd
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