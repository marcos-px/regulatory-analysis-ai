import os
import re
import spacy
import json
# import pandas as pd
# import numpy as np
import networkx as nx
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
from openai import AzureOpenAI

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

    
    
        