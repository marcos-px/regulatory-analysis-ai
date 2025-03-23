import os
import json
import aiohttp
import requests
import time
import random
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AIProviderManager")

class AIProvider:
    """Classe base para provedores de IA generativa"""
    def __init__(self, name, endpoint, api_key, model, api_version="2023-05-15"):
        self.name = name
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.api_version = api_version
        self.last_error = None
        self.error_count = 0
        self.last_success = datetime.now()
        self.usage_count = 0
        self.available = True
    
    async def generate_completion(self, prompt, system_message="Você é um especialista em análise de tendências regulatórias.", temperature=0.7, max_tokens=800):
        """Método a ser sobrescrito por classes derivadas"""
        raise NotImplementedError("Este método deve ser implementado pelas subclasses")
    
    async def generate_embedding(self, text):
        """Método a ser sobrescrito por classes derivadas"""
        raise NotImplementedError("Este método deve ser implementado pelas subclasses")
    
    def mark_error(self, error_msg):
        """Marcar um erro neste provedor"""
        self.error_count += 1
        self.last_error = error_msg
        if self.error_count >= 3:
            self.available = False
            logger.warning(f"Provedor {self.name} marcado como indisponível após {self.error_count} erros. Último erro: {error_msg}")
    
    def mark_success(self):
        """Marcar um sucesso neste provedor"""
        self.error_count = 0
        self.last_success = datetime.now()
        self.usage_count += 1
        if not self.available:
            self.available = True
            logger.info(f"Provedor {self.name} restaurado após sucesso")
    
    def get_status(self):
        """Obter status do provedor"""
        return {
            "name": self.name,
            "available": self.available,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "usage_count": self.usage_count
        }

class AzureOpenAIProvider(AIProvider):
    """Provedor de IA usando Azure OpenAI API"""
    
    async def generate_completion(self, prompt, system_message="Você é um especialista em análise de tendências regulatórias.", temperature=0.7, max_tokens=800):
        """Gerar texto usando Azure OpenAI API"""
        try:
            endpoint = f"{self.endpoint}/openai/deployments/{self.model}/chat/completions?api-version={self.api_version}"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
            
            data = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            logger.info(f"Chamando API do provedor {self.name}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    status_code = response.status
                    response_text = await response.text()
                    
                    if status_code == 200:
                        result = json.loads(response_text)
                        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                        self.mark_success()
                        return content
                    else:
                        error_msg = f"Erro na API {self.name}: {status_code} - {response_text[:100]}"
                        logger.error(error_msg)
                        self.mark_error(error_msg)
                        return None
        except Exception as e:
            error_msg = f"Exceção ao chamar {self.name}: {str(e)}"
            logger.error(error_msg)
            self.mark_error(error_msg)
            return None
    
    async def generate_embedding(self, text):
        """Gerar embeddings usando Azure OpenAI API"""
        try:
            endpoint = f"{self.endpoint}/openai/deployments/{self.model}/embeddings?api-version={self.api_version}"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
            
            data = {
                "input": text
            }
            
            logger.info(f"Chamando API de embeddings do provedor {self.name}")
            
            response = requests.post(endpoint, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                self.mark_success()
                return result['data'][0]['embedding']
            else:
                error_msg = f"Erro na API de embeddings {self.name}: {response.status_code} - {response.text[:100]}"
                logger.error(error_msg)
                self.mark_error(error_msg)
                return None
        except Exception as e:
            error_msg = f"Exceção ao chamar embeddings {self.name}: {str(e)}"
            logger.error(error_msg)
            self.mark_error(error_msg)
            return None

class AzureAIServicesProvider(AIProvider):
    """Provedor de IA usando o novo Azure AI Services"""
    
    async def generate_completion(self, prompt, system_message="Você é um especialista em análise de tendências regulatórias.", temperature=0.7, max_tokens=800):
        """Gerar texto usando Azure AI Services API"""
        try:
            endpoint = f"{self.endpoint}/models/chat/completions?api-version={self.api_version}"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
            
            data = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            logger.info(f"Chamando API do provedor {self.name}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    status_code = response.status
                    response_text = await response.text()
                    
                    if status_code == 200:
                        result = json.loads(response_text)
                        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                        self.mark_success()
                        return content
                    else:
                        error_msg = f"Erro na API {self.name}: {status_code} - {response_text[:100]}"
                        logger.error(error_msg)
                        self.mark_error(error_msg)
                        return None
        except Exception as e:
            error_msg = f"Exceção ao chamar {self.name}: {str(e)}"
            logger.error(error_msg)
            self.mark_error(error_msg)
            return None
    
    async def generate_embedding(self, text):
        """Implementação básica - substituir pelo endpoint correto quando disponível"""
        logger.warning(f"Embeddings não implementado para {self.name} - usando método falso")
        return None

class AIProviderManager:
    """Gerenciador para múltiplos provedores de IA"""
    
    def __init__(self):
        self.completion_providers = []
        self.embedding_providers = []
        self.last_provider_index = 0
        
        self._load_providers_from_env()
    
    def _load_providers_from_env(self):
        """Carregar provedores a partir de variáveis de ambiente"""
        azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_openai_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_openai_model = os.environ.get("AZURE_OPENAI_COMPLETION_MODEL", "gpt-4o")
        
        if azure_openai_endpoint and azure_openai_key:
            provider = AzureOpenAIProvider(
                name="Azure OpenAI Primary",
                endpoint=azure_openai_endpoint,
                api_key=azure_openai_key,
                model=azure_openai_model
            )
            self.completion_providers.append(provider)
            
            azure_embedding_model = os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            if azure_embedding_model:
                embedding_provider = AzureOpenAIProvider(
                    name="Azure OpenAI Embedding",
                    endpoint=azure_openai_endpoint,
                    api_key=azure_openai_key,
                    model=azure_embedding_model
                )
                self.embedding_providers.append(embedding_provider)
        
        azure_openai_endpoint2 = os.environ.get("AZURE_OPENAI_ENDPOINT_2")
        azure_openai_key2 = os.environ.get("AZURE_OPENAI_API_KEY_2")
        azure_openai_model2 = os.environ.get("AZURE_OPENAI_COMPLETION_MODEL_2", "gpt-4o")
        
        if azure_openai_endpoint2 and azure_openai_key2:
            provider2 = AzureOpenAIProvider(
                name="Azure OpenAI Secondary",
                endpoint=azure_openai_endpoint2,
                api_key=azure_openai_key2,
                model=azure_openai_model2
            )
            self.completion_providers.append(provider2)
        
        azure_ai_endpoint = os.environ.get("AZURE_AI_SERVICES_ENDPOINT")
        azure_ai_key = os.environ.get("AZURE_AI_SERVICES_KEY")
        azure_ai_version = os.environ.get("AZURE_AI_SERVICES_VERSION", "2024-05-01-preview")
        
        if azure_ai_endpoint and azure_ai_key:
            ai_provider = AzureAIServicesProvider(
                name="Azure AI Services",
                endpoint=azure_ai_endpoint,
                api_key=azure_ai_key,
                model="chat",
                api_version=azure_ai_version
            )
            self.completion_providers.append(ai_provider)
        
        if not self.completion_providers:
            logger.warning("Nenhum provedor de IA configurado - criando provedor simulado para testes")
            dummy_provider = AzureOpenAIProvider(
                name="Dummy Provider",
                endpoint="https://example.com",
                api_key="dummy-key",
                model="dummy-model"
            )
            self.completion_providers.append(dummy_provider)
        
        if not self.embedding_providers and self.completion_providers:
            logger.warning("Nenhum provedor de embeddings configurado - usando o primeiro provedor de completion")
            self.embedding_providers.append(self.completion_providers[0])
        
        logger.info(f"Inicializado com {len(self.completion_providers)} provedores de completion e {len(self.embedding_providers)} provedores de embedding")
    
    def add_completion_provider(self, provider):
        """Adicionar um novo provedor de completion"""
        self.completion_providers.append(provider)
    
    def add_embedding_provider(self, provider):
        """Adicionar um novo provedor de embedding"""
        self.embedding_providers.append(provider)
    
    def _get_next_available_provider(self, providers):
        """Obter o próximo provedor disponível usando round-robin"""
        if not providers:
            return None
        
        available_providers = [p for p in providers if p.available]
        if not available_providers:
            logger.warning("Nenhum provedor disponível, tentando reativar todos")
            for p in providers:
                p.available = True
                p.error_count = 0
            available_providers = providers
        
        index = self.last_provider_index % len(available_providers)
        provider = available_providers[index]
        self.last_provider_index += 1
        
        return provider
    
    async def generate_completion(self, prompt, system_message="Você é um especialista em análise de tendências regulatórias.", temperature=0.7, max_tokens=800, max_attempts=3):
        """Gerar completion usando o próximo provedor disponível"""
        attempts = 0
        
        while attempts < max_attempts:
            provider = self._get_next_available_provider(self.completion_providers)
            
            if not provider:
                logger.error("Nenhum provedor de completion disponível")
                return None
            
            logger.info(f"Tentativa {attempts+1}/{max_attempts} com provedor {provider.name}")
            result = await provider.generate_completion(prompt, system_message, temperature, max_tokens)
            
            if result is not None:
                return result
            
            attempts += 1
        
        logger.error(f"Falha em todas as {max_attempts} tentativas de completion")
        return None
    
    async def generate_embedding(self, text, max_attempts=3):
        """Gerar embedding usando o próximo provedor disponível"""
        attempts = 0
        
        while attempts < max_attempts:
            provider = self._get_next_available_provider(self.embedding_providers)
            
            if not provider:
                logger.error("Nenhum provedor de embedding disponível")
                return None
            
            logger.info(f"Tentativa de embedding {attempts+1}/{max_attempts} com provedor {provider.name}")
            result = await provider.generate_embedding(text)
            
            if result is not None:
                return result
            
            attempts += 1
        
        logger.error(f"Falha em todas as {max_attempts} tentativas de embedding")
        return None
    
    def get_status(self):
        """Obter status de todos os provedores"""
        completion_status = [p.get_status() for p in self.completion_providers]
        embedding_status = [p.get_status() for p in self.embedding_providers]
        
        return {
            "completion_providers": completion_status,
            "embedding_providers": embedding_status,
            "last_provider_index": self.last_provider_index
        }