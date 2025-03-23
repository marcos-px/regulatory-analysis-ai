import io
import aiohttp

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

async def _get_comparison_analysis_with_gpt(self, text1, text2, similarity, key_differences):
    """Obter análise detalhada da comparação usando GPT-4o"""
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

        endpoint = f"{self.azure_openai_endpoint}/openai/deployments/{self.azure_openai_completion_model}/chat/completions?api-version=2023-05-15"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.azure_openai_key
        }
        
        data = {
            "messages": [
                {"role": "system", "content": "Você é um especialista em análise de tendências regulatórias."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 800
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
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
                        print(f"Erro ao processar análise GPT: {str(e)}")
                        return None
                else:
                    error_text = await response.text()
                    print(f"Erro na API OpenAI: {response.status} - {error_text}")
                    return None
    except Exception as e:
        print(f"Erro ao obter análise GPT: {str(e)}")
        return None