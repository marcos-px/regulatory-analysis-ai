from fastapi import FastAPI, HTTPException, Body, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import io
import os
import json
from datetime import datetime
import base64
import uvicorn
from .models.regulation_model import Regulation
from .models.prediction_request_model import PredictionRequest
from .models.regulation_comparison_model import RegulationComparison
from .models.similarity_response_model import SimilarityResponse
from dotenv import load_dotenv
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analyzer.regulatory_analysis import RegulatoryChangeAnalyzer

load_dotenv()

app = FastAPI(
    title="API de Análise Regulatória",
    description="API para análise e previsão de mudanças em textos regulatórios",
    version="1.0.0"
)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = None

def get_analyzer():
    global analyzer
    if analyzer is None:
        analyzer = RegulatoryChangeAnalyzer()
    return analyzer

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.get("/regulations")
async def get_regulations(analyzer: RegulatoryChangeAnalyzer = Depends(get_analyzer)):
    try:
        regulations = await analyzer.get_all_regulations()
        if not regulations:
            regulations = []
            for node in analyzer.knowledge_graph.nodes:
                node_data = analyzer.knowledge_graph.nodes[node]
                regulations.append({
                    "id": node,
                    "date": node_data.get("date"),
                    "text": node_data.get("text"),
                    "key_phrases": node_data.get("key_phrases", [])
                })
        
        return {
            "status": "success",
            "count": len(regulations),
            "regulations": regulations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/regulations/{regulation_id}")
async def get_regulation(regulation_id: str, analyzer: RegulatoryChangeAnalyzer = Depends(get_analyzer)):
    """Obter uma regulação específica pelo ID"""
    try:
        query = f"g.V().has('regulation', 'id', '{regulation_id}')" + \
                ".project('id', 'date', 'text', 'key_phrases')" + \
                ".by('id').by('date').by('text').by('key_phrases')"
        
        result = None
        
        try:
            results = await analyzer.g.V().has('regulation', 'id', regulation_id) \
                    .project('id', 'date', 'text', 'key_phrases') \
                    .by('id').by('date').by('text').by('key_phrases') \
                    .toList()
            if results and len(results) > 0:
                result = results[0]
        except:
            pass
            
        if not result and regulation_id in analyzer.knowledge_graph.nodes:
            node_data = analyzer.knowledge_graph.nodes[regulation_id]
            result = {
                "id": regulation_id,
                "date": node_data.get("date"),
                "text": node_data.get("text"),
                "key_phrases": node_data.get("key_phrases", [])
            }
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Regulação {regulation_id} não encontrada")
        
        predecessors = []
        successors = []
        
        try:
            predecessors = await analyzer.g.V().has('regulation', 'id', regulation_id) \
                        .in_('changed_to').values('id').toList()
            successors = await analyzer.g.V().has('regulation', 'id', regulation_id) \
                        .out('changed_to').values('id').toList()
        except:
            predecessors = list(analyzer.knowledge_graph.predecessors(regulation_id))
            successors = list(analyzer.knowledge_graph.successors(regulation_id))
        
        result["previous_regulations"] = predecessors
        result["next_regulations"] = successors
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/compare")
async def compare_regulations(
    comparison: RegulationComparison, 
    use_gpt: bool = True,
    analyzer: RegulatoryChangeAnalyzer = Depends(get_analyzer)
):
    """Comparar dois textos regulatórios e calcular similaridade"""
    try:
        text1 = comparison.text1
        text2 = comparison.text2
        
        similarity = analyzer.calculate_similarity(text1, text2)
        
        changes = analyzer.extract_key_changes(text1, text2)
        
        key_differences = []
        
        diff_blocks = changes.get("text_diff_blocks", [])
        if isinstance(diff_blocks, list):
            for diff_block in diff_blocks:
                if isinstance(diff_block, dict):
                    opcode = diff_block.get("opcode")
                    old_text = diff_block.get("old_text", "")
                    new_text = diff_block.get("new_text", "")
                    
                    if opcode == "replace":
                        key_differences.append(f"Substituição: '{old_text}' -> '{new_text}'")
                    elif opcode == "insert":
                        key_differences.append(f"Inserção: '{new_text}'")
                    elif opcode == "delete":
                        key_differences.append(f"Remoção: '{old_text}'")
        
        numerical_changes = changes.get("numerical_changes", [])
        if isinstance(numerical_changes, list):
            for pair in numerical_changes:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    old_val, new_val = pair
                    key_differences.append(f"Alteração numérica: {old_val} -> {new_val}")
        
        analysis = None
        if use_gpt:
            analysis = await analyzer._get_comparison_analysis_with_gpt(text1, text2, similarity, key_differences)
        
        return {
            "similarity": similarity,
            "key_differences": key_differences,
            "raw_changes": changes,
            "gpt_analysis": analysis
        }
    except Exception as e:
        import traceback
        print(f"Erro ao comparar regulações: {str(e)}")
        print(traceback.format_exc())
        
        return {
            "similarity": 0.5,
            "key_differences": ["Erro ao processar diferenças: " + str(e)],
            "raw_changes": {},
            "gpt_analysis": None
        }

@app.post("/predictions")
async def get_predictions(
    request: PredictionRequest = Body(...),
    analyzer: RegulatoryChangeAnalyzer = Depends(get_analyzer)
):
    """Obter previsões de mudanças futuras com base em um texto específico"""
    try:
        text = request.text if hasattr(request, 'text') else None
        num_predictions = request.num_predictions or 1

        if text:
            predictions = await analyzer.predict_future_changes_with_text(text, num_predictions)
        else:
            predictions = await analyzer.predict_future_changes_with_gpt(num_predictions)
            
        return predictions
    except Exception as e:
        import traceback
        print(f"Erro ao gerar previsões: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-graph")
async def get_knowledge_graph(analyzer: RegulatoryChangeAnalyzer = Depends(get_analyzer)):
    """Obter o grafo de conhecimento em formato JSON"""
    try:
        return await analyzer.get_knowledge_graph()
    except Exception as e:
        print(f"Erro ao obter grafo de conhecimento: {str(e)}")
        return {"nodes": [], "edges": []}


@app.get("/visualize")
async def visualize_graph(analyzer: RegulatoryChangeAnalyzer = Depends(get_analyzer)):
    """Gerar e obter uma visualização do grafo de conhecimento"""
    try:
        result = analyzer.visualize_knowledge_graph()
        
        graph_image = None
        if os.path.exists("knowledge_graph.png"):
            with open("knowledge_graph.png", "rb") as img_file:
                graph_image = base64.b64encode(img_file.read()).decode('utf-8')
        
        return {
            "status": "success",
            "message": result,
            "graph_image": graph_image
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deploy")
async def deploy_to_azure(analyzer: RegulatoryChangeAnalyzer = Depends(get_analyzer)):
    """Implantar o modelo na Azure"""
    try:
        result = analyzer.deploy_to_azure()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-regulation")
async def upload_regulation(
    file: UploadFile = File(...),
    regulation_id: Optional[str] = None,
    date: Optional[str] = None,
    previous_id: Optional[str] = None,
    analyzer: RegulatoryChangeAnalyzer = Depends(get_analyzer)
):
    """Fazer upload de um arquivo contendo texto regulatório"""
    try:
        content = await file.read()
        text = content.decode("utf-8")
        
        if regulation_id is None:
            regulation_id = f"reg_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        if date is None:
            date = datetime.now().strftime("%B de %Y")
        
        await analyzer.add_to_knowledge_graph(regulation_id, date, text, previous_id)
        
        return {
            "status": "success",
            "message": f"Regulação {regulation_id} adicionada com sucesso",
            "regulation_id": regulation_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Eventos na inicialização da API"""
    global analyzer
    analyzer = RegulatoryChangeAnalyzer()
    
    print("Inicializando API de Análise Regulatória")
    
    try:
        existing_regulations = await analyzer.get_all_regulations()
        
        if existing_regulations:
            print(f"Carregadas {len(existing_regulations)} regulações existentes")
        else:
            print("Nenhuma regulação encontrada")
            
        print("Inicialização concluída com sucesso!")
    except Exception as e:
        print(f"Erro ao inicializar: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Eventos no encerramento da API"""
    global analyzer
    if analyzer:
        await analyzer.close()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)