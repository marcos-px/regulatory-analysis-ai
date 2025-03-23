const API_BASE_URL = 'http://191.232.163.4:8000';

let state = {
    regulations: [],
    graphData: null,
    selectedRegulation: null,
    predictions: [],
    selectedText: null
};

function checkElementExists(id) {
    const element = document.getElementById(id);
    if (!element) {
        console.error(`Elemento com ID "${id}" não foi encontrado!`);
        return false;
    }
    console.log(`Elemento com ID "${id}" encontrado.`);
    return true;
}

async function getPredictionsWithoutModal() {
    try {
        const text = document.getElementById('text2').value;
        if (!text) {
            throw new Error('Texto modificado não disponível para previsão');
        }
        
        state.selectedText = text;
        
        const response = await fetch(`${API_BASE_URL}/predictions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                num_predictions: 3
            })
        });
        
        if (!response.ok) {
            throw new Error(`Erro na API: ${response.status} - ${response.statusText}`);
        }
        
        const predictions = await response.json();
        console.log('Previsões recebidas:', predictions);
        
        state.predictions = Array.isArray(predictions) ? predictions : [predictions];
        
        displayPredictions(state.predictions);
        updateRecentPredictions();
        
    } catch (error) {
        console.error('Erro ao gerar previsões:', error);
        
        const container = document.getElementById('predictions-container');
        if (container) {
            container.innerHTML = `
                <div class="alert alert-warning">
                    <h5><i class="bi bi-exclamation-triangle me-2"></i>Não foi possível gerar previsões</h5>
                    <p>Ocorreu um erro ao tentar gerar previsões: ${error.message}</p>
                    <p>Por favor, tente novamente ou contate o suporte se o problema persistir.</p>
                </div>
            `;
        }
    }
}


document.addEventListener('DOMContentLoaded', function() {

    const gnnVisualizer = new GNNVisualizer('gnn-visualizer');
        gnnVisualizer.fetchGNNData().then(() => {
            gnnVisualizer.visualize();
        });
    
        document.getElementById('train-gnn-btn').addEventListener('click', async function() {
            console.log("Botão de treinar GNN foi clicado!");
            try {
                showLoadingModal("Treinando modelo GNN...");
                
                console.log("Enviando requisição para enriquecer embeddings...");
                const response = await fetch(`${API_BASE_URL}/enrich-embeddings`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    }
                });
                
                console.log("Resposta recebida:", response.status);
                
                if (!response.ok) {
                    const errorText = await response.text();
                    console.error(`Erro HTTP: ${response.status} - ${errorText}`);
                    throw new Error(`Erro ao treinar GNN: ${response.status}. Detalhes: ${errorText}`);
                }
                
                const result = await response.json();
                console.log("Resultado do treinamento:", result);
                
                alert(`Modelo GNN treinado com sucesso! Processados ${result.node_count} nós.`);
                
                // Initialize the GNN visualizer again with the new data
                console.log("Reinicializando o visualizador GNN...");
                const gnnVisualizer = new GNNVisualizer('gnn-visualizer');
                await gnnVisualizer.fetchGNNData();
                gnnVisualizer.visualize();
                
                hideLoadingModal();
            } catch (error) {
                console.error("Erro ao treinar GNN:", error);
                alert(`Erro ao treinar o modelo GNN: ${error.message}`);
                hideLoadingModal();
            }
        });
        
    const cancelBtn = document.getElementById('cancel-loading-btn');
    if (cancelBtn) {
        cancelBtn.addEventListener('click', hideLoadingModal);
    }
    
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape' || event.keyCode === 27) {
        }
    });
    
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const section = this.getAttribute('data-section');
            navigateTo(section);
        });
    });
    
    const uploadBtn = document.getElementById('upload-btn');
    const uploadModal = new bootstrap.Modal(document.getElementById('upload-modal'));
    
    uploadBtn.addEventListener('click', function() {
        const previousRegSelect = document.getElementById('previous-regulation');
        previousRegSelect.innerHTML = '<option value="">Nenhuma (nova regulação)</option>';
        
        state.regulations.forEach(reg => {
            const option = document.createElement('option');
            option.value = reg.id;
            option.textContent = `${reg.id} (${reg.date})`;
            previousRegSelect.appendChild(option);
        });
        
        uploadModal.show();
    });
    
    document.getElementById('submit-upload').addEventListener('click', function() {
        const regulationId = document.getElementById('regulation-id').value;
        const date = document.getElementById('regulation-date').value;
        const previousId = document.getElementById('previous-regulation').value;
        const file = document.getElementById('regulation-file').files[0];
        const text = document.getElementById('regulation-text').value;
        
        if (!file && !text) {
            alert('Por favor, forneça um arquivo ou texto da regulação.');
            return;
        }
        
        const uploadModalElement = document.getElementById('upload-modal');
        const uploadModalInstance = bootstrap.Modal.getInstance(uploadModalElement);
        if (uploadModalInstance) {
            uploadModalInstance.hide();
        }
        
        const container = document.getElementById('mini-graph-container');
        container.innerHTML = `
            <div class="text-center my-5">
                <div class="spinner-border text-primary mb-3" role="status">
                    <span class="visually-hidden">Carregando...</span>
                </div>
                <h5>Enviando regulação...</h5>
            </div>
        `;
        
        if (file) {
            uploadRegulationFile(file, regulationId, date, previousId);
        } else {
            uploadRegulationText(text, regulationId, date, previousId);
        }
    });
    
    document.getElementById('compare-btn').addEventListener('click', function() {
        const text1 = document.getElementById('text1').value;
        const text2 = document.getElementById('text2').value;
        
        if (!text1 || !text2) {
            alert('Por favor, insira ambos os textos para comparação.');
            return;
        }
        
        compareTexts(text1, text2);
    });

    document.getElementById('generate-predictions-btn').addEventListener('click', function() {
        const text2 = document.getElementById('text2').value;
        
        if (!text2) {
            alert("Por favor, insira o texto modificado para gerar previsões.");
            return;
        }
        
        const container = document.getElementById('predictions-container');
        container.innerHTML = `
            <div class="text-center my-5">
                <div class="spinner-border text-primary mb-3" role="status">
                    <span class="visually-hidden">Carregando...</span>
                </div>
                <h5>Gerando previsões...</h5>
            </div>
        `;
        
        getPredictionsWithoutModal();
    });
    
    document.getElementById('reg-select-1').addEventListener('change', function() {
        const selectedId = this.value;
        if (selectedId) {
            const regulation = state.regulations.find(reg => reg.id === selectedId);
            document.getElementById('text1').value = regulation.text;
        }
    });
    
    document.getElementById('reg-select-2').addEventListener('change', function() {
        const selectedId = this.value;
        if (selectedId) {
            const regulation = state.regulations.find(reg => reg.id === selectedId);
            document.getElementById('text2').value = regulation.text;
        }
    });
    
    document.getElementById('prediction-source').addEventListener('change', function() {
        const selectedId = this.value;
        if (selectedId) {
            const regulation = state.regulations.find(reg => reg.id === selectedId);
            document.getElementById('prediction-text').value = regulation.text;
            state.selectedRegulation = regulation;
        } else {
            document.getElementById('prediction-text').value = '';
            state.selectedRegulation = null;
        }
    });
    
    document.getElementById('predict-btn').addEventListener('click', function() {
        if (!state.selectedRegulation) {
            alert('Por favor, selecione uma regulação para fazer previsões.');
            return;
        }
        
        const container = document.getElementById('predictions-container');
        container.innerHTML = `
            <div class="text-center my-5">
                <div class="spinner-border text-primary mb-3" role="status">
                    <span class="visually-hidden">Carregando...</span>
                </div>
                <h5>Gerando previsões...</h5>
            </div>
        `;
        
        getPredictionsWithoutModal();
    });
    
    document.getElementById('reset-view-btn').addEventListener('click', function() {
        initializeKnowledgeGraph(state.graphData);
    });
    
    document.getElementById('zoom-range').addEventListener('input', function() {
        const zoom = parseFloat(this.value);
        updateGraphZoom(zoom);
    });
    
    document.getElementById('show-labels').addEventListener('change', function() {
        const showLabels = this.checked;
        toggleGraphLabels(showLabels);
    });
    
    document.getElementById('highlight-changes').addEventListener('change', function() {
        const highlightChanges = this.checked;
        toggleHighlightChanges(highlightChanges);
    });
    
    fetchRegulations();
    fetchGraphData();
    
    updateStatistics();
});

function navigateTo(sectionId) {
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    
    document.getElementById(sectionId).classList.add('active');
    
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('data-section') === sectionId) {
            link.classList.add('active');
        }
    });
    
    if (sectionId === 'regulations') {
        updateRegulationsTable();
    } else if (sectionId === 'visualize') {
        if (state.graphData) {
            initializeKnowledgeGraph(state.graphData);
        } else {
            fetchGraphData();
        }
    } else if (sectionId === 'analyze') {
        initializeAnalytics();
    } else if (sectionId === 'compare') {
        document.getElementById('comparison-results').style.display = 'none';
        document.getElementById('predictions-results').style.display = 'none';
    }
}

async function fetchRegulations() {
    try {
        console.log("Fetching regulations from API...");
        const response = await fetch(`${API_BASE_URL}/regulations`);
        
        if (!response.ok) {
            console.error(`API Error: ${response.status} - ${response.statusText}`);
            throw new Error(`Failed to fetch regulations: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log(`Loaded ${data.regulations.length} regulations`);
        state.regulations = data.regulations;
        
        updateRegulationSelects();
        if (document.getElementById('regulations').classList.contains('active')) {
            updateRegulationsTable();
        }
        updateStatistics();
        initializeMiniGraph();
        
        return data.regulations;
    } catch (error) {
        console.error('Error fetching regulations:', error);
        
        const container = document.getElementById('mini-graph-container');
        if (container) {
            container.innerHTML = `
                <div class="alert alert-danger">
                    <h5><i class="bi bi-exclamation-triangle me-2"></i>Error Loading Data</h5>
                    <p>${error.message}</p>
                    <button class="btn btn-primary mt-2" onclick="fetchRegulations()">Retry</button>
                </div>
            `;
        }
        
        return [];
    }
}

async function fetchGraphData() {
    try {
        const response = await fetch(`${API_BASE_URL}/knowledge-graph`);
        if (!response.ok) throw new Error('Falha ao obter dados do grafo');
        
        const data = await response.json();
        state.graphData = data;
        
        if (document.getElementById('visualize').classList.contains('active')) {
            initializeKnowledgeGraph(data);
        }
        
        initializeMiniGraph();
    } catch (error) {
        console.error('Erro ao buscar dados do grafo:', error);
        document.getElementById('knowledge-graph-container').innerHTML = 
            '<div class="alert alert-danger">Erro ao carregar o grafo de conhecimento.</div>';
    }
}

async function getPredictions() {
    return getPredictionsWithoutModal();
}

async function compareTexts(text1, text2) {
    const resultsDiv = document.getElementById('comparison-results');
    if (!resultsDiv) {
        console.error('Elemento comparison-results não encontrado');
        return;
    }
    
    resultsDiv.style.display = 'block';
    resultsDiv.innerHTML = `
        <div class="text-center my-5">
            <div class="spinner-border text-primary mb-3" role="status">
                <span class="visually-hidden">Carregando...</span>
            </div>
            <h5>Comparando textos...</h5>
        </div>
    `;
    
    try {
        const response = await fetch(`${API_BASE_URL}/compare`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text1: text1,
                text2: text2
            })
        });
        
        if (!response.ok) throw new Error('Falha ao comparar textos');
        
        const result = await response.json();
        displayComparisonResults(result);
    } catch (error) {
        console.error('Erro ao comparar textos:', error);
        
        if (resultsDiv) {
            resultsDiv.innerHTML = `
                <div class="alert alert-danger">
                    <h5><i class="bi bi-exclamation-triangle me-2"></i>Erro</h5>
                    <p>Não foi possível comparar os textos: ${error.message}</p>
                    <button class="btn btn-primary mt-2" onclick="document.getElementById('compare-btn').click()">Tentar Novamente</button>
                </div>
            `;
        }
    }
}

async function uploadRegulationFile(file, regulationId, date, previousId) {
    const formData = new FormData();
    formData.append('file', file);
    
    if (regulationId) formData.append('regulation_id', regulationId);
    if (date) formData.append('date', date);
    if (previousId) formData.append('previous_id', previousId);
    
    try {
        const response = await fetch(`${API_BASE_URL}/upload-regulation`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Falha ao enviar regulação');
        
        const result = await response.json();
        alert(`Regulação ${result.regulation_id} adicionada com sucesso!`);
        
        fetchRegulations();
        fetchGraphData();
    } catch (error) {
        console.error('Erro ao enviar regulação:', error);
        alert('Não foi possível enviar a regulação. Verifique o console para mais detalhes.');
    }
}

async function uploadRegulationText(text, regulationId, date, previousId) {
    try {
        const textBlob = new Blob([text], { type: 'text/plain' });
        
        const textFile = new File([textBlob], "regulation.txt", { type: 'text/plain' });
        
        const formData = new FormData();
        formData.append('file', textFile);
        
        if (regulationId) formData.append('regulation_id', regulationId);
        if (date) formData.append('date', date);
        if (previousId) formData.append('previous_id', previousId);
        
        const response = await fetch(`${API_BASE_URL}/upload-regulation`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Falha ao enviar regulação: ${response.status} - ${errorText}`);
        }
        
        const result = await response.json();
        alert(`Regulação ${result.regulation_id} adicionada com sucesso!`);
        
        fetchRegulations();
        fetchGraphData();
    } catch (error) {
        console.error('Erro ao enviar regulação:', error);
        alert('Não foi possível enviar a regulação. Verifique o console para mais detalhes.');
    }
}

function updateRegulationSelects() {
    const regSelect1 = document.getElementById('reg-select-1');
    const regSelect2 = document.getElementById('reg-select-2');
    const predictionSource = document.getElementById('prediction-source');
    
    regSelect1.innerHTML = '<option value="">Selecionar regulação...</option>';
    regSelect2.innerHTML = '<option value="">Selecionar regulação...</option>';
    predictionSource.innerHTML = '<option value="">Selecione uma regulação...</option>';
    
    state.regulations.forEach(reg => {
        const option1 = document.createElement('option');
        option1.value = reg.id;
        option1.textContent = `${reg.id} (${reg.date})`;
        regSelect1.appendChild(option1);
        
        const option2 = document.createElement('option');
        option2.value = reg.id;
        option2.textContent = `${reg.id} (${reg.date})`;
        regSelect2.appendChild(option2);
        
        const option3 = document.createElement('option');
        option3.value = reg.id;
        option3.textContent = `${reg.id} (${reg.date})`;
        predictionSource.appendChild(option3);
    });
}

function updateRegulationsTable() {
    const tableBody = document.getElementById('regulations-table');
    tableBody.innerHTML = '';
    
    if (state.regulations.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="5" class="text-center">Nenhuma regulação encontrada</td></tr>';
        return;
    }
    
    state.regulations.forEach(reg => {
        const row = document.createElement('tr');
        
        const idCell = document.createElement('td');
        idCell.textContent = reg.id;
        row.appendChild(idCell);
        
        const dateCell = document.createElement('td');
        dateCell.textContent = reg.date;
        row.appendChild(dateCell);
        
        const contentCell = document.createElement('td');
        contentCell.textContent = reg.text.length > 100 ? reg.text.substring(0, 100) + '...' : reg.text;
        row.appendChild(contentCell);
        
        const phrasesCell = document.createElement('td');
        if (reg.key_phrases && reg.key_phrases.length > 0) {
            const phrasesList = document.createElement('ul');
            phrasesList.className = 'mb-0';
            reg.key_phrases.slice(0, 3).forEach(phrase => {
                const item = document.createElement('li');
                item.textContent = phrase;
                phrasesList.appendChild(item);
            });
            if (reg.key_phrases.length > 3) {
                const more = document.createElement('li');
                more.textContent = `+ ${reg.key_phrases.length - 3} mais...`;
                more.className = 'text-muted';
                phrasesList.appendChild(more);
            }
            phrasesCell.appendChild(phrasesList);
        } else {
            phrasesCell.textContent = 'Nenhuma frase-chave disponível';
            phrasesCell.className = 'text-muted';
        }
        row.appendChild(phrasesCell);
        
        const actionsCell = document.createElement('td');
        
        const viewBtn = document.createElement('button');
        viewBtn.className = 'btn btn-sm btn-primary me-1';
        viewBtn.innerHTML = '<i class="bi bi-eye"></i>';
        viewBtn.addEventListener('click', () => showRegulationDetails(reg));
        actionsCell.appendChild(viewBtn);
        
        const compareBtn = document.createElement('button');
        compareBtn.className = 'btn btn-sm btn-info me-1';
        compareBtn.innerHTML = '<i class="bi bi-arrow-left-right"></i>';
        compareBtn.addEventListener('click', () => {
            document.getElementById('text1').value = reg.text;
            navigateTo('compare');
        });
        actionsCell.appendChild(compareBtn);
        
        const predictBtn = document.createElement('button');
        predictBtn.className = 'btn btn-sm btn-warning';
        predictBtn.innerHTML = '<i class="bi bi-lightning"></i>';
        predictBtn.addEventListener('click', () => {
            document.getElementById('text2').value = reg.text;
            
            if (reg.previous_regulations && reg.previous_regulations.length > 0) {
                const prevId = reg.previous_regulations[0];
                const prevReg = state.regulations.find(r => r.id === prevId);
                if (prevReg) {
                    document.getElementById('text1').value = prevReg.text;
                }
            }
            
            navigateTo('compare');
            
            document.getElementById('compare-btn').click();
        });
        actionsCell.appendChild(predictBtn);
        
        row.appendChild(actionsCell);
        
        tableBody.appendChild(row);
    });
}

function updateStatistics() {
    document.getElementById('reg-count').textContent = state.regulations.length;
    
    let changesCount = 0;
    if (state.graphData && state.graphData.edges) {
        changesCount = state.graphData.edges.length;
    }
    document.getElementById('changes-count').textContent = changesCount;
    
    const lastUpdate = state.regulations.length > 0 
        ? state.regulations[state.regulations.length - 1].date
        : 'N/A';
    document.getElementById('last-update').textContent = lastUpdate;
}

function updateRecentPredictions() {
    const container = document.getElementById('recent-predictions');
    
    if (!state.predictions || state.predictions.length === 0) {
        container.innerHTML = '<p class="text-center text-muted">Nenhuma previsão disponível</p>';
        return;
    }
    
    container.innerHTML = '';
    const predictions = state.predictions.slice(0, 3);
    
    predictions.forEach(prediction => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        
        let content = '';
        if (prediction.current_value) {
            content = `
                <div><strong>${prediction.current_value}</strong> → <strong class="text-primary">${prediction.predicted_value}</strong></div>
                <div class="small text-muted">Confiança: ${(prediction.confidence * 100).toFixed(1)}%</div>
                <div class="confidence-bar">
                    <div class="confidence-value" style="width: ${prediction.confidence * 100}%"></div>
                </div>
            `;
        } else {
            content = `
                <div class="text-truncate"><strong>"${prediction.current_text}"</strong></div>
                <div class="text-truncate text-primary">→ <strong>"${prediction.predicted_text}"</strong></div>
                <div class="small text-muted">Confiança: ${(prediction.confidence * 100).toFixed(1)}%</div>
                <div class="confidence-bar">
                    <div class="confidence-value" style="width: ${prediction.confidence * 100}%"></div>
                </div>
            `;
        }
        
        item.innerHTML = content;
        container.appendChild(item);
    });
}

function displayComparisonResults(result) {
    const resultsDiv = document.getElementById('comparison-results');
    if (!resultsDiv) {
        console.error('Elemento comparison-results não encontrado');
        return;
    }
    
    resultsDiv.style.display = 'block';
    
    resultsDiv.innerHTML = `
        <h4>Resultados da Comparação</h4>
        <div class="row">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">Similaridade</div>
                    <div class="card-body text-center">
                        <div class="similarity-gauge" id="similarity-gauge">
                            <div class="gauge-value">${(result.similarity * 100).toFixed(1)}%</div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">Diferenças Principais</div>
                    <div class="card-body">
                        <ul id="differences-list" class="list-group">
                            ${result.key_differences && result.key_differences.length > 0 ? 
                                result.key_differences.map(diff => 
                                    `<li class="list-group-item">${diff}</li>`
                                ).join('') 
                                : 
                                '<li class="list-group-item text-muted">Nenhuma diferença significativa detectada</li>'
                            }
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    const gauge = document.getElementById('similarity-gauge');
    if (gauge) {
        const similarity = result.similarity * 100;
        let color;
        if (similarity > 80) {
            color = '#198754';
        } else if (similarity > 50) {
            color = '#ffc107';
        } else {
            color = '#dc3545';
        }
        
        gauge.style.background = `conic-gradient(${color} 0% ${similarity}%, #e2e8f0 ${similarity}% 100%)`;
    }
    
    if (result.gpt_analysis) {
        const gptAnalysisDiv = document.createElement('div');
        gptAnalysisDiv.className = 'mt-4';
        gptAnalysisDiv.innerHTML = `
            <h4>Análise Inteligente</h4>
            <div class="card">
                <div class="card-header">Interpretação das Mudanças</div>
                <div class="card-body">
                    <h5>Principais Mudanças</h5>
                    <p>${result.gpt_analysis.main_changes}</p>
                    
                    <h5>Propósito</h5>
                    <p>${result.gpt_analysis.purpose}</p>
                    
                    <h5>Impacto Potencial</h5>
                    <p>${result.gpt_analysis.impact}</p>
                    
                    <h5>Tendências Futuras</h5>
                    <p>${result.gpt_analysis.future_trends}</p>
                </div>
            </div>
        `;
        
        resultsDiv.appendChild(gptAnalysisDiv);
    }
    
    const predictionsResults = document.getElementById('predictions-results');
    if (predictionsResults) {
        predictionsResults.style.display = 'block';
    }
}

function displayPredictions(predictions) {
    const container = document.getElementById('predictions-container');
    
    if (!predictions || predictions.length === 0) {
        container.innerHTML = '<p class="text-center text-muted">Nenhuma previsão disponível</p>';
        return;
    }
    
    container.innerHTML = '<h5 class="mb-4">Baseado na análise do texto atual, prováveis mudanças futuras:</h5>';
    
    predictions.forEach((prediction, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item mb-4 p-3 border rounded fade-in';
        
        const predictionType = prediction.type || (prediction.current_value ? "numerical" : "textual");
        
        if (predictionType === "numerical" || prediction.current_value) {
            const explanation = prediction.explanation ? 
                `<div class="mt-3 pt-3 border-top">
                    <h6 class="mb-2">Por que esperamos esta mudança:</h6>
                    <p class="mb-0 text-muted">${prediction.explanation}</p>
                </div>` : '';
                
            const content = `
                <div class="d-flex align-items-center mb-3">
                    <span class="badge bg-primary me-2">${(prediction.confidence * 100).toFixed(0)}% confiança</span>
                    <h5 class="mb-0">Alteração Numérica Prevista</h5>
                </div>
                <div class="d-flex justify-content-between align-items-center p-3 bg-light rounded">
                    <div class="h4">${prediction.current_value}</div>
                    <div class="px-3"><i class="bi bi-arrow-right fs-3"></i></div>
                    <div class="h4 text-primary fw-bold">${prediction.predicted_value}</div>
                </div>
                ${explanation}
            `;
            
            item.innerHTML = content;
        } else if (predictionType === "textual" || prediction.current_text) {
            let currentText = prediction.current_text;
            let predictedText = prediction.predicted_text;
            
            const explanation = prediction.explanation ? 
                `<div class="mt-3 pt-3 border-top">
                    <h6 class="mb-2">Por que esperamos esta mudança:</h6>
                    <p class="mb-0 text-muted">${prediction.explanation}</p>
                </div>` : '';
                
            const content = `
                <div class="d-flex align-items-center mb-3">
                    <span class="badge bg-primary me-2">${(prediction.confidence * 100).toFixed(0)}% confiança</span>
                    <h5 class="mb-0">Alteração Textual Prevista</h5>
                </div>
                <div class="mb-3 p-3 bg-light rounded">
                    <div class="fw-bold mb-2">Texto Atual:</div>
                    <div class="p-2 border-start border-4 border-secondary">${currentText}</div>
                </div>
                <div class="mb-3 p-3 bg-light rounded">
                    <div class="fw-bold mb-2">Provável Alteração Futura:</div>
                    <div class="p-2 border-start border-4 border-primary">${predictedText}</div>
                </div>
                ${explanation}
            `;
            
            item.innerHTML = content;
        } else {
            item.innerHTML = `<div class="alert alert-warning">Formato de previsão não reconhecido</div>`;
        }
        
        container.appendChild(item);
    });
}

function showRegulationDetails(regulation) {
    const modal = new bootstrap.Modal(document.getElementById('regulation-details-modal'));
    const title = document.getElementById('regulation-details-title');
    const body = document.getElementById('regulation-details-body');
    
    title.textContent = `Regulação ${regulation.id} (${regulation.date})`;
    
    let content = `
        <div class="mb-3">
            <h5>Texto da Regulação</h5>
            <div class="p-3 bg-light rounded">${regulation.text}</div>
        </div>
    `;
    
    if (regulation.key_phrases && regulation.key_phrases.length > 0) {
        content += `
            <div class="mb-3">
                <h5>Frases-chave</h5>
                <ul class="list-group">
        `;
        
        regulation.key_phrases.forEach(phrase => {
            content += `<li class="list-group-item">${phrase}</li>`;
        });
        
        content += `
                </ul>
            </div>
        `;
    }
    
    if (regulation.previous_regulations && regulation.previous_regulations.length > 0) {
        content += `
            <div class="mb-3">
                <h5>Regulações Anteriores</h5>
                <ul class="list-group">
        `;
        
        regulation.previous_regulations.forEach(prevId => {
            const prevReg = state.regulations.find(r => r.id === prevId);
            if (prevReg) {
                content += `<li class="list-group-item">${prevReg.id} (${prevReg.date})</li>`;
            }
        });
        
        content += `
                </ul>
            </div>
        `;
    }
    
    if (regulation.next_regulations && regulation.next_regulations.length > 0) {
        content += `
            <div class="mb-3">
                <h5>Regulações Subsequentes</h5>
                <ul class="list-group">
        `;
        
        regulation.next_regulations.forEach(nextId => {
            const nextReg = state.regulations.find(r => r.id === nextId);
            if (nextReg) {
                content += `<li class="list-group-item">${nextReg.id} (${nextReg.date})</li>`;
            }
        });
        
        content += `
                </ul>
            </div>
        `;
    }
    
    body.innerHTML = content;
    modal.show();
}

function initializeMiniGraph() {
    const container = document.getElementById('mini-graph-container');
    
    if (!state.graphData || !state.graphData.nodes || state.graphData.nodes.length === 0) {
        container.innerHTML = '<p class="text-center text-muted">Nenhum dado disponível para visualização</p>';
        return;
    }
    
    container.innerHTML = '';
    
    const width = container.clientWidth;
    const height = 200;
    
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    const simulation = d3.forceSimulation(state.graphData.nodes)
        .force('link', d3.forceLink(state.graphData.edges).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-200))
        .force('center', d3.forceCenter(width / 2, height / 2));
    
    const link = svg.append('g')
        .selectAll('line')
        .data(state.graphData.edges)
        .enter()
        .append('line')
        .attr('class', 'link')
        .attr('stroke-width', d => 1 + d.similarity * 3);
    
    const node = svg.append('g')
        .selectAll('circle')
        .data(state.graphData.nodes)
        .enter()
        .append('circle')
        .attr('class', 'node')
        .attr('r', 8)
        .attr('fill', '#4299e1')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));
    
    const labels = svg.append('g')
        .selectAll('text')
        .data(state.graphData.nodes)
        .enter()
        .append('text')
        .text(d => d.id)
        .attr('font-size', '8px')
        .attr('dx', 10)
        .attr('dy', 3);
    
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
        
        labels
            .attr('x', d => d.x)
            .attr('y', d => d.y);
    });
    
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

function initializeKnowledgeGraph(data) {
    const container = document.getElementById('knowledge-graph-container');
    
    if (!data || !data.nodes || data.nodes.length === 0) {
        container.innerHTML = '<div class="alert alert-info">Nenhum dado disponível para visualização do grafo</div>';
        return;
    }
    
    container.innerHTML = '';
    
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    const g = svg.append('g');
    
    svg.call(d3.zoom()
        .scaleExtent([0.5, 3])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        }));
    
    const simulation = d3.forceSimulation(data.nodes)
        .force('link', d3.forceLink(data.edges).id(d => d.id).distance(150))
        .force('charge', d3.forceManyBody().strength(-400))
        .force('center', d3.forceCenter(width / 2, height / 2));
    
    const link = g.append('g')
        .selectAll('g')
        .data(data.edges)
        .enter()
        .append('g');
    
    link.append('line')
        .attr('class', 'link')
        .attr('stroke-width', d => 1 + d.similarity * 3);
    
    link.append('polygon')
        .attr('points', '0,-3 6,0 0,3')
        .attr('fill', '#a0aec0')
        .attr('class', 'arrow');
    
    const node = g.append('g')
        .selectAll('.node')
        .data(data.nodes)
        .enter()
        .append('g')
        .attr('class', 'node')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));
    
    node.append('circle')
        .attr('r', 15)
        .attr('fill', '#4299e1')
        .attr('stroke', '#2b6cb0')
        .attr('stroke-width', 2);
    
    node.append('text')
        .attr('dy', 5)
        .attr('text-anchor', 'middle')
        .text(d => d.id)
        .attr('font-size', '10px')
        .attr('fill', 'white');
    
    node.append('text')
        .attr('dy', 30)
        .attr('text-anchor', 'middle')
        .text(d => d.date)
        .attr('font-size', '9px')
        .attr('class', 'date-label');
    
    node.append('title')
        .text(d => `${d.id} (${d.date})`);
    
    simulation.on('tick', () => {
        link.select('line')
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        link.select('polygon')
            .attr('transform', d => {
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const angle = Math.atan2(dy, dx) * 180 / Math.PI;
                const dist = Math.sqrt(dx * dx + dy * dy);
                return `translate(${d.source.x + dx * (1 - 15 / dist)}, ${d.source.y + dy * (1 - 15 / dist)}) rotate(${angle})`;
            });
        
        node.attr('transform', d => `translate(${d.x}, ${d.y})`);
    });
    
    node.on('click', (event, d) => {
        node.classed('highlighted', false);
        link.classed('highlighted', false);
        
        d3.select(event.currentTarget).classed('highlighted', true);
        
        link.each(function(l) {
            if (l.source.id === d.id || l.target.id === d.id) {
                d3.select(this).classed('highlighted', true);
                
                node.each(function(n) {
                    if ((l.source.id === d.id && l.target.id === n.id) ||
                        (l.target.id === d.id && l.source.id === n.id)) {
                        d3.select(this).classed('highlighted', true);
                    }
                });
            }
        });
        
        const regulation = state.regulations.find(r => r.id === d.id);
        if (regulation) {
            showRegulationDetails(regulation);
        }
    });
    
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
    
    state.simulation = simulation;
    state.svg = svg;
}

function updateGraphZoom(zoom) {
    if (state.svg) {
        state.svg.transition().duration(300).call(
            d3.zoom().transform,
            d3.zoomIdentity.scale(zoom)
        );
    }
}

function toggleGraphLabels(show) {
    if (state.svg) {
        state.svg.selectAll('.date-label')
            .style('display', show ? 'block' : 'none');
    }
}

function toggleHighlightChanges(highlight) {
    if (state.svg) {
        if (highlight) {
            state.svg.selectAll('.link')
                .attr('stroke-width', d => 1 + d.similarity * 3)
                .attr('stroke-opacity', d => 0.2 + d.similarity * 0.8);
        } else {
            state.svg.selectAll('.link')
                .attr('stroke-width', 1.5)
                .attr('stroke-opacity', 0.6);
        }
    }
}

function initializeAnalytics() {
    if (!state.regulations || state.regulations.length < 2) {
        document.getElementById('analyze').innerHTML = 
            '<div class="alert alert-info">Dados insuficientes para análise. Adicione pelo menos duas regulações.</div>';
        return;
    }
    
    initializeNumericTrendsChart();
    initializeKeywordTrendsChart();
    initializeTimeline();
}

function initializeNumericTrendsChart() {
    const ctx = document.getElementById('numeric-trends-chart').getContext('2d');
    
    const numericData = [];
    
    const numberRegex = /(\d+(?:\.\d+)?)\s*%?/g;
    
    state.regulations.forEach(reg => {
        const matches = [...reg.text.matchAll(numberRegex)];
        const values = matches.map(m => parseFloat(m[1]));
        
        numericData.push({
            id: reg.id,
            date: reg.date,
            values: values
        });
    });
    
    const allValues = numericData.map(d => d.values).flat();
    const valueCounts = {};
    
    allValues.forEach(val => {
        valueCounts[val] = (valueCounts[val] || 0) + 1;
    });
    
    const commonValues = Object.keys(valueCounts)
        .filter(val => valueCounts[val] > 1)
        .map(Number);
    
    if (commonValues.length === 0) {
        document.getElementById('numeric-trends-chart').closest('.col-md-6').innerHTML = 
            '<div class="alert alert-info">Nenhuma tendência numérica identificada entre as regulações.</div>';
        return;
    }
    
    const datasets = [];
    
    commonValues.forEach((value, index) => {
        const data = [];
        
        numericData.forEach(d => {
            if (d.values.includes(value)) {
                data.push({
                    x: d.date,
                    y: value
                });
            }
        });
        
        if (data.length > 1) {
            datasets.push({
                label: `Valor ${index + 1}`,
                data: data,
                borderColor: getRandomColor(),
                fill: false,
                tension: 0.1
            });
        }
    });
    
    new Chart(ctx, {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    type: 'category',
                    title: {
                        display: true,
                        text: 'Data'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Valor'
                    }
                }
            }
        }
    });
}

function initializeKeywordTrendsChart() {
    const ctx = document.getElementById('keyword-trends-chart').getContext('2d');
    
    const keyPhrases = {};
    
    state.regulations.forEach(reg => {
        if (reg.key_phrases && reg.key_phrases.length > 0) {
            reg.key_phrases.forEach(phrase => {
                if (!keyPhrases[phrase]) {
                    keyPhrases[phrase] = [];
                }
                keyPhrases[phrase].push(reg.date);
            });
        }
    });
    
    const commonPhrases = Object.keys(keyPhrases)
        .filter(phrase => keyPhrases[phrase].length > 1);
    
    if (commonPhrases.length === 0) {
        document.getElementById('keyword-trends-chart').closest('.col-md-6').innerHTML = 
            '<div class="alert alert-info">Nenhuma tendência de termos-chave identificada entre as regulações.</div>';
        return;
    }
    
    const topPhrases = commonPhrases
        .sort((a, b) => keyPhrases[b].length - keyPhrases[a].length)
        .slice(0, 5);
    
    const labels = state.regulations.map(reg => reg.date);
    const datasets = [];
    
    topPhrases.forEach((phrase, index) => {
        const data = state.regulations.map(reg => {
            return (reg.key_phrases && reg.key_phrases.includes(phrase)) ? 1 : 0;
        });
        
        datasets.push({
            label: phrase.length > 20 ? phrase.substring(0, 20) + '...' : phrase,
            data: data,
            backgroundColor: getRandomColor(),
            borderColor: 'rgba(0, 0, 0, 0.1)',
            borderWidth: 1
        });
    });
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Data'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Presença'
                    },
                    min: 0,
                    max: 1,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}

function initializeTimeline() {
    const container = document.getElementById('timeline-container');
    container.innerHTML = '';
    
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    const sortedRegulations = [...state.regulations]
        .sort((a, b) => new Date(a.date) - new Date(b.date));
    
    if (sortedRegulations.length === 0) {
        container.innerHTML = '<div class="alert alert-info">Nenhuma regulação disponível para exibir na linha do tempo.</div>';
        return;
    }
    
    const line = document.createElement('div');
    line.className = 'timeline-line';
    line.style.top = height / 2 + 'px';
    line.style.width = '100%';
    container.appendChild(line);
    
    const itemWidth = Math.min(width / (sortedRegulations.length + 1), 100);
    
    sortedRegulations.forEach((reg, index) => {
        const x = itemWidth * (index + 1);
        const y = height / 2;
        
        const item = document.createElement('div');
        item.className = 'timeline-item';
        item.style.left = x + 'px';
        item.style.top = y + 'px';
        item.setAttribute('data-reg-id', reg.id);
        item.title = reg.id;
        container.appendChild(item);
        
        const label = document.createElement('div');
        label.className = 'timeline-label';
        label.textContent = reg.date;
        label.style.left = x + 'px';
        label.style.top = y + 'px';
        container.appendChild(label);
        
        item.addEventListener('click', function() {
            document.querySelectorAll('.timeline-item').forEach(el => {
                el.classList.remove('selected');
            });
            this.classList.add('selected');
            
            const regulation = state.regulations.find(r => r.id === reg.id);
            if (regulation) {
                showRegulationDetails(regulation);
            }
        });
    });
}

function getRandomColor() {
    const colors = [
        '#4299e1', 
        '#48bb78', 
        '#ed8936', 
        '#e53e3e', 
        '#9f7aea', 
        '#f6e05e', 
        '#667eea', 
        '#ed64a6'  
    ];
    
    return colors[Math.floor(Math.random() * colors.length)];
}

function showLoadingModal(message) {
    console.log("Operação iniciada: " + message);
}

function hideLoadingModal() {
    console.log("Operação concluída");
}