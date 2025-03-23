// gnn_visualizer.js - Visualização de Grafo Enriquecido por GNN

class GNNVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.width = this.container.clientWidth;
        this.height = 500;
        this.gnnData = null;
        this.colorScale = d3.scaleOrdinal(d3.schemeCategory10);
    }

    async fetchGNNData() {
        try {
            // Primeiro, verificar se já temos embeddings GNN
            const response = await fetch(`${API_BASE_URL}/gnn-status`);
            const status = await response.json();
            
            if (!status.has_gnn_embeddings) {
                // Iniciar treinamento GNN se não tiver embeddings
                await this.trainGNN();
            }
            
            // Buscar dados do grafo enriquecido
            const dataResponse = await fetch(`${API_BASE_URL}/gnn-graph-data`);
            this.gnnData = await dataResponse.json();
            
            return this.gnnData;
        } catch (error) {
            console.error("Erro ao buscar dados GNN:", error);
            return null;
        }
    }
    
    async trainGNN() {
        try {
            const loadingModal = new bootstrap.Modal(document.getElementById('loading-modal'));
            document.getElementById('loading-message').textContent = 'Treinando modelo GNN...';
            loadingModal.show();
            
            const response = await fetch(`${API_BASE_URL}/enrich-embeddings`, {
                method: 'POST'
            });
            
            loadingModal.hide();
            
            if (!response.ok) {
                throw new Error(`Erro ao treinar GNN: ${response.status}`);
            }
            
            const result = await response.json();
            return result;
        } catch (error) {
            console.error("Erro ao treinar GNN:", error);
            return null;
        }
    }
    
    visualize() {
        if (!this.gnnData) {
            this.container.innerHTML = '<div class="alert alert-warning">Nenhum dado GNN disponível.</div>';
            return;
        }
        
        this.container.innerHTML = '';
        
        const svg = d3.select(this.container)
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height);
        
        // Criar grupo principal
        const g = svg.append('g');
        
        // Adicionar zoom
        svg.call(d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            }));
        
        // Extrair nós e links dos dados
        const nodes = this.gnnData.nodes.map(node => ({
            id: node.id,
            date: node.date,
            cluster: node.cluster || 0,
            projection: node.projection || 0,
            key_phrases: node.key_phrases || []
        }));
        
        const links = this.gnnData.edges.map(edge => ({
            source: edge.source,
            target: edge.target,
            similarity: edge.similarity || 0.5
        }));
        
        // Criar simulação de força
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2));
        
        // Desenhar links
        const link = g.append('g')
            .selectAll('line')
            .data(links)
            .enter()
            .append('line')
            .attr('stroke-width', d => 1 + d.similarity * 3)
            .attr('stroke-opacity', d => 0.2 + d.similarity * 0.8)
            .attr('stroke', '#999');
        
        // Desenhar nós
        const node = g.append('g')
            .selectAll('.node')
            .data(nodes)
            .enter()
            .append('g')
            .attr('class', 'node')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));
        
        // Adicionar círculos aos nós
        node.append('circle')
            .attr('r', 10)
            .attr('fill', d => this.colorScale(d.cluster))
            .attr('stroke', '#fff')
            .attr('stroke-width', 1.5);
        
        // Adicionar rótulos
        node.append('text')
            .attr('dx', 12)
            .attr('dy', 4)
            .text(d => d.id)
            .attr('font-size', '10px');
        
        // Adicionar tooltips
        node.append('title')
            .text(d => {
                let tooltip = `ID: ${d.id}\nData: ${d.date}\n`;
                if (d.key_phrases && d.key_phrases.length > 0) {
                    tooltip += `Frases-chave: ${d.key_phrases.slice(0, 3).join(', ')}`;
                }
                if (d.cluster !== undefined) {
                    tooltip += `\nGrupo: ${d.cluster}`;
                }
                if (d.projection !== undefined) {
                    tooltip += `\nProjeção: ${d.projection.toFixed(2)}`;
                }
                return tooltip;
            });
        
        // Atualização da simulação
        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            node.attr('transform', d => `translate(${d.x},${d.y})`);
        });
        
        // Adicionar legenda de clusters
        const legendData = Array.from(new Set(nodes.map(n => n.cluster)))
            .map(cluster => ({ cluster, color: this.colorScale(cluster) }));
        
        const legend = svg.append('g')
            .attr('class', 'legend')
            .attr('transform', 'translate(20, 20)');
        
        legend.selectAll('rect')
            .data(legendData)
            .enter()
            .append('rect')
            .attr('x', 0)
            .attr('y', (d, i) => i * 20)
            .attr('width', 10)
            .attr('height', 10)
            .attr('fill', d => d.color);
        
        legend.selectAll('text')
            .data(legendData)
            .enter()
            .append('text')
            .attr('x', 15)
            .attr('y', (d, i) => i * 20 + 9)
            .text(d => `Grupo ${d.cluster}`)
            .attr('font-size', '12px');
        
        // Adicionar controles de visualização
        this.addControls();
        
        // Funções para arrastar
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
    
    addControls() {
        // Adicionar controles para alternar entre diferentes visualizações GNN
        const controlsDiv = document.createElement('div');
        controlsDiv.className = 'mb-3 mt-3';
        controlsDiv.innerHTML = `
            <div class="btn-group" role="group">
                <button class="btn btn-sm btn-outline-primary active" id="cluster-view">
                    Visualização de Clusters
                </button>
                <button class="btn btn-sm btn-outline-primary" id="evolution-view">
                    Visualização de Evolução
                </button>
            </div>
        `;
        
        this.container.parentNode.insertBefore(controlsDiv, this.container);
        
        // Adicionar eventos aos botões
        document.getElementById('cluster-view').addEventListener('click', (e) => {
            e.target.classList.add('active');
            document.getElementById('evolution-view').classList.remove('active');
            this.visualizeClusterView();
        });
        
        document.getElementById('evolution-view').addEventListener('click', (e) => {
            e.target.classList.add('active');
            document.getElementById('cluster-view').classList.remove('active');
            this.visualizeEvolutionView();
        });
    }
    
    visualizeClusterView() {
        // Colorir nós por cluster
        d3.select(this.container)
            .selectAll('.node circle')
            .transition()
            .duration(500)
            .attr('fill', d => this.colorScale(d.cluster));
        
        // Atualizar força para agrupar por cluster
        const simulation = d3.select(this.container).select('svg g').datum();
        if (simulation) {
            simulation
                .force('x', d3.forceX(this.width / 2).strength(0.1))
                .force('y', d3.forceY(this.height / 2).strength(0.1))
                .alpha(0.3)
                .restart();
        }
    }
    
    visualizeEvolutionView() {
        // Ordenar pela projeção direcional e colorir em gradiente
        const projections = this.gnnData.nodes.map(node => node.projection || 0);
        const minProj = Math.min(...projections);
        const maxProj = Math.max(...projections);
        
        // Criar escala de cores para projeção
        const projColorScale = d3.scaleSequential()
            .domain([minProj, maxProj])
            .interpolator(d3.interpolateViridis);
        
        // Atualizar cores dos nós
        d3.select(this.container)
            .selectAll('.node circle')
            .transition()
            .duration(500)
            .attr('fill', d => projColorScale(d.projection || 0));
        
        // Ajustar força para mostrar evolução linear
        const simulation = d3.select(this.container).select('svg g').datum();
        if (simulation) {
            simulation
                .force('x', d3.forceX(d => (this.width * 0.1) + (d.projection - minProj) / (maxProj - minProj) * (this.width * 0.8)).strength(0.5))
                .force('y', d3.forceY(this.height / 2).strength(0.1))
                .alpha(0.5)
                .restart();
        }
        
        // Adicionar legenda de gradiente para evolução temporal
        this.addEvolutionLegend(minProj, maxProj, projColorScale);
    }
    
    addEvolutionLegend(minProj, maxProj, colorScale) {
        // Remover legenda anterior
        d3.select(this.container).select('.evolution-legend').remove();
        
        const svg = d3.select(this.container).select('svg');
        const legend = svg.append('g')
            .attr('class', 'evolution-legend')
            .attr('transform', `translate(${this.width - 180}, 20)`);
        
        // Gradiente
        const defs = svg.append('defs');
        const gradient = defs.append('linearGradient')
            .attr('id', 'evolution-gradient')
            .attr('x1', '0%')
            .attr('y1', '0%')
            .attr('x2', '100%')
            .attr('y2', '0%');
        
        // Adicionar stops ao gradiente
        const steps = 10;
        for (let i = 0; i <= steps; i++) {
            const val = minProj + (i / steps) * (maxProj - minProj);
            gradient.append('stop')
                .attr('offset', `${i * 100 / steps}%`)
                .attr('stop-color', colorScale(val));
        }
        
        // Retângulo com gradiente
        legend.append('rect')
            .attr('width', 150)
            .attr('height', 15)
            .style('fill', 'url(#evolution-gradient)');
        
        // Texto para início e fim
        legend.append('text')
            .attr('x', 0)
            .attr('y', 30)
            .text('Mais Antigo')
            .attr('font-size', '10px');
        
        legend.append('text')
            .attr('x', 150)
            .attr('y', 30)
            .attr('text-anchor', 'end')
            .text('Mais Recente')
            .attr('font-size', '10px');
    }
}