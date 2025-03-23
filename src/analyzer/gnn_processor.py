import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data
import numpy as np
import networkx as nx

class RegulationGNN(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, output_dim=64, model_type='gcn'):
        super(RegulationGNN, self).__init__()
        
        self.model_type = model_type
        
        if model_type == 'gcn':
            self.conv1 = GCNConv(input_dim, hidden_dim)
        elif model_type == 'sage':
            self.conv1 = SAGEConv(input_dim, hidden_dim)
        elif model_type == 'gat':
            self.conv1 = GATConv(input_dim, hidden_dim)
        else:
            raise ValueError(f"Tipo de modelo não suportado: {model_type}")
            
        if model_type == 'gcn':
            self.conv2 = GCNConv(hidden_dim, output_dim)
        elif model_type == 'sage':
            self.conv2 = SAGEConv(hidden_dim, output_dim)
        elif model_type == 'gat':
            self.conv2 = GATConv(hidden_dim, output_dim)
            
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_attr)
        
        return x


class GNNProcessor:

    def __init__(self, embedding_dim=1024, hidden_dim=256, output_dim=128):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def networkx_to_pytorch_geometric(self, G):

        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        
        node_embeddings = []
        for node in G.nodes():
            if 'embedding' in G.nodes[node]:
                emb = G.nodes[node]['embedding']
            else:
                emb = np.zeros(self.embedding_dim)
            node_embeddings.append(emb)
            
        node_features = torch.FloatTensor(np.array(node_embeddings))
        
        edge_indices = []
        edge_weights = []
        
        for source, target, data in G.edges(data=True):
            source_idx = node_mapping[source]
            target_idx = node_mapping[target]
            
            edge_indices.append([source_idx, target_idx])
            
            weight = data.get('similarity', 0.5)
            edge_weights.append(weight)
            
        edge_index = torch.LongTensor(edge_indices).t().contiguous()
        edge_attr = torch.FloatTensor(edge_weights).view(-1, 1)
        
        data = Data(x=node_features, 
                   edge_index=edge_index, 
                   edge_attr=edge_attr,
                   num_nodes=len(G.nodes()))
        
        return data, node_mapping
    
    def init_model(self, model_type='gcn'):

        self.model = RegulationGNN(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            model_type=model_type
        ).to(self.device)
        
    def train(self, graph_data, epochs=100, lr=0.01):

        if self.model is None:
            self.init_model()
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        graph_data = graph_data.to(self.device)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            node_embeddings = self.model(
                graph_data.x, 
                graph_data.edge_index, 
                graph_data.edge_attr
            )

            
            reconstructed_adj = torch.mm(node_embeddings, node_embeddings.t())
            
            adj = torch.zeros((graph_data.num_nodes, graph_data.num_nodes), 
                              device=self.device)
            for i in range(graph_data.edge_index.size(1)):
                src, dst = graph_data.edge_index[:, i]
                adj[src, dst] = graph_data.edge_attr[i]
            
            loss = F.mse_loss(reconstructed_adj, adj)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
        
        return self.model
    
    def get_node_embeddings(self, graph_data, node_mapping):

        if self.model is None:
            raise ValueError("Modelo não inicializado. Execute init_model() primeiro.")
            
        self.model.eval()
        with torch.no_grad():
            node_embeddings = self.model(
                graph_data.x.to(self.device), 
                graph_data.edge_index.to(self.device), 
                graph_data.edge_attr.to(self.device)
            )
            
        inverse_mapping = {idx: node_id for node_id, idx in node_mapping.items()}
        result = {inverse_mapping[i]: emb.cpu().numpy() 
                 for i, emb in enumerate(node_embeddings)}
        
        return result
    
    def predict_similarity(self, node_a_emb, node_b_emb):

        if isinstance(node_a_emb, np.ndarray):
            node_a_emb = torch.FloatTensor(node_a_emb)
        if isinstance(node_b_emb, np.ndarray):
            node_b_emb = torch.FloatTensor(node_b_emb)
            
        node_a_emb = F.normalize(node_a_emb, p=2, dim=0)
        node_b_emb = F.normalize(node_b_emb, p=2, dim=0)
        
        similarity = torch.dot(node_a_emb, node_b_emb).item()
        
        return similarity