"""
Temporal/Dynamic Graph Neural Network Baselines
"""
import torch
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
import logging
import os
import pickle
import warnings
import torch.nn as nn
import time
from tqdm import tqdm

warnings.filterwarnings("ignore")

from LETAGNN import set_seed, normalize_features
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, RGCNConv, global_mean_pool, global_add_pool
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import glob

NODE_CATEGORIES = ["Address", "TokenContract", "DEX", "CEX", "DepositAddress"]


def load_hetero_graphs(input_dir, max_file_size_mb=80):
    """Load heterogeneous graphs from .pt files"""
    graphs = []
    labels = []
    pt_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".pt")])
    
    for fname in tqdm(pt_files, desc=f"Loading hetero graphs from {os.path.basename(input_dir)}"):
        fpath = os.path.join(input_dir, fname)
        fsize = os.path.getsize(fpath) / (1024 * 1024)
        if fsize > max_file_size_mb:
            continue
        
        g = torch.load(fpath, map_location="cpu", weights_only=False)
        
        if isinstance(g, list):
            for gg in g:
                graphs.append(gg)
                lab = _label_from_hetero(gg, fname)
                labels.append(0 if lab is None else lab)
        else:
            graphs.append(g)
            lab = _label_from_hetero(g, fname)
            labels.append(0 if lab is None else lab)
    
    return graphs, labels


def _label_from_hetero(g, fname: str):
    """Extract label from heterogeneous graph"""
    if hasattr(g, "y") and g.y is not None:
        v = g.y
        return int(v.item()) if hasattr(v, "item") else int(v)
    if hasattr(g, "center_node_type") and hasattr(g, "center_node_idx"):
        nt, idx = g.center_node_type, g.center_node_idx
        if hasattr(g, nt) and hasattr(g[nt], "y") and g[nt].y is not None:
            v = g[nt].y[idx]
            return int(v.item()) if hasattr(v, "item") else int(v)
    if "_" in fname and fname.endswith(".pt"):
        suff = fname.replace(".pt", "").split("_")[-1]
        if suff in ("0", "1"):
            return int(suff)
    return None


def get_hetero_metadata(hetero_graphs):
    """Extract metadata: collect all edge types from all graphs."""
    node_types = sorted(NODE_CATEGORIES)
    all_edge_types = set()
    node_type_dims = {}
    
    for g in hetero_graphs:
        all_edge_types.update(g.edge_types)
        if not node_type_dims:
            for ntype in g.node_types:
                if hasattr(g[ntype], 'x') and g[ntype].x is not None:
                    node_type_dims[ntype] = g[ntype].x.size(1)
    
    edge_types = sorted(all_edge_types)
    default_dim = next(iter(node_type_dims.values()), None) if node_type_dims else None
    for ntype in NODE_CATEGORIES:
        if ntype not in node_type_dims and default_dim is not None:
            node_type_dims[ntype] = default_dim
    return node_types, edge_types, node_type_dims


def normalize_hetero_features(hetero_graphs):
    """Normalize node features for each node type in heterogeneous graphs."""
    logging.info("Normalizing heterogeneous graph features...")
    
    # Collect statistics for each node type
    node_type_stats = {}
    for g in tqdm(hetero_graphs, desc="Computing stats"):
        for ntype in g.node_types:
            if not hasattr(g[ntype], 'x') or g[ntype].x is None:
                continue
            if ntype not in node_type_stats:
                node_type_stats[ntype] = {'sum': 0, 'sum_sq': 0, 'count': 0, 'dim': g[ntype].x.size(1)}
            x = g[ntype].x
            node_type_stats[ntype]['sum'] += x.sum(dim=0)
            node_type_stats[ntype]['sum_sq'] += (x ** 2).sum(dim=0)
            node_type_stats[ntype]['count'] += x.size(0)
    
    # Compute mean and std for each node type
    node_type_norm = {}
    for ntype, stats in node_type_stats.items():
        mean = stats['sum'] / stats['count']
        std = torch.sqrt(stats['sum_sq'] / stats['count'] - mean ** 2)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        node_type_norm[ntype] = {'mean': mean, 'std': std}
    
    # Apply normalization
    for g in tqdm(hetero_graphs, desc="Applying normalization"):
        for ntype in g.node_types:
            if ntype not in node_type_norm:
                continue
            if hasattr(g[ntype], 'x') and g[ntype].x is not None:
                g[ntype].x = (g[ntype].x - node_type_norm[ntype]['mean']) / node_type_norm[ntype]['std']
    
    return hetero_graphs


class TGATModel(torch.nn.Module):
    """Temporal Graph Attention Network with node-level temporal attention"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=4, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.temporal_enc = nn.Sequential(
            nn.Linear(1, hidden_channels // 4),
            nn.ReLU(),
            nn.Linear(hidden_channels // 4, hidden_channels // 4)
        )
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels + hidden_channels // 4, hidden_channels, heads=heads, dropout=dropout, add_self_loops=False))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, add_self_loops=False))
        
        self.lin = nn.Linear(hidden_channels * heads, out_channels)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_times = data.edge_times_normalized if hasattr(data, 'edge_times_normalized') else torch.zeros(edge_index.size(1), 1, device=x.device)
        
        if edge_times.dim() == 1:
            edge_times = edge_times.unsqueeze(-1)
        
        temporal_feat = self.temporal_enc(edge_times)
        edge_src = edge_index[0]
        node_temporal = torch.zeros(x.size(0), temporal_feat.size(1), device=x.device)
        node_temporal.index_add_(0, edge_src, temporal_feat)
        
        x = torch.cat([x, node_temporal], dim=1)
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        return self.lin(x)


class TGNModel(torch.nn.Module):
    """Temporal Graph Network with memory and temporal encoding"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        
        self.time_enc = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        self.msg_fn = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.gru = nn.GRUCell(hidden_channels, hidden_channels)
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.lin = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_times = data.edge_times_normalized if hasattr(data, 'edge_times_normalized') else torch.zeros(edge_index.size(1), 1, device=x.device)
        
        if edge_times.dim() == 1:
            edge_times = edge_times.unsqueeze(-1)
        
        time_enc = self.time_enc(edge_times)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        
        memory = torch.zeros(x.size(0), self.hidden_channels, device=x.device)
        msg = torch.zeros(x.size(0), self.hidden_channels, device=x.device)
        
        time_msg = torch.zeros_like(msg)
        time_msg.index_add_(0, edge_src, time_enc)
        
        x_msg = self.msg_fn(torch.cat([x, time_msg], dim=1))
        # Aggregate messages: for each edge (src,dst), add x_msg[src] to msg[dst]
        msg.index_add_(0, edge_dst, x_msg[edge_src])
        
        memory = self.gru(msg, memory)
        
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = h + memory
        h = global_mean_pool(h, batch)
        return self.lin(h)


class EvolveGCNModel(torch.nn.Module):
    """EvolveGCN: Evolving GCN parameters over time"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.lstm_list = nn.ModuleList()
        for _ in range(num_layers):
            self.lstm_list.append(nn.LSTMCell(hidden_channels, hidden_channels))
        
        self.weight_evolver = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.Tanh()
        )
        
        self.lin = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_times = data.edge_times_normalized if hasattr(data, 'edge_times_normalized') else torch.zeros(1, 1, device=x.device)
        
        if edge_times.dim() == 1:
            edge_times = edge_times.unsqueeze(-1)
        
        mean_time = edge_times.mean().view(1, 1)
        time_emb = self.weight_evolver(mean_time).squeeze(0)
        
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            
            h_agg = global_mean_pool(h, batch)
            if h_agg.size(0) > 0:
                h_t = h_agg.mean(0, keepdim=True).expand(1, self.hidden_channels)
                ht, ct = self.lstm_list[i](h_t, (time_emb.unsqueeze(0), time_emb.unsqueeze(0)))
                scale = torch.sigmoid(ht).expand_as(h)
                h = h * scale
            
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = global_mean_pool(h, batch)
        return self.lin(h)


class RGCNModel(torch.nn.Module):
    """Relational GCN for heterogeneous graphs"""
    def __init__(self, node_type_dims, hidden_channels, out_channels, canonical_edge_types, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.node_types = list(node_type_dims.keys())
        self.canonical_edge_types = list(canonical_edge_types)
        num_relations = len(self.canonical_edge_types)
        
        self.input_lins = nn.ModuleDict()
        for ntype, in_dim in node_type_dims.items():
            self.input_lins[ntype] = nn.Linear(in_dim, hidden_channels)
        
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations))
        
        self.lin = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, hetero_data):
        x_dict = {}
        for ntype in hetero_data.node_types:
            if ntype not in self.input_lins:
                continue
            if hasattr(hetero_data[ntype], 'x') and hetero_data[ntype].x is not None:
                x_dict[ntype] = self.input_lins[ntype](hetero_data[ntype].x)
        
        x_all = torch.cat([x_dict[ntype] for ntype in self.node_types if ntype in x_dict], dim=0)
        
        edge_indices = []
        edge_type_tensors = []
        offset = 0
        node_offsets = {}
        for ntype in self.node_types:
            if ntype in x_dict:
                node_offsets[ntype] = offset
                offset += x_dict[ntype].size(0)
        
        for edge_type in self.canonical_edge_types:
            if edge_type not in hetero_data.edge_types:
                continue
            src_type, _, dst_type = edge_type
            if src_type not in node_offsets or dst_type not in node_offsets:
                continue
            rel_idx = self.canonical_edge_types.index(edge_type)
            edge_index = hetero_data[edge_type].edge_index
            edge_index_shifted = edge_index.clone()
            edge_index_shifted[0] += node_offsets[src_type]
            edge_index_shifted[1] += node_offsets[dst_type]
            edge_indices.append(edge_index_shifted)
            edge_type_tensors.append(torch.full((edge_index.size(1),), rel_idx, dtype=torch.long, device=edge_index.device))
        
        if len(edge_indices) > 0:
            edge_index_all = torch.cat(edge_indices, dim=1)
            edge_type_all = torch.cat(edge_type_tensors, dim=0)
        else:
            # No edges: return zero output with gradient support
            graph_emb = x_all.mean(dim=0, keepdim=True)
            return self.lin(graph_emb)
        
        x = x_all
        for conv in self.convs:
            x = conv(x, edge_index_all, edge_type_all)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Aggregate: pool all node types and sum their contributions
        if hasattr(hetero_data, 'batch_dict') and hetero_data.batch_dict:
            num_graphs = max(b.max().item() for b in hetero_data.batch_dict.values()) + 1
            graph_emb = torch.zeros(num_graphs, x.size(1), device=x.device)
            for ntype in self.node_types:
                if ntype in node_offsets and ntype in hetero_data.batch_dict:
                    start = node_offsets[ntype]
                    end = start + hetero_data.batch_dict[ntype].size(0)
                    ntype_emb = global_mean_pool(x[start:end], hetero_data.batch_dict[ntype], size=num_graphs)
                    graph_emb += ntype_emb
        else:
            graph_emb = x.mean(dim=0, keepdim=True)
        
        return self.lin(graph_emb)


class HGTModel(torch.nn.Module):
    """Heterogeneous Graph Transformer"""
    def __init__(self, node_type_dims, hidden_channels, out_channels, metadata, num_layers=2, num_heads=4, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.node_types = metadata[0]
        self.registered_edge_types = set(metadata[1])
        
        self.input_lins = nn.ModuleDict()
        for ntype, in_dim in node_type_dims.items():
            self.input_lins[ntype] = Linear(in_dim, hidden_channels)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)
        
        self.lin_out = Linear(hidden_channels, out_channels)
    
    def forward(self, hetero_data):
        x_dict = {}
        for ntype in hetero_data.node_types:
            if ntype not in self.input_lins:
                continue
            if hasattr(hetero_data[ntype], 'x') and hetero_data[ntype].x is not None:
                x_dict[ntype] = self.input_lins[ntype](hetero_data[ntype].x)
        
        edge_index_dict = {}
        for edge_type in hetero_data.edge_types:
            if edge_type in self.registered_edge_types and hasattr(hetero_data[edge_type], 'edge_index'):
                edge_index_dict[edge_type] = hetero_data[edge_type].edge_index
        
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(val) for key, val in x_dict.items()}
            x_dict = {key: F.dropout(val, p=self.dropout, training=self.training) for key, val in x_dict.items()}
        
        # Aggregate: pool all node types and sum their contributions
        if hasattr(hetero_data, 'batch_dict') and hetero_data.batch_dict:
            hidden_dim = next(iter(x_dict.values())).size(1)
            num_graphs = max(b.max().item() for b in hetero_data.batch_dict.values()) + 1
            graph_emb = torch.zeros(num_graphs, hidden_dim, device=next(iter(x_dict.values())).device)
            for ntype in self.node_types:
                if ntype in x_dict and ntype in hetero_data.batch_dict:
                    ntype_emb = global_mean_pool(x_dict[ntype], hetero_data.batch_dict[ntype], size=num_graphs)
                    graph_emb += ntype_emb
        else:
            first_type = next(iter(x_dict.keys()))
            graph_emb = x_dict[first_type].mean(dim=0, keepdim=True)
        
        return self.lin_out(graph_emb)


class ieHGCNModel(torch.nn.Module):
    """incremental evolving Heterogeneous Graph Convolutional Network"""
    def __init__(self, node_type_dims, hidden_channels, out_channels, canonical_edge_types, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.node_types = list(node_type_dims.keys())
        self.canonical_edge_types = list(canonical_edge_types)
        self.hidden_channels = hidden_channels
        num_relations = len(self.canonical_edge_types)
        
        # Node type embedding layers
        self.input_lins = nn.ModuleDict()
        for ntype, in_dim in node_type_dims.items():
            self.input_lins[ntype] = nn.Linear(in_dim, hidden_channels)
        
        # Temporal encoding for incremental evolution
        self.temporal_enc = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Evolution memory for each node type (GRU-based)
        self.evolution_memory = nn.ModuleDict()
        for ntype in self.node_types:
            self.evolution_memory[ntype] = nn.GRUCell(hidden_channels, hidden_channels)
        
        # Heterogeneous graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations))
        
        self.lin = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, hetero_data):
        # Extract temporal information
        edge_times = None
        for edge_type in hetero_data.edge_types:
            if hasattr(hetero_data[edge_type], 'edge_attr'):
                edge_times = hetero_data[edge_type].edge_attr
                break
        
        if edge_times is None:
            # Use normalized timestamps if available
            for edge_type in hetero_data.edge_types:
                if hasattr(hetero_data[edge_type], 'edge_times_normalized'):
                    edge_times = hetero_data[edge_type].edge_times_normalized
                    break
        
        if edge_times is None:
            device = next(iter(self.input_lins.values())).weight.device
            edge_times = torch.zeros(1, 1, device=device)
        
        if edge_times.dim() == 1:
            edge_times = edge_times.unsqueeze(-1)
        
        # Temporal encoding
        mean_time = edge_times.mean().view(1, 1)
        time_emb = self.temporal_enc(mean_time).squeeze(0)
        
        # Node type embedding
        x_dict = {}
        memory_dict = {}
        for ntype in hetero_data.node_types:
            if ntype not in self.input_lins:
                continue
            if hasattr(hetero_data[ntype], 'x') and hetero_data[ntype].x is not None:
                x_dict[ntype] = self.input_lins[ntype](hetero_data[ntype].x)
                
                # Initialize memory for incremental evolution
                num_nodes = x_dict[ntype].size(0)
                memory_dict[ntype] = torch.zeros(num_nodes, self.hidden_channels, device=x_dict[ntype].device)
                
                # Update memory with temporal information
                time_emb_expanded = time_emb.unsqueeze(0).expand(num_nodes, -1)
                memory_dict[ntype] = self.evolution_memory[ntype](time_emb_expanded, memory_dict[ntype])
                
                # Combine node features with evolution memory
                x_dict[ntype] = x_dict[ntype] + memory_dict[ntype]
        
        # Flatten heterogeneous graph for RGCN
        x_all = torch.cat([x_dict[ntype] for ntype in self.node_types if ntype in x_dict], dim=0)
        
        edge_indices = []
        edge_type_tensors = []
        offset = 0
        node_offsets = {}
        for ntype in self.node_types:
            if ntype in x_dict:
                node_offsets[ntype] = offset
                offset += x_dict[ntype].size(0)
        
        for edge_type in self.canonical_edge_types:
            if edge_type not in hetero_data.edge_types:
                continue
            src_type, _, dst_type = edge_type
            if src_type not in node_offsets or dst_type not in node_offsets:
                continue
            rel_idx = self.canonical_edge_types.index(edge_type)
            edge_index = hetero_data[edge_type].edge_index
            edge_index_shifted = edge_index.clone()
            edge_index_shifted[0] += node_offsets[src_type]
            edge_index_shifted[1] += node_offsets[dst_type]
            edge_indices.append(edge_index_shifted)
            edge_type_tensors.append(torch.full((edge_index.size(1),), rel_idx, dtype=torch.long, device=edge_index.device))
        
        if len(edge_indices) > 0:
            edge_index_all = torch.cat(edge_indices, dim=1)
            edge_type_all = torch.cat(edge_type_tensors, dim=0)
        else:
            # No edges: return zero output with gradient support
            graph_emb = x_all.mean(dim=0, keepdim=True)
            return self.lin(graph_emb)
        
        # RGCN layers with incremental evolution
        x = x_all
        for conv in self.convs:
            x = conv(x, edge_index_all, edge_type_all)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Type-specific aggregation (simplified without attention to avoid bugs)
        if hasattr(hetero_data, 'batch_dict') and hetero_data.batch_dict:
            num_graphs = max(b.max().item() for b in hetero_data.batch_dict.values()) + 1
            graph_emb = torch.zeros(num_graphs, x.size(1), device=x.device)
            
            for ntype in self.node_types:
                if ntype in node_offsets and ntype in hetero_data.batch_dict:
                    start = node_offsets[ntype]
                    end = start + hetero_data.batch_dict[ntype].size(0)
                    ntype_x = x[start:end]
                    
                    # Simple mean pooling for each node type
                    ntype_emb = global_mean_pool(ntype_x, hetero_data.batch_dict[ntype], size=num_graphs)
                    graph_emb += ntype_emb
        else:
            graph_emb = x.mean(dim=0, keepdim=True)
        
        return self.lin(graph_emb)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}


def run_baseline_homo(graphs, model_name, args):
    """Run baseline on homogeneous graphs (TGAT, TGN, EvolveGCN)"""
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}')
    
    graphs = normalize_features(graphs)
    labels = [g.y.item() for g in graphs]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_results = []
    
    in_channels = graphs[0].x.size(1)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        logging.info(f"Fold {fold+1}/5")
        
        train_data = [graphs[i] for i in train_idx]
        test_data = [graphs[i] for i in test_idx]
        
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
        
        if model_name == 'TGAT':
            model = TGATModel(in_channels, 64, 2, num_layers=2, heads=4, dropout=0.3)
        elif model_name == 'TGN':
            model = TGNModel(in_channels, 64, 2, num_layers=2, dropout=0.3)
        elif model_name == 'EvolveGCN':
            model = EvolveGCNModel(in_channels, 64, 2, num_layers=2, dropout=0.3)
        else:
            raise ValueError(f"Unknown homo model: {model_name}")
        
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_f1 = 0
        patience = 15
        patience_counter = 0
        history = {'train_loss': [], 'test_precision': [], 'test_recall': [], 'test_f1': []}
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            test_metrics = evaluate(model, test_loader, device)
            history['train_loss'].append(train_loss)
            history['test_precision'].append(test_metrics['precision'])
            history['test_recall'].append(test_metrics['recall'])
            history['test_f1'].append(test_metrics['f1'])
            
            if test_metrics['f1'] > best_f1:
                best_f1 = test_metrics['f1']
                best_metrics = test_metrics
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
        
        best_idx = int(np.argmax(history['test_f1']))
        min_loss_idx = int(np.argmin(history['train_loss'])) if history['train_loss'] else 0
        fold_result = {
            'final_precision': history['test_precision'][best_idx],
            'final_recall': history['test_recall'][best_idx],
            'final_f1': history['test_f1'][best_idx],
            'p_at_min_loss': history['test_precision'][min_loss_idx],
            'r_at_min_loss': history['test_recall'][min_loss_idx],
            'f1_at_min_loss': history['test_f1'][min_loss_idx],
        }
        fold_results.append(fold_result)
        logging.info(f"Fold {fold+1} Best F1: {best_f1:.4f} | @min_train_loss: P={fold_result['p_at_min_loss']:.4f}, R={fold_result['r_at_min_loss']:.4f}, F1={fold_result['f1_at_min_loss']:.4f}")
    
    avg_results = {
        'avg_precision': np.mean([r['final_precision'] for r in fold_results]),
        'std_precision': np.std([r['final_precision'] for r in fold_results]),
        'avg_recall': np.mean([r['final_recall'] for r in fold_results]),
        'std_recall': np.std([r['final_recall'] for r in fold_results]),
        'avg_f1': np.mean([r['final_f1'] for r in fold_results]),
        'std_f1': np.std([r['final_f1'] for r in fold_results]),
        'avg_p_at_min_loss': np.mean([r['p_at_min_loss'] for r in fold_results]),
        'std_p_at_min_loss': np.std([r['p_at_min_loss'] for r in fold_results]),
        'avg_r_at_min_loss': np.mean([r['r_at_min_loss'] for r in fold_results]),
        'std_r_at_min_loss': np.std([r['r_at_min_loss'] for r in fold_results]),
        'avg_f1_at_min_loss': np.mean([r['f1_at_min_loss'] for r in fold_results]),
        'std_f1_at_min_loss': np.std([r['f1_at_min_loss'] for r in fold_results]),
    }
    return avg_results


def run_baseline_hetero(hetero_graphs, labels, model_name, args):
    """Run baseline on heterogeneous graphs (RGCN, HGT)"""
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}')
    
    hetero_graphs = normalize_hetero_features(hetero_graphs)
    node_types, edge_types, node_type_dims = get_hetero_metadata(hetero_graphs)
    metadata = (node_types, edge_types)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        logging.info(f"Fold {fold+1}/5")
        
        train_data = [hetero_graphs[i] for i in train_idx]
        test_data = [hetero_graphs[i] for i in test_idx]
        
        for i, g in enumerate(train_data):
            g.y = torch.tensor([labels[train_idx[i]]], dtype=torch.long)
        for i, g in enumerate(test_data):
            g.y = torch.tensor([labels[test_idx[i]]], dtype=torch.long)
        
        # Use DataLoader for batching (speed optimization)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        
        if model_name == 'RGCN':
            model = RGCNModel(node_type_dims, 64, 2, canonical_edge_types=edge_types, num_layers=2, dropout=0.3)
        elif model_name == 'HGT':
            model = HGTModel(node_type_dims, 64, 2, metadata, num_layers=2, num_heads=4, dropout=0.3)
        elif model_name == 'ieHGCN':
            model = ieHGCNModel(node_type_dims, 64, 2, canonical_edge_types=edge_types, num_layers=2, dropout=0.3)
        else:
            raise ValueError(f"Unknown hetero model: {model_name}")
        
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_f1 = 0
        patience = 15
        patience_counter = 0
        history = {'train_loss': [], 'test_precision': [], 'test_recall': [], 'test_f1': []}
        
        for epoch in range(args.epochs):
            logging.info(f"  Epoch {epoch+1}/{args.epochs} training ...")
            model.train()
            total_loss = 0
            num_samples = 0
            for batch in tqdm(train_loader, desc=f"Fold{fold+1} E{epoch+1}", leave=False):
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs
                num_samples += batch.num_graphs
            
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    pred = out.argmax(dim=1)
                    all_preds.append(pred.cpu().numpy())
                    all_labels.append(batch.y.cpu().numpy())
            
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            
            acc = accuracy_score(all_labels, all_preds)
            p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
            test_metrics = {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}
            avg_loss = total_loss / max(num_samples, 1)
            history['train_loss'].append(avg_loss)
            history['test_precision'].append(p)
            history['test_recall'].append(r)
            history['test_f1'].append(f1)
            
            logging.info(f"  Epoch {epoch+1} loss={avg_loss:.4f} test_f1={f1:.4f} P={p:.4f} R={r:.4f}")
            
            if test_metrics['f1'] > best_f1:
                best_f1 = test_metrics['f1']
                best_metrics = test_metrics
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
        
        best_idx = int(np.argmax(history['test_f1']))
        min_loss_idx = int(np.argmin(history['train_loss'])) if history['train_loss'] else 0
        fold_result = {
            'final_precision': history['test_precision'][best_idx],
            'final_recall': history['test_recall'][best_idx],
            'final_f1': history['test_f1'][best_idx],
            'p_at_min_loss': history['test_precision'][min_loss_idx],
            'r_at_min_loss': history['test_recall'][min_loss_idx],
            'f1_at_min_loss': history['test_f1'][min_loss_idx],
        }
        fold_results.append(fold_result)
        logging.info(f"Fold {fold+1} Best F1: {best_f1:.4f} | @min_train_loss: P={fold_result['p_at_min_loss']:.4f}, R={fold_result['r_at_min_loss']:.4f}, F1={fold_result['f1_at_min_loss']:.4f}")
    
    avg_results = {
        'avg_precision': np.mean([r['final_precision'] for r in fold_results]),
        'std_precision': np.std([r['final_precision'] for r in fold_results]),
        'avg_recall': np.mean([r['final_recall'] for r in fold_results]),
        'std_recall': np.std([r['final_recall'] for r in fold_results]),
        'avg_f1': np.mean([r['final_f1'] for r in fold_results]),
        'std_f1': np.std([r['final_f1'] for r in fold_results]),
        'avg_p_at_min_loss': np.mean([r['p_at_min_loss'] for r in fold_results]),
        'std_p_at_min_loss': np.std([r['p_at_min_loss'] for r in fold_results]),
        'avg_r_at_min_loss': np.mean([r['r_at_min_loss'] for r in fold_results]),
        'std_r_at_min_loss': np.std([r['r_at_min_loss'] for r in fold_results]),
        'avg_f1_at_min_loss': np.mean([r['f1_at_min_loss'] for r in fold_results]),
        'std_f1_at_min_loss': np.std([r['f1_at_min_loss'] for r in fold_results]),
    }
    return avg_results


def main():
    parser = argparse.ArgumentParser(description='Temporal/Dynamic GNN Baselines')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--models', type=str, default='TGAT,TGN,EvolveGCN,RGCN,HGT,ieHGCN',
                       help='Comma-separated list of models to run')
    args = parser.parse_args()
    
    os.makedirs('logs', exist_ok=True)
    os.makedirs('Result', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/temporal_baseline_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )
    
    datasets = ['Meta', 'ZipZap', '2DynEth']
    homo_graphs_paths = {
        '2DynEth': "Dataset/PhishCombine/graphs_home_bak0201/homo_2DynEth_graphs.pkl",
        'Meta': "Dataset/PhishCombine/graphs_home_bak0201/homo_Meta_graphs.pkl",
        'ZipZap': "Dataset/PhishCombine/graphs_home_bak0201/homo_ZipZap_graphs.pkl"
    }
    hetero_graphs_dirs = {
        '2DynEth': "Dataset/PhishCombine/graphs_2DynEth",
        'Meta': "Dataset/PhishCombine/graphs_Meta",
        'ZipZap': "Dataset/PhishCombine/graphs_ZipZap"
    }
    
    model_list = args.models.split(',')
    homo_models = ['TGAT', 'TGN', 'EvolveGCN']
    hetero_models = ['RGCN', 'HGT', 'ieHGCN']
    out_cols = ['dataset', 'model', 'Precision', 'Recall', 'F1-Score',
                'P@min_train_loss', 'R@min_train_loss', 'F1@min_train_loss']
    results_file = f"Result/temporal_baseline_{timestamp}.csv"
    first_row = True

    for dataset_name in datasets:
        logging.info(f"========== Processing Dataset: {dataset_name} ==========")
        
        homo_graphs = None
        hetero_graphs = None
        hetero_labels = None
        
        for model_name in model_list:
            model_name = model_name.strip()
            logging.info(f"--- Running {model_name} on {dataset_name} ---")
            
            if model_name in homo_models:
                if homo_graphs is None:
                    with open(homo_graphs_paths[dataset_name], 'rb') as f:
                        homo_graphs = pickle.load(f)
                result = run_baseline_homo(homo_graphs, model_name, args)
            
            elif model_name in hetero_models:
                if hetero_graphs is None:
                    hetero_graphs, hetero_labels = load_hetero_graphs(hetero_graphs_dirs[dataset_name])
                result = run_baseline_hetero(hetero_graphs, hetero_labels, model_name, args)
            
            else:
                logging.warning(f"Unknown model: {model_name}")
                continue
            
            row = {
                'dataset': dataset_name,
                'model': model_name,
                'Precision': f"{result['avg_precision']*100:.2f}±{result['std_precision']*100:.2f}",
                'Recall': f"{result['avg_recall']*100:.2f}±{result['std_recall']*100:.2f}",
                'F1-Score': f"{result['avg_f1']*100:.2f}±{result['std_f1']*100:.2f}",
                'P@min_train_loss': f"{result['avg_p_at_min_loss']*100:.2f}±{result['std_p_at_min_loss']*100:.2f}",
                'R@min_train_loss': f"{result['avg_r_at_min_loss']*100:.2f}±{result['std_r_at_min_loss']*100:.2f}",
                'F1@min_train_loss': f"{result['avg_f1_at_min_loss']*100:.2f}±{result['std_f1_at_min_loss']*100:.2f}",
            }
            df_row = pd.DataFrame([row])[out_cols]
            df_row.to_csv(results_file, mode='a' if not first_row else 'w', header=first_row, index=False)
            first_row = False
            logging.info(f"Appended result to {results_file}")
            print(df_row)
            
            logging.info(f"{model_name} on {dataset_name} - F1: {result['avg_f1']*100:.2f}±{result['std_f1']*100:.2f}")

    logging.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
