"""
LETAGNN Model Definitions
=========================
- HomoGAT: Basic GAT for homogeneous graphs
- LETAGNN: Windowed GAT encoder + attention aggregation
- TemporalGATEncoder: Encoder used by LETAGNN
- Ablations: LETAGNN_NoWindow, GraphSAGE_Model, GAT_Model
"""

import torch
import numpy as np
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_features(graphs):
    """Normalize node and edge features across graphs."""
    print("Normalizing graph features...")
    
    max_node_dim = max(graph.x.size(1) for graph in graphs)
    print(f"Unified node feature dimension: {max_node_dim}")
    
    for i, graph in enumerate(graphs):
        if graph.x.size(1) < max_node_dim:
            padding = torch.zeros(graph.x.size(0), max_node_dim - graph.x.size(1))
            graphs[i].x = torch.cat([graph.x, padding], dim=1)
    
    has_edge_attr = any(hasattr(graph, 'edge_attr') and graph.edge_attr is not None for graph in graphs)
    if has_edge_attr:
        max_edge_dim = max(graph.edge_attr.size(1) for graph in graphs 
                          if hasattr(graph, 'edge_attr') and graph.edge_attr is not None)
        print(f"Unified edge feature dimension: {max_edge_dim}")
        
        for i, graph in enumerate(graphs):
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                if graph.edge_attr.size(1) < max_edge_dim:
                    padding = torch.zeros(graph.edge_attr.size(0), max_edge_dim - graph.edge_attr.size(1))
                    graphs[i].edge_attr = torch.cat([graph.edge_attr, padding], dim=1)
    
    node_sum = torch.zeros(max_node_dim)
    node_sum_sq = torch.zeros(max_node_dim)
    node_count = 0
    
    edge_sum = torch.zeros(max_edge_dim) if has_edge_attr else None
    edge_sum_sq = torch.zeros(max_edge_dim) if has_edge_attr else None
    edge_count = 0
    
    for graph in tqdm(graphs, desc="Computing feature statistics"):
        node_sum += graph.x.sum(dim=0)
        node_sum_sq += (graph.x ** 2).sum(dim=0)
        node_count += graph.x.size(0)
        
        if has_edge_attr and hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            edge_sum += graph.edge_attr.sum(dim=0)
            edge_sum_sq += (graph.edge_attr ** 2).sum(dim=0)
            edge_count += graph.edge_attr.size(0)
    
    node_mean = node_sum / max(1, node_count)
    node_std = torch.sqrt((node_sum_sq / max(1, node_count)) - (node_mean ** 2))
    node_std = torch.where(node_std > 1e-6, node_std, torch.ones_like(node_std))
    
    if has_edge_attr:
        edge_mean = edge_sum / max(1, edge_count)
        edge_std = torch.sqrt((edge_sum_sq / max(1, edge_count)) - (edge_mean ** 2))
        edge_std = torch.where(edge_std > 1e-6, edge_std, torch.ones_like(edge_std))
    
    for graph in tqdm(graphs, desc="Applying normalization"):
        graph.x = (graph.x - node_mean) / node_std
        graph.x = torch.clamp(graph.x, -5.0, 5.0)
        
        if has_edge_attr and hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            graph.edge_attr = (graph.edge_attr - edge_mean) / edge_std
            graph.edge_attr = torch.clamp(graph.edge_attr, -5.0, 5.0)
    
    return graphs


def split_graph_by_time_windows(graph, max_windows=10, min_window_size=50):
    """Split a graph into temporal windows based on edge order."""
    num_edges = graph.edge_index.size(1)
    if num_edges == 0:
        return [graph]

    max_edges_per_window = 10000
    size_from_max_windows = int(np.ceil(num_edges / max_windows))
    window_size = max(min_window_size, size_from_max_windows)
    window_size = min(window_size, max_edges_per_window)

    temporal_subgraphs = []
    for start_idx in range(0, num_edges, window_size):
        end_idx = min(start_idx + window_size, num_edges)
        if start_idx >= end_idx:
            continue
        
        window_edge_index = graph.edge_index[:, start_idx:end_idx]
        unique_nodes, remapped_edge_index = torch.unique(window_edge_index, return_inverse=True)
        remapped_edge_index = remapped_edge_index.reshape(2, -1)
        window_x = graph.x[unique_nodes]
        window_edge_attr = graph.edge_attr[start_idx:end_idx] if graph.edge_attr is not None else None
        if hasattr(graph, 'edge_times_normalized') and graph.edge_times_normalized is not None:
            window_edge_times = graph.edge_times_normalized[start_idx:end_idx]
        else:
            window_edge_times = torch.zeros(remapped_edge_index.size(1), device=graph.x.device)

        window_graph = Data(
            x=window_x,
            edge_index=remapped_edge_index,
            edge_attr=window_edge_attr,
            y=graph.y,
            edge_times_normalized=window_edge_times
        )
        temporal_subgraphs.append(window_graph)
    
    if len(temporal_subgraphs) > max_windows:
        indices = np.linspace(0, len(temporal_subgraphs) - 1, num=max_windows, dtype=int)
        temporal_subgraphs = [temporal_subgraphs[i] for i in indices]

    return temporal_subgraphs if temporal_subgraphs else [graph]


class WindowAttentionAggregator(nn.Module):
    """Attention-based aggregation over window embeddings."""
    def __init__(self, embed_dim, hidden_dim=64):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, window_embeds):
        attn_scores = self.attn_mlp(window_embeds)
        attn_weights = torch.softmax(attn_scores, dim=0)
        graph_embed = (attn_weights * window_embeds).sum(dim=0)
        return graph_embed, attn_weights.squeeze(-1)


class HomoGAT(torch.nn.Module):
    """Basic GAT model for homogeneous graphs."""
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, num_layers=2, heads=4, dropout=0.3):
        super(HomoGAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.convs = torch.nn.ModuleList([
            GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            for _ in range(num_layers - 1)
        ])
        
        self.lin1 = torch.nn.Linear(hidden_channels * heads, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        
        self.batch_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(hidden_channels * heads) for _ in range(num_layers)
        ])
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.batch_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x


class TemporalGATEncoder(torch.nn.Module):
    """Encode a graph/window to node embeddings with GAT and edge-time features."""
    def __init__(self, in_channels, hidden_channels=64, num_layers=2, heads=4, dropout=0.3, edge_dim=None):
        super(TemporalGATEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.heads = heads
        
        self.temporal_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_channels // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels // 4, hidden_channels // 4)
        )
        self.edge_processor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels // 4, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        self.edge_attr_projection = None
        if edge_dim is not None and edge_dim > 0:
            self.edge_attr_projection = torch.nn.Linear(edge_dim, hidden_channels // 2)

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=hidden_channels // 2)
        self.convs = torch.nn.ModuleList([
            GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, edge_dim=hidden_channels // 2)
            for _ in range(num_layers - 1)
        ])

        self.res_linears = torch.nn.ModuleList([
            torch.nn.Linear(hidden_channels * heads, hidden_channels * heads)
            for _ in range(num_layers - 1)
        ])
        self.layer_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(hidden_channels * heads) for _ in range(num_layers)
        ])

        self.jk_linear = torch.nn.Linear(hidden_channels * heads * num_layers, hidden_channels * heads)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(module.weight, 1.0)
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x, edge_index, edge_times_normalized=None, edge_attr=None):
        """Return final node embeddings for a graph/window."""
        temporal_features = None
        if edge_times_normalized is not None and edge_times_normalized.numel() > 0:
            temporal_features = self.temporal_encoder(edge_times_normalized.unsqueeze(-1))
            temporal_features = self.edge_processor(temporal_features)
        
        edge_features = self._process_edge_features(edge_attr, temporal_features)

        layer_outputs = []
        
        x = self.conv1(x, edge_index, edge_attr=edge_features)
        x = self.layer_norms[0](x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_outputs.append(x)
        
        prev_x = x
        for i, conv in enumerate(self.convs):
            x_main = conv(x, edge_index, edge_attr=edge_features)
            x_main = self.layer_norms[i+1](x_main)
            x_res = self.res_linears[i](prev_x)
            x = x_main + x_res
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)
            prev_x = x
            
        if len(layer_outputs) > 1:
            jk_x = torch.cat(layer_outputs, dim=1)
            x = self.jk_linear(jk_x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        return x

    def _process_edge_features(self, edge_attr, temporal_features):
        """Combine raw edge attributes and temporal features."""
        if edge_attr is not None and temporal_features is not None:
            if self.edge_attr_projection is not None:
                projected_edge_attr = self.edge_attr_projection(edge_attr)
                edge_features = projected_edge_attr + temporal_features
            else:
                edge_features = temporal_features
        elif temporal_features is not None:
            edge_features = temporal_features
        elif edge_attr is not None:
            if self.edge_attr_projection is not None:
                edge_features = self.edge_attr_projection(edge_attr)
            else:
                edge_features = None
        else:
            edge_features = None
        return edge_features


class LETAGNN(torch.nn.Module):
    """GAT-based encoder over temporal windows + attention aggregation."""
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, num_layers=2, heads=4, dropout=0.3,
                 window_batch_size=32, edge_dim=None):
        super(LETAGNN, self).__init__()
        self.window_batch_size = window_batch_size
        
        self.encoder = TemporalGATEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim
        )
        
        encoder_out_dim = hidden_channels * heads
        
        self.attention_aggregator = WindowAttentionAggregator(
            embed_dim=encoder_out_dim, 
            hidden_dim=hidden_channels
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(encoder_out_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, window_batch, graph_window_indices, num_graphs):
        """Process pre-split windows and classify each original graph."""
        device = window_batch.x.device

        window_embeds_list = []
        window_data_list = window_batch.to_data_list()
        window_chunks = [window_data_list[i:i + self.window_batch_size] for i in range(0, len(window_data_list), self.window_batch_size)]

        for chunk in window_chunks:
            chunk_batch = Batch.from_data_list(chunk).to(device)
            node_embeds = self.encoder(
                chunk_batch.x, chunk_batch.edge_index,
                chunk_batch.edge_times_normalized, chunk_batch.edge_attr
            )
            pooled_embeds = global_mean_pool(node_embeds, chunk_batch.batch)
            window_embeds_list.append(pooled_embeds)
        
        all_window_embeds = torch.cat(window_embeds_list, dim=0)

        batch_graph_embeds = []
        all_attn_weights = []
        for i in range(num_graphs):
            current_graph_window_embeds = all_window_embeds[graph_window_indices == i]
            
            if current_graph_window_embeds.shape[0] <= 1:
                if current_graph_window_embeds.shape[0] == 0:
                    embed_dim = self.encoder.jk_linear.out_features
                    graph_embed = torch.zeros(embed_dim, device=device)
                    attn_weights = torch.tensor([1.0], device=device)
                else:
                    graph_embed = current_graph_window_embeds.squeeze(0)
                    attn_weights = torch.tensor([1.0], device=device)
            else:
                graph_embed, attn_weights = self.attention_aggregator(current_graph_window_embeds)
            
            batch_graph_embeds.append(graph_embed)
            all_attn_weights.append(attn_weights.cpu().detach().numpy())
            
        final_batch_embeds = torch.stack(batch_graph_embeds, dim=0)
        out = self.classifier(final_batch_embeds)
        
        if not self.training:
            return out, all_attn_weights
        return out


class TemporalGAT(torch.nn.Module):
    """DEPRECATED: Use LETAGNN instead."""
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, num_layers=2, heads=4, dropout=0.3):
        super(TemporalGAT, self).__init__()
        raise NotImplementedError(
            "TemporalGAT is deprecated and has been replaced by the LETAGNN class. "
            "Please use LETAGNN, which implements windowing, encoding, and attention."
        )


class FocalLoss(torch.nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.1, balance_weight=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.balance_weight = balance_weight
        
    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        smooth_targets = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
        
        if self.label_smoothing > 0:
            smooth_targets = smooth_targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        
        log_probs = F.log_softmax(inputs, dim=1)
        loss = -(smooth_targets * log_probs).sum(dim=1)
        
        probs = torch.exp(log_probs)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weights = (1 - pt) ** self.gamma
        focal_loss = focal_weights * loss
        
        if self.alpha is not None:
            batch_alpha = self.alpha.gather(0, targets)
            focal_loss = batch_alpha * focal_loss
        
        if self.balance_weight > 0:
            with torch.no_grad():
                predicted = (probs[:, 1] > 0.5).long()
                tp = ((predicted == 1) & (targets == 1)).sum().float()
                fp = ((predicted == 1) & (targets == 0)).sum().float()
                fn = ((predicted == 0) & (targets == 1)).sum().float()
                dice_coefficient = (2 * tp) / (2 * tp + fp + fn + 1e-8)
                dice_loss = 1.0 - dice_coefficient
            focal_loss = focal_loss + self.balance_weight * dice_loss
        
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum() if self.reduction == 'sum' else focal_loss 


class LETAGNN_NoWindow(torch.nn.Module):
    """Ablation: process the whole graph without windowing."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=4, dropout=0.3, edge_dim=None):
        super().__init__()
        self.encoder = TemporalGATEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim
        )
        encoder_out_dim = hidden_channels * heads
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(encoder_out_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, batch_data):
        """Process a batch of graphs without windowing."""
        node_embeds = self.encoder(
            batch_data.x, batch_data.edge_index,
            batch_data.edge_times_normalized, batch_data.edge_attr
        )
        graph_embed = global_mean_pool(node_embeds, batch_data.batch)
        out = self.classifier(graph_embed)
        return out


class GraphSAGE_Model(torch.nn.Module):
    """Ablation: GraphSAGE encoder baseline."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.lns.append(torch.nn.LayerNorm(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        """Process a batch with GraphSAGE encoder."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.lns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x


class GAT_Model(torch.nn.Module):
    """Ablation: GAT encoder baseline."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=4, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.lns.append(torch.nn.LayerNorm(hidden_channels * heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.lns.append(torch.nn.LayerNorm(hidden_channels * heads))
        self.lin = torch.nn.Linear(hidden_channels * heads, out_channels)
        self.dropout = dropout

    def forward(self, data):
        """Process a batch with GAT encoder."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.lns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x 