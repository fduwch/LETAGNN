import torch
import numpy as np
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_features(graphs):
    """Normalize graph features"""
    print("Normalizing graph features...")
    
    # Unify feature dimensions
    max_node_dim = max(graph.x.size(1) for graph in graphs)
    print(f"Unified node feature dimension: {max_node_dim}")
    
    for i, graph in enumerate(graphs):
        if graph.x.size(1) < max_node_dim:
            padding = torch.zeros(graph.x.size(0), max_node_dim - graph.x.size(1))
            graphs[i].x = torch.cat([graph.x, padding], dim=1)
    
    # Handle edge features
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
    
    # Compute statistics
    node_sum = torch.zeros(max_node_dim)
    node_sum_sq = torch.zeros(max_node_dim)
    node_count = 0
    
    edge_sum = torch.zeros(max_edge_dim) if has_edge_attr else None
    edge_sum_sq = torch.zeros(max_edge_dim) if has_edge_attr else None
    edge_count = 0
    
    # First pass: compute mean and std
    for graph in tqdm(graphs, desc="Computing feature statistics"):
        node_sum += graph.x.sum(dim=0)
        node_sum_sq += (graph.x ** 2).sum(dim=0)
        node_count += graph.x.size(0)
        
        if has_edge_attr and hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            edge_sum += graph.edge_attr.sum(dim=0)
            edge_sum_sq += (graph.edge_attr ** 2).sum(dim=0)
            edge_count += graph.edge_attr.size(0)
    
    # Calculate mean and std
    node_mean = node_sum / max(1, node_count)
    node_std = torch.sqrt((node_sum_sq / max(1, node_count)) - (node_mean ** 2))
    node_std = torch.where(node_std > 1e-6, node_std, torch.ones_like(node_std))
    
    if has_edge_attr:
        edge_mean = edge_sum / max(1, edge_count)
        edge_std = torch.sqrt((edge_sum_sq / max(1, edge_count)) - (edge_mean ** 2))
        edge_std = torch.where(edge_std > 1e-6, edge_std, torch.ones_like(edge_std))
    
    # Second pass: apply normalization
    for graph in tqdm(graphs, desc="Applying normalization"):
        graph.x = (graph.x - node_mean) / node_std
        graph.x = torch.clamp(graph.x, -5.0, 5.0)
        
        if has_edge_attr and hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            graph.edge_attr = (graph.edge_attr - edge_mean) / edge_std
            graph.edge_attr = torch.clamp(graph.edge_attr, -5.0, 5.0)
    
    return graphs


def split_graph_by_time_gap(graph, time_gap_days=3, min_edges_to_split=300, max_windows=30):
    """
    Splits a graph into temporal windows based on time gaps between transactions.
    Creates a new window when the time difference exceeds the specified threshold.
    
    Args:
        graph: Input graph with edge_times_normalized
        time_gap_days: Time gap threshold in days to trigger a new window
        min_edges_to_split: Minimum edges to consider splitting
        max_windows: Maximum number of windows to create
    
    Returns:
        List of temporal subgraphs (windows)
    """
    num_edges = graph.edge_index.size(1)
    
    # Handle empty graph
    if num_edges == 0:
        empty_t = torch.zeros(0, dtype=torch.float, device=graph.x.device)
        w = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr,
                 y=graph.y, edge_times_normalized=empty_t)
        return [w]
    
    # Don't split if too few edges
    if num_edges <= min_edges_to_split:
        et = graph.edge_times_normalized if hasattr(graph, 'edge_times_normalized') and graph.edge_times_normalized is not None else torch.zeros(num_edges, dtype=torch.float, device=graph.x.device)
        w = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr,
                 y=graph.y, edge_times_normalized=et)
        return [w]
    
    # Get edge timestamps (assume edge_times_normalized are in days or normalized format)
    if hasattr(graph, 'edge_times_normalized') and graph.edge_times_normalized is not None:
        edge_times = graph.edge_times_normalized
    else:
        # Fallback: use uniform spacing
        edge_times = torch.linspace(0, 1, num_edges, device=graph.x.device)
    
    # Find split points based on time gaps
    split_indices = [0]
    for i in range(1, num_edges):
        time_diff = abs(edge_times[i] - edge_times[i-1])
        # Normalize time_gap_days to match edge_times scale (assume edge_times normalized to [0,1])
        if edge_times.max() <= 1.0:
            # If normalized, convert days to fraction of time range
            normalized_gap = time_gap_days / 365.0  # Rough normalization
        else:
            normalized_gap = time_gap_days
        
        if time_diff > normalized_gap:
            split_indices.append(i)
    
    split_indices.append(num_edges)
    
    # Create windows based on split points
    temporal_subgraphs = []
    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]
        
        if start_idx >= end_idx:
            continue
        
        # Extract edges for the current window
        window_edge_index = graph.edge_index[:, start_idx:end_idx]
        unique_nodes, remapped_edge_index = torch.unique(window_edge_index, return_inverse=True)
        remapped_edge_index = remapped_edge_index.reshape(2, -1)
        
        window_x = graph.x[unique_nodes]
        window_edge_attr = graph.edge_attr[start_idx:end_idx] if graph.edge_attr is not None else None
        window_edge_times = edge_times[start_idx:end_idx]
        
        window_graph = Data(
            x=window_x,
            edge_index=remapped_edge_index,
            edge_attr=window_edge_attr,
            y=graph.y,
            edge_times_normalized=window_edge_times
        )
        temporal_subgraphs.append(window_graph)
    
    # Limit to max_windows if necessary
    if len(temporal_subgraphs) > max_windows:
        indices = np.linspace(0, len(temporal_subgraphs) - 1, num=max_windows, dtype=int)
        temporal_subgraphs = [temporal_subgraphs[i] for i in indices]
    
    return temporal_subgraphs if temporal_subgraphs else [graph]


def split_graph_by_time_windows(graph, max_windows=10, min_window_size=50, min_edges_to_split=300):
    """
    Splits a single graph into multiple temporal subgraphs (windows) based on edge order.
    For each window, it creates a subgraph containing only the nodes and edges
    present in that window, which is crucial for memory efficiency with large graphs.

    min_edges_to_split: only split into windows when num_edges > min_edges_to_split;
    otherwise the graph is returned as a single window (no temporal split).
    """
    num_edges = graph.edge_index.size(1)
    if num_edges == 0:
        # Return a single window with same keys as normal windows (no edge_times)
        # so Batch.from_data_list does not KeyError when mixing with other windows.
        empty_t = torch.zeros(0, dtype=torch.float, device=graph.x.device)
        w = Data(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            y=graph.y,
            edge_times_normalized=empty_t,
        )
        return [w]

    # If edge count does not exceed threshold, do not split: return graph as a single window.
    if num_edges <= min_edges_to_split:
        if hasattr(graph, 'edge_times_normalized') and graph.edge_times_normalized is not None:
            et = graph.edge_times_normalized
        else:
            et = torch.zeros(num_edges, dtype=torch.float, device=graph.x.device)
        w = Data(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            y=graph.y,
            edge_times_normalized=et,
        )
        return [w]

    # Heuristic to prevent extremely large windows for graphs with massive numbers of edges.
    max_edges_per_window = 10000

    # Determine window size, ensuring it's not smaller than min_window_size or excessively large.
    size_from_max_windows = int(np.ceil(num_edges / max_windows))
    window_size = max(min_window_size, size_from_max_windows)
    window_size = min(window_size, max_edges_per_window)

    temporal_subgraphs = []
    for start_idx in range(0, num_edges, window_size):
        end_idx = min(start_idx + window_size, num_edges)
        
        # This check prevents creating empty windows at the end.
        if start_idx >= end_idx:
            continue
            
        # Extract edges for the current window
        window_edge_index = graph.edge_index[:, start_idx:end_idx]
        
        # Get unique nodes in the current window and create a subgraph
        unique_nodes, remapped_edge_index = torch.unique(window_edge_index, return_inverse=True)
        
        # The remapped_edge_index is flat, so reshape it.
        remapped_edge_index = remapped_edge_index.reshape(2, -1)
        
        # Subset the node features for the nodes in this window
        window_x = graph.x[unique_nodes]

        # Extract corresponding edge attributes and temporal data
        window_edge_attr = graph.edge_attr[start_idx:end_idx] if graph.edge_attr is not None else None
        if hasattr(graph, 'edge_times_normalized') and graph.edge_times_normalized is not None:
            window_edge_times = graph.edge_times_normalized[start_idx:end_idx]
        else:
            window_edge_times = torch.zeros(remapped_edge_index.size(1), device=graph.x.device)

        # Create a new Data object for the subgraph
        window_graph = Data(
            x=window_x,
            edge_index=remapped_edge_index,
            edge_attr=window_edge_attr,
            y=graph.y,
            edge_times_normalized=window_edge_times
        )
        temporal_subgraphs.append(window_graph)
    
    # If capping window_size resulted in more windows than max_windows, sample them.
    if len(temporal_subgraphs) > max_windows:
        # Sample windows evenly across the temporal span
        indices = np.linspace(0, len(temporal_subgraphs) - 1, num=max_windows, dtype=int)
        temporal_subgraphs = [temporal_subgraphs[i] for i in indices]

    # Ensure we always return a list of graphs. If no subgraphs were created, return the original.
    return temporal_subgraphs if temporal_subgraphs else [graph]


class LightWindowEncoder(nn.Module):
    """Cheap per-window embedding: mean pool of node features + linear. No GNN, so speed ~= NoWindow when used as aux."""
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        pooled = global_mean_pool(x, batch)  # (num_windows, in_channels)
        return self.proj(pooled)  # (num_windows, embed_dim)


class LightWindowEncoderGNN(nn.Module):
    """1-layer GNN per window for structure-aware window embedding; still light (one layer, small hidden)."""
    def __init__(self, in_channels: int, embed_dim: int, hidden_small: int = 32):
        super().__init__()
        self.conv = SAGEConv(in_channels, hidden_small)
        self.norm = nn.LayerNorm(hidden_small)
        self.proj = nn.Linear(hidden_small, embed_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index)
        h = self.norm(h)
        h = F.relu(h)
        pooled = global_mean_pool(h, batch)
        return self.proj(pooled)


class BatchedWindowAttentionAggregator(nn.Module):
    """Batched attention over window embeddings. (B, max_windows, D) + (B, max_windows) mask -> (B, D).
    Includes residual from mean-pooled windows for stability."""
    def __init__(self, embed_dim, hidden_dim=64):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Learnable residual weight: fallback to mean when attention is weak
        self.residual_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, padded, mask):
        B, W, D = padded.shape
        scores = self.attn_mlp(padded).squeeze(-1).masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=1)
        out = (attn.unsqueeze(-1) * padded).sum(dim=1)
        # Residual from mean-pooled windows (masked mean)
        mean_pooled = (padded * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        out = out + self.residual_alpha * mean_pooled
        return out, attn


class HomoGAT(torch.nn.Module):
    """Basic GAT model for homogeneous graphs"""
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
        # First layer
        x = self.conv1(x, edge_index)
        x = self.batch_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling and output
        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        return x


class TemporalGATEncoder(torch.nn.Module):
    """
    Encodes a single graph (or subgraph/window) to produce node embeddings.
    This is a refactored version of the original TemporalGAT, focused on encoding.
    """
    def __init__(self, in_channels, hidden_channels=64, num_layers=2, heads=4, dropout=0.3, edge_dim=None):
        super(TemporalGATEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.heads = heads
        
        # Temporal feature processing
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
        
        # Static edge feature processing
        self.edge_attr_projection = None
        if edge_dim is not None and edge_dim > 0:
            # This projection is now created in __init__
            self.edge_attr_projection = torch.nn.Linear(edge_dim, hidden_channels // 2)

        # GAT layers
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=hidden_channels // 2)
        self.convs = torch.nn.ModuleList([
            GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, edge_dim=hidden_channels // 2)
            for _ in range(num_layers - 1)
        ])

        # Residual connections and normalization
        self.res_linears = torch.nn.ModuleList([
            torch.nn.Linear(hidden_channels * heads, hidden_channels * heads)
            for _ in range(num_layers - 1)
        ])
        self.layer_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(hidden_channels * heads) for _ in range(num_layers)
        ])

        # Jumping Knowledge
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
        """
        Forward pass that returns final node embeddings.
        """
        # 1. Process edge features (temporal and otherwise)
        temporal_features = None
        if edge_times_normalized is not None and edge_times_normalized.numel() > 0:
            temporal_features = self.temporal_encoder(edge_times_normalized.unsqueeze(-1))
            temporal_features = self.edge_processor(temporal_features)
        
        edge_features = self._process_edge_features(edge_attr, temporal_features)

        # 2. GAT forward pass
        layer_outputs = []
        
        # First GAT layer
        x = self.conv1(x, edge_index, edge_attr=edge_features)
        x = self.layer_norms[0](x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_outputs.append(x)
        
        # Subsequent GAT layers with residual connections
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
            
        # 3. Jumping Knowledge connection
        if len(layer_outputs) > 1:
            jk_x = torch.cat(layer_outputs, dim=1)
            x = self.jk_linear(jk_x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        return x # Return final node embeddings

    def _process_edge_features(self, edge_attr, temporal_features):
        """
        Combines raw edge attributes with temporal features.
        """
        if edge_attr is not None and temporal_features is not None:
            # Project edge_attr to match temporal_features dimension if necessary
            if self.edge_attr_projection is not None:
                projected_edge_attr = self.edge_attr_projection(edge_attr)
                edge_features = projected_edge_attr + temporal_features
            else:
                # Fallback if edge_attr exists but projection was not created (e.g., edge_dim=0)
                edge_features = temporal_features

        elif temporal_features is not None:
            edge_features = temporal_features
        
        elif edge_attr is not None:
            if self.edge_attr_projection is not None:
                edge_features = self.edge_attr_projection(edge_attr)
            else:
                # This case should ideally not be hit if edge_dim > 0 was handled in init
                edge_features = None
        else:
            edge_features = None
            
        return edge_features

    def _initialize_weights_for_layer(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)


class LETAGNN(torch.nn.Module):
    """
    LETAGNN: Temporal-Window GAT with Attention Aggregation.
    This is the main model that implements the full phishing detection pipeline.
    It replaces the old TemporalGAT class.
    """
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, num_layers=2, heads=4, dropout=0.3,
                 window_batch_size=32, edge_dim=None, use_global_residual=True):
        super(LETAGNN, self).__init__()
        self.window_batch_size = window_batch_size # For micro-batching windows
        self.use_global_residual = use_global_residual  # Reduce info loss: residual from full-graph encoding
        
        # Subgraph encoder
        self.encoder = TemporalGATEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim
        )
        
        encoder_out_dim = hidden_channels * heads
        
        # Window position embedding (max 64 windows) so aggregator knows temporal order
        self.max_window_positions = 64
        self.window_pos_embed = nn.Embedding(self.max_window_positions, encoder_out_dim)
        
        self.attention_aggregator = BatchedWindowAttentionAggregator(
            embed_dim=encoder_out_dim,
            hidden_dim=hidden_channels,
        )
        
        # Global residual: learnable weight for full-graph embedding (reduces windowing info loss)
        if use_global_residual:
            self.global_residual_alpha = nn.Parameter(torch.tensor(0.2))
        
        # Final classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(encoder_out_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, window_batch, graph_window_indices, num_graphs, full_graph_batch=None):
        device = window_batch.x.device
        B = num_graphs
        gwi = graph_window_indices.long()

        window_list = window_batch.to_data_list()
        embeds_list = []
        for i in range(0, len(window_list), self.window_batch_size):
            chunk = Batch.from_data_list(window_list[i : i + self.window_batch_size]).to(device)
            h = self.encoder(
                chunk.x, chunk.edge_index,
                chunk.edge_times_normalized, chunk.edge_attr,
            )
            embeds_list.append(global_mean_pool(h, chunk.batch))
        E = torch.cat(embeds_list, dim=0)
        D = E.size(-1)

        counts = torch.bincount(gwi, minlength=B)
        max_w = int(counts.max().item())
        padded = torch.zeros(B, max_w, D, device=device, dtype=E.dtype)
        mask = torch.zeros(B, max_w, dtype=torch.bool, device=device)
        for i in range(B):
            n = int(counts[i].item())
            idx = (gwi == i).nonzero(as_tuple=True)[0]
            padded[i, :n] = E[idx]
            mask[i, :n] = True

        # Add window position encoding (temporal order)
        pos_idx = torch.arange(max_w, device=device).clamp(max=self.max_window_positions - 1)
        padded = padded + self.window_pos_embed(pos_idx).unsqueeze(0)

        # Single-window shortcut: no need for attention over one vector
        if max_w == 1:
            graph_embeds = padded[:, 0, :]
            attn = mask.float()
        else:
            graph_embeds, attn = self.attention_aggregator(padded, mask)

        # Global residual: add full-graph encoding to reduce windowing info loss
        if self.use_global_residual and full_graph_batch is not None:
            full_graph_batch = full_graph_batch.to(device)
            h_full = self.encoder(
                full_graph_batch.x, full_graph_batch.edge_index,
                full_graph_batch.edge_times_normalized, full_graph_batch.edge_attr,
            )
            full_embeds = global_mean_pool(h_full, full_graph_batch.batch)  # (B, D)
            graph_embeds = graph_embeds + self.global_residual_alpha * full_embeds

        out = self.classifier(graph_embeds)

        if not self.training:
            attn_list = [attn[i, : int(counts[i].item())].cpu().numpy() for i in range(B)]
            return out, attn_list
        return out


class TemporalGAT(torch.nn.Module):
    """
    DEPRECATED: This class is preserved for reference but is replaced by LETAGNN.
    The new LETAGNN model encapsulates the full logic of windowing and attention.
    """
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, num_layers=2, heads=4, dropout=0.3):
        super(TemporalGAT, self).__init__()
        raise NotImplementedError(
            "TemporalGAT is deprecated and has been replaced by the LETAGNN class. "
            "Please update your code to use LETAGNN, which implements the complete "
            "windowing, encoding, and attention mechanism."
        )

class FocalLoss(torch.nn.Module):
    """Focal Loss for handling class imbalance"""
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
        
        # Precision-Recall balance constraint using Dice Loss
        if self.balance_weight > 0:
            with torch.no_grad():
                predicted = (probs[:, 1] > 0.5).long()
                
                # Calculate components for Dice Loss
                tp = ((predicted == 1) & (targets == 1)).sum().float()
                fp = ((predicted == 1) & (targets == 0)).sum().float()
                fn = ((predicted == 0) & (targets == 1)).sum().float()
                
                # Dice Loss is 1 - Dice Coefficient
                dice_coefficient = (2 * tp) / (2 * tp + fp + fn + 1e-8)
                dice_loss = 1.0 - dice_coefficient
            
            # Add the balanced loss component
            focal_loss = focal_loss + self.balance_weight * dice_loss
        
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum() if self.reduction == 'sum' else focal_loss 


class LETAGNN_NoWindow(torch.nn.Module):
    """
    Ablation model for LETAGNN that removes the temporal windowing mechanism.
    It processes the entire graph at once using the TemporalGATEncoder.
    """
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
        """Processes a batch of graphs without windowing."""
        node_embeds = self.encoder(
            batch_data.x, batch_data.edge_index,
            batch_data.edge_times_normalized, batch_data.edge_attr
        )
        # Pool node embeddings to get a graph-level embedding
        graph_embed = global_mean_pool(node_embeds, batch_data.batch)
        out = self.classifier(graph_embed)
        return out


class LETAGNN_LightWindow(torch.nn.Module):
    """
    Enhances no-window baseline with a light temporal-window branch (paper innovation).
    - Main path: one full-graph encode (same as NoWindow) -> fast, keeps performance.
    - Window path: 1-layer GNN per window + window attention -> temporal signal.
    - Fuse: concat(main, window) -> Linear(2D, D) with init so output ≈ main at start; training can add window signal.
    """
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, num_layers=2, heads=4, dropout=0.3, edge_dim=None):
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
        self.light_window_encoder = LightWindowEncoderGNN(in_channels, encoder_out_dim)
        self.max_window_positions = 64
        self.window_pos_embed = nn.Embedding(self.max_window_positions, encoder_out_dim)
        self.window_aggregator = BatchedWindowAttentionAggregator(
            embed_dim=encoder_out_dim,
            hidden_dim=hidden_channels,
        )
        # Concat fusion: init so output ≈ main at start (no window noise); training can add window signal
        self.fusion_proj = nn.Linear(encoder_out_dim * 2, encoder_out_dim)
        with torch.no_grad():
            self.fusion_proj.weight[:, :encoder_out_dim] = torch.eye(encoder_out_dim)
            self.fusion_proj.weight[:, encoder_out_dim:] = 0.0
            self.fusion_proj.bias.zero_()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(encoder_out_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, window_batch, graph_window_indices, num_graphs, full_graph_batch=None):
        device = window_batch.x.device
        B = num_graphs
        gwi = graph_window_indices.long()

        # Main path: full-graph encode (one forward, same as NoWindow)
        if full_graph_batch is None:
            raise ValueError("LETAGNN_LightWindow requires full_graph_batch (main path).")
        full_graph_batch = full_graph_batch.to(device)
        h_main = self.encoder(
            full_graph_batch.x, full_graph_batch.edge_index,
            full_graph_batch.edge_times_normalized, full_graph_batch.edge_attr,
        )
        graph_embed_main = global_mean_pool(h_main, full_graph_batch.batch)

        # Window path: 1-layer GNN per window + attention
        E_light = self.light_window_encoder(
            window_batch.x, window_batch.edge_index, window_batch.batch
        )
        D = E_light.size(-1)
        counts = torch.bincount(gwi, minlength=B)
        max_w = int(counts.max().item())
        padded = torch.zeros(B, max_w, D, device=device, dtype=E_light.dtype)
        mask = torch.zeros(B, max_w, dtype=torch.bool, device=device)
        for i in range(B):
            n = int(counts[i].item())
            idx = (gwi == i).nonzero(as_tuple=True)[0]
            padded[i, :n] = E_light[idx]
            mask[i, :n] = True
        pos_idx = torch.arange(max_w, device=device).clamp(max=self.max_window_positions - 1)
        padded = padded + self.window_pos_embed(pos_idx).unsqueeze(0)
        if max_w == 1:
            graph_embed_window = padded[:, 0, :]
        else:
            graph_embed_window, _ = self.window_aggregator(padded, mask)

        graph_embed = self.fusion_proj(
            torch.cat([graph_embed_main, graph_embed_window], dim=1)
        )
        out = self.classifier(graph_embed)
        if not self.training:
            return out, [[]] * B  # no per-window attn; compatible with evaluate_model
        return out


class GraphSAGE_Model(torch.nn.Module):
    """
    Ablation model using GraphSAGE as the encoder.
    This provides a baseline to evaluate the effectiveness of the GAT-based encoder.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.lns.append(torch.nn.LayerNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
        
        # Classifier
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        """Processes a batch of graphs using GraphSAGE."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.lns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pool node embeddings to get a graph-level embedding
        x = global_mean_pool(x, batch)
        
        # Final classification
        x = self.lin(x)
        return x


class GAT_Model(torch.nn.Module):
    """
    Ablation model using GAT as the encoder.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=4, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()

        # Input layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.lns.append(torch.nn.LayerNorm(hidden_channels * heads))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.lns.append(torch.nn.LayerNorm(hidden_channels * heads))
        
        # Classifier
        self.lin = torch.nn.Linear(hidden_channels * heads, out_channels)
        self.dropout = dropout

    def forward(self, data):
        """Processes a batch of graphs using GAT."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.lns[i](x)
            x = F.elu(x) # Using ELU for GAT
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pool node embeddings to get a graph-level embedding
        x = global_mean_pool(x, batch)
        
        # Final classification
        x = self.lin(x)
        return x 