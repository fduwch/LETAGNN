"""Temporal aggregation comparison: LSTM, GRU, and attention."""

import torch
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
import logging
import os
import pickle
import warnings
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

warnings.filterwarnings("ignore")

from LETAGNN import (
    set_seed, normalize_features, TemporalGATEncoder, 
    split_graph_by_time_windows, LightWindowEncoderGNN
)
from TrainLETA import train_model, collate_preprocessed_graphs
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold


def setup_logging(script_name):
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/{script_name}_experiments_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
        force=True
    )
    logging.info(f"Logging initialized. Output saved to {log_filename}")
    return log_filename


class LSTMAggregator(nn.Module):
    """
    LSTM-based temporal aggregation for window embeddings.
    Processes the sequence of window embeddings using LSTM to capture 
    long-term and short-term temporal dependencies.
    """
    def __init__(self, embed_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, padded, mask):
        """
        Args:
            padded: (B, max_windows, D) window embeddings
            mask: (B, max_windows) boolean mask
        Returns:
            out: (B, D) aggregated embedding
            weights: (B, max_windows) dummy weights for compatibility
        """
        B, W, D = padded.shape
        
        # Get sequence lengths for pack_padded_sequence
        lengths = mask.sum(dim=1).cpu()  # (B,)
        
        # Sort by length (required for pack_padded_sequence)
        lengths_sorted, sorted_idx = lengths.sort(descending=True)
        padded_sorted = padded[sorted_idx]
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            padded_sorted, lengths_sorted.clamp(min=1), batch_first=True, enforce_sorted=True
        )
        
        # LSTM forward
        _, (h_n, _) = self.lstm(packed)  # h_n: (num_layers, B, hidden_dim)
        
        # Use the last layer's hidden state
        last_hidden = h_n[-1]  # (B, hidden_dim)
        
        # Unsort to original order
        _, unsort_idx = sorted_idx.sort()
        last_hidden = last_hidden[unsort_idx]
        
        # Project back to embed_dim
        out = self.output_proj(last_hidden)  # (B, D)
        
        # Return dummy attention weights for interface compatibility
        dummy_weights = mask.float() / mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        return out, dummy_weights


class GRUAggregator(nn.Module):
    """
    GRU-based temporal aggregation for window embeddings.
    Uses gated recurrent units to model sequential dependencies 
    with fewer parameters than LSTM.
    """
    def __init__(self, embed_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, padded, mask):
        """
        Args:
            padded: (B, max_windows, D) window embeddings
            mask: (B, max_windows) boolean mask
        Returns:
            out: (B, D) aggregated embedding
            weights: (B, max_windows) dummy weights for compatibility
        """
        B, W, D = padded.shape
        
        # Get sequence lengths
        lengths = mask.sum(dim=1).cpu()  # (B,)
        
        # Sort by length
        lengths_sorted, sorted_idx = lengths.sort(descending=True)
        padded_sorted = padded[sorted_idx]
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            padded_sorted, lengths_sorted.clamp(min=1), batch_first=True, enforce_sorted=True
        )
        
        # GRU forward
        _, h_n = self.gru(packed)  # h_n: (num_layers, B, hidden_dim)
        
        # Use the last layer's hidden state
        last_hidden = h_n[-1]  # (B, hidden_dim)
        
        # Unsort to original order
        _, unsort_idx = sorted_idx.sort()
        last_hidden = last_hidden[unsort_idx]
        
        # Project back to embed_dim
        out = self.output_proj(last_hidden)  # (B, D)
        
        # Return dummy attention weights for interface compatibility
        dummy_weights = mask.float() / mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        return out, dummy_weights


class AttentionAggregator(nn.Module):
    """
    Attention-based temporal aggregation for window embeddings.
    Uses transformer-style attention to dynamically weight window importance.
    This is the method used in LETAGNN (our proposed approach).
    """
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
        """
        Args:
            padded: (B, max_windows, D) window embeddings
            mask: (B, max_windows) boolean mask
        Returns:
            out: (B, D) aggregated embedding
            attn: (B, max_windows) attention weights
        """
        B, W, D = padded.shape
        scores = self.attn_mlp(padded).squeeze(-1).masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=1)
        out = (attn.unsqueeze(-1) * padded).sum(dim=1)
        # Residual from mean-pooled windows (masked mean)
        mean_pooled = (padded * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        out = out + self.residual_alpha * mean_pooled
        return out, attn


class TemporalAggregationModel(nn.Module):
    """
    Unified model for comparing temporal aggregation methods.
    Shares encoder and classifier, only differs in aggregation strategy.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, 
                 heads, dropout, edge_dim, aggregator_type='attention'):
        super().__init__()
        self.aggregator_type = aggregator_type
        
        # Shared encoder (same for all variants)
        self.encoder = TemporalGATEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim
        )
        encoder_out_dim = hidden_channels * heads
        
        # Light window encoder for efficient window embedding
        self.light_window_encoder = LightWindowEncoderGNN(in_channels, encoder_out_dim)
        
        # Window position encoding
        self.max_window_positions = 64
        self.window_pos_embed = nn.Embedding(self.max_window_positions, encoder_out_dim)
        
        # Temporal aggregator (LSTM, GRU, or Attention)
        if aggregator_type == 'lstm':
            self.aggregator = LSTMAggregator(encoder_out_dim, hidden_dim=128, num_layers=2, dropout=dropout)
        elif aggregator_type == 'gru':
            self.aggregator = GRUAggregator(encoder_out_dim, hidden_dim=128, num_layers=2, dropout=dropout)
        elif aggregator_type == 'attention':
            self.aggregator = AttentionAggregator(encoder_out_dim, hidden_dim=64)
        else:
            raise ValueError(f"Unknown aggregator_type: {aggregator_type}")
        
        # Global residual connection (optional, reduces windowing information loss)
        self.use_global_residual = True
        self.global_residual_alpha = nn.Parameter(torch.tensor(0.2))
        
        # Shared classifier
        self.classifier = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, window_batch, graph_window_indices, num_graphs, full_graph_batch=None):
        """
        Args:
            window_batch: Batched windows from all graphs
            graph_window_indices: Mapping from window to graph index
            num_graphs: Number of graphs in batch
            full_graph_batch: Optional full graphs for global residual
        """
        device = window_batch.x.device
        B = num_graphs
        
        # Encode windows using light encoder
        window_embeds = self.light_window_encoder(
            window_batch.x, window_batch.edge_index, window_batch.batch
        )  # (total_windows, D)
        
        # Group window embeddings by graph
        max_w = max((graph_window_indices == i).sum().item() for i in range(B))
        D = window_embeds.size(1)
        
        padded = torch.zeros(B, max_w, D, device=device)
        mask = torch.zeros(B, max_w, dtype=torch.bool, device=device)
        counts = torch.zeros(B, device=device)
        
        for i in range(B):
            graph_mask = (graph_window_indices == i)
            graph_windows = window_embeds[graph_mask]
            n_windows = graph_windows.size(0)
            padded[i, :n_windows] = graph_windows
            mask[i, :n_windows] = True
            counts[i] = n_windows
        
        # Add window position encoding (temporal order)
        pos_idx = torch.arange(max_w, device=device).clamp(max=self.max_window_positions - 1)
        padded = padded + self.window_pos_embed(pos_idx).unsqueeze(0)
        
        # Single-window shortcut
        if max_w == 1:
            graph_embeds = padded[:, 0, :]
            attn = mask.float()
        else:
            # Apply temporal aggregator (LSTM, GRU, or Attention)
            graph_embeds, attn = self.aggregator(padded, mask)
        
        # Global residual connection (optional)
        if self.use_global_residual and full_graph_batch is not None:
            full_graph_batch = full_graph_batch.to(device)
            h_full = self.encoder(
                full_graph_batch.x, full_graph_batch.edge_index,
                full_graph_batch.edge_times_normalized, full_graph_batch.edge_attr,
            )
            full_embeds = global_mean_pool(h_full, full_graph_batch.batch)  # (B, D)
            graph_embeds = graph_embeds + self.global_residual_alpha * full_embeds
        
        # Classify
        out = self.classifier(graph_embeds)
        
        if not self.training:
            attn_list = [attn[i, : int(counts[i].item())].cpu().numpy() for i in range(B)]
            return out, attn_list
        return out


def run_temporal_comparison(homo_graphs, args):
    """Run 5-fold CV comparing different temporal aggregation methods."""
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}')
    aggregator_type = args.aggregator_type
    epochs = args.epochs
    
    homo_graphs = normalize_features(homo_graphs)
    max_windows = 10
    min_window_size = 50
    min_edges_to_split = 300
    
    logging.info(f"Preprocessing graphs (aggregator={aggregator_type}, epochs={epochs})")
    preprocessed_graphs = []
    for graph in tqdm(homo_graphs, desc="Windowing graphs"):
        windows = split_graph_by_time_windows(
            graph, max_windows=max_windows, min_window_size=min_window_size, 
            min_edges_to_split=min_edges_to_split
        )
        preprocessed_graphs.append((windows, graph, graph.y.item()))
    
    all_labels = np.array([label for _, _, label in preprocessed_graphs])
    n_pos_total = (all_labels == 1).sum()
    n_neg_total = (all_labels == 0).sum()
    logging.info(f"Dataset: {n_pos_total} positive, {n_neg_total} negative")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
        logging.info(f"Fold {fold+1}/5: Training with {aggregator_type} aggregator")
        
        train_data = [preprocessed_graphs[i] for i in train_idx]
        test_data = [preprocessed_graphs[i] for i in test_idx]
        
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, 
                                 pin_memory=True, collate_fn=collate_preprocessed_graphs)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, 
                                pin_memory=True, collate_fn=collate_preprocessed_graphs)
        
        first_window = preprocessed_graphs[0][0][0]
        in_channels = first_window.x.size(1)
        edge_dim = first_window.edge_attr.size(1) if first_window.edge_attr is not None else 0
        
        # Create model with specified aggregator
        model = TemporalAggregationModel(
            in_channels=in_channels, hidden_channels=128, out_channels=2,
            num_layers=2, heads=4, dropout=0.3, edge_dim=edge_dim,
            aggregator_type=aggregator_type
        )
        
        train_config = {'device': device, 'num_epochs': epochs, 'lr': 0.001, 'patience': 10}
        model, history, _ = train_model(train_loader, test_loader, model, train_config)
        
        best_idx = np.argmax(history['test_f1']) if history['test_f1'] else 0
        precision_best = history['test_precision'][best_idx]
        recall_best = history['test_recall'][best_idx]
        f1_best = history['test_f1'][best_idx]
        
        min_loss_idx = int(np.argmin(history['train_loss'])) if history.get('train_loss') else 0
        p_at_min = history['test_precision'][min_loss_idx]
        r_at_min = history['test_recall'][min_loss_idx]
        f1_at_min = history['test_f1'][min_loss_idx]
        
        fold_results.append({
            'precision': precision_best, 'recall': recall_best, 'f1': f1_best,
            'p_at_min_loss': p_at_min, 'r_at_min_loss': r_at_min, 'f1_at_min_loss': f1_at_min,
        })
        logging.info(f"Fold {fold+1} Best F1: {f1_best:.4f} | @min_train_loss F1: {f1_at_min:.4f}")
    
    return {
        'avg_precision': np.mean([r['precision'] for r in fold_results]),
        'std_precision': np.std([r['precision'] for r in fold_results]),
        'avg_recall': np.mean([r['recall'] for r in fold_results]),
        'std_recall': np.std([r['recall'] for r in fold_results]),
        'avg_f1': np.mean([r['f1'] for r in fold_results]),
        'std_f1': np.std([r['f1'] for r in fold_results]),
        'avg_p_at_min_loss': np.mean([r['p_at_min_loss'] for r in fold_results]),
        'std_p_at_min_loss': np.std([r['p_at_min_loss'] for r in fold_results]),
        'avg_r_at_min_loss': np.mean([r['r_at_min_loss'] for r in fold_results]),
        'std_r_at_min_loss': np.std([r['r_at_min_loss'] for r in fold_results]),
        'avg_f1_at_min_loss': np.mean([r['f1_at_min_loss'] for r in fold_results]),
        'std_f1_at_min_loss': np.std([r['f1_at_min_loss'] for r in fold_results]),
    }


def main():
    parser = argparse.ArgumentParser(description='Temporal Aggregation Comparison: LSTM vs GRU vs Attention')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    
    setup_logging('temporal_aggregation')
    
    # Three aggregation methods to compare
    aggregator_types = ['lstm', 'gru', 'attention']
    
    datasets = ['Meta', 'ZipZap', '2DynEth']
    homo_graphs_paths = {
        '2DynEth': "Dataset/PhishCombine/graphs_home_bak0201/homo_2DynEth_graphs.pkl",
        'Meta': "Dataset/PhishCombine/graphs_home_bak0201/homo_Meta_graphs.pkl",
        'ZipZap': "Dataset/PhishCombine/graphs_home_bak0201/homo_ZipZap_graphs.pkl"
    }
    
    out_cols = ['dataset', 'aggregator', 'epochs', 'Precision', 'Recall', 'F1-Score',
                'P@min_train_loss', 'R@min_train_loss', 'F1@min_train_loss']
    os.makedirs('Result', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"Result/temporal_aggregation_{ts}.csv"
    first_row = True
    
    for dataset_name in datasets:
        logging.info(f"========== Dataset: {dataset_name} ==========")
        with open(homo_graphs_paths[dataset_name], 'rb') as f:
            graphs = pickle.load(f)
        
        # Clean up graph attributes
        for graph in graphs:
            standard_attrs = {'x', 'edge_index', 'edge_attr', 'y', 'edge_times_normalized', 'address'}
            current_attrs = set(graph.keys())
            for attr in current_attrs - standard_attrs:
                if hasattr(graph, attr):
                    delattr(graph, attr)
            if not hasattr(graph, 'edge_times_normalized') or graph.edge_times_normalized is None:
                graph.edge_times_normalized = torch.zeros(graph.edge_index.size(1))
        
        for aggregator_type in aggregator_types:
            current_args = argparse.Namespace(**vars(args))
            current_args.dataset = dataset_name
            current_args.aggregator_type = aggregator_type
            logging.info(f"--- Aggregator: {aggregator_type} ---")
            result = run_temporal_comparison(graphs, current_args)
            
            row = {
                'dataset': dataset_name,
                'aggregator': aggregator_type.upper(),
                'epochs': args.epochs,
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
    
    if first_row:
        logging.warning("No results to save.")
    else:
        df_final = pd.read_csv(results_file)
        df_final.sort_values(by=['dataset', 'aggregator'], inplace=True)
        df_final.to_csv(results_file, index=False)
        logging.info(f"Results saved to {results_file}")
        print("\n" + "="*80)
        print("FINAL RESULTS: Temporal Aggregation Comparison")
        print("="*80)
        print(df_final.to_string(index=False))
        print("="*80)


if __name__ == "__main__":
    main()
