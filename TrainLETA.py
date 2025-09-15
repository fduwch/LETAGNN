"""
LETAGNN Training Script - Temporal Window Splitting + Transformer Attention
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

# Optional GPU memory cap (uncomment to enable)
# if torch.cuda.is_available():
#     torch.cuda.set_per_process_memory_fraction(0.2)

warnings.filterwarnings("ignore")

from LETAGNN import (
    set_seed, normalize_features, LETAGNN, FocalLoss, LETAGNN_NoWindow, GraphSAGE_Model,
    split_graph_by_time_windows
)
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

def setup_logging(script_name):
    """Setup a single log file for the entire script run."""
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/{script_name}_experiments_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ],
        force=True 
    )
    logging.info(f"Logging initialized. All output will be saved to {log_filename}")
    return log_filename

def collate_preprocessed_graphs(batch):
    """
    Collate function for pre-windowed graphs.
    Each batch item: (list_of_windows, label).
    Returns: (window_batch, graph_window_indices, labels, num_graphs)
    """
    all_windows = [item for sublist in [b[0] for b in batch] for item in sublist]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    
    graph_window_indices = []
    for i, (windows, _) in enumerate(batch):
        graph_window_indices.extend([i] * len(windows))
        
    window_batch = Batch.from_data_list(all_windows)
    
    return window_batch, torch.tensor(graph_window_indices), labels, len(batch)

def train_model(train_loader, test_loader, model, config):
    device = config['device']
    num_epochs = config['num_epochs']
    lr = config.get('lr', 0.0005)
    weight_decay = config.get('weight_decay', 5e-4)
    patience = config.get('patience', 15)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=lr/20
    )

    # Use provided class weights or compute from training data
    if 'class_weights' in config and config['class_weights'] is not None:
        class_weights = config['class_weights']
    else:
        class_weights = _calculate_class_weights(train_loader, device)
    print(f"Using class_weights: {class_weights}")
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    history = _initialize_history()
    early_stop_counter = 0
    best_f1 = 0
    inference_time_at_best_f1 = 0
    best_model_state = None

    for epoch in range(num_epochs):
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        model.train()
        total_loss = 0
        num_graphs_processed = 0
        history['lr'].append(optimizer.param_groups[0]['lr'])

        for window_batch, graph_window_indices, labels, num_graphs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            window_batch = window_batch.to(device)
            graph_window_indices = graph_window_indices.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(window_batch, graph_window_indices, num_graphs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * num_graphs
            num_graphs_processed += num_graphs
            
        scheduler.step()
        
        avg_loss = total_loss / max(1, num_graphs_processed)
        history['train_loss'].append(avg_loss)
        
        train_metrics, _ = evaluate_model(model, train_loader, device)
        test_metrics, _ = evaluate_model(model, test_loader, device)
        
        current_inference_time = test_metrics.pop('inference_time_s')

        for key in ['accuracy', 'precision', 'recall', 'f1']:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'test_{key}'].append(test_metrics[key])
            
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            inference_time_at_best_f1 = current_inference_time
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        if epoch % 2 == 0:
            logging.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Train F1={train_metrics['f1']:.4f}, Test Precision={test_metrics['precision']:.4f}, Test Recall={test_metrics['recall']:.4f}, Test F1={test_metrics['f1']:.4f}")
            
        if early_stop_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    return model, history, inference_time_at_best_f1


def evaluate_model(model, loader, device):
    """Evaluate model performance on a given dataset."""
    model.eval()
    all_preds, all_labels, all_attention_weights = [], [], []
    total_inference_time = 0
    graph_idx_counter = 0
    
    with torch.no_grad():
        for window_batch, graph_window_indices, labels, num_graphs in loader:
            window_batch = window_batch.to(device)
            graph_window_indices = graph_window_indices.to(device)
            labels = labels.to(device)
            
            start_time = time.time()
            outputs, attn_weights = model(window_batch, graph_window_indices, num_graphs)
            total_inference_time += time.time() - start_time
            
            probs = F.softmax(outputs, dim=1)[:, 1]
            preds = (probs > 0.5).long()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            for i in range(num_graphs):
                all_attention_weights.append({
                    'graph_index': graph_idx_counter + i,
                    'weights': attn_weights[i]
                })
            graph_idx_counter += num_graphs

    if not all_labels:
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'inference_time_s': 0}, []

    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    num_samples = len(loader.dataset)
    avg_inference_time = total_inference_time / num_samples if num_samples > 0 else 0

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
        'inference_time_s': avg_inference_time
    }
    return metrics, all_attention_weights


def _calculate_class_weights(train_loader, device, is_preprocessed=True):
    """Calculate class weights for balanced training on preprocessed data."""
    class_counts = torch.zeros(2)
    for _, _, labels, _ in train_loader:
        for i in range(2):
            class_counts[i] += (labels == i).sum().item()
    
    weights = class_counts.sum() / (2 * class_counts + 1e-6)
    return weights.to(device)


def _initialize_history():
    """Initialize training history dictionary."""
    return {
        'train_loss': [], 'train_accuracy': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
        'test_accuracy': [], 'test_precision': [], 'test_recall': [], 'test_f1': [], 'lr': []
    }


def run_letagnn_training(homo_graphs, args):
    """Run LETAGNN training with 5-fold cross-validation."""
    set_seed(args.seed)
    device = args.device
    
    # Normalize then window
    homo_graphs = normalize_features(homo_graphs)

    logging.info("Preprocessing all graphs into windows...")
    preprocessed_graphs = []
    for graph in tqdm(homo_graphs, desc="Windowing graphs"):
        windows = split_graph_by_time_windows(
            graph, max_windows=10, min_window_size=50
        )
        preprocessed_graphs.append((windows, graph.y.item()))

    all_labels = [label for _, label in preprocessed_graphs]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
        logging.info(f"Starting Fold {fold+1}/5")
        
        train_data = [preprocessed_graphs[i] for i in train_idx]
        test_data = [preprocessed_graphs[i] for i in test_idx]
        
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_preprocessed_graphs)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_preprocessed_graphs)
        
        first_window = preprocessed_graphs[0][0][0]
        in_channels = first_window.x.size(1)
        edge_dim = first_window.edge_attr.size(1) if first_window.edge_attr is not None else 0
        
        model = LETAGNN(
            in_channels=in_channels,
            hidden_channels=128,
            out_channels=2,
            num_layers=2,
            heads=4,
            dropout=0.3,
            window_batch_size=128,
            edge_dim=edge_dim
        )
        
        train_config = {
            'device': device,
            'num_epochs': args.epochs,
            'lr': 0.001,
            'weight_decay': 5e-4,
            'patience': 15
        }
        model, history, inference_time = train_model(train_loader, test_loader, model, train_config)
        
        logging.info("Evaluating final model to get attention weights...")
        test_metrics, attention_weights = evaluate_model(model, test_loader, device)
        
        for aw in attention_weights:
            original_graph_idx = test_idx[aw['graph_index']]
            try:
                aw['identifier'] = homo_graphs[original_graph_idx].address
            except AttributeError:
                aw['identifier'] = f"graph_index_{original_graph_idx}"
        
        weights_save_path = f"logs/attention_weights_{args.dataset}_fold{fold+1}.pkl"
        with open(weights_save_path, 'wb') as f:
            pickle.dump(attention_weights, f)
        logging.info(f"Saved attention weights for fold {fold+1} to {weights_save_path}")
        
        best_f1_epoch_idx = np.argmax(history['test_f1']) if history['test_f1'] else 0
        fold_result = {
            'fold': fold + 1,
            'best_accuracy': history['test_accuracy'][best_f1_epoch_idx],
            'best_precision': history['test_precision'][best_f1_epoch_idx],
            'best_recall': history['test_recall'][best_f1_epoch_idx],
            'best_f1': history['test_f1'][best_f1_epoch_idx],
            'best_epoch': best_f1_epoch_idx + 1,
            'final_accuracy': history['test_accuracy'][-1],
            'final_precision': history['test_precision'][-1],
            'final_recall': history['test_recall'][-1],
            'final_f1': history['test_f1'][-1],
            'inference_time_s': inference_time,
        }
        fold_results.append(fold_result)
        logging.info(f"Fold {fold+1} Results:")
        logging.info(f"  Best Test F1 (Epoch {fold_result['best_epoch']}): {fold_result['best_f1']:.4f} (P: {fold_result['best_precision']:.4f}, R: {fold_result['best_recall']:.4f}, Acc: {fold_result['best_accuracy']:.4f})")
        logging.info(f"  Final Epoch Test F1: {fold_result['final_f1']:.4f} (P: {fold_result['final_precision']:.4f}, R: {fold_result['final_recall']:.4f}, Acc: {fold_result['final_accuracy']:.4f})")
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    avg_results = {
        'num_graphs': len(homo_graphs),
        'node_feat_dim': homo_graphs[0].x.size(1) if len(homo_graphs) > 0 else 0,
        'edge_feat_dim': homo_graphs[0].edge_attr.size(1) if len(homo_graphs) > 0 and homo_graphs[0].edge_attr is not None else 0,
        'avg_best_test_f1': np.mean([r['best_f1'] for r in fold_results]),
        'std_best_test_f1': np.std([r['best_f1'] for r in fold_results]),
        'avg_best_test_precision': np.mean([r['best_precision'] for r in fold_results]),
        'std_best_test_precision': np.std([r['best_precision'] for r in fold_results]),
        'avg_best_test_recall': np.mean([r['best_recall'] for r in fold_results]),
        'std_best_test_recall': np.std([r['best_recall'] for r in fold_results]),
        'avg_best_test_accuracy': np.mean([r['best_accuracy'] for r in fold_results]),
        'std_best_test_accuracy': np.std([r['best_accuracy'] for r in fold_results]),
        'avg_best_epoch': np.mean([r['best_epoch'] for r in fold_results]),
        'std_best_epoch': np.std([r['best_epoch'] for r in fold_results]),
        'avg_final_test_f1': np.mean([r['final_f1'] for r in fold_results]),
        'std_final_test_f1': np.std([r['final_f1'] for r in fold_results]),
        'avg_final_test_precision': np.mean([r['final_precision'] for r in fold_results]),
        'std_final_test_precision': np.std([r['final_precision'] for r in fold_results]),
        'avg_final_test_recall': np.mean([r['final_recall'] for r in fold_results]),
        'std_final_test_recall': np.std([r['final_recall'] for r in fold_results]),
        'avg_final_test_accuracy': np.mean([r['final_accuracy'] for r in fold_results]),
        'std_final_test_accuracy': np.std([r['final_accuracy'] for r in fold_results]),
        'avg_inference_time_s': np.mean([r['inference_time_s'] for r in fold_results]),
        'std_inference_time_s': np.std([r['inference_time_s'] for r in fold_results]),
    }
    logging.info("LETAGNN Training Complete - Final Results:")
    logging.info(f"  Best Test F1:  {avg_results['avg_best_test_f1']:.4f} ± {avg_results['std_best_test_f1']:.4f} (P: {avg_results['avg_best_test_precision']:.4f}±{avg_results['std_best_test_precision']:.4f}, R: {avg_results['avg_best_test_recall']:.4f}±{avg_results['std_best_test_recall']:.4f}, Acc: {avg_results['avg_best_test_accuracy']:.4f}±{avg_results['std_best_test_accuracy']:.4f}) @ Epoch {avg_results['avg_best_epoch']:.1f}±{avg_results['std_best_epoch']:.1f}")
    logging.info(f"  Final Epoch Test F1: {avg_results['avg_final_test_f1']:.4f} ± {avg_results['std_final_test_f1']:.4f} (P: {avg_results['avg_final_test_precision']:.4f}±{avg_results['std_final_test_precision']:.4f}, R: {avg_results['avg_final_test_recall']:.4f}±{avg_results['std_final_test_recall']:.4f}, Acc: {avg_results['avg_final_test_accuracy']:.4f}±{avg_results['std_final_test_accuracy']:.4f})")
    
    return avg_results


def main():
    """Run LETAGNN training across all specified datasets."""
    parser = argparse.ArgumentParser(description='LETAGNN Training with Temporal Window Attention')
    parser.add_argument('--epochs', type=int, default=51, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=24, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, choices=[0, 1, 2, 3], help='GPU device to use')
    args = parser.parse_args()

    setup_logging('TrainLETA')

    datasets = ['2DynEth', 'Meta', 'ZipZap']
    all_results = []
    
    homo_graphs_paths = {
        '2DynEth': "Dataset/graphs_home/homo_2DynEth_graphs.pkl",
        'Meta': "Dataset/graphs_home/homo_Meta_graphs.pkl", 
        'ZipZap': "Dataset/graphs_home/homo_ZipZap_graphs.pkl"
    }
    
    for dataset_name in datasets:
        logging.info(f"========== Processing Dataset: {dataset_name} ==========")
        current_args = argparse.Namespace(**vars(args))
        current_args.dataset = dataset_name
        device = torch.device(f'cuda:{current_args.gpu}')

        logging.info(f"Loading data from: {homo_graphs_paths[dataset_name]}")
        with open(homo_graphs_paths[dataset_name], 'rb') as f:
            homo_graphs = pickle.load(f)

        # Keep only essential attributes
        for graph in homo_graphs:
            standard_attrs = {'x', 'edge_index', 'edge_attr', 'y', 'edge_times_normalized', 'address'}
            current_attrs = set(graph.keys())
            for attr in current_attrs - standard_attrs:
                if hasattr(graph, attr):
                    delattr(graph, attr)
            
            if not hasattr(graph, 'edge_times_normalized') or graph.edge_times_normalized is None:
                graph.edge_times_normalized = torch.zeros(graph.edge_index.size(1))

        logging.info(f"Loaded {len(homo_graphs)} graphs for {dataset_name}")
        
        current_args.device = device
        result = run_letagnn_training(homo_graphs, current_args)
        
        if result is not None:
            all_results.append({'dataset': dataset_name, **result})
        else:
            logging.error(f"Training failed for dataset {dataset_name}.")

    if all_results:
        results_file = "logs/letagnn_results.csv"
        df_final = pd.DataFrame(all_results)
        
        df_final['F1-Score'] = df_final.apply(lambda r: f"{r['avg_best_test_f1']*100:.2f}±{r['std_best_test_f1']*100:.2f}", axis=1)
        df_final['Precision'] = df_final.apply(lambda r: f"{r['avg_best_test_precision']*100:.2f}±{r['std_best_test_precision']*100:.2f}", axis=1)
        df_final['Recall'] = df_final.apply(lambda r: f"{r['avg_best_test_recall']*100:.2f}±{r['std_best_test_recall']*100:.2f}", axis=1)
        df_final['Accuracy'] = df_final.apply(lambda r: f"{r['avg_best_test_accuracy']*100:.2f}±{r['std_best_test_accuracy']*100:.2f}", axis=1)
        df_final['Inference Time (ms)'] = df_final.apply(lambda r: f"{r['avg_inference_time_s']*1000:.4f}±{r['std_inference_time_s']*1000:.4f}", axis=1)

        final_columns = ['dataset', 'F1-Score', 'Precision', 'Recall', 'Accuracy', 'avg_best_epoch', 'Inference Time (ms)']
        df_to_save = df_final[final_columns]
        
        df_to_save.sort_values(by='dataset', inplace=True)
        df_to_save.to_csv(results_file, index=False)
        logging.info(f"--- All LETAGNN training results saved to {results_file} ---")
        print(df_to_save)


if __name__ == "__main__":
    main() 