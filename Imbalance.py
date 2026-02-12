"""Class imbalance evaluation for LETAGNN under different positive ratios."""

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

warnings.filterwarnings("ignore")

from LETAGNN import set_seed, normalize_features, LETAGNN_LightWindow, split_graph_by_time_windows
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


def subsample_to_positive_ratio(indices, labels, positive_ratio, rng):
    """
    Subsample positive samples while keeping all negative samples.
    positive_ratio: fraction of positive samples to keep (0-1).
    labels[i] corresponds to indices[i]; labels are 0 (neg) or 1 (pos).
    Returns subset of indices.
    """
    labels = np.asarray(labels)
    pos_mask = labels == 1
    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(~pos_mask)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    # Keep all negative samples
    neg_keep = neg_idx
    
    # Subsample positive samples by the given ratio
    if positive_ratio <= 0:
        pos_keep = np.array([], dtype=int)
    elif positive_ratio >= 1:
        pos_keep = pos_idx
    else:
        n_pos_keep = max(1, int(n_pos * positive_ratio))
        pos_keep = rng.choice(pos_idx, size=min(n_pos_keep, n_pos), replace=False)

    kept = np.concatenate([pos_keep, neg_keep])
    rng.shuffle(kept)
    return indices[kept]


def run_imbalance_analysis(homo_graphs, args):
    """Run 5-fold CV with training set subsampled to target positive ratio."""
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}')
    positive_ratio = args.config['positive_ratio']
    epochs = args.config.get('epochs', 30)
    rng = np.random.default_rng(args.seed)

    homo_graphs = normalize_features(homo_graphs)
    max_windows = 10
    min_window_size = 50
    min_edges_to_split = 300

    logging.info(f"Preprocessing graphs (positive_ratio={positive_ratio:.0%}, epochs={epochs})")
    preprocessed_graphs = []
    for graph in tqdm(homo_graphs, desc="Windowing graphs"):
        windows = split_graph_by_time_windows(
            graph, max_windows=max_windows, min_window_size=min_window_size, min_edges_to_split=min_edges_to_split
        )
        preprocessed_graphs.append((windows, graph, graph.y.item()))

    all_labels = np.array([label for _, _, label in preprocessed_graphs])
    n_pos_total = (all_labels == 1).sum()
    n_neg_total = (all_labels == 0).sum()
    logging.info(f"Dataset: {n_pos_total} positive, {n_neg_total} negative")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (train_idx_full, test_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
        train_idx_full = np.array(train_idx_full)
        train_labels = all_labels[train_idx_full]
        train_idx = subsample_to_positive_ratio(train_idx_full, train_labels, positive_ratio, rng)

        train_data = [preprocessed_graphs[i] for i in train_idx]
        test_data = [preprocessed_graphs[i] for i in test_idx]
        n_pos_train = sum(1 for _, _, l in train_data if l == 1)
        n_neg_train = sum(1 for _, _, l in train_data if l == 0)
        actual_ratio = n_pos_train / (n_pos_train + n_neg_train) if (n_pos_train + n_neg_train) > 0 else 0
        logging.info(f"Fold {fold+1}/5: train size={len(train_data)} (pos={n_pos_train}, neg={n_neg_train}, ratio={actual_ratio:.2%})")

        train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_preprocessed_graphs)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_preprocessed_graphs)

        first_window = preprocessed_graphs[0][0][0]
        in_channels = first_window.x.size(1)
        edge_dim = first_window.edge_attr.size(1) if first_window.edge_attr is not None else 0

        model = LETAGNN_LightWindow(
            in_channels=in_channels, hidden_channels=128, out_channels=2,
            num_layers=2, heads=4, dropout=0.3, edge_dim=edge_dim
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
    parser = argparse.ArgumentParser(description='Class imbalance evaluation for LETAGNN')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    setup_logging('imbalance')

    # Positive ratio = fraction of positive samples to keep (negative samples are all kept)
    experiment_configs = [
        # {'positive_ratio': 0.01, 'epochs': 30},
        # {'positive_ratio': 0.05, 'epochs': 30},
        # {'positive_ratio': 0.10, 'epochs': 30},
        # {'positive_ratio': 0.30, 'epochs': 30},
        # {'positive_ratio': 0.50, 'epochs': 30},
        {'positive_ratio': 0.80, 'epochs': 30},
    ]

    datasets = ['2DynEth'] # Meta, ZipZap
    homo_graphs_paths = {
        '2DynEth': "Dataset/PhishCombine/graphs_home_bak0201/homo_2DynEth_graphs.pkl",
        'Meta': "Dataset/PhishCombine/graphs_home_bak0201/homo_Meta_graphs.pkl",
        'ZipZap': "Dataset/PhishCombine/graphs_home_bak0201/homo_ZipZap_graphs.pkl"
    }

    out_cols = ['dataset', 'positive_ratio', 'epochs', 'Precision', 'Recall', 'F1-Score',
                'P@min_train_loss', 'R@min_train_loss', 'F1@min_train_loss']
    os.makedirs('Result', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"Result/imbalance_{ts}.csv"
    first_row = True

    for dataset_name in datasets:
        logging.info(f"========== Dataset: {dataset_name} ==========")
        with open(homo_graphs_paths[dataset_name], 'rb') as f:
            graphs = pickle.load(f)
        for graph in graphs:
            standard_attrs = {'x', 'edge_index', 'edge_attr', 'y', 'edge_times_normalized', 'address'}
            current_attrs = set(graph.keys())
            for attr in current_attrs - standard_attrs:
                if hasattr(graph, attr):
                    delattr(graph, attr)
            if not hasattr(graph, 'edge_times_normalized') or graph.edge_times_normalized is None:
                graph.edge_times_normalized = torch.zeros(graph.edge_index.size(1))

        for config in experiment_configs:
            current_args = argparse.Namespace(**vars(args))
            current_args.dataset = dataset_name
            current_args.config = config
            logging.info(f"--- positive_ratio={config['positive_ratio']:.0%} ---")
            result = run_imbalance_analysis(graphs, current_args)

            row = {
                'dataset': dataset_name,
                'positive_ratio': f"{config['positive_ratio']:.0%}",
                'epochs': config['epochs'],
                'Precision': f"{result['avg_precision']*100:.2f}±{result['std_precision']*100:.2f}",
                'Recall': f"{result['avg_recall']*100:.2f}±{result['std_recall']*100:.2f}",
                'F1-Score': f"{result['avg_f1']*100:.2f}±{result['std_f1']*100:.2f}",
                'P@min_train_loss': f"{result['avg_p_at_min_loss']*100:.2f}±{result['std_p_at_min_loss']*100:.2f}",
                'R@min_train_loss': f"{result['avg_r_at_min_loss']*100:.2f}±{result['std_r_at_min_loss']*100:.2f}",
                'F1@min_train_loss': f"{result['avg_f1_at_min_loss']*100:.2f}±{result['std_f1_at_min_loss']*100:.2f}",
            }
            pd.DataFrame([row])[out_cols].to_csv(results_file, mode='a', header=first_row, index=False)
            first_row = False

    df_final = pd.read_csv(results_file)
    df_final.sort_values(by=['dataset', 'positive_ratio', 'epochs'], inplace=True)
    df_final.to_csv(results_file, index=False)
    logging.info(f"Results saved to {results_file}")
    print(df_final)


if __name__ == "__main__":
    main()
