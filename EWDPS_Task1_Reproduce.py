"""EWDPS Task 1: phishing address identification with Random Forest."""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import networkx as nx

from ConvertGraphs import load_graph_files, convert_hetero_to_homo, NODE_DIM, EDGE_DIM

# EWDPS constants from paper (Table II features now from homo_graph.table2_features)
DEFAULT_T_SLICES = 5
TFE_LSTM_HIDDEN = 64
TFE_OUT_DIM = 64
ACCOUNT_FUNCS = ["sum", "max", "min", "mean", "std"]
ACCOUNT_DIM = 10  # 5 (value) + 5 (timestamp)
NETWORK_DIM = 7  # out_deg, in_deg, num_tx_per_unit, clustering, EC, BC, CC
RF_N_ESTIMATORS = 100
RF_DUAL_SAMPLING_B = 50
SUBSAMPLE_NEG_RATIO = 0.5


def time_slice_graph(edge_index, edge_times, edge_attr, T: int):
    """Split edges into T time slices by quantiles."""
    if edge_times is None or edge_times.numel() == 0:
        return []
    et = edge_times.numpy()
    q = np.linspace(0, 1, T + 1)
    quantiles = np.quantile(et, q)
    out = []
    for i in range(T):
        low, high = quantiles[i], quantiles[i + 1]
        mask = (et >= low) & (et < high) if i < T - 1 else (et >= low) & (et <= high)
        mask = torch.tensor(mask, dtype=torch.bool)
        if mask.sum() == 0:
            continue
        out.append(
            (
                edge_index[:, mask],
                edge_times[mask],
                edge_attr[mask] if edge_attr is not None else None,
                mask,
            )
        )
    return out


def get_afin_subgraph(edge_index, num_nodes, center_idx: int, max_hops: int = 2):
    """Extract AFIN subgraph (max_hops from center)."""
    row, col = edge_index[0].numpy(), edge_index[1].numpy()
    neighbors = {center_idx}
    frontier = {center_idx}
    for _ in range(max_hops):
        next_frontier = set()
        for u in frontier:
            for i in range(edge_index.size(1)):
                if row[i] == u:
                    next_frontier.add(col[i])
                if col[i] == u:
                    next_frontier.add(row[i])
        neighbors.update(next_frontier)
        frontier = next_frontier

    node_list = sorted(neighbors)
    old_to_new = {n: i for i, n in enumerate(node_list)}
    edge_mask = []
    for i in range(edge_index.size(1)):
        u, v = row[i], col[i]
        if u in old_to_new and v in old_to_new:
            edge_mask.append(True)
        else:
            edge_mask.append(False)
    edge_mask = torch.tensor(edge_mask, dtype=torch.bool)
    node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for n in node_list:
        node_mask[n] = True
    return node_mask, edge_mask, node_list, old_to_new


def account_features_per_node(edge_index, edge_times, edge_attr, num_nodes, center_idx_in_sub, old_to_new):
    """Compute account features: 5 stats on value + 5 on timestamp."""
    row, col = edge_index[0].numpy(), edge_index[1].numpy()
    n_sub = len(old_to_new)
    value = edge_attr[:, 0].numpy() if edge_attr is not None else np.zeros(edge_index.size(1))
    time_arr = edge_times.numpy()

    in_val, in_ts = [[] for _ in range(n_sub)], [[] for _ in range(n_sub)]
    out_val, out_ts = [[] for _ in range(n_sub)], [[] for _ in range(n_sub)]
    for i in range(edge_index.size(1)):
        u, v = int(row[i]), int(col[i])
        if u not in old_to_new or v not in old_to_new:
            continue
        ui, vi = old_to_new[u], old_to_new[v]
        val, t = float(value[i]), float(time_arr[i])
        out_val[ui].append(val)
        out_ts[ui].append(t)
        in_val[vi].append(val)
        in_ts[vi].append(t)

    def stats(x):
        if not x:
            return [0.0] * 5
        x = np.array(x, dtype=np.float64)
        return [
            float(np.sum(x)),
            float(np.max(x)),
            float(np.min(x)),
            float(np.mean(x)),
            float(np.std(x)) if len(x) > 1 else 0.0,
        ]

    feats = []
    for i in range(n_sub):
        all_val = in_val[i] + out_val[i]
        all_ts = in_ts[i] + out_ts[i]
        feats.append(stats(all_val) + stats(all_ts))
    return np.array(feats, dtype=np.float32)


def network_features_subgraph(edge_index, num_nodes_sub, old_to_new_inv):
    """Compute network features: degree, clustering, centrality (7 dims)."""
    row, col = edge_index[0].numpy(), edge_index[1].numpy()
    G = nx.DiGraph()
    for i in range(edge_index.size(1)):
        u, v = int(row[i]), int(col[i])
        G.add_edge(u, v)
    nodes = sorted(G.nodes())
    n = len(nodes)
    if n == 0:
        return np.zeros((num_nodes_sub, NETWORK_DIM), dtype=np.float32)
    mapping = {nd: i for i, nd in enumerate(nodes)}
    G = nx.relabel_nodes(G, mapping)
    out_deg = np.array([G.out_degree(i) for i in range(n)], dtype=np.float32)
    in_deg = np.array([G.in_degree(i) for i in range(n)], dtype=np.float32)
    num_tx = out_deg + in_deg
    try:
        clust = nx.clustering(nx.Graph(G.to_undirected()))
        clust = np.array([clust.get(i, 0.0) for i in range(n)], dtype=np.float32)
    except Exception:
        clust = np.zeros(n, dtype=np.float32)
    try:
        ec = nx.eigenvector_centrality_numpy(nx.Graph(G.to_undirected()), max_iter=500)
        ec = np.array([ec.get(i, 0.0) for i in range(n)], dtype=np.float32)
    except Exception:
        ec = np.zeros(n, dtype=np.float32)
    try:
        bc = nx.betweenness_centrality(G)
        bc = np.array([bc.get(i, 0.0) for i in range(n)], dtype=np.float32)
    except Exception:
        bc = np.zeros(n, dtype=np.float32)
    try:
        cc = nx.closeness_centrality(G)
        cc = np.array([cc.get(i, 0.0) for i in range(n)], dtype=np.float32)
    except Exception:
        cc = np.zeros(n, dtype=np.float32)
    net_feat = np.stack([out_deg, in_deg, num_tx, clust, ec, bc, cc], axis=1)
    if n != num_nodes_sub:
        pad = np.zeros((num_nodes_sub - n, NETWORK_DIM), dtype=np.float32)
        net_feat = np.vstack([net_feat, pad])
    return net_feat.astype(np.float32)


def temporal_feature_lstm(edge_index, edge_times, edge_attr, table2_feat, num_nodes_sub, old_to_new, lstm_hidden, out_dim):
    """TFE: LSTM on temporal edge sequences (direction, value, time, neighbor_feat).
    
    Args:
        table2_feat: Table II features (num_nodes, 8) from ConvertGraphs
    """
    device = next(
        (t.device for t in [edge_index, edge_times, edge_attr, table2_feat] if t is not None and t.device.type != "cpu"),
        torch.device("cpu"),
    )
    if device.type == "cpu":
        device = torch.device("cpu")
    row, col = edge_index[0], edge_index[1]
    n_sub = num_nodes_sub
    value = edge_attr[:, 0:1] if edge_attr is not None else torch.zeros(edge_index.size(1), 1, device=edge_index.device)
    time_ = edge_times.unsqueeze(1)
    
    # Use Table II features (8 dims) as neighbor features
    x_8 = table2_feat.clone() if table2_feat is not None else torch.zeros(num_nodes_sub, 8)
    if x_8.size(1) < 8:
        x_8 = torch.cat([x_8, torch.zeros(x_8.size(0), 8 - x_8.size(1), device=x_8.device)], dim=1)
    
    seqs = [[] for _ in range(n_sub)]
    for i in range(edge_index.size(1)):
        u, v = int(row[i].item()), int(col[i].item())
        if u not in old_to_new or v not in old_to_new:
            continue
        ui, vi = old_to_new[u], old_to_new[v]
        v_val = value[i].flatten().to(device)
        t_val = time_[i].flatten().to(device)
        vec_out = torch.cat(
            [
                torch.tensor([-1.0], device=device),
                v_val,
                t_val,
                x_8[v].to(device),
            ],
            dim=0,
        )
        vec_in = torch.cat(
            [
                torch.tensor([1.0], device=device),
                v_val,
                t_val,
                x_8[u].to(device),
            ],
            dim=0,
        )
        seqs[ui].append((float(time_[i].item()), vec_out))
        seqs[vi].append((float(time_[i].item()), vec_in))
    feat_dim = 11
    lstm = torch.nn.LSTM(feat_dim, lstm_hidden, batch_first=True).to(device)
    out_proj = torch.nn.Linear(lstm_hidden, out_dim).to(device)
    out_feats = []
    for i in range(n_sub):
        if not seqs[i]:
            out_feats.append(torch.zeros(out_dim, device=device))
            continue
        seqs[i].sort(key=lambda x: x[0])
        seq = torch.stack([s[1] for s in seqs[i]], dim=0).unsqueeze(0).to(device)
        _, (h, _) = lstm(seq)
        out_feats.append(out_proj(h.squeeze(0).squeeze(0)))
    return torch.stack(out_feats, dim=0).detach().cpu().numpy().astype(np.float32)


def build_ewdps_center_features(homo_graph, T: int, use_tfe: bool = True):
    """Extract EWDPS features for center node across T time slices."""
    center_idx = getattr(homo_graph, "center_idx", None)
    if center_idx is None:
        return None
    edge_index = homo_graph.edge_index
    edge_times = getattr(homo_graph, "edge_times", None)
    edge_attr = getattr(homo_graph, "edge_attr", None)
    x = homo_graph.x
    table2_feat = getattr(homo_graph, "table2_features", None)
    num_nodes = x.size(0)
    
    # Fallback if table2_features not available
    if table2_feat is None:
        table2_feat = torch.zeros(num_nodes, 8)
    
    if edge_times is None or edge_times.numel() == 0:
        edge_times = torch.zeros(edge_index.size(1), dtype=torch.float)
    slices = time_slice_graph(edge_index, edge_times, edge_attr, T)
    if not slices:
        return None
    list_center_feat = []
    for edge_index_t, edge_times_t, edge_attr_t, _ in slices:
        node_mask, edge_mask, node_list, old_to_new = get_afin_subgraph(
            edge_index_t, num_nodes, center_idx, max_hops=2
        )
        if center_idx not in old_to_new:
            continue
        center_idx_sub = old_to_new[center_idx]
        n_sub = len(node_list)
        row, col = edge_index_t[0].numpy(), edge_index_t[1].numpy()
        ei_list, et_list, ea_list = [], [], []
        for i in range(edge_index_t.size(1)):
            u, v = int(row[i]), int(col[i])
            if u in old_to_new and v in old_to_new:
                ei_list.append([old_to_new[u], old_to_new[v]])
                et_list.append(edge_times_t[i].item())
                ea_list.append(edge_attr_t[i] if edge_attr_t is not None else torch.zeros(EDGE_DIM))
        if not ei_list:
            continue
        ei_sub = torch.tensor(ei_list, dtype=torch.long).t()
        et_sub = torch.tensor(et_list, dtype=torch.float)
        ea_sub = torch.stack(ea_list) if ea_list else torch.zeros(len(ei_list), EDGE_DIM)
        if ea_sub.dim() == 1:
            ea_sub = ea_sub.unsqueeze(0)
        
        # Extract Table II features for AFIN subgraph nodes
        table2_sub = table2_feat[node_list]
        
        acc = account_features_per_node(ei_sub, et_sub, ea_sub, n_sub, center_idx_sub, {i: i for i in range(n_sub)})
        net = network_features_subgraph(ei_sub, n_sub, None)
        if use_tfe:
            tfe = temporal_feature_lstm(ei_sub, et_sub, ea_sub, table2_sub, n_sub, {i: i for i in range(n_sub)}, TFE_LSTM_HIDDEN, TFE_OUT_DIM)
        else:
            tfe = np.zeros((n_sub, TFE_OUT_DIM), dtype=np.float32)
        c_acc = acc[center_idx_sub]
        c_net = net[center_idx_sub] if center_idx_sub < net.shape[0] else np.zeros(NETWORK_DIM, dtype=np.float32)
        c_tfe = tfe[center_idx_sub]
        list_center_feat.append(np.concatenate([c_acc, c_net, c_tfe]))
    if not list_center_feat:
        return None
    return np.mean(list_center_feat, axis=0).astype(np.float32)


def rf_dual_sampling_train(X_train, y_train, n_estimators=RF_N_ESTIMATORS, b=RF_DUAL_SAMPLING_B, subsample_neg=SUBSAMPLE_NEG_RATIO):
    """Train RF with dual-sampling: subsample negatives b times and ensemble."""
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    if n_neg == 0 or n_pos == 0:
        clf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train)
        return clf
    k_neg = max(1, int(n_neg * subsample_neg))
    preds = []
    for _ in range(b):
        neg_samp = np.random.choice(neg_idx, size=min(k_neg, n_neg), replace=False)
        idx = np.concatenate([pos_idx, neg_samp])
        np.random.shuffle(idx)
        X_b, y_b = X_train[idx], y_train[idx]
        tree = RandomForestClassifier(n_estimators=min(10, n_estimators // 5), class_weight="balanced", random_state=None)
        tree.fit(X_b, y_b)
        preds.append(tree)
    
    class Ensemble:
        def __init__(self, models):
            self.models = models

        def predict(self, X):
            votes = np.stack([m.predict(X) for m in self.models], axis=1)
            return (votes.mean(axis=1) >= 0.5).astype(int)
    return Ensemble(preds)


def run_ewdps_training(args, dataset_name=None, feature_cache=None):
    """Run EWDPS training on a specific dataset with k-fold CV. feature_cache: dict key->(feat,) for cross-dataset reuse."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.isdir(args.graph_dir):
        print(f"Graph dir not found: {args.graph_dir}. Use e.g. Dataset/PhishCombine/graphs_Meta or graphs.")
        return
    print("Loading hetero graphs from", args.graph_dir)
    out = load_graph_files(args.graph_dir, max_file_size_mb=80, return_keys=(feature_cache is not None))
    hetero_graphs, labels = out[0], out[1]
    graph_keys = out[2] if len(out) > 2 else [None] * len(hetero_graphs)
    if args.max_graphs is not None:
        hetero_graphs = hetero_graphs[: args.max_graphs]
        labels = labels[: args.max_graphs]
        graph_keys = graph_keys[: args.max_graphs]
    print(f"Loaded {len(hetero_graphs)} graphs, labels: {np.bincount(labels)}")
    prune = args.prune_keep_ratio if 0 < args.prune_keep_ratio <= 1.0 else None
    print("Converting to homo (prune_keep_ratio=%s) ..." % (prune if prune else "None"))
    homo_list, label_list, key_list = [], [], []
    for i, (h, lab) in enumerate(tqdm(zip(hetero_graphs, labels), total=len(hetero_graphs), desc="Convert")):
        g = convert_hetero_to_homo(h, lab, prune_keep_ratio=prune)
        if g is not None and getattr(g, "center_idx", None) is not None:
            homo_list.append(g)
            label_list.append(lab)
            key_list.append(graph_keys[i])
    print(f"Converted {len(homo_list)} homo graphs with center_idx")
    if len(homo_list) == 0:
        print("No graphs with center_idx. Abort.")
        return
    
    if args.features_cache and os.path.exists(args.features_cache):
        print(f"Loading feature cache: {args.features_cache}")
        with open(args.features_cache, 'rb') as f:
            cache_data = pickle.load(f)
            X = cache_data['X']
            y = cache_data['y']
            cache_config = cache_data.get('config', {})
        print(f"Loaded feature matrix shape: {X.shape}, labels: {np.bincount(y)}")
        print(f"Cache config: T={cache_config.get('T', 'N/A')}, use_tfe={cache_config.get('use_tfe', 'N/A')}")
    else:
        print("Building EWDPS features (account + network + temporal) ...")
        X_list, y_list = [], []
        n_hit = 0
        for g, lab, key in tqdm(zip(homo_list, label_list, key_list), total=len(homo_list), desc="EWDPS features"):
            if feature_cache is not None and key is not None and key in feature_cache:
                X_list.append(feature_cache[key][0])
                y_list.append(lab)
                n_hit += 1
            else:
                feat = build_ewdps_center_features(g, T=args.T, use_tfe=not args.no_tfe)
                if feat is not None:
                    X_list.append(feat)
                    y_list.append(lab)
                    if feature_cache is not None and key is not None:
                        feature_cache[key] = (feat,)
        if n_hit:
            print(f"Feature cache hits: {n_hit}/{len(homo_list)}")
        X = np.vstack(X_list)
        y = np.array(y_list, dtype=np.int64)
        print(f"Feature matrix shape: {X.shape}, labels: {np.bincount(y)}")
        
        if args.features_cache:
            print(f"Saving features to cache: {args.features_cache}")
            os.makedirs(os.path.dirname(args.features_cache) if os.path.dirname(args.features_cache) else '.', exist_ok=True)
            cache_data = {
                'X': X,
                'y': y,
                'config': {
                    'T': args.T,
                    'use_tfe': not args.no_tfe,
                    'graph_dir': args.graph_dir,
                    'max_graphs': args.max_graphs,
                    'prune_keep_ratio': args.prune_keep_ratio
                }
            }
            with open(args.features_cache, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Features saved to {args.features_cache}")
    
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    fold_results = []
    print(f"\nStarting {args.n_folds}-fold cross-validation...")
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold_idx}/{args.n_folds} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"Train set: {len(y_train)} samples, label distribution: {np.bincount(y_train)}")
        print(f"Test set: {len(y_test)} samples, label distribution: {np.bincount(y_test)}")
        
        if args.no_dual_sampling:
            clf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, class_weight="balanced", random_state=args.seed)
            clf.fit(X_train, y_train)
        else:
            clf = rf_dual_sampling_train(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        fold_results.append({
            'fold': fold_idx,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
        
        print(f"Fold {fold_idx} Results:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1:        {f1:.4f}")
    
    print("\n" + "="*50)
    print(f"{args.n_folds}-fold Cross-Validation Overall Results:")
    print("="*50)
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        values = [r[metric] for r in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric.capitalize():10s}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\nDetailed Results per Fold:")
    print("-"*70)
    print(f"{'Fold':<6} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*70)
    for r in fold_results:
        print(f"{r['fold']:<6} {r['accuracy']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f}")
    
    avg_results = {
        'num_graphs': len(homo_list),
        'num_samples': len(X),
        'feature_dim': X.shape[1],
        'avg_accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'std_accuracy': np.std([r['accuracy'] for r in fold_results]),
        'avg_precision': np.mean([r['precision'] for r in fold_results]),
        'std_precision': np.std([r['precision'] for r in fold_results]),
        'avg_recall': np.mean([r['recall'] for r in fold_results]),
        'std_recall': np.std([r['recall'] for r in fold_results]),
        'avg_f1': np.mean([r['f1'] for r in fold_results]),
        'std_f1': np.std([r['f1'] for r in fold_results]),
    }
    return avg_results


def main():
    parser = argparse.ArgumentParser(description="EWDPS Task 1: Phishing Address Identification")
    parser.add_argument("--graph-dir", type=str, default=None, help="Directory of .pt hetero graphs (overrides dataset)")
    parser.add_argument("--T", type=int, default=DEFAULT_T_SLICES, help="Number of time slices")
    parser.add_argument("--max-graphs", type=int, default=None, help="Max graphs to load (default all)")
    parser.add_argument("--no-tfe", action="store_true", help="Disable LSTM temporal features")
    parser.add_argument("--no-dual-sampling", action="store_true", help="Use single RF instead of dual-sampling")
    parser.add_argument("--prune-keep-ratio", type=float, default=0.0, help="Prune ratio (0 = no prune)")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--features-cache-dir", type=str, default="cache", help="Directory to store feature caches")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device number (0-3)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"Set CUDA_VISIBLE_DEVICES={args.gpu}")
    
    # 2DynEth first so its features populate cache; Meta/ZipZap are subsets and reuse cache
    datasets = ['2DynEth', 'Meta', 'ZipZap']
    graph_dirs = {
        'Meta': "Dataset/PhishCombine/graphs_Meta",
        'ZipZap': "Dataset/PhishCombine/graphs_ZipZap",
        '2DynEth': "Dataset/PhishCombine/graphs_2DynEth"
    }
    
    os.makedirs('Result', exist_ok=True)
    os.makedirs(args.features_cache_dir, exist_ok=True)
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"Result/EWDPS_Task1_{ts}.csv"
    out_cols = ['dataset', 'T', 'n_folds', 'seed', 'num_graphs', 'num_samples', 'feature_dim',
                'Precision', 'Recall', 'F1-Score', 'Accuracy']
    first_row = True
    
    def fmt_pct(m, s):
        return f"{m * 100:.2f}±{s * 100:.2f}"
    
    feature_cache = {}  # key -> (feat,); reuse across datasets (2DynEth first, then Meta/ZipZap)
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        current_args = argparse.Namespace(**vars(args))
        current_args.graph_dir = args.graph_dir if args.graph_dir else graph_dirs[dataset_name]
        current_args.features_cache = os.path.join(args.features_cache_dir, f"ewdps_features_{dataset_name}_T{args.T}.pkl")
        
        try:
            result = run_ewdps_training(current_args, dataset_name, feature_cache=feature_cache)
            
            if result is None:
                print(f"Training failed for dataset {dataset_name}.")
                continue
            
            row = {
                'dataset': dataset_name,
                'T': args.T,
                'n_folds': args.n_folds,
                'seed': args.seed,
                'num_graphs': result['num_graphs'],
                'num_samples': result['num_samples'],
                'feature_dim': result['feature_dim'],
                'Precision': fmt_pct(result['avg_precision'], result['std_precision']),
                'Recall': fmt_pct(result['avg_recall'], result['std_recall']),
                'F1-Score': fmt_pct(result['avg_f1'], result['std_f1']),
                'Accuracy': fmt_pct(result['avg_accuracy'], result['std_accuracy']),
            }
            
            df_row = pd.DataFrame([row])[out_cols]
            df_row.to_csv(results_file, mode='a' if not first_row else 'w', header=first_row, index=False)
            first_row = False
            print(f"\nResults appended to {results_file}")
            print(df_row.to_string(index=False))
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if first_row:
        print("\nNo results to save.")
    else:
        print(f"\nAll results saved to {results_file}")


if __name__ == "__main__":
    main()
