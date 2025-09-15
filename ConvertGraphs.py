"""
Simple tool to convert heterogeneous graphs to homogeneous graphs.
Reads from `graphs_Meta` and writes to `graphs_Meta_home`.
"""

import torch
import os
import pickle
from tqdm import tqdm
from torch_geometric.data import Data

# Load all .pt graph files from a directory and return (graphs, labels)
def load_graph_files(input_dir, max_file_size_mb=80):
    graphs = []
    labels = []
    pt_files = [fname for fname in os.listdir(input_dir) if fname.endswith('.pt')]
    for fname in tqdm(pt_files, desc='Loading .pt graph files'):
        fpath = os.path.join(input_dir, fname)
        if os.path.getsize(fpath) < max_file_size_mb * 1024 * 1024:
            g = torch.load(fpath, map_location='cpu')
            if isinstance(g, list):
                for gg in g:
                    graphs.append(gg)
                    if hasattr(gg, 'y'):
                        labels.append(int(gg.y.item()) if hasattr(gg.y, 'item') else int(gg.y))
                    else:
                        labels.append(0)
            else:
                graphs.append(g)
                if hasattr(g, 'y'):
                    labels.append(int(g.y.item()) if hasattr(g.y, 'item') else int(g.y))
                else:
                    labels.append(0)
    return graphs, labels


def convert_hetero_to_homo(hetero_graph, label):
    """Convert a heterogeneous graph to a homogeneous graph."""
    node_features, edge_indices, edge_attr = [], [], []
    node_count = 0
    node_map = {}
    
    # Nodes
    for node_type in hetero_graph.node_types:
        if hasattr(hetero_graph[node_type], 'x') and hetero_graph[node_type].x is not None:
            num_nodes = hetero_graph[node_type].x.size(0)
            
            # Node type one-hot indicator
            type_indicator = torch.zeros(num_nodes, len(hetero_graph.node_types))
            type_idx = list(hetero_graph.node_types).index(node_type)
            type_indicator[:, type_idx] = 1
            
            if hetero_graph[node_type].x.size(1) > 0:
                node_feat = torch.cat([hetero_graph[node_type].x, type_indicator], dim=1)
            else:
                node_feat = type_indicator
            
            # Map (type, local_idx) -> global_idx
            for i in range(num_nodes):
                node_map[(node_type, i)] = node_count + i
            
            node_count += num_nodes
            node_features.append(node_feat)
    
    # Concatenate and pad node features to the same dim
    if node_features:
        max_dim = max(feat.size(1) for feat in node_features)
        padded_features = []
        for feat in node_features:
            if feat.size(1) < max_dim:
                padding = torch.zeros(feat.size(0), max_dim - feat.size(1))
                padded_features.append(torch.cat([feat, padding], dim=1))
            else:
                padded_features.append(feat)
        all_features = torch.cat(padded_features, dim=0)
    else:
        all_features = torch.zeros(node_count, 1)
    
    # Edges (aggregate by relation type only)
    edge_times = []
    relation_types = set()
    
    # Collect unique relation types
    for edge_type in hetero_graph.edge_types:
        src_type, rel_type, dst_type = edge_type
        relation_types.add(rel_type)
    
    relation_types = sorted(list(relation_types))
    
    for edge_type in hetero_graph.edge_types:
        if hasattr(hetero_graph[edge_type], 'edge_index'):
            src_type, rel_type, dst_type = edge_type
            edge_index = hetero_graph[edge_type].edge_index
            
            for i in range(edge_index.size(1)):
                src_idx = edge_index[0, i].item()
                dst_idx = edge_index[1, i].item()
                
                if (src_type, src_idx) in node_map and (dst_type, dst_idx) in node_map:
                    new_src_idx = node_map[(src_type, src_idx)]
                    new_dst_idx = node_map[(dst_type, dst_idx)]
                    edge_indices.append(torch.tensor([[new_src_idx], [new_dst_idx]]))
                    
                    # Edge features
                    if hasattr(hetero_graph[edge_type], 'edge_attr') and hetero_graph[edge_type].edge_attr is not None:
                        edge_feature = hetero_graph[edge_type].edge_attr[i]
                        
                        # Relation one-hot
                        relation_onehot = torch.zeros(len(relation_types))
                        relation_idx = relation_types.index(rel_type)
                        relation_onehot[relation_idx] = 1
                        
                        if edge_feature.dim() > 0:
                            edge_feat = torch.cat([edge_feature, relation_onehot])
                        else:
                            edge_feat = relation_onehot
                        
                        # Timestamp (5th feature if available)
                        edge_times.append(edge_feature[4].item() if edge_feature.size(0) > 4 else 0.0)
                        edge_attr.append(edge_feat.unsqueeze(0))
                    else:
                        edge_times.append(0.0)
    
    # Build homogeneous graph
    if edge_indices:
        all_edge_indices = torch.cat(edge_indices, dim=1)
        
        if edge_attr:
            max_dim = max(attr.size(1) for attr in edge_attr)
            padded_attrs = []
            for attr in edge_attr:
                if attr.size(1) < max_dim:
                    padding = torch.zeros(attr.size(0), max_dim - attr.size(1))
                    padded_attrs.append(torch.cat([attr, padding], dim=1))
                else:
                    padded_attrs.append(attr)
            all_edge_attr = torch.cat(padded_attrs, dim=0)
        else:
            all_edge_attr = None
        
        homo_graph = Data(
            x=all_features,
            edge_index=all_edge_indices,
            edge_attr=all_edge_attr,
            y=torch.tensor([label], dtype=torch.long)
        )
        
        # Keep minimal but useful metadata
        homo_graph.original_node_feat_dim = hetero_graph[list(hetero_graph.node_types)[0]].x.size(1) if len(hetero_graph.node_types) > 0 and hasattr(hetero_graph[list(hetero_graph.node_types)[0]], 'x') else 0
        homo_graph.node_type_feat_dim = len(hetero_graph.node_types)
        homo_graph.original_edge_feat_dim = 5
        homo_graph.edge_type_feat_dim = len(relation_types)
        
        if edge_times:
            homo_graph.edge_times = torch.tensor(edge_times, dtype=torch.float)
            homo_graph.edge_times_normalized = torch.zeros(len(edge_times), dtype=torch.float)
        else:
            num_edges = homo_graph.edge_index.size(1)
            homo_graph.edge_times = torch.zeros(num_edges, dtype=torch.float)
            homo_graph.edge_times_normalized = torch.zeros(num_edges, dtype=torch.float)
        
        homo_graph.is_multi_temporal = False
        homo_graph.num_temporal_windows = 1
        
        return homo_graph
    
    return None


def main():
    input_dir = "Dataset/PhishCombine/graphs_2DynEth"
    output_dir = "Dataset/PhishCombine/graphs_home"
    output_file = os.path.join(output_dir, "homo_2DynEth_graphs.pkl")
    
    print("Converting heterogeneous graphs to homogeneous graphs...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading heterogeneous graphs...")
    hetero_graphs, labels = load_graph_files(input_dir, max_file_size_mb=80)
    print(f"Loaded {len(hetero_graphs)} heterogeneous graphs")
    
    homo_graphs = []
    print("Converting...")
    for idx, hetero_graph in enumerate(tqdm(hetero_graphs, desc="Converting")):
        homo_graph = convert_hetero_to_homo(hetero_graph, labels[idx])
        if homo_graph is not None:
            homo_graphs.append(homo_graph)
    
    print(f"Successfully converted {len(homo_graphs)} homogeneous graphs")
    
    print(f"Saving to: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(homo_graphs, f)
    
    print("Done!")
    print(f"Results saved at: {output_file}")


if __name__ == "__main__":
    main() 