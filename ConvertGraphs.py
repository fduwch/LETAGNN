"""
简单的异构图到同构图转换工具
从graphs_Meta文件夹读取异构图，转换后保存到graphs_Meta_home文件夹
"""

import torch
import os
import pickle
from tqdm import tqdm
from torch_geometric.data import Data

# Local implementation to load all graph files from a directory
# Returns (graphs, labels)
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
    """将异构图转换为同构图"""
    node_features, edge_indices, edge_attr = [], [], []
    node_count = 0
    node_map = {}
    
    # 处理所有节点
    for node_type in hetero_graph.node_types:
        if hasattr(hetero_graph[node_type], 'x') and hetero_graph[node_type].x is not None:
            num_nodes = hetero_graph[node_type].x.size(0)
            
            # 包含节点类型指示器
            type_indicator = torch.zeros(num_nodes, len(hetero_graph.node_types))
            type_idx = list(hetero_graph.node_types).index(node_type)
            type_indicator[:, type_idx] = 1
            
            if hetero_graph[node_type].x.size(1) > 0:
                node_feat = torch.cat([hetero_graph[node_type].x, type_indicator], dim=1)
            else:
                node_feat = type_indicator
            
            # 保存节点映射
            for i in range(num_nodes):
                node_map[(node_type, i)] = node_count + i
            
            node_count += num_nodes
            node_features.append(node_feat)
    
    # 合并所有节点特征
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
    
    # 处理所有边 - 只看关系类型，不区分源目标节点类型
    edge_times = []
    relation_types = set()
    
    # 首先收集所有唯一的关系类型
    for edge_type in hetero_graph.edge_types:
        src_type, rel_type, dst_type = edge_type
        relation_types.add(rel_type)
    
    relation_types = sorted(list(relation_types))  # 保证顺序一致
    
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
                    
                    # 提取边特征
                    if hasattr(hetero_graph[edge_type], 'edge_attr') and hetero_graph[edge_type].edge_attr is not None:
                        edge_feature = hetero_graph[edge_type].edge_attr[i]
                        
                        # 只基于关系类型编码，不考虑源目标节点类型
                        relation_onehot = torch.zeros(len(relation_types))
                        relation_idx = relation_types.index(rel_type)
                        relation_onehot[relation_idx] = 1
                        
                        if edge_feature.dim() > 0:
                            edge_feat = torch.cat([edge_feature, relation_onehot])
                        else:
                            edge_feat = relation_onehot
                        
                        # 提取时间戳
                        edge_times.append(edge_feature[4].item() if edge_feature.size(0) > 4 else 0.0)
                        edge_attr.append(edge_feat.unsqueeze(0))
                    else:
                        edge_times.append(0.0)
    
    # 创建同构图
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
        
        # Save feature dimension info for ablation
        homo_graph.original_node_feat_dim = hetero_graph[list(hetero_graph.node_types)[0]].x.size(1) if len(hetero_graph.node_types) > 0 and hasattr(hetero_graph[list(hetero_graph.node_types)[0]], 'x') else 0
        homo_graph.node_type_feat_dim = len(hetero_graph.node_types)
        homo_graph.original_edge_feat_dim = 5  # Assuming 5 original edge features
        homo_graph.edge_type_feat_dim = len(relation_types)
        
        # 添加时间信息
        if edge_times:
            homo_graph.edge_times = torch.tensor(edge_times, dtype=torch.float)
            # 不再归一化时间，edge_times_normalized仅为占位
            homo_graph.edge_times_normalized = torch.zeros(len(edge_times), dtype=torch.float)
        else:
            # Ensure all graphs have edge_times attributes
            num_edges = homo_graph.edge_index.size(1)
            homo_graph.edge_times = torch.zeros(num_edges, dtype=torch.float)
            homo_graph.edge_times_normalized = torch.zeros(num_edges, dtype=torch.float)
        
        # Add basic temporal attributes for consistent batching
        homo_graph.is_multi_temporal = False
        homo_graph.num_temporal_windows = 1
        
        return homo_graph
    
    return None

def main():
    input_dir = "Dataset/PhishCombine/graphs_2DynEth"
    output_dir = "Dataset/PhishCombine/graphs_home"
    output_file = os.path.join(output_dir, "homo_2DynEth_graphs.pkl")
    
    print("开始转换异构图到同构图...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载异构图
    print("加载异构图数据...")
    hetero_graphs, labels = load_graph_files(input_dir, max_file_size_mb=80)
    print(f"加载了 {len(hetero_graphs)} 个异构图")
    
    # 转换为同构图
    homo_graphs = []
    print("转换中...")
    for idx, hetero_graph in enumerate(tqdm(hetero_graphs, desc="转换进度")):
        homo_graph = convert_hetero_to_homo(hetero_graph, labels[idx])
        if homo_graph is not None:
            homo_graphs.append(homo_graph)
    
    print(f"成功转换 {len(homo_graphs)} 个同构图")
    
    # 保存到单个文件
    print(f"保存到文件: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(homo_graphs, f)
    
    print("转换完成!")
    print(f"结果保存在: {output_file}")

if __name__ == "__main__":
    main() 