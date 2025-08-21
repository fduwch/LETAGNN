from torch_geometric.data import Dataset, HeteroData
import torch
import pandas as pd
import os
import json
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Set
import warnings
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import time
import glob
import math
import logging
import datetime
import gc

# Setup logging
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'graph_processing_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# Transaction and node type definitions
TX_TYPES = ['Normal', 'Internal', 'ERC20']
EDGE_TYPES = ['normal', 'internal', 'erc20']
NODE_CATEGORIES = ["Address", "TokenContract", "DEX", "CEX", "DepositAddress"]

# Address category label mapping
LABEL_MAPPING = {
    "Token Contract": "TokenContract",
    "Uniswap": "DEX",
    "SushiSwap": "DEX",
    "Binance": "CEX",
    "Exchange": "CEX",
    "OKX": "CEX",
    "Deposit Address": "DepositAddress"
}

# Default parameters
DEFAULT_TIMEOUT = 3600
DEFAULT_MAX_EDGES = 20
DEFAULT_WORKERS = min(8, multiprocessing.cpu_count())
DEFAULT_BATCH_SIZE = 4000

ADDRESS_DATA_PATH = 'Dataset/PhishCombine/addresses_with_transactions.json'
ADDRESS_GRAPH_DIR = 'Dataset/PhishCombine/graphs'

class PhishingGraphDataset(Dataset):
    """
    Heterogeneous graph dataset for phishing detection on Ethereum blockchain.
    Constructs multiple heterogeneous graphs, one for each address.
    """
    
    def __init__(self, root: str, transform=None, pre_transform=None, n_addresses: Optional[int] = None, 
                 use_cached_features: bool = True, include_neighbor_edges: bool = True, batch_id: int = 0,
                 batch_size: int = DEFAULT_BATCH_SIZE):
        self.n_addresses = n_addresses
        self.use_cached_features = use_cached_features
        self.include_neighbor_edges = include_neighbor_edges
        self.batch_id = batch_id
        self.batch_size = batch_size
        self._address_category_cache = {}
        self._labels_dict = None
        self._node_features_cache = {}
        self._transaction_cache = {}
        
        # Create directories
        self.feature_cache_dir = os.path.join(root, 'feature_cache')
        os.makedirs(self.feature_cache_dir, exist_ok=True)
        self.graphs_dir = ADDRESS_GRAPH_DIR
        os.makedirs(self.graphs_dir, exist_ok=True)
        self.node_features_file = os.path.join(self.feature_cache_dir, 'node_features.csv')
        
        if use_cached_features:
            self._load_feature_caches()
            
        super(PhishingGraphDataset, self).__init__(root, transform, pre_transform)
        
    def _load_feature_caches(self):
        """Load existing feature caches from CSV files."""
        if not os.path.exists(self.node_features_file):
            return
            
        node_df = pd.read_csv(
            self.node_features_file,
            engine='c',
            dtype={'address': str, 'is_center': int, 'features': str},
            low_memory=True,
            usecols=['address', 'is_center', 'features'],
            memory_map=True,
            on_bad_lines='skip'
        )
        logger.info(f"Loaded {len(node_df)} cached node features")
        
        for _, row in node_df.iterrows():
            addr, is_center, features_str = row['address'], bool(row['is_center']), row['features']
            features = np.fromstring(features_str, sep=',', dtype=np.float32)
            features_tensor = torch.tensor(features, dtype=torch.float)
            cache_key = f"{addr}_{1 if is_center else 0}"
            self._node_features_cache[cache_key] = features_tensor
                
    def _save_features_to_csv(self):
        """Save cached node features to a CSV file."""
        node_data = [{
            'address': cache_key.split('_')[0],
            'is_center': int(cache_key.split('_')[1]),
            'features': ','.join(map(str, features.cpu().numpy().tolist()))
        } for cache_key, features in self._node_features_cache.items()]
        
        if node_data:
            node_df = pd.DataFrame(node_data)
            node_df.to_csv(
                self.node_features_file,
                index=False,
                compression='gzip' if len(node_data) > 10000 else None,
                chunksize=10000 if len(node_data) > 20000 else None
            )
            logger.info(f"Saved {len(node_data)} node features to {self.node_features_file}")
        
    @property
    def raw_file_names(self) -> List[str]:
        return [ADDRESS_DATA_PATH]
    
    @property
    def processed_file_names(self) -> List[str]:
        n_suffix = f"_n{self.n_addresses}" if self.n_addresses else ""
        batch_suffix = f"_batch{self.batch_id}"
        return [f'data_list{n_suffix}{batch_suffix}.pt', f'addresses{n_suffix}{batch_suffix}.pt']
    
    def download(self):
        pass
    
    def process(self):
        """Process raw data to generate heterogeneous graphs."""
        processed_path_data, processed_path_addresses = self.processed_paths[0], self.processed_paths[1]
        
        for path in [processed_path_data, processed_path_addresses]:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Removed existing processed file: {path}")
        
        with open(ADDRESS_DATA_PATH, 'r') as f:
            all_addresses_data = json.load(f)
        
        all_addresses = list(all_addresses_data.keys())
        total_addresses = len(all_addresses)
        
        start_idx = self.batch_id * self.batch_size
        end_idx = min(start_idx + self.batch_size, total_addresses)
        
        if start_idx >= total_addresses:
            logger.warning(f"Batch {self.batch_id} start index is out of bounds.")
            addresses_data = {}
        else:
            addresses_keys = all_addresses[start_idx:end_idx]
            addresses_data = {addr: all_addresses_data[addr] for addr in addresses_keys}
            logger.info(f"Processing batch {self.batch_id}: {len(addresses_keys)} addresses from index {start_idx} to {end_idx-1}")
        
        self._load_etherscan_labels()
        
        if self.n_addresses:
            addresses_data = {addr: info for addr, info in list(addresses_data.items())[:self.n_addresses]}
            logger.info(f"Limiting processing to {len(addresses_data)} addresses as specified.")
        
        processed_addresses = self._get_processed_addresses()
        addresses_to_process = {addr: info for addr, info in addresses_data.items() if addr not in processed_addresses}
        
        logger.info(f"Found {len(processed_addresses)} already processed graphs. Processing {len(addresses_to_process)} new addresses.")
        
        data_list, address_list = self._build_graphs(addresses_to_process, TX_TYPES, EDGE_TYPES)
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            
        torch.save(data_list, processed_path_data)
        torch.save(address_list, processed_path_addresses)
        
        if self.use_cached_features:
            self._save_features_to_csv()
        
        self._transaction_cache.clear()
    
    def _get_processed_addresses(self) -> Set[str]:
        """Get set of addresses that already have processed graph files."""
        processed = set()
        graph_files = glob.glob(os.path.join(self.graphs_dir, "*.pt"))
        for file_path in graph_files:
            filename = os.path.basename(file_path)
            if '_' in filename:
                processed.add(filename.split('_')[0])
        return processed
    
    def _load_etherscan_labels(self):
        """Load and process Etherscan labels."""
        etherscan_labels_path = os.path.join('Dataset', 'PhishCombine', 'etherscan_labels.csv')
        if not os.path.exists(etherscan_labels_path):
            logger.warning(f"{etherscan_labels_path} not found. Proceeding without category labels.")
            return
        
        df = pd.read_csv(
            etherscan_labels_path,
            engine='c',
            low_memory=True,
            dtype={'Address': str, 'SpanLabel': str},
            usecols=['Address', 'SpanLabel'],
            memory_map=True,
            on_bad_lines='skip',
            compression='infer'
        )
        df['Address'] = df['Address'].str.lower()
        if 'SpanLabel' in df.columns:
            self._labels_dict = dict(zip(df['Address'], df['SpanLabel']))
    
    def _get_address_category(self, address: str) -> str:
        """Get the category of an address based on Etherscan labels."""
        address_lower = address.lower()
        if address_lower in self._address_category_cache:
            return self._address_category_cache[address_lower]
        
        category = "Address"
        if self._labels_dict:
            span_label = self._labels_dict.get(address_lower)
            if span_label and pd.notna(span_label):
                last_part = span_label.split(';')[-1].strip()
                category = LABEL_MAPPING.get(last_part, "Address")
        
        self._address_category_cache[address_lower] = category
        return category
    
    def _load_transactions(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load transaction data from a CSV file."""
        if file_path in self._transaction_cache:
            return self._transaction_cache[file_path]
            
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            self._transaction_cache[file_path] = None
            return None
        
        try:
            df = pd.read_csv(
                file_path,
                engine='c',
                low_memory=True,
                dtype={'from': 'str', 'to': 'str'},
                na_values=['null', 'nan', ''],
                on_bad_lines='skip'
            )
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            self._transaction_cache[file_path] = None
            return None

        if df.empty or 'from' not in df.columns or 'to' not in df.columns:
            self._transaction_cache[file_path] = None
            return None
            
        df['from'] = df['from'].fillna('').astype(str).str.lower()
        df['to'] = df['to'].fillna('').astype(str).str.lower()
        
        if 'timeStamp' in df.columns:
            df['timeStamp'] = pd.to_numeric(df['timeStamp'], errors='coerce')
            df.sort_values('timeStamp', ascending=False, inplace=True)
            df = df.head(20000)
                
        self._transaction_cache[file_path] = df
        return df
    
    def _save_processing_state(self, success_addresses, timeout_addresses, error_addresses, stats):
        """Save current processing state to JSON files."""
        state_dir = self.feature_cache_dir
        os.makedirs(state_dir, exist_ok=True)
        
        state = {
            'success_addresses': success_addresses,
            'timeout_addresses': timeout_addresses,
            'error_addresses': error_addresses,
            'stats': {k: v for k, v in stats.items() if isinstance(v, (int, float, str, bool))}
        }
        
        for key, data in state.items():
            with open(os.path.join(state_dir, f'{key}.json'), 'w') as f:
                json.dump(data, f, indent=4)
                
        logger.info(f"Saved processing state to {state_dir}")

    def _build_graphs(self, addresses_data: Dict, tx_types: List[str], 
                     edge_types: List[str]) -> Tuple[List[HeteroData], List[str]]:
        """Build heterogeneous graphs for each address in the dataset."""
        logger.info("Building heterogeneous graphs...")
        
        manager = multiprocessing.Manager()
        shared_address_categories = manager.dict(self._address_category_cache)
        
        num_workers = DEFAULT_WORKERS
        logger.info(f"Using {num_workers} workers for parallel processing.")
        
        task_queue = manager.Queue()
        result_queue = manager.Queue()
        
        for i, (address, info) in enumerate(addresses_data.items()):
            task_queue.put((i, address, info))
        for _ in range(num_workers):
            task_queue.put(None)
            
        processes = []
        for worker_id in range(num_workers):
            p = multiprocessing.Process(
                target=self._worker_process,
                args=(worker_id, task_queue, result_queue, tx_types, edge_types, 
                      shared_address_categories, self.include_neighbor_edges, 
                      DEFAULT_MAX_EDGES, DEFAULT_TIMEOUT)
            )
            processes.append(p)
            p.start()
            
        data_list, address_list = [], []
        success_addresses, timeout_addresses, error_addresses = [], [], []
        stats = defaultdict(int)
        stats['start_time'] = time.time()
        
        total_tasks = len(addresses_data)
        with tqdm(total=total_tasks, desc="Processing addresses") as pbar:
            for i in range(total_tasks):
                try:
                    result = result_queue.get(timeout=DEFAULT_TIMEOUT + 60)
                except Exception:
                    logger.error("Result queue timeout. Some workers might be stuck.")
                    break
                
                pbar.update(1)
                status = result['status']
                stats[status] += 1
                address = result['address']

                if status == 'completed' and result['success']:
                    data_list.append(result['graph'])
                    address_list.append(address)
                    success_addresses.append(address)
                    is_phish = addresses_data[address].get('VerifiedLabel', 0)
                    graph_filename = f"{address}_{int(is_phish)}.pt"
                    graph_path = os.path.join(self.graphs_dir, graph_filename)
                    if not os.path.exists(graph_path):
                        torch.save(result['graph'], graph_path)
                elif status == 'timeout':
                    timeout_addresses.append(address)
                    logger.warning(f"Address {address} timed out after {DEFAULT_TIMEOUT}s")
                elif status == 'error':
                    error_addresses.append(address)
                    logger.error(f"Error processing address {address}: {result.get('error', 'Unknown error')}")
                
                pbar.set_postfix({'success': stats['completed'], 'timeout': stats['timeout'], 'error': stats['error']})
                
                if (i + 1) % 50 == 0:
                    self._save_processing_state(success_addresses, timeout_addresses, error_addresses, stats)

        stats['end_time'] = time.time()
        stats['elapsed_time'] = stats['end_time'] - stats['start_time']
        
        self._cleanup_processes(processes)
        self._save_processing_state(success_addresses, timeout_addresses, error_addresses, stats)
        
        logger.info(f"Processing completed. Success: {stats['completed']}, Timeout: {stats['timeout']}, Error: {stats['error']}")
        logger.info(f"Total time elapsed: {stats['elapsed_time']:.2f} seconds")
        
        return data_list, address_list

    def _cleanup_processes(self, processes: List[multiprocessing.Process]):
        """Clean up worker processes."""
        for p in processes:
            try:
                if p.is_alive():
                    p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
            except Exception as e:
                logger.error(f"Error cleaning up process {p.pid}: {e}")

    def _worker_process(self, worker_id: int, task_queue: multiprocessing.Queue, 
                        result_queue: multiprocessing.Queue, tx_types: List[str], 
                        edge_types: List[str], shared_address_categories: Dict, 
                        include_neighbor_edges: bool, max_edges_per_pair: int, 
                        address_timeout: int):
        """Worker process function that processes addresses from the queue."""
        logger.debug(f"Worker {worker_id} started.")
        preloaded_files = {}

        while True:
            task = task_queue.get()
            if task is None:
                break
                
            task_id, address, info = task
            logger.debug(f"Worker {worker_id} processing address {address}")
            
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._process_single_address, address, info, tx_types, 
                                         edge_types, shared_address_categories, preloaded_files, 
                                         include_neighbor_edges, max_edges_per_pair)
                
                try:
                    graph, success = future.result(timeout=address_timeout)
                    result_queue.put({'status': 'completed', 'task_id': task_id, 'address': address, 'graph': graph, 'success': success})
                except TimeoutError:
                    result_queue.put({'status': 'timeout', 'task_id': task_id, 'address': address})
                except Exception as e:
                    logger.error(f"Worker {worker_id} error on address {address}: {e}")
                    result_queue.put({'status': 'error', 'task_id': task_id, 'address': address, 'error': str(e)})
        
        logger.debug(f"Worker {worker_id} shutting down.")

    def _process_single_address(self, center_address: str, info: Dict, 
                               tx_types: List[str], edge_types: List[str],
                               shared_address_categories: Dict,
                               preloaded_files: Dict,
                               include_neighbor_edges: bool = True,
                               max_edges_per_pair: int = 20) -> Tuple[Optional[HeteroData], bool]:
        """Process a single address to build a heterogeneous graph."""
        center_lower = center_address.lower()
        
        # Load transactions and identify neighborhood
        neighborhood, tx_dataframes = self._load_center_transactions(center_address, center_lower, tx_types, preloaded_files)
        if not tx_dataframes or len(neighborhood) <= 1:
            logger.debug(f"No transactions or neighbors found for {center_address}, skipping.")
            return None, False
        
        # Load neighbor transactions if specified
        neighbor_tx_dataframes = {}
        if include_neighbor_edges:
            important_neighbors = self._find_important_neighbors(center_lower, tx_dataframes)
            if important_neighbors:
                neighbor_tx_dataframes = self._load_neighbor_transactions(
                    center_lower, neighborhood, important_neighbors, tx_types, preloaded_files
                )
        
        # Categorize nodes and build graph structure
        data = HeteroData()
        data.address = center_address
        addresses_by_category, addr_to_index, center_node_info = self._categorize_nodes(neighborhood, shared_address_categories, center_lower)
        
        if not self._add_nodes_to_graph(data, addresses_by_category, center_lower, center_node_info, info, tx_types):
            logger.warning(f"Failed to add nodes for {center_address}, graph may be incomplete.")
            return data, False
        
        # Process transactions to create edges
        edge_dict, edge_attr_dict = defaultdict(list), defaultdict(list)
        
        self._add_edges_from_transactions(
            tx_dataframes, tx_types, edge_types, addr_to_index, edge_dict, edge_attr_dict, 
            is_neighbor_tx=False, max_edges_per_pair=None, sort_by_timestamp=True
        )
        
        if include_neighbor_edges and neighbor_tx_dataframes:
            self._add_edges_from_transactions(
                neighbor_tx_dataframes, tx_types, edge_types, addr_to_index, edge_dict, 
                edge_attr_dict, is_neighbor_tx=True, max_edges_per_pair=max_edges_per_pair
            )
        
        # Add edges to the graph
        added_edges = self._add_edges_to_graph(data, edge_dict, edge_attr_dict)
        logger.debug(f"Finished processing {center_address}. Added edges: {added_edges}")
        
        return data, added_edges

    def _load_center_transactions(self, center_address: str, center_lower: str, 
                                 tx_types: List[str], preloaded_files: Dict) -> Tuple[Set[str], Dict[str, pd.DataFrame]]:
        """Load transaction data for the center address and identify its neighbors."""
        neighborhood = {center_lower}
        tx_dataframes = {}
        
        for tx_type in tx_types:
            tx_file = os.path.join('Dataset', 'PhishCombine', 'RelatedTransactions', tx_type, f"{center_address}.csv")
            
            if tx_file not in preloaded_files:
                preloaded_files[tx_file] = self._load_transactions(tx_file)
            df = preloaded_files[tx_file]
                
            if df is not None:
                tx_dataframes[tx_type] = df
                outgoing = df[df['from'] == center_lower]['to'].dropna().unique()
                incoming = df[df['to'] == center_lower]['from'].dropna().unique()
                neighborhood.update(outgoing)
                neighborhood.update(incoming)
                
        return neighborhood, tx_dataframes
        
    def _load_neighbor_transactions(self, center_lower: str, neighborhood: Set[str], 
                                  important_neighbors: List[str], tx_types: List[str], 
                                  preloaded_files: Dict) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Load transactions for important neighbors."""
        neighbor_tx_dataframes = {}
        for neighbor in important_neighbors:
            for tx_type in tx_types:
                neighbor_tx_file = os.path.join('Dataset', 'PhishCombine', 'RelatedAddressTransactions', tx_type, f"{neighbor}.csv")
                
                if neighbor_tx_file not in preloaded_files:
                    preloaded_files[neighbor_tx_file] = self._load_transactions(neighbor_tx_file)
                df = preloaded_files[neighbor_tx_file]
                    
                if df is not None:
                    mask = (df['from'].isin(neighborhood) | df['to'].isin(neighborhood)) & \
                           (df['from'] != center_lower) & (df['to'] != center_lower) & \
                           (df['from'] != df['to'])
                    
                    if mask.any():
                        neighbor_tx_dataframes[(neighbor, tx_type)] = df[mask]
                        
        return neighbor_tx_dataframes
        
    def _find_important_neighbors(self, center_lower: str, tx_dataframes: Dict, 
                                  max_neighbors: int = 20) -> List[str]:
        """Identify important neighbors based on transaction frequency and value."""
        neighbor_scores = defaultdict(lambda: {'count': 0, 'value': 0.0})
        
        all_txs_list = [df for df in tx_dataframes.values() if df is not None]
        if not all_txs_list:
            return []
        
        all_txs = pd.concat(all_txs_list, ignore_index=True)
        if all_txs.empty:
            return []

        center_txs = all_txs[(all_txs['from'] == center_lower) | (all_txs['to'] == center_lower)].copy()
        if center_txs.empty:
            return []

        center_txs.loc[:, 'value'] = pd.to_numeric(center_txs['value'], errors='coerce').fillna(0)
        
        # Aggregate outgoing transactions
        outgoing = center_txs[center_txs['from'] == center_lower]
        if not outgoing.empty:
            out_agg = outgoing.groupby('to').agg(count=('to', 'size'), value=('value', 'sum')).reset_index()
            for _, row in out_agg.iterrows():
                addr = row['to']
                if addr and addr != center_lower:
                    neighbor_scores[addr]['count'] += row['count']
                    neighbor_scores[addr]['value'] += row['value']

        # Aggregate incoming transactions
        incoming = center_txs[center_txs['to'] == center_lower]
        if not incoming.empty:
            in_agg = incoming.groupby('from').agg(count=('from', 'size'), value=('value', 'sum')).reset_index()
            for _, row in in_agg.iterrows():
                addr = row['from']
                if addr and addr != center_lower:
                    neighbor_scores[addr]['count'] += row['count']
                    neighbor_scores[addr]['value'] += row['value']

        if not neighbor_scores:
            return []
        
        scored_neighbors = [
            (addr, data['count'] * 0.7 + math.log1p(data['value']) * 0.3 if data['value'] > 0 else data['count'] * 0.7)
            for addr, data in neighbor_scores.items()
        ]
        
        return [addr for addr, _ in sorted(scored_neighbors, key=lambda x: x[1], reverse=True)[:max_neighbors]]

    def _add_edges_from_transactions(self, tx_dataframes: Dict, tx_types: List[str], 
                                     edge_types: List[str], addr_to_index: Dict, 
                                     edge_dict: Dict, edge_attr_dict: Dict, 
                                     is_neighbor_tx: bool = False, max_edges_per_pair: Optional[int] = 20, 
                                     sort_by_timestamp: bool = False):
        """Extract edges from transaction dataframes and add them to the graph dictionaries."""
        for key, df in tx_dataframes.items():
            tx_type = key[1] if is_neighbor_tx else key
            
            if tx_type in tx_types:
                edge_type_idx = tx_types.index(tx_type)
                if edge_type_idx < len(edge_types):
                    edge_type = edge_types[edge_type_idx]
                    self._extract_edges_from_df(
                        df, addr_to_index, edge_type, edge_dict, edge_attr_dict, 
                        max_edges_per_pair, sort_by_timestamp
                    )

    def _extract_edges_from_df(self, df: pd.DataFrame, addr_to_index: Dict, edge_type: str, 
                               edge_dict: Dict, edge_attr_dict: Dict,
                               max_edges_per_pair: Optional[int] = 20, 
                               sort_by_timestamp: bool = False):
        """Helper to extract edge information from a DataFrame."""
        if df.empty: return

        if sort_by_timestamp and 'timeStamp' in df.columns:
            df = df.sort_values('timeStamp', ascending=False)

        valid_addresses = set(addr_to_index.keys())
        valid_df = df[df['from'].isin(valid_addresses) & df['to'].isin(valid_addresses)]
        if valid_df.empty: return
        
        for (from_addr, to_addr), group in valid_df.groupby(['from', 'to'], sort=False):
            if from_addr == to_addr: continue

            source_type, source_idx = addr_to_index[from_addr]
            target_type, target_idx = addr_to_index[to_addr]
            edge_key = (source_type, edge_type, target_type)
            
            transactions_to_process = group.head(max_edges_per_pair) if max_edges_per_pair is not None else group
            
            for _, row in transactions_to_process.iterrows():
                edge_dict[edge_key].append((source_idx, target_idx))
                edge_attr_dict[edge_key].append(self._extract_edge_features(row, edge_type))

    def _extract_edge_features(self, row: pd.Series, tx_type: str) -> torch.Tensor:
        """Extract edge features for different transaction types."""
        def safe_numeric(val, default=0.0):
            if pd.isna(val) or val == '': return default
            v = float(val)
            return np.log1p(abs(v)) * np.sign(v) if abs(v) > 1e10 else v
        
        basic_features = [
            safe_numeric(row.get('value')),
            safe_numeric(row.get('gasPrice')),
            safe_numeric(row.get('gas')),
            safe_numeric(row.get('gasUsed')),
            safe_numeric(row.get('timeStamp'), 0.0) / 1e9
        ]
        
        tx_type_lower = tx_type.lower()
        if tx_type_lower == 'erc20':
            type_specific = [safe_numeric(row.get('tokenDecimal'))]
        elif tx_type_lower == 'normal':
            inp = row.get('input', '')
            has_call = float(inp is not None and inp != '0x' and inp != '')
            type_specific = [0.0, has_call, safe_numeric(row.get('isError'))]
        elif tx_type_lower == 'internal':
            is_call = float(row.get('type') == 'call')
            type_specific = [0.0, is_call, safe_numeric(row.get('isError'))]
        else:
            type_specific = []
        
        features = np.array(basic_features + type_specific, dtype=np.float32)
        return torch.from_numpy(np.nan_to_num(features))

    def _extract_node_features(self, address: str, is_center: bool, tx_types: List[str]) -> torch.Tensor:
        """Extract node features for a given address."""
        cache_key = f"{address}_{1 if is_center else 0}"
        if cache_key in self._node_features_cache:
            return self._node_features_cache[cache_key]
            
        address_lower = address.lower()
        tx_base_path = 'Dataset/PhishCombine/RelatedTransactions' if is_center else 'Dataset/PhishCombine/RelatedAddressTransactions'
        
        total_stats = defaultdict(float)
        total_stats['unique_addresses'] = set()
        total_stats['first_tx_time'] = float('inf')
        total_stats['tx_type_stats'] = {tx_type.lower(): defaultdict(float) for tx_type in tx_types}

        for tx_type in tx_types:
            df = self._load_transactions(os.path.join(tx_base_path, tx_type, f"{address}.csv"))
            if df is None: continue
            self._update_node_stats_from_df(df, address_lower, tx_type, total_stats)

        final_features = self._build_feature_vector(total_stats, is_center, tx_types)
        
        features_tensor = torch.tensor(final_features, dtype=torch.float)
        self._node_features_cache[cache_key] = features_tensor
        return features_tensor

    def _update_node_stats_from_df(self, df: pd.DataFrame, address_lower: str, tx_type: str, stats: Dict):
        """Update node statistics from a transaction dataframe."""
        num_txs = len(df)
        tx_type_lower = tx_type.lower()
        stats['txs'] += num_txs
        stats['tx_type_stats'][tx_type_lower]['count'] = num_txs

        outgoing_mask = df['from'] == address_lower
        incoming_mask = df['to'] == address_lower
        
        outgoing_count = outgoing_mask.sum()
        incoming_count = incoming_mask.sum()
        stats['tx_type_stats'][tx_type_lower]['outgoing_count'] = outgoing_count
        stats['tx_type_stats'][tx_type_lower]['incoming_count'] = incoming_count

        if 'value' in df.columns:
            df_value = pd.to_numeric(df['value'], errors='coerce').fillna(0)
            outgoing_value = df_value[outgoing_mask].sum()
            incoming_value = df_value[incoming_mask].sum()
            stats['tx_type_stats'][tx_type_lower]['outgoing_value'] = outgoing_value
            stats['tx_type_stats'][tx_type_lower]['incoming_value'] = incoming_value
            stats['sent'] += outgoing_value
            stats['received'] += incoming_value

        if outgoing_mask.any(): stats['unique_addresses'].update(df.loc[outgoing_mask, 'to'].dropna())
        if incoming_mask.any(): stats['unique_addresses'].update(df.loc[incoming_mask, 'from'].dropna())

        if 'timeStamp' in df.columns:
            timestamps = pd.to_numeric(df['timeStamp'], errors='coerce').dropna()
            if not timestamps.empty:
                stats['first_tx_time'] = min(stats['first_tx_time'], timestamps.min())
                stats['last_tx_time'] = max(stats['last_tx_time'], timestamps.max())
        
        if 'isError' in df.columns: stats['error_txs'] += pd.to_numeric(df['isError'], errors='coerce').sum()
        if 'gasUsed' in df.columns and 'gasPrice' in df.columns:
            gas_cost = (pd.to_numeric(df['gasUsed'], errors='coerce').fillna(0) * pd.to_numeric(df['gasPrice'], errors='coerce').fillna(0)).sum()
            stats['gas_spent'] += gas_cost
        
        if tx_type_lower == 'normal' and 'input' in df.columns:
            stats['contract_calls'] += df['input'].apply(lambda x: isinstance(x, str) and x != '0x' and x != '').sum()
        elif tx_type_lower == 'erc20':
            stats['token_txs'] += num_txs

    def _build_feature_vector(self, stats: Dict, is_center: bool, tx_types: List[str]) -> np.ndarray:
        """Build the final feature vector from aggregated statistics."""
        first_tx = 0.0 if stats['first_tx_time'] == float('inf') else stats['first_tx_time']
        active_duration = max(0, stats['last_tx_time'] - first_tx)
        error_rate = 1.0 - (stats['error_txs'] / max(1, stats['txs']))

        final_features = [
            float(is_center), stats['txs'], stats['sent'], stats['received'],
            float(len(stats['unique_addresses'])), first_tx, stats['last_tx_time'],
            active_duration, stats['contract_calls'], stats['token_txs'],
            error_rate, stats['gas_spent']
        ]
        
        for tx_type in tx_types:
            tx_stats = stats['tx_type_stats'].get(tx_type.lower(), defaultdict(float))
            final_features.extend([
                tx_stats['count'], tx_stats['outgoing_value'], tx_stats['incoming_value'],
                tx_stats['outgoing_count'], tx_stats['incoming_count'],
                tx_stats['count'] / max(1, stats['txs'])
            ])
        
        features = np.array(final_features, dtype=np.float32)
        np.nan_to_num(features, copy=False)
        large_mask = np.abs(features) > 1e10
        features[large_mask] = np.log1p(np.abs(features[large_mask])) * np.sign(features[large_mask])
        np.clip(features, -1e9, 1e9, out=features)
        
        return features

    def len(self) -> int:
        """Return the number of graphs in the dataset."""
        address_file = self.processed_paths[1]
        return len(torch.load(address_file)) if os.path.exists(address_file) else 0
    
    def get(self, idx: int) -> HeteroData:
        """Get a graph from the dataset."""
        data_file = self.processed_paths[0]
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Processed data file not found: {data_file}")
        
        data_list = torch.load(data_file)
        if not 0 <= idx < len(data_list):
            raise IndexError(f"Index {idx} out of range for dataset with {len(data_list)} items")
        
        return data_list[idx]
    
    def get_address(self, idx: int) -> Optional[str]:
        """Get the Ethereum address corresponding to the graph at index idx."""
        address_file = self.processed_paths[1]
        if not os.path.exists(address_file): return None
        addresses = torch.load(address_file)
        return addresses[idx] if 0 <= idx < len(addresses) else None
    
    def get_addresses(self) -> List[str]:
        """Get all Ethereum addresses in the dataset."""
        address_file = self.processed_paths[1]
        return torch.load(address_file) if os.path.exists(address_file) else []

    def _categorize_nodes(self, neighborhood: Set[str], shared_address_categories: Dict, 
                          center_lower: str) -> Tuple[Dict, Dict, Optional[Tuple]]:
        """Categorize all nodes in the neighborhood."""
        addresses_by_category = defaultdict(list)
        addr_to_index = {}
        center_node_info = None
        
        for addr in neighborhood:
            if not addr: continue
            category = shared_address_categories.get(addr) or self._get_address_category(addr)
            shared_address_categories[addr] = category
            addresses_by_category[category].append(addr)
        
        for node_type, addresses in addresses_by_category.items():
            for local_idx, addr in enumerate(addresses):
                addr_to_index[addr] = (node_type, local_idx)
                if addr == center_lower:
                    center_node_info = (node_type, local_idx)
        
        return addresses_by_category, addr_to_index, center_node_info

    def _add_nodes_to_graph(self, data: HeteroData, addresses_by_category: Dict,
                            center_lower: str, center_node_info: Optional[Tuple], 
                            info: Dict, tx_types: List[str]) -> bool:
        """Add all nodes and their features to the graph."""
        for node_type in NODE_CATEGORIES:
            addresses = addresses_by_category.get(node_type, [])
            if not addresses: continue
            
            features, labels = [], []
            for local_idx, addr in enumerate(addresses):
                is_center = (addr == center_lower)
                if is_center and center_node_info is None:
                    center_node_info = (node_type, local_idx)
                
                features.append(self._extract_node_features(addr, is_center, tx_types))
                labels.append(info.get('VerifiedLabel', 0) if is_center else -1)
            
            if features:
                data[node_type].x = self._create_padded_tensor(features)
                data[node_type].y = torch.tensor(labels, dtype=torch.long)
                data[node_type].original_addresses = addresses
        
        if center_node_info:
            data.center_node_type, data.center_node_idx = center_node_info
            return True
        return False
        
    def _create_padded_tensor(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Create a tensor from a list of tensors, padding to a consistent size if needed."""
        if not features:
            return torch.empty(0, 0, dtype=torch.float)
            
        feature_dims = {f.size(0) for f in features}
        if len(feature_dims) == 1:
            return torch.stack(features)
        
        max_dim = max(feature_dims)
        padded_features = [
            torch.cat([f, torch.zeros(max_dim - f.size(0), dtype=f.dtype)]) if f.size(0) < max_dim else f
            for f in features
        ]
        return torch.stack(padded_features)
        
    def _add_edges_to_graph(self, data: HeteroData, edge_dict: Dict, edge_attr_dict: Dict) -> bool:
        """Add extracted edges and their attributes to the graph."""
        added_edges = False
        for edge_key, edge_indices in edge_dict.items():
            if not edge_indices: continue
            
            source_type, edge_type, target_type = edge_key
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            
            features = edge_attr_dict.get(edge_key, [])
            if features:
                edge_attr = self._create_padded_tensor(features)
                data[source_type, edge_type, target_type].edge_attr = edge_attr
            
            data[source_type, edge_type, target_type].edge_index = edge_index
            added_edges = True
        
        return added_edges

def print_dataset_summary(dataset: PhishingGraphDataset, num_graphs_to_show: Optional[int] = 5):
    """Print summary information about the dataset."""
    logger.info(f"\n=== Dataset Summary ===")
    dataset_len = len(dataset)
    logger.info(f"Dataset contains {dataset_len} graphs")
    
    if dataset_len == 0: return
    
    num_to_show = min(num_graphs_to_show, dataset_len)
    
    total_node_counts = defaultdict(int)
    total_edge_counts = defaultdict(int)

    for i in range(num_to_show):
        graph = dataset[i]
        address = dataset.get_address(i)
        
        node_counts = {nt: graph[nt].x.size(0) for nt in graph.node_types if hasattr(graph[nt], 'x')}
        edge_counts = {f"{et[0]}-{et[1]}->{et[2]}": graph[et].edge_index.size(1) for et in graph.edge_types}
        
        total_nodes = sum(node_counts.values())
        total_edges = sum(edge_counts.values())
        
        for nt, count in node_counts.items(): total_node_counts[nt] += count
        for et, count in edge_counts.items(): total_edge_counts[et] += count
        
        label_text = "Unknown"
        if hasattr(graph, 'center_node_type') and hasattr(graph, 'center_node_idx'):
            label = graph[graph.center_node_type].y[graph.center_node_idx].item()
            label_text = "Phishing" if label == 1 else "Non-phishing"
            
        logger.info(f"Graph {i} ({address}): {total_nodes} nodes, {total_edges} edges, Label: {label_text}")

    logger.info(f"\n=== Aggregated Stats for first {num_to_show} graphs ===")
    logger.info(f"Node distribution: {dict(total_node_counts)}")
    logger.info(f"Edge distribution: {dict(total_edge_counts)}")

if __name__ == "__main__":
    logger.info(f"\n=== Running PhishingGraphDataset Processing ===")
    
    with open(ADDRESS_DATA_PATH, 'r') as f:
        total_addresses = len(json.load(f))
    
    batch_size = DEFAULT_BATCH_SIZE
    total_batches = (total_addresses + batch_size - 1) // batch_size
    
    logger.info(f"Total addresses: {total_addresses}, Batch size: {batch_size}, Total batches: {total_batches}")
    
    for batch_id in range(total_batches):
        logger.info(f"\n=== Processing Batch {batch_id}/{total_batches-1} ===")
        
        dataset = PhishingGraphDataset(
            root="Dataset/PhishCombine/", 
            use_cached_features=True,
            include_neighbor_edges=True,
            batch_id=batch_id,
            batch_size=batch_size
        )
        
        print_dataset_summary(dataset, num_graphs_to_show=5)
        
        logger.info("Cleaning up memory...")
        del dataset
        gc.collect()
        
        logger.info(f"Batch {batch_id} completed successfully.")
        
    logger.info("\n=== All batches completed ===")
