import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

import pandas as pd
import re
import tqdm
import time
from Utils import *
import tqdm
import json
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from collections import defaultdict

# Constants imported from Graph.py
TX_TYPES = ['Normal', 'Internal', 'ERC20']

def VerifyAddressLabel():
    """
    Verify address labels using Etherscan and update dataset.

    Reads addresses from combined_phishing_dataset.csv. For rows with SpanLabel == 'To_Be_Verified',
    fetch labels from Etherscan and update SpanLabel, TitleLabel, and VerifiedLabel (1 = phishing, 0 = non-phishing).
    Saves a checkpoint every 10 records and writes the final result to verified_phishing_dataset.csv.
    """
    cdataset_df = pd.read_csv('Dataset/PhishCombine/combined_phishing_dataset.csv')
    
    # Ensure required columns exist
    if 'SpanLabel' not in cdataset_df.columns:
        cdataset_df['SpanLabel'] = ""
    if 'TitleLabel' not in cdataset_df.columns:
        cdataset_df['TitleLabel'] = ""
    if 'VerifiedLabel' not in cdataset_df.columns:
        cdataset_df['VerifiedLabel'] = cdataset_df['Label']
    
    to_verify_df = cdataset_df[cdataset_df['SpanLabel'] == 'To_Be_Verified']
    
    for index, row in tqdm.tqdm(to_verify_df.iterrows(), total=len(to_verify_df), desc="Processing addresses to verify"):
        address = row['Address']
        span_label, title_label = getAddressLabelFromEthereumPage(address)
        cdataset_df.at[index, 'SpanLabel'] = span_label
        cdataset_df.at[index, 'TitleLabel'] = title_label
        
        if span_label and re.search(r'phish', span_label, re.IGNORECASE):
            cdataset_df.at[index, 'VerifiedLabel'] = 1
        else:
            cdataset_df.at[index, 'VerifiedLabel'] = 0
    
        # Save checkpoint regularly
        if index > 0 and (index % 10 == 0 or index == len(to_verify_df) - 1):
            cdataset_df.to_csv('Dataset/PhishCombine/verified_phishing_dataset_checkpoint.csv', index=False)
    
    cdataset_df.to_csv('Dataset/PhishCombine/verified_phishing_dataset.csv', index=False)

def GetAddressRelatedTransactions():
    """
    Download related transactions for each address (Normal, Internal, ERC20) and save to CSV files.
    Reads verified_phishing_dataset.csv (starting from row 10500) and fetches missing transaction files
    under Dataset/PhishCombine/RelatedTransactions/.
    """
    relatedTransactionPath = "Dataset/PhishCombine/RelatedTransactions/"
    
    data = DataSource()
    
    if not os.path.exists(relatedTransactionPath):
        os.makedirs(relatedTransactionPath)
    
    for directory in ['Normal', 'Internal', 'ERC20']:
        path = os.path.join(relatedTransactionPath, directory)
        if not os.path.exists(path):
            os.makedirs(path)
            
    phishing_df = pd.read_csv("Dataset/PhishCombine/verified_phishing_dataset.csv", skiprows=10500)
    addresses = phishing_df.iloc[:, 0].tolist()
    
    for address in tqdm(addresses, desc="Processing addresses"):
        for tt in ['Normal/', 'Internal/', 'ERC20/']:
            file_path = f"{relatedTransactionPath}{tt}{address}.csv"
            if not os.path.exists(file_path):
                try:
                    data.getTotalDatafromScan(address, tt, f"{relatedTransactionPath}{tt}")
                    time.sleep(data.timeStep)
                except Exception as e:
                    print(f"Error processing {address} for {tt}: {e}")

def GetRelatedAddressLabel(reverse=False):
    relatedTransactionPath = "Dataset/PhishCombine/RelatedTransactions/"
    
    labels_json_path = "Dataset/PhishCombine/address_labels_reverse.json" if reverse else "Dataset/PhishCombine/address_labels.json"
    skipped_addresses_path = "Dataset/PhishCombine/skipped_addresses_reverse.json" if reverse else "Dataset/PhishCombine/skipped_addresses.json"
    print("Using reverse mode" if reverse else "Using normal mode")
    
    # Load existing data if available
    if os.path.exists(labels_json_path):
        with open(labels_json_path, 'r') as f:
            address_labels = json.load(f)
    else:
        address_labels = {}
    
    if os.path.exists(skipped_addresses_path):
        with open(skipped_addresses_path, 'r') as f:
            skipped_addresses = json.load(f)
    else:
        skipped_addresses = {}
    
    df = pd.read_csv("Dataset/PhishCombine/verified_phishing_dataset.csv")
    if reverse:
        df = df.iloc[::-1]
    
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        address = row['Address']
        label = row['VerifiedLabel']
        
        related_addresses = set()
        
        transaction_counts = {'Normal': 0, 'Internal': 0, 'ERC20': 0}
        for tt in ['Normal/', 'Internal/', 'ERC20/']:
            file_path = f"{relatedTransactionPath}{tt}{address}.csv"
            if os.path.exists(file_path):
                transaction_df = pd.read_csv(file_path)
                # Collect neighbors while excluding the address itself
                related_addresses.update([addr for addr in transaction_df['from'].tolist() if isinstance(addr, str) and addr.lower() != address.lower()])
                related_addresses.update([addr for addr in transaction_df['to'].tolist() if isinstance(addr, str) and addr.lower() != address.lower()])
                transaction_counts[tt.replace('/', '')] = len(transaction_df)
         
        num_related = len(related_addresses)
        tqdm.tqdm.write(f"Address {address} (Label={label}): {num_related} related, Normal: {transaction_counts['Normal']}, Internal: {transaction_counts['Internal']}, ERC20: {transaction_counts['ERC20']}")
        
        if num_related > 1000:
            skipped_addresses[address] = num_related
            with open(skipped_addresses_path, 'w') as f:
                json.dump(skipped_addresses, f)
            tqdm.tqdm.write(f"Skip {address}: too many related addresses (> 1000)")
            continue
        
        new_labels_added = False
        for related_addr in tqdm.tqdm(related_addresses, leave=False):
            if related_addr.lower() not in address_labels:
                span_label, title_label, label_text = getAddressLabelFromEthereumPage(related_addr)
                address_labels[related_addr.lower()] = {
                    "span_label": span_label,
                    "title_label": title_label,
                    "label_text": label_text
                }
                new_labels_added = True
        
        if new_labels_added:
            with open(labels_json_path, 'w') as f:
                json.dump(address_labels, f)


def GetRelatedAddressTransactions(reverse=False, use_reverse_file=False):
    """
    Download transactions for related addresses listed in address_labels.json (or reverse version).
    Skips an address if any of its transaction files already exists.
    """
    relatedAddressTransactionPath = "Dataset/PhishCombine/RelatedAddressTransactions/"
    labels_json_path = "Dataset/PhishCombine/address_labels_reverse.json" if use_reverse_file else "Dataset/PhishCombine/address_labels.json"
    
    data = DataSource()
    
    os.makedirs(relatedAddressTransactionPath, exist_ok=True)
    for directory in ['Normal', 'Internal', 'ERC20']:
        path = os.path.join(relatedAddressTransactionPath, directory)
        os.makedirs(path, exist_ok=True)
    
    if not os.path.exists(labels_json_path):
        return
    
    with open(labels_json_path, 'r') as f:
        address_labels = json.load(f)
    
    related_addresses = list(address_labels.keys())
    if reverse:
        related_addresses = related_addresses[::-1]
        print(f"Total {len(related_addresses)} related addresses (reverse order)")
    else:
        print(f"Total {len(related_addresses)} related addresses")
    
    for address in tqdm.tqdm(related_addresses):
        transaction_types = ['Normal/', 'Internal/', 'ERC20/']
        existing_files = [tt for tt in transaction_types if os.path.exists(f"{relatedAddressTransactionPath}{tt}{address}.csv")]
        
        if existing_files:
            continue
            
        for tt in transaction_types:
            try:
                data.getTotalDatafromScan(address, tt, f"{relatedAddressTransactionPath}{tt}")
            except Exception as e:
                print(f"Failed to fetch {tt} for {address}: {e}")

# Feature extraction

def extract_node_features(addr, is_center, tx_types, cache_dict=None):
    """
    Extract node features from transaction data.
    Returns a feature tensor.
    """
    addr_lower = addr.lower()
    is_center_bool = bool(is_center)
    
    cache_key = f"{addr_lower}_{1 if is_center_bool else 0}"
    
    # Read from cache if available
    if cache_dict is not None and cache_key in cache_dict:
        return cache_dict[cache_key]
    
    tx_dir = os.path.join('Dataset', 'PhishCombine', 
                         'RelatedTransactions' if is_center_bool else 'RelatedAddressTransactions')
    
    tx_counts = np.zeros(len(tx_types), dtype=np.float32)
    tx_values = np.zeros(len(tx_types), dtype=np.float32)
    tx_age = np.zeros(len(tx_types), dtype=np.float32)
    
    columns_to_read = ['value', 'timeStamp']
    max_rows = 20000
    
    for i, tx_type in enumerate(tx_types):
        tx_file = os.path.join(tx_dir, tx_type, f"{addr}.csv")
        
        if not os.path.exists(tx_file) or os.path.getsize(tx_file) == 0:
            continue
        
        try:
            with open(tx_file, 'r') as f:
                header = f.readline().strip().split(',')
                valid_cols = [col for col in columns_to_read if col in header]
                
                if not valid_cols:
                    continue
            
            tx_df = pd.read_csv(
                tx_file, 
                usecols=valid_cols,
                nrows=max_rows,
                engine='c',
                dtype={
                    'value': np.float64,
                    'timeStamp': np.float64
                }
            )
            
            tx_count = len(tx_df)
            tx_counts[i] = tx_count
            
            if 'value' in tx_df.columns and tx_count > 0:
                values = pd.to_numeric(tx_df['value'], errors='coerce')
                tx_values[i] = values.sum(skipna=True)
                
            if 'timeStamp' in tx_df.columns and tx_count > 0:
                timestamps = pd.to_numeric(tx_df['timeStamp'], errors='coerce')
                if not timestamps.isna().all():
                    tx_age[i] = timestamps.max(skipna=True)
                
        except Exception as e:
            print(f"Error reading transaction file {tx_file}: {e}")
    
    total_tx_count = np.sum(tx_counts)
    total_value = np.sum(tx_values)
    tx_count_ratios = tx_counts / (total_tx_count + 1e-8)
    
    all_features = np.concatenate([
        tx_counts,
        tx_count_ratios,
        tx_values,
        [total_tx_count],
        [total_value],
        tx_age
    ])
    
    mask = all_features > 1000000
    all_features[mask] = np.log1p(all_features[mask])
    
    features_tensor = torch.tensor(all_features, dtype=torch.float)
    
    if cache_dict is not None:
        cache_dict[cache_key] = features_tensor
    
    return features_tensor

def process_address_features(addr, feature_cache_dir, is_center):
    """Process feature extraction for a single address."""
    tx_dir = os.path.join('Dataset', 'PhishCombine', 
                         'RelatedTransactions' if is_center else 'RelatedAddressTransactions')
    
    has_tx_files = False
    for tx_type in TX_TYPES:
        tx_file = os.path.join(tx_dir, tx_type, f"{addr}.csv")
        if os.path.exists(tx_file) and os.path.getsize(tx_file) > 0:
            has_tx_files = True
            break
    
    if not has_tx_files:
        feature_size = len(TX_TYPES) * 3 + 2 + len(TX_TYPES)
        features = torch.zeros(feature_size, dtype=torch.float)
        return addr, is_center, features
    
    features = extract_node_features(addr, is_center, TX_TYPES)
    return addr, is_center, features

def extract_and_save_features(is_center=True, batch_size=200, num_workers=None):
    """
    Extract and save node features in parallel to CSV files.
    """
    feature_cache_dir = os.path.join('Dataset', 'PhishCombine', 'feature_cache')
    os.makedirs(feature_cache_dir, exist_ok=True)
    
    node_features_file = os.path.join(feature_cache_dir, 'node_features.csv')
    
    node_features_cache = {}
    
    if os.path.exists(node_features_file):
        try:
            node_df = pd.read_csv(node_features_file)
            print(f"Loaded {len(node_df)} cached node features")
            for _, row in node_df.iterrows():
                if bool(row['is_center']) == is_center:
                    addr = row['address']
                    features = torch.tensor([float(x) for x in row['features'].split(',')], dtype=torch.float)
                    node_features_cache[f"{addr}_{int(is_center)}"] = features
        except Exception as e:
            print(f"Error loading node features cache: {e}")
    
    if is_center:
        print("Processing center node features")
        df = pd.read_csv("Dataset/PhishCombine/verified_phishing_dataset.csv")
        addresses = [addr for addr in df['Address'].tolist() if any(
            os.path.exists(os.path.join('Dataset', 'PhishCombine', 'RelatedTransactions', tx_type, f"{addr}.csv"))
            for tx_type in TX_TYPES
        )]
    else:
        print("Processing neighbor node features")
        neighborhood_file = os.path.join(feature_cache_dir, 'neighborhood.json')
        
        if os.path.exists(neighborhood_file):
            try:
                print(f"Loading neighborhood information from {neighborhood_file}")
                with open(neighborhood_file, 'r') as f:
                    neighborhood_data = json.load(f)
                
                neighbor_addresses = set()
                for _, neighbors in neighborhood_data.items():
                    neighbor_addresses.update(neighbors)
                
                addresses = list(neighbor_addresses)
                print(f"Found {len(addresses)} unique neighbor addresses from neighborhood.json")
            except Exception as e:
                print(f"Error loading neighborhood data: {e}, falling back to directory scanning")
                addresses = []
        else:
            print("No neighborhood.json found, scanning directories for neighbor addresses")
            addresses = []
        
        if not addresses:
            neighbor_addresses = set()
            for tx_type in TX_TYPES:
                tx_dir = os.path.join('Dataset', 'PhishCombine', 'RelatedAddressTransactions', tx_type)
                if os.path.exists(tx_dir):
                    addrs = [os.path.splitext(f)[0] for f in os.listdir(tx_dir) if f.endswith('.csv')]
                    neighbor_addresses.update(addrs)
            addresses = list(neighbor_addresses)
            print(f"Found {len(addresses)} neighbor addresses from directory scanning")
        
        if len(addresses) > 100:
            print("Pre-filtering addresses with transaction files...")
            tx_base_dir = 'Dataset/PhishCombine/RelatedAddressTransactions'
            
            check_workers = min(16, multiprocessing.cpu_count()) if num_workers is None else num_workers
            
            def has_tx_files(addr):
                """Check if address has any valid transaction files."""
                return any(os.path.exists(os.path.join(tx_base_dir, tx_type, f"{addr}.csv")) and 
                          os.path.getsize(os.path.join(tx_base_dir, tx_type, f"{addr}.csv")) > 0
                          for tx_type in TX_TYPES)
            
            valid_addresses = []
            with ThreadPoolExecutor(max_workers=check_workers) as executor:
                futures = [executor.submit(has_tx_files, addr) for addr in addresses]
                for i, future in enumerate(tqdm.tqdm(as_completed(futures), total=len(futures), 
                                           desc="Checking transaction files")):
                    if future.result():
                        valid_addresses.append(addresses[i])
            
            print(f"Filtered down to {len(valid_addresses)} addresses with transaction files")
            addresses = valid_addresses
    
    if os.path.exists(node_features_file):
        node_df = pd.read_csv(node_features_file)
        processed_addresses = set(row['address'] for _, row in node_df.iterrows() 
                                if bool(row['is_center']) == is_center)
    else:
        processed_addresses = set()
    
    addresses_to_process = [addr for addr in addresses if addr not in processed_addresses]
    
    print(f"Total addresses: {len(addresses)}")
    print(f"Already processed: {len(processed_addresses)}")
    print(f"To process: {len(addresses_to_process)}")
    
    if not addresses_to_process:
        print("No new addresses to process")
        return
    
    num_workers = min(16, multiprocessing.cpu_count()) if num_workers is None else num_workers
    print(f"Using {num_workers} workers for feature extraction")
    
    lock = multiprocessing.Manager().Lock()
    
    total_addresses = len(addresses_to_process)
    new_features = []
    
    for batch_start in range(0, total_addresses, batch_size):
        batch_end = min(batch_start + batch_size, total_addresses)
        current_batch = addresses_to_process[batch_start:batch_end]
        batch_features = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_address_features, addr, feature_cache_dir, is_center) 
                      for addr in current_batch]
            
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), 
                                   desc=f"Batch {batch_start//batch_size + 1}/{(total_addresses+batch_size-1)//batch_size}"):
                try:
                    addr, is_c, features = future.result()
                    features_str = ','.join(map(str, features.numpy().tolist()))
                    
                    batch_features.append({
                        'address': addr,
                        'is_center': int(is_c),
                        'features': features_str
                    })
                    
                    node_features_cache[f"{addr}_{1 if is_c else 0}"] = features
                except Exception as e:
                    print(f"Error processing address: {e}")
        
        new_features.extend(batch_features)
        
        with lock:
            new_df = pd.DataFrame(new_features)
            
            if os.path.exists(node_features_file):
                try:
                    old_df = pd.read_csv(node_features_file)
                    combined_df = pd.concat([old_df, new_df], ignore_index=True)
                    combined_df.drop_duplicates(subset=['address', 'is_center'], inplace=True)
                    combined_df.to_csv(node_features_file, index=False)
                except Exception:
                    new_df.to_csv(node_features_file, index=False)
            else:
                new_df.to_csv(node_features_file, index=False)
            
            new_features = []
    
    print(f"Feature extraction completed for {'center' if is_center else 'neighbor'} nodes")
    print(f"Features saved to {node_features_file}")

def extract_and_save_neighborhood_information():
    """
    Extract and save neighbor lists for each center node to a JSON file.
    """
    print("Extracting center-to-neighbor relationships...")
    
    feature_cache_dir = os.path.join('Dataset', 'PhishCombine', 'feature_cache')
    os.makedirs(feature_cache_dir, exist_ok=True)
    neighborhood_file = os.path.join(feature_cache_dir, 'neighborhood.json')
    
    if os.path.exists(neighborhood_file):
        try:
            with open(neighborhood_file, 'r') as f:
                neighborhood_data = json.load(f)
                print(f"Loaded existing neighborhood data for {len(neighborhood_data)} center nodes")
        except Exception as e:
            print(f"Error reading existing neighborhood file: {e}")
            neighborhood_data = {}
    else:
        neighborhood_data = {}
    
    df = pd.read_csv("Dataset/PhishCombine/verified_phishing_dataset.csv")
    addresses = df['Address'].tolist()
    
    valid_addresses = [
        address for address in addresses if any(
            os.path.exists(os.path.join('Dataset', 'PhishCombine', 'RelatedTransactions', tx_type, f"{address}.csv")) 
            for tx_type in TX_TYPES
        )
    ]
    
    print(f"Found {len(valid_addresses)} valid center nodes that need neighbor extraction")
    
    addresses_to_process = [addr for addr in valid_addresses if addr not in neighborhood_data]
    print(f"Of which {len(addresses_to_process)} nodes need new neighbor extraction")
    
    if not addresses_to_process:
        print("No new addresses to process")
        return neighborhood_data
        
    num_workers = min(16, multiprocessing.cpu_count())
    batch_size = 100
    
    lock = multiprocessing.Manager().Lock()
    
    for batch_start in range(0, len(addresses_to_process), batch_size):
        batch_end = min(batch_start + batch_size, len(addresses_to_process))
        current_batch = addresses_to_process[batch_start:batch_end]
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(extract_address_neighbors, address) for address in current_batch]
                
            new_neighbors = {}
            for future in tqdm.tqdm(
                as_completed(futures), 
                total=len(futures), 
                desc=f"Batch {batch_start//batch_size + 1}/{(len(addresses_to_process)+batch_size-1)//batch_size}"
            ):
                try:
                    address, neighbors = future.result()
                    new_neighbors[address] = list(neighbors)
                except Exception as e:
                    print(f"Error processing address neighbors: {e}")
            
            with lock:
                neighborhood_data.update(new_neighbors)
                with open(neighborhood_file, 'w') as f:
                    json.dump(neighborhood_data, f)
                    
    print(f"Neighborhood extraction complete, data saved to {neighborhood_file}")
    return neighborhood_data

def extract_address_neighbors(address):
    """
    Extract unique neighbor addresses from the center address's transactions.
    Returns (address, set_of_neighbors).
    """
    neighbors = set()
    address_lower = address.lower()
    
    for tx_type in TX_TYPES:
        tx_file = os.path.join(
            'Dataset', 'PhishCombine', 'RelatedTransactions', 
            tx_type, f"{address}.csv"
        )
        
        if not os.path.exists(tx_file) or os.path.getsize(tx_file) == 0:
            continue
            
        try:
            tx_df = pd.read_csv(
                tx_file, 
                usecols=['from', 'to'] if 'from' in pd.read_csv(tx_file, nrows=0).columns else None,
                engine='c',
                dtype={'from': 'str', 'to': 'str'}
            )
            
            if 'from' in tx_df.columns and 'to' in tx_df.columns:
                outgoing_mask = tx_df['from'].str.lower() == address_lower
                outgoing_neighbors = tx_df.loc[outgoing_mask, 'to'].dropna().tolist()
                neighbors.update(outgoing_neighbors)
                
                incoming_mask = tx_df['to'].str.lower() == address_lower
                incoming_neighbors = tx_df.loc[incoming_mask, 'from'].dropna().tolist()
                neighbors.update(incoming_neighbors)
                
        except Exception as e:
            print(f"Error reading transaction file {tx_file}: {e}")
    
    return address, neighbors

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data processing utilities")
    parser.add_argument("--verify", action="store_true", help="Verify address labels")
    parser.add_argument("--get_transactions", action="store_true", help="Fetch related transactions for addresses")
    parser.add_argument("--get_labels", action="store_true", help="Collect labels for related addresses")
    parser.add_argument("--get_address_transactions", action="store_true", help="Fetch transactions for related addresses")
    parser.add_argument("--extract_center_features", action="store_true", help="Extract center node features")
    parser.add_argument("--extract_neighbor_features", action="store_true", help="Extract neighbor node features")
    parser.add_argument("--extract_neighbors", action="store_true", help="Extract and save neighbor relations")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker threads")
    parser.add_argument("--batch_size", type=int, default=200, help="Addresses per batch")
    parser.add_argument("--reverse", action="store_true", help="Process in reverse order")
    parser.add_argument("--use_reverse_file", action="store_true", help="Use address_labels_reverse.json")
    
    args = parser.parse_args()
    
    if args.verify:
        VerifyAddressLabel()
    if args.get_transactions:
        GetAddressRelatedTransactions()
    if args.get_labels:
        GetRelatedAddressLabel(args.reverse)
    if args.get_address_transactions:
        GetRelatedAddressTransactions(args.reverse, args.use_reverse_file)
    if args.extract_center_features:
        extract_and_save_features(is_center=True, batch_size=args.batch_size, num_workers=args.workers)
    if args.extract_neighbor_features:
        extract_and_save_features(is_center=False, batch_size=args.batch_size, num_workers=args.workers)
    if args.extract_neighbors:
        extract_and_save_neighborhood_information()