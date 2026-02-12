## LETAGNN: Label-Enhanced Time-Aware Graph Neural Network

Implementation of **LETAGNN** for phishing address detection on the Ethereum transaction network.
The codebase also includes baseline models and ablation studies used in the paper.

All comments and identifiers are in English to ease open-source use and reproduction.

---

## 1. Environment

- Python 3.8+ is recommended.
- Install dependencies:

```bash
pip install -r requirements.txt
```

GPU (CUDA) is strongly recommended for training, but not strictly required for running the scripts.

---

## 2. Data Preparation

You have **two options**:

### 2.1 Download pre-built label-enhanced graphs (recommended)

We provide pre-processed homogeneous graphs for three datasets:

- `Meta`
- `ZipZap`
- `2DynEth`

Download from OneDrive:

- Public folder link: `https://1drv.ms/f/c/cdc0f83bf736d892/EpAFMmHdQrxLuN5zcStBc_ABQZZGcasSHpjmCnWNs2n4Tg?e=04Et3Q`
- Inside that folder you will find `homo_*.pkl` files: `homo_Meta_graphs.pkl`, `homo_ZipZap_graphs.pkl`, `homo_2DynEth_graphs.pkl`

After download, place them under:

```text
Dataset/PhishCombine/graphs_home_bak0201/
```

The default training scripts (e.g. `TrainLETA.py`, `TemporalBaseline.py`,
`TemporalAggregationComparison.py`, `Imbalance.py`) expect this directory structure.

### 2.2 Build graphs from raw data (full pipeline)

If you want to reproduce the full data pipeline:

1. **Configure Etherscan access in `Utils.py`**

   - Set your Etherscan API key list:

   ```python
   self.apikeys = ["YOUR_ETHERSCAN_API_KEY"]
   ```

   - Optionally set headers for HTML scraping in `getAddressLabelFromEthereumPage`:

   ```python
   headers = {
       'cookie': 'YOUR_ETHERSCAN_COOKIES_HERE',
       'user-agent': 'YOUR_USER_AGENT_STRING_HERE',
       'origin': 'https://etherscan.io'
   }
   ```

   For open-source safety, the repository only contains placeholders, **no real keys or cookies**.

2. **Run `Dataset.py`**

   - Download/verify phishing address labels.
   - Download related transactions for each address.
   - Extract node-level features and neighbor relationships.

3. **Run `Graph.py`**

   - Construct label-enhanced heterogeneous graphs (`HeteroData`) for each address.

4. **Run `ConvertGraphs.py`**

   - Convert heterogeneous graphs to homogeneous graphs (`torch_geometric.data.Data`).
   - Save as `homo_*.pkl` under `Dataset/PhishCombine/graphs_home/` (or similar).

Once the homogeneous graphs are built, you can use all training and evaluation scripts as below.

---

## 3. Training LETAGNN

The main training entry point is `TrainLETA.py`, which runs 5-fold cross-validation on all three datasets.

Basic usage:

```bash
python TrainLETA.py --epochs 51 --seed 42 --gpu 0
```

Key options:

- `--epochs`: number of training epochs (default: 51)
- `--seed`: random seed (default: 42)
- `--gpu`: GPU id, e.g. `0`
- `--light-window`: use `LETAGNN_LightWindow` (full-graph encoder + light temporal branch)

Outputs:

- Per-fold logs under `logs/`
- Attention weights for temporal windows in `logs/attention_weights_<dataset>_fold*.pkl`
- Summary CSV under `Result/` (e.g. `TrainLETA_*.csv`)

---

## 4. Baselines and Ablation Scripts

The repository includes several scripts used in the experiments:

- `TemporalBaseline.py`  
  Temporal / dynamic GNN baselines on both homogeneous and heterogeneous graphs:
  - Homogeneous: `TGAT`, `TGN`, `EvolveGCN`
  - Heterogeneous: `RGCN`, `HGT`, `ieHGCN`

- `TemporalAggregationComparison.py`  
  Compares temporal aggregation mechanisms **LSTM**, **GRU**, and **attention** under a shared encoder.

- `Imbalance.py`  
  Evaluates LETAGNN under different positive-sample ratios to simulate class imbalance.

- `EWDPS_Task1_Reproduce.py`  
  Reimplementation of the EWDPS Task 1 setting using EWDPS-style features + Random Forest, for comparison.

Each script writes its results to a CSV file under `Result/` with timestamps in the filename.

