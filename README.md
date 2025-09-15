# LETAGNN: Label-Enhanced Time-Aware Graph Neural Network for Phishing Detection in Ethereum

## Requirements

To set up the environment, ensure you have Python installed. Then, install the required packages using:

```bash
pip install -r requirements.txt
```

This will install all necessary dependencies for running the scripts.

## Running the Code

To run the code, follow these steps:

1. **Download the Pre-built Label-Enhanced Graph Dataset**
   - You can download the dataset from OneDrive (https://1drv.ms/f/c/cdc0f83bf736d892/EpAFMmHdQrxLuN5zcStBc_ABQZZGcasSHpjmCnWNs2n4Tg?e=04Et3Q).

   **OR**

1.1 **Run Dataset.py**
   - This script will download the raw address dataset, including related transactions and address labels.

1.2 **Run Graph.py**
   - This script constructs the label-enhanced heterogeneous graph.

1.3 **Run ConvertGraphs.py**
   - This script converts the heterogeneous graph into a homogeneous graph.

2. **Run TrainLETA.py**
   - After constructing the graph, use this script to train the LETAGNN model.
