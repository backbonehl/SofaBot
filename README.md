# SofaBot
This is a PyTorch implementation of the Sofabot.

## Requirements
* python3.10
* networkx==2.8.8 
* numpy==1.24.0 
* PyGCL==0.1.2 
* scikit-learn==1.2.2 
* torch==1.11.0 
* torch-geometric==2.0.4 
* torch-scatter==2.0.9 
* torch-sparse==0.6.14 
* torchmetrics==0.11.0
* scipy==1.9.3


## Dataset 
Datasets used in the paper are all publicly available datasets. The dataset access instructions and data split configurations are located in `data`

### Quick Start For Node Classification:
Just execute the following command for source model pre-training:
```
python main_src.py
```
Then, execute the following command for adaptation:
```
python main_tgt.py
```
