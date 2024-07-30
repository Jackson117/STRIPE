# Spatial-temporal Memory Enhanced Graph Autoencoder for Anomaly Detection in Dynamic Graphs
Source Code of ICDE'25 submitted paper "Spatial-temporal Memory Enhanced Graph Autoencoder for Anomaly Detection in Dynamic Graphs"(https://arxiv.org/abs/2403.09039)

  ## Setup

### Dependencies
+ python==3.10.13
+ pyg==2.5.2
+ torch==2.4.0
+ torch-scatter==2.1.0
+ torch-sparse==0.6.18
+ networkx==3.3
+ numpy==1.23.5
+ scikit-learn==1.5.0
+ scipy==1.10.1
+ tqdm==4.66.4

We also prepare the configuration for creating the environment with conda in *env.yml*:
```bash
conda env create -f env.yml
```

## Experiment
### Usage
To train and evaluate on DBLP3:
```
python train.py --config ./config/DBLP3.yaml
```
To train and evaluate on Reddit:
```
python train.py --config ./config/REDDIT.yaml
```
