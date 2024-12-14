# Architecture Experiments

This directory contains experiments exploring different graph-based neural network architectures and their impact on predicting nucleotide-level RNA degradation. We implement and compare several models to understand how architectural choices influence performance:

## Architectures Explored

1. **Baseline GNN (GCN)**:  
   A straightforward Graph Convolutional Network (GCN) that uses simple message-passing layers. This model serves as a reference point.

2. **Advanced GAT**:  
   Employs Graph Attention Networks (GAT) with multi-head attention. This aims to learn more expressive node embeddings by paying selective attention to specific edges.

3. **Hybrid GNN + XGBoost**:  
   A two-stage pipeline:
   - First, extract graph embeddings from a TransformerConv-based GNN.
   - Second, train an XGBoost regressor on these embeddings to predict degradation rates.
   
   This approach attempts to leverage both deep learning representations and the strong tabular handling capabilities of XGBoost.

4. **Pretrain-Then-Supervised Approach**:  
   Uses a large TransformerConv-based GNN. It is first pretrained on unsupervised tasks (MLM and link prediction) and then fine-tuned on the supervised RNA degradation prediction task. The goal is to learn better initial representations that lead to improved downstream performance.

## Results

We record validation losses for each architecture and report them in a results table (see the main report and `architecture_experiments_results.txt` for details). The key takeaway is that more advanced and pretraining-based approaches generally outperform simpler methods on validation sets, although certain baselines (like DegScore) remain strong competitors on specific stability metrics.

## Files

- `baseline_gnn.py`: Implementation of the GCN baseline model.
- `advanced_gat.py`: GAT-based model and training script.
- `hybrid_gnn_xgboost.py`: The hybrid approach combining graph embeddings with XGBoost.
- `pretrain_then_supervised.py`: The pretrain-then-supervised transformer GNN model training script.
- `architecture_experiments_results.txt`: Contains recorded final validation losses and notes on performance differences.

## Usage

Run each script directly (e.g., `python baseline_gnn.py`) to reproduce the corresponding architecture’s training and evaluation. Check the code comments for configuration details.

## Notes

- These quick experiments helped identify which architectures and training strategies perform best for our dataset.
- For more details on comparisons, see the main project report.