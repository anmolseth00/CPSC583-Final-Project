# CPSC 583 Final Project

This repository contains the code and data for a project focused on predicting RNA degradation rates at the nucleotide level using graph-based deep learning methods. Our work builds on datasets and concepts from the [Stanford COVID-19 Vaccine competition](https://www.kaggle.com/c/stanford-covid-vaccine), [Eterna’s OpenVaccine challenges](https://eternagame.org/challenges/10845741), and [Combinatorial optimization of mRNA structure, stability, and translation for RNA-based therapeutics](https://www.nature.com/articles/s41467-022-28776-w). By integrating modern graph neural network architectures, transformer-style attention, pretraining strategies, and extensive hyperparameter tuning, we aim to advance the state of the art in RNA stability prediction by directly modeling sequence and structure.

## Repository Structure

- **raw_data/**:  
  Contains raw input files, including the Stanford COVID-19 Vaccine dataset and other experiment data.  
  See the `raw_data/README.md` for detailed descriptions of data sources and experimental conditions.

- **architecture_experiments/**:  
  Implements and compares different GNN-based architectures, including baseline GCN, advanced GAT, hybrid GNN+XGBoost, and a pretrain-then-supervised transformer GNN approach.  
  See `architecture_experiments/README.md` for a summary of each model and the associated results.

- **hyperparameter_optimization/**:  
  Contains scripts and logs for performing hyperparameter tuning using Optuna.  
  See `hyperparameter_optimization/README.md` for instructions and details on the optimization process.

- **results/**:  
  Stores training curves, best model checkpoints, and logs from various runs.

- **tmp/**:  
  A scratch directory used for temporary files during experiments.

## Requirements

- **DegScore**:  
  We rely on [DegScore](https://github.com/eternagame/DegScore) to provide a baseline and additional analytical tools for evaluating RNA degradation predictions.  
  Please follow the [DegScore installation instructions](https://github.com/eternagame/DegScore/blob/master/README.md) before running our code.

- **Arnie**:  
  [Arnie](https://github.com/DasLab/arnie) could be required for RNA secondary structure calculations and related preprocessing steps.  
  Install Arnie following the instructions in its repository.

## Running the Code

1. **Set up the environment**:  
   You can replicate the development environment used for this project by creating a new conda environment from the provided .yml file. Simply run:
```bash
conda env create -f cpsc583_final-project_environment.yml
conda activate cpsc583_environment
```

3. **Prepare data**:  
   Download and place all required data files into the `raw_data/` directory as described in `raw_data/README.md`.  
   Make sure the `train.json`, `test.json`, and related files from the Stanford COVID-19 Vaccine dataset are placed correctly.

4. **Model training and experiments**:  
   - For architecture comparisons, go to `architecture_experiments/` and run scripts such as `python baseline_gnn.py`.  
   - For hyperparameter optimization, navigate to `hyperparameter_optimization/` and run `python hyperparameter_optimization.py`.  
   - Check `results/` for output logs, best model checkpoints, and performance plots.

## Contributions

Our approach introduces:
- Pretraining with masked-language-modeling and link prediction on RNA graph representations.
- Advanced GNN architectures (e.g., TransformerConv layers and multi-head attention) applied directly to RNA secondary structures.
- A two-stage hybrid approach combining deep graph embeddings with classical ML methods (XGBoost).

For further details on experimental results and model comparisons, refer to the main project report and the supplementary READMEs in each subdirectory.