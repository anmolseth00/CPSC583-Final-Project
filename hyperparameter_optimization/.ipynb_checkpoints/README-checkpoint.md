# Hyperparameter Optimization

This directory focuses on hyperparameter tuning using Optuna to improve the performance of our GNN-based RNA degradation prediction model. By systematically exploring the search space of key parameters, we aim to identify an optimal set of hyperparameters that yields better validation performance and training stability.

## Approach

We use Optuna’s TPESampler and MedianPruner to efficiently navigate the hyperparameter search space. Candidates include:

- **Hidden dimensions (e.g., 64, 128, 256)**  
- **Number of layers (e.g., 2–4)**  
- **Number of heads (e.g., 2, 4, 8)**  
- **Dropout rates (0.0–0.5)**  
- **Learning rate (log-scaled range)**

We run multiple trials, each evaluating the resulting model’s validation loss. Trials that underperform early are pruned to save computation, focusing efforts on more promising configurations.

## Files

- `hyperparameter_optimization.py`: The main Optuna-driven script that:
  - Loads the RNA dataset.
  - Splits into train/validation sets.
  - Trains and evaluates candidate models for each trial.
  - Logs intermediate and final results.

- `results_hpo/`: Directory created automatically to store:
  - `run.log`: Logging information from the optimization run.
  - `trials.csv`: A record of all trials, their parameters, and scores.
  - `best_params.json`: The best-found hyperparameters.
  - Plots (if generated) like `optimization_history.png` and `param_importances.png` to visualize the search process and parameter importance.

## Usage

Run `python hyperparameter_optimization.py` to start the optimization process. After all trials finish, inspect `results_hpo/best_params.json` for the best hyperparameters. Also review `param_importances.png` and `parallel_coordinates.png` for insights into parameter interactions.

## Outcome

The tuned hyperparameters are used to configure the model in the final experiments. This process helps ensure that the chosen GNN configuration is not only theoretically robust but also empirically well-tuned for the specific RNA degradation prediction task.

## Key Results

![Optimization History](results_hpo/optimization_history.png)
![Parallel Coordinates](results_hpo/parallel_coordinates.png)
![Param Importances](results_hpo/param_importances.png)
