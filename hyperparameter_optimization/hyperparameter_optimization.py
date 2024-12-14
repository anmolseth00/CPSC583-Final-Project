#!/usr/bin/env python3
import os
import json
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import optuna
import matplotlib.pyplot as plt

from tqdm import trange
from torch.utils.data import random_split, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import TransformerConv, LayerNorm
from torch_geometric.utils import negative_sampling

#######################################
# Configuration
#######################################
SEED = 42
TRAIN_JSON = "../raw_data/stanford-covid-vaccine/train.json"
RESULTS_DIR = "results_hpo"
N_TRIALS = 50
EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 16
LR_BASE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0
DEVICE = "cpu"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

logging.basicConfig(filename=os.path.join(RESULTS_DIR, 'run.log'),
                    level=logging.INFO, format='%(asctime)s %(message)s')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)

NUCS = ['A','G','C','U']

#######################################
# Helper functions
#######################################
def one_hot_nucleotide(seq_char):
    mapping = {'A':0,'G':1,'C':2,'U':3}
    vec = np.zeros(4)
    vec[mapping.get(seq_char,0)] = 1
    return vec

def one_hot_loop_type(lc):
    types = ['E','S','H','I','M','B','X']
    vec = np.zeros(len(types))
    if lc in types:
        vec[types.index(lc)] = 1
    return vec

def parse_structure(structure):
    stack = []
    pairs = []
    for i, ch in enumerate(structure):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            j = stack.pop()
            pairs.append((i,j))
    return pairs

def positional_embedding(seq_length):
    pos_embed = []
    for i in range(seq_length):
        pos_embed.append([math.sin(i/1000), math.cos(i/1000), math.sin(i/100), math.cos(i/100)])
    return np.array(pos_embed, dtype=np.float32)

#######################################
# Dataset
#######################################
class LabeledRNADataset(torch.utils.data.Dataset):
    def __init__(self, json_path):
        self.data = []
        with open(json_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        seq = item['sequence']
        structure = item['structure']
        loop_type = item['predicted_loop_type']
        seq_length = item['seq_length']
        seq_scored = item['seq_scored']

        reactivity = np.array(item['reactivity'])
        deg_Mg_pH10 = np.array(item['deg_Mg_pH10'])
        deg_Mg_50C = np.array(item['deg_Mg_50C'])
        deg_pH10 = np.array(item['deg_pH10'])
        deg_50C = np.array(item['deg_50C'])

        pos_embed = positional_embedding(seq_length)
        node_feats = []
        for i in range(seq_length):
            nt_feat = one_hot_nucleotide(seq[i])
            lt_feat = one_hot_loop_type(loop_type[i])
            node_feat = np.concatenate([nt_feat, lt_feat, pos_embed[i]])
            node_feats.append(node_feat)
        node_feats = np.array(node_feats, dtype=np.float32)

        pairs = parse_structure(structure)
        edge_index = []
        # Only edges from structure pairs
        for (p,q) in pairs:
            edge_index.append([p,q])
            edge_index.append([q,p])
        if len(edge_index)==0:
            edge_index = torch.zeros((2,0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        targets = np.full((seq_length,5), np.nan, dtype=np.float32)
        targets[:seq_scored,0] = reactivity
        targets[:seq_scored,1] = deg_Mg_pH10
        targets[:seq_scored,2] = deg_Mg_50C
        targets[:seq_scored,3] = deg_pH10
        targets[:seq_scored,4] = deg_50C

        data = Data(
            x=torch.tensor(node_feats, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(targets, dtype=torch.float)
        )
        data.seq_scored = seq_scored
        return data

def custom_collate_labeled(data_list):
    seq_scored_list = [d.seq_scored for d in data_list]
    batch = Batch.from_data_list(data_list)
    seq_scored_tensor = torch.tensor(seq_scored_list, dtype=torch.long)
    return batch, seq_scored_tensor

#######################################
# Model
#######################################
class ResidualTransformerLayer(nn.Module):
    def __init__(self, hid_dim, heads=4, dropout=0.1):
        super().__init__()
        self.conv = TransformerConv(hid_dim, hid_dim, heads=heads, beta=True, dropout=dropout)
        self.norm = LayerNorm(hid_dim*heads)
        self.proj = nn.Linear(hid_dim*heads, hid_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = self.norm(h)
        h = self.proj(h)
        h = self.dropout(h)
        x = h + x
        x = self.act(x)
        return x

class GNNModel(nn.Module):
    def __init__(self, in_dim=15, out_dim=5, hid_dim=128, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.input_lin = nn.Linear(in_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([ResidualTransformerLayer(hid_dim, heads, dropout) for _ in range(layers)])
        self.out_lin = nn.Linear(hid_dim, out_dim)

    def forward_gnn(self, x, edge_index):
        x = self.input_lin(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

    def forward(self, x, edge_index):
        x = self.forward_gnn(x, edge_index)
        return self.out_lin(x)

#######################################
# Training & Evaluation
#######################################
def mask_mse_loss(pred, y, batch_graph, seq_scored_list):
    device = pred.device
    loss_sum = 0.0
    count = 0
    num_graphs = seq_scored_list.shape[0]

    for g in range(num_graphs):
        mask_nodes = (batch_graph == g)
        g_pred = pred[mask_nodes]
        g_y = y[mask_nodes]
        g_seq_scored = seq_scored_list[g].item()
        n_g = g_pred.size(0)
        mask = torch.arange(n_g,device=device) < g_seq_scored
        mask = mask.unsqueeze(-1).expand_as(g_pred)
        mask_f = mask & (~torch.isnan(g_y))
        diff = (g_pred[mask_f]-g_y[mask_f])**2
        loss_sum += diff.sum()
        count += diff.numel()

    if count == 0:
        return torch.tensor(0.0, device=device)
    return loss_sum/count

def train_epoch(model, loader, optimizer, grad_clip):
    model.train()
    total_loss = 0.0
    steps = 0
    for (batch, seq_scored_tensor) in loader:
        batch = batch.to(DEVICE)
        seq_scored_tensor = seq_scored_tensor.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index)
        loss = mask_mse_loss(pred, batch.y, batch.batch, seq_scored_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    return total_loss/steps if steps>0 else 0.0

def eval_epoch(model, loader):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for (batch, seq_scored_tensor) in loader:
            batch = batch.to(DEVICE)
            seq_scored_tensor = seq_scored_tensor.to(DEVICE)
            pred = model(batch.x, batch.edge_index)
            loss = mask_mse_loss(pred, batch.y, batch.batch, seq_scored_tensor)
            total_loss += loss.item()
            steps += 1
    return total_loss/steps if steps>0 else 0.0

#######################################
# Objective function for Optuna
#######################################
def objective(trial):
    # Hyperparams to tune
    hid_dim = trial.suggest_categorical('hid_dim', [64,128,256])
    layers = trial.suggest_int('layers', 2,6)
    heads = trial.suggest_categorical('heads', [2,4,8,16])
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    grad_clip = trial.suggest_float('grad_clip', 1.0, 10.0, step=1.0)

    model = GNNModel(in_dim=15, out_dim=5, hid_dim=hid_dim, layers=layers, heads=heads, dropout=dropout).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in trange(EPOCHS, desc=f"Trial {trial.number+1}/{N_TRIALS}"):
        train_loss = train_epoch(model, train_loader, optimizer, grad_clip)
        val_loss = eval_epoch(model, val_loader)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return -best_val_loss  # We want to minimize val_loss, so return negative for maximize direction

#######################################
# Main
#######################################
def main():
    logging.info("Loading dataset...")
    ds = LabeledRNADataset(TRAIN_JSON)
    val_size = int(0.1*len(ds))
    train_size = len(ds)-val_size
    global train_loader, val_loader
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_labeled)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_labeled)

    # Optuna study
    logging.info("Starting hyperparameter optimization...")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS)

    best_trial = study.best_trial
    logging.info(f"Best trial: Value={best_trial.value}, Params={best_trial.params}")

    df = study.trials_dataframe()
    df.to_csv(os.path.join(RESULTS_DIR, 'trials.csv'), index=False)

    with open(os.path.join(RESULTS_DIR, 'best_params.json'), 'w') as f:
        json.dump(best_trial.params, f, indent=2)

    ax1 = optuna.visualization.matplotlib.plot_optimization_history(study)
    ax1.figure.savefig(os.path.join(RESULTS_DIR, 'optimization_history.png'))
    plt.close(ax1.figure)
    
    ax2 = optuna.visualization.matplotlib.plot_param_importances(study)
    ax2.figure.savefig(os.path.join(RESULTS_DIR, 'param_importances.png'))
    plt.close(ax2.figure)
    
    ax3 = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    ax3.figure.savefig(os.path.join(RESULTS_DIR, 'parallel_coordinates.png'))
    plt.close(ax3.figure)

    print("Hyperparameter optimization complete.")
    print(f"Best validation metric (negative val loss): {best_trial.value:.4f}")
    print("Best hyperparameters:")
    for k,v in best_trial.params.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()