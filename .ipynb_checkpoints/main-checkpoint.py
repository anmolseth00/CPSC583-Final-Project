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
from torch.utils.data import Dataset, random_split, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import TransformerConv, LayerNorm
from torch_geometric.utils import negative_sampling
import matplotlib.pyplot as plt
import logging
import ast
from DegScore import DegScore

#######################################
# Configuration
#######################################
TRAIN_JSON = "raw_data/stanford-covid-vaccine/train.json"
TEST_JSON = "raw_data/stanford-covid-vaccine/test.json"
PRIVATE_TEST_LABELS = "raw_data/stanford-covid-vaccine/post_deadline_files/private_test_labels.csv"  # For final evaluation
RESULTS_DIR = "results"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

BATCH_SIZE = 16
PRETRAIN_EPOCHS = 150    # Increase for richer embedding learning
SUPERVISED_EPOCHS = 150  # Longer supervised training
LR = 1e-3
HIDDEN_DIM = 256         # Larger dimension for richer representation
HEADS = 8                # More heads for better attention
LAYERS = 6               # More layers for deeper architecture
DROPOUT = 0.1
WEIGHT_DECAY = 1e-4
PATIENCE = 10
GRAD_CLIP = 5.0
DEVICE = "cpu"

NUCS = ['A','G','C','U']

# Set seeds globally in a function for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
# Datasets
#######################################
class UnlabeledRNADataset(Dataset):
    # Pretraining uses both train and test JSON files
    # We'll load both and combine
    def __init__(self, json_paths, mlm_mask_ratio=0.1, neg_edge_ratio=0.5):
        self.data = []
        if not isinstance(json_paths, list):
            json_paths = [json_paths]
        for jp in json_paths:
            with open(jp, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    self.data.append(item)
        self.mlm_mask_ratio = mlm_mask_ratio
        self.neg_edge_ratio = neg_edge_ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        seq = item['sequence']
        structure = item['structure']
        loop_type = item['predicted_loop_type']
        seq_length = item['seq_length']

        pos_embed = positional_embedding(seq_length)
        node_feats = []
        for i in range(seq_length):
            nt_feat = one_hot_nucleotide(seq[i])
            lt_feat = one_hot_loop_type(loop_type[i])
            node_feat = np.concatenate([nt_feat, lt_feat, pos_embed[i]])
            node_feats.append(node_feat)
        node_feats = np.array(node_feats, dtype=np.float32)

        # Edges: ONLY from structure
        # No linear chain edges per requirements
        pairs = parse_structure(structure)
        edge_index = []
        for (p,q) in pairs:
            edge_index.append([p,q])
            edge_index.append([q,p])
        if len(edge_index) == 0:
            # If no pairs, just empty edges
            edge_index = torch.zeros((2,0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # MLM
        targets_mlm = np.full(seq_length, -1, dtype=np.int64)
        num_mask = max(1, int(seq_length*self.mlm_mask_ratio))
        mask_indices = random.sample(range(seq_length), k=num_mask)
        for mi in mask_indices:
            nt = seq[mi]
            targets_mlm[mi] = NUCS.index(nt)

        # Link Prediction
        original_edge_index = edge_index
        num_edges = original_edge_index.size(1) if original_edge_index.size(1)>0 else 0
        if num_edges > 0:
            neg_samples = int(num_edges * self.neg_edge_ratio)
            neg_edge_index = negative_sampling(original_edge_index, num_nodes=seq_length, num_neg_samples=neg_samples)
            pos_y = torch.ones(num_edges, dtype=torch.long)
            neg_y = torch.zeros(neg_edge_index.size(1), dtype=torch.long)
            lp_edge_index = torch.cat([original_edge_index, neg_edge_index], dim=1)
            lp_y = torch.cat([pos_y, neg_y], dim=0)
        else:
            # No edges: just produce dummy lp data
            lp_edge_index = torch.zeros((2,0), dtype=torch.long)
            lp_y = torch.zeros(0, dtype=torch.long)

        data = Data(
            x=torch.tensor(node_feats, dtype=torch.float),
            edge_index=original_edge_index,
            y=torch.tensor(targets_mlm, dtype=torch.long)
        )
        data.seq = seq
        data.lp_edge_index = lp_edge_index
        data.lp_y = lp_y
        return data

class LabeledRNADataset(Dataset):
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

        # Edges from structure only
        pairs = parse_structure(structure)
        edge_index = []
        for (p,q) in pairs:
            edge_index.append([p,q])
            edge_index.append([q,p])
        if len(edge_index) == 0:
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

#######################################
# Model
#######################################
class ResidualTransformerLayer(nn.Module):
    def __init__(self, hid_dim, heads=8, dropout=0.1):
        super().__init__()
        self.conv = TransformerConv(hid_dim, hid_dim, heads=heads, beta=True, dropout=dropout)
        self.norm = LayerNorm(hid_dim*heads)
        self.proj = nn.Linear(hid_dim*heads, hid_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        h = self.conv(x, edge_index, edge_attr=edge_attr)
        h = self.norm(h)
        h = self.proj(h)
        h = self.dropout(h)
        x = h + x
        x = self.act(x)
        return x

class FoundationGNN(nn.Module):
    """A large model for pretraining & supervised fine-tuning"""
    def __init__(self, in_dim=15, hid_dim=256, out_dim=4, layers=6, heads=8, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.input_lin = nn.Linear(in_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([ResidualTransformerLayer(hid_dim, heads, dropout) for _ in range(layers)])

        # MLM head:
        self.mlm_head = nn.Linear(hid_dim, 4)

        # Link Pred head:
        self.lp_mlp = nn.Sequential(
            nn.Linear(hid_dim*2, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, 2)
        )

    def forward_gnn(self, x, edge_index, edge_attr=None):
        x = self.input_lin(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x

    def forward_mlm(self, x, edge_index):
        return self.mlm_head(x)

    def forward_lp(self, x, lp_edge_index):
        if lp_edge_index.size(1) == 0:
            # No edges, return empty
            return torch.zeros((0,2), device=x.device)
        src = lp_edge_index[0]
        dst = lp_edge_index[1]
        src_emb = x[src]
        dst_emb = x[dst]
        pair_emb = torch.cat([src_emb,dst_emb], dim=-1)
        out = self.lp_mlp(pair_emb)
        return out

def mlm_loss(pred, y):
    mask = (y!=-1)
    if not torch.any(mask):
        return torch.tensor(0.0, device=pred.device)
    return nn.CrossEntropyLoss()(pred[mask], y[mask])

def lp_loss(pred, y):
    if pred.size(0) == 0:
        return torch.tensor(0.0, device=y.device)
    return nn.CrossEntropyLoss()(pred, y)

def pretrain_epoch(model, loader, optimizer=None):
    is_train = (optimizer is not None)
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    steps = 0
    with torch.set_grad_enabled(is_train):
        for data in loader:
            data = data.to(DEVICE)
            x = model.forward_gnn(data.x, data.edge_index)
            pred_mlm = model.forward_mlm(x, data.edge_index)
            loss_m = mlm_loss(pred_mlm, data.y)

            pred_lp = model.forward_lp(x, data.lp_edge_index)
            loss_l = lp_loss(pred_lp, data.lp_y)

            loss = loss_m + loss_l
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            total_loss += loss.item()
            steps += 1
    return total_loss/steps if steps>0 else 0.0

def custom_collate_pretrain(data_list):
    return Batch.from_data_list(data_list)

def custom_collate_labeled(data_list):
    seq_scored_list = [d.seq_scored for d in data_list]
    batch = Batch.from_data_list(data_list)
    seq_scored_tensor = torch.tensor(seq_scored_list, dtype=torch.long)
    return batch, seq_scored_tensor

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

class FineTuneModel(nn.Module):
    def __init__(self, in_dim=15, hid_dim=256, out_dim=5, layers=6, heads=8, dropout=0.1, pretrained_model=None):
        super().__init__()
        self.input_lin = nn.Linear(in_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([ResidualTransformerLayer(hid_dim, heads, dropout) for _ in range(layers)])
        self.out_lin = nn.Linear(hid_dim, out_dim)

        if pretrained_model is not None:
            self.input_lin.load_state_dict(pretrained_model.input_lin.state_dict())
            for i in range(layers):
                self.layers[i].load_state_dict(pretrained_model.layers[i].state_dict())

    def forward_gnn(self, x, edge_index):
        x = self.input_lin(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

    def forward(self, x, edge_index):
        x = self.forward_gnn(x, edge_index)
        return self.out_lin(x)

def supervised_epoch(model, loader, optimizer=None):
    is_train = (optimizer is not None)
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    steps = 0
    with torch.set_grad_enabled(is_train):
        for (batch, seq_scored_tensor) in loader:
            batch = batch.to(DEVICE)
            seq_scored_tensor = seq_scored_tensor.to(DEVICE)
            pred = model(batch.x, batch.edge_index)
            loss = mask_mse_loss(pred, batch.y, batch.batch, seq_scored_tensor)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            total_loss += loss.item()
            steps += 1
    return total_loss/steps if steps>0 else 0.0

def run_experiment(run_id, seeds=[0,1,2]):
    # Setup logging
    log_path = os.path.join(RESULTS_DIR, f"run_{run_id}.log")
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info("Starting experiment run_id=%s", run_id)

    # Load datasets
    # Pretraining uses both TRAIN_JSON and TEST_JSON
    pretrain_ds = UnlabeledRNADataset([TRAIN_JSON, TEST_JSON], mlm_mask_ratio=0.1, neg_edge_ratio=0.5)
    # Just split a small val set
    val_size = int(0.1*len(pretrain_ds))
    train_size = len(pretrain_ds)-val_size
    pretrain_train_ds, pretrain_val_ds = random_split(pretrain_ds, [train_size, val_size])

    pretrain_train_loader = DataLoader(
        pretrain_train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_pretrain
    )
    pretrain_val_loader = DataLoader(
        pretrain_val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_pretrain
    )

    # Supervised dataset
    sup_ds = LabeledRNADataset(TRAIN_JSON)
    val_size = int(0.1*len(sup_ds))
    train_size = len(sup_ds)-val_size
    finetune_train_ds, finetune_val_ds = random_split(sup_ds,[train_size,val_size])
    finetune_train_loader = DataLoader(finetune_train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_labeled)
    finetune_val_loader = DataLoader(finetune_val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_labeled)

    # Runs
    final_results = []
    for seed in seeds:
        set_seed(seed)
        logging.info("Running seed %d", seed)

        # Pretrain
        foundation_model = FoundationGNN(in_dim=15, hid_dim=HIDDEN_DIM, out_dim=4, layers=LAYERS, heads=HEADS, dropout=DROPOUT).to(DEVICE)
        optimizer = optim.Adam(foundation_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(PRETRAIN_EPOCHS):
            train_loss = pretrain_epoch(foundation_model, pretrain_train_loader, optimizer)
            val_loss = pretrain_epoch(foundation_model, pretrain_val_loader, optimizer=None)
            logging.info("[Pretraining] Seed %d Epoch %d/%d: Train Loss %.4f, Val Loss %.4f", seed, epoch+1, PRETRAIN_EPOCHS, train_loss, val_loss)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(foundation_model.state_dict(), os.path.join(RESULTS_DIR, f"foundation_seed{seed}.pt"))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logging.info("Early stopping pretraining seed %d", seed)
                    break

        # Load best pretrain
        foundation_model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, f"foundation_seed{seed}.pt"), map_location=DEVICE, weights_only=True))

        # Supervised
        finetune_model = FineTuneModel(in_dim=15, hid_dim=HIDDEN_DIM, out_dim=5, layers=LAYERS, heads=HEADS, dropout=DROPOUT, pretrained_model=foundation_model).to(DEVICE)
        optimizer = optim.Adam(finetune_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

        best_val_loss = float('inf')
        patience_counter = 0
        train_loss_curve = []
        val_loss_curve = []
        for epoch in range(SUPERVISED_EPOCHS):
            t_loss = supervised_epoch(finetune_model, finetune_train_loader, optimizer)
            v_loss = supervised_epoch(finetune_model, finetune_val_loader, optimizer=None)
            train_loss_curve.append(t_loss)
            val_loss_curve.append(v_loss)
            logging.info("[Supervised] Seed %d Epoch %d/%d: Train Loss %.4f, Val Loss %.4f", seed, epoch+1, SUPERVISED_EPOCHS, t_loss, v_loss)
            scheduler.step(v_loss)
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save(finetune_model.state_dict(), os.path.join(RESULTS_DIR, f"supervised_seed{seed}.pt"))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logging.info("Early stopping supervised seed %d", seed)
                    break

        # Plot training curves
        plt.figure()
        plt.plot(train_loss_curve, label='Train')
        plt.plot(val_loss_curve, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f"Training Curve (Seed={seed})")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, f"training_curve_seed{seed}.png"))
        plt.close()

        # Load best supervised
        finetune_model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, f"supervised_seed{seed}.pt"), map_location=DEVICE, weights_only=True))
        finetune_model.eval()

        # Evaluate on private_test_labels.csv
        dfp = pd.read_csv(PRIVATE_TEST_LABELS)
        mae_reactivity = []
        mae_Mg_pH10 = []
        mae_Mg_50C = []

        degscore_mae_Mg_pH10 = []
        degscore_mae_Mg_50C = []

        for idx, row in dfp.iterrows():
            sequence = row['sequence']
            structure = row['structure']
            loop_type = row['predicted_loop_type']

            reactivity_true = ast.literal_eval(row['reactivity'])
            deg_Mg_pH10_true = ast.literal_eval(row['deg_Mg_pH10'])
            deg_Mg_50C_true = ast.literal_eval(row['deg_Mg_50C'])

            seq_length = len(sequence)
            pos_embed = positional_embedding(seq_length)

            node_feats = []
            for i in range(seq_length):
                nt_feat = one_hot_nucleotide(sequence[i])
                lt_feat = one_hot_loop_type(loop_type[i])
                node_feat = np.concatenate([nt_feat, lt_feat, pos_embed[i]])
                node_feats.append(node_feat)
            node_feats = np.array(node_feats, dtype=np.float32)

            pairs = parse_structure(structure)
            edge_index = []
            for (p,q) in pairs:
                edge_index.append([p,q])
                edge_index.append([q,p])
            if len(edge_index)==0:
                edge_index = torch.zeros((2,0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            data = Data(
                x=torch.tensor(node_feats, dtype=torch.float),
                edge_index=edge_index
            )

            with torch.no_grad():
                pred = finetune_model(data.x, data.edge_index).cpu().numpy()
            scored_length = len(reactivity_true)
            pred_reactivity = pred[:scored_length,0]
            pred_deg_Mg_pH10 = pred[:scored_length,1]
            pred_deg_Mg_50C = pred[:scored_length,2]

            true_sum_reactivity = np.nansum(reactivity_true)
            true_sum_Mg_pH10 = np.nansum(deg_Mg_pH10_true)
            true_sum_Mg_50C = np.nansum(deg_Mg_50C_true)

            pred_sum_reactivity = np.nansum(pred_reactivity)
            pred_sum_Mg_pH10 = np.nansum(pred_deg_Mg_pH10)
            pred_sum_Mg_50C = np.nansum(pred_deg_Mg_50C)

            mae_reactivity.append(abs(pred_sum_reactivity - true_sum_reactivity))
            mae_Mg_pH10.append(abs(pred_sum_Mg_pH10 - true_sum_Mg_pH10))
            mae_Mg_50C.append(abs(pred_sum_Mg_50C - true_sum_Mg_50C))

            # DegScore baseline
            mdl = DegScore(sequence, structure=structure)
            degscore = mdl.degscore

            degscore_mae_Mg_pH10.append(abs(degscore - true_sum_Mg_pH10))
            degscore_mae_Mg_50C.append(abs(degscore - true_sum_Mg_50C))

        final_mae_react = np.mean(mae_reactivity)
        final_mae_Mg_pH10 = np.mean(mae_Mg_pH10)
        final_mae_Mg_50C = np.mean(mae_Mg_50C)

        degscore_final_mae_Mg_pH10 = np.mean(degscore_mae_Mg_pH10)
        degscore_final_mae_Mg_50C = np.mean(degscore_mae_Mg_50C)

        result = {
            'seed': seed,
            'model_mae_reactivity': final_mae_react,
            'model_mae_Mg_pH10': final_mae_Mg_pH10,
            'model_mae_Mg_50C': final_mae_Mg_50C,
            'degscore_mae_Mg_pH10': degscore_final_mae_Mg_pH10,
            'degscore_mae_Mg_50C': degscore_final_mae_Mg_50C
        }
        final_results.append(result)

    # Compute mean/std
    model_react_all = [r['model_mae_reactivity'] for r in final_results]
    model_Mg_pH10_all = [r['model_mae_Mg_pH10'] for r in final_results]
    model_Mg_50C_all = [r['model_mae_Mg_50C'] for r in final_results]

    degscore_Mg_pH10_all = [r['degscore_mae_Mg_pH10'] for r in final_results]
    degscore_Mg_50C_all = [r['degscore_mae_Mg_50C'] for r in final_results]

    summary = {
        'model_mae_reactivity_mean': np.mean(model_react_all),
        'model_mae_reactivity_std': np.std(model_react_all),
        'model_mae_Mg_pH10_mean': np.mean(model_Mg_pH10_all),
        'model_mae_Mg_pH10_std': np.std(model_Mg_pH10_all),
        'model_mae_Mg_50C_mean': np.mean(model_Mg_50C_all),
        'model_mae_Mg_50C_std': np.std(model_Mg_50C_all),
        'degscore_mae_Mg_pH10_mean': np.mean(degscore_Mg_pH10_all),
        'degscore_mae_Mg_pH10_std': np.std(degscore_Mg_pH10_all),
        'degscore_mae_Mg_50C_mean': np.mean(degscore_Mg_50C_all),
        'degscore_mae_Mg_50C_std': np.std(degscore_Mg_50C_all)
    }

    df_res = pd.DataFrame(final_results)
    df_res.to_csv(os.path.join(RESULTS_DIR, f"final_results_{run_id}.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, f"summary_{run_id}.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    # Simple comparison plot
    labels = ['Mg_pH10', 'Mg_50C']
    model_means = [summary['model_mae_Mg_pH10_mean'], summary['model_mae_Mg_50C_mean']]
    model_err = [summary['model_mae_Mg_pH10_std'], summary['model_mae_Mg_50C_std']]
    degscore_means = [summary['degscore_mae_Mg_pH10_mean'], summary['degscore_mae_Mg_50C_mean']]
    degscore_err = [summary['degscore_mae_Mg_pH10_std'], summary['degscore_mae_Mg_50C_std']]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, model_means, width, yerr=model_err, label='Model', capsize=5)
    plt.bar(x + width/2, degscore_means, width, yerr=degscore_err, label='DegScore', capsize=5)
    plt.xticks(x, labels)
    plt.ylabel('MAE')
    plt.title('Model vs DegScore on Mg_pH10 and Mg_50C (mean ± std over 3 runs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"comparison_{run_id}.png"))
    plt.close()

    logging.info("Experiment complete. Results saved.")

def main():
    run_experiment(run_id="final_run", seeds=[0,1,2])

if __name__ == "__main__":
    main()