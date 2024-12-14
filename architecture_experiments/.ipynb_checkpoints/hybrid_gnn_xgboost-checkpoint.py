#!/usr/bin/env python3
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, TransformerConv, LayerNorm
import xgboost as xgb

#######################################
# Configuration
#######################################
TRAIN_JSON = "raw_data/stanford-covid-vaccine/train.json"
BPP_PATH = "raw_data/stanford-covid-vaccine/bpps"
BATCH_SIZE = 16
MAX_EPOCHS_GNN = 100
EPOCHS_XGB = 100
LR = 1e-3
HIDDEN_DIM = 64
HEADS = 4
PATIENCE = 10
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0
DEVICE = "cpu"

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

def gc_content(sequence):
    gc_count = sequence.count('G')+sequence.count('C')
    return gc_count/len(sequence)

def compute_bpp_features(bpp):
    mean_bpp = bpp.mean()
    max_bpp = bpp.max()
    nonzero_frac = (bpp>0).sum()/(bpp.size)
    return mean_bpp, max_bpp, nonzero_frac

def compute_structure_features(structure):
    pairs = parse_structure(structure)
    num_pairs = len(pairs)
    return num_pairs

def positional_embedding(seq_length):
    # Sinusoidal positional embeddings: [seq_length,4]
    pos_embed = []
    for i in range(seq_length):
        pos_embed.append([math.sin(i/1000), math.cos(i/1000), math.sin(i/100), math.cos(i/100)])
    return np.array(pos_embed, dtype=np.float32)

#######################################
# Dataset
#######################################
class RNADataset(Dataset):
    def __init__(self, json_path, bpp_path):
        self.data = []
        with open(json_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
        self.bpp_path = bpp_path

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

        # Node features: 4(nt)+7(lt)=11
        node_feats = []
        for i in range(seq_length):
            nt_feat = one_hot_nucleotide(seq[i])
            lt_feat = one_hot_loop_type(loop_type[i])
            node_feat = np.concatenate([nt_feat, lt_feat])
            node_feats.append(node_feat)
        node_feats = np.array(node_feats, dtype=np.float32)

        edge_index = []
        for i in range(seq_length-1):
            edge_index.append([i,i+1])
            edge_index.append([i+1,i])
        pairs = parse_structure(structure)
        for (p,q) in pairs:
            edge_index.append([p,q])
            edge_index.append([q,p])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        targets = np.full((seq_length,5), np.nan, dtype=np.float32)
        targets[:seq_scored,0] = reactivity
        targets[:seq_scored,1] = deg_Mg_pH10
        targets[:seq_scored,2] = deg_Mg_50C
        targets[:seq_scored,3] = deg_pH10
        targets[:seq_scored,4] = deg_50C

        bpp_file = os.path.join(self.bpp_path, item['id'] + '.npy')
        if os.path.exists(bpp_file):
            bpp = np.load(bpp_file)
        else:
            bpp = np.zeros((seq_length, seq_length), dtype=np.float32)

        data = Data(
            x=torch.tensor(node_feats, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(targets, dtype=torch.float)
        )
        data.seq_scored = seq_scored
        data.sequence = seq
        data.structure = structure
        data.bpp = bpp
        data.id = item['id']
        return data

#######################################
# Enhanced Transformer-based GNN with residual etc.
#######################################
class ResidualTransformerLayer(nn.Module):
    def __init__(self, hid_dim, heads=4, dropout=0.2):
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

class HybridTransformerGNN(nn.Module):
    def __init__(self, in_dim=11, hid_dim=64, out_dim=5, heads=4, layers=3, dropout=0.2):
        super().__init__()
        self.input_lin = nn.Linear(in_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([ResidualTransformerLayer(hid_dim, heads, dropout) for _ in range(layers)])
        self.out_lin = nn.Linear(hid_dim, out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.input_lin(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        x = self.out_lin(x)
        return x

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

def custom_collate(data_list):
    seq_scored_list = [d.seq_scored for d in data_list]
    batch = Batch.from_data_list(data_list)
    seq_scored_tensor = torch.tensor(seq_scored_list, dtype=torch.long)
    return batch, seq_scored_tensor

#######################################
# Training GNN first
#######################################
def train_gnn(train_dataset, val_dataset):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)

    model = HybridTransformerGNN(in_dim=11, hid_dim=HIDDEN_DIM, out_dim=5, heads=HEADS, layers=3, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(MAX_EPOCHS_GNN):
        model.train()
        train_loss_acc = 0
        steps = 0
        for (batch, seq_scored_tensor) in train_loader:
            batch = batch.to(DEVICE)
            seq_scored_tensor = seq_scored_tensor.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr if hasattr(batch,'edge_attr') else None)
            loss = mask_mse_loss(pred, batch.y, batch.batch, seq_scored_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss_acc += loss.item()
            steps += 1
        train_loss = train_loss_acc/steps

        model.eval()
        val_loss_acc = 0
        val_steps = 0
        with torch.no_grad():
            for (batch, seq_scored_tensor) in val_loader:
                batch = batch.to(DEVICE)
                seq_scored_tensor = seq_scored_tensor.to(DEVICE)
                pred = model(batch.x, batch.edge_index, batch.edge_attr if hasattr(batch,'edge_attr') else None)
                loss = mask_mse_loss(pred, batch.y, batch.batch, seq_scored_tensor)
                val_loss_acc += loss.item()
                val_steps += 1
        val_loss = val_loss_acc/val_steps
        print(f"[Hybrid GNN+XGB GNN Stage] Epoch {epoch+1}/{MAX_EPOCHS_GNN}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "hybrid_gnn_xgboost.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered in GNN training.")
                break

    print("Best GNN val loss:", best_val_loss)
    model_state = torch.load("hybrid_gnn_xgboost.pt", map_location=DEVICE)
    model.load_state_dict(model_state)
    return model

#######################################
# Extract features for XGBoost
#######################################
def get_embeddings(model, x, edge_index, edge_attr):
    # replicate forward logic to get embeddings before out_lin
    with torch.no_grad():
        x = model.input_lin(x)
        x = model.dropout(x)
        residual = x
        for layer in model.layers:
            h = layer.conv(x, edge_index, edge_attr=edge_attr)
            h = layer.norm(h)
            h = layer.proj(h)
            h = model.dropout(h)
            x = h + residual
            x = layer.act(x)
            residual = x
        # x is [N,hid_dim]
        return x

def extract_features(model, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
    X_features = []
    Y_targets = []
    IDs = []

    model.eval()
    with torch.no_grad():
        for (batch, seq_scored_tensor) in loader:
            batch = batch.to(DEVICE)
            emb = get_embeddings(model, batch.x, batch.edge_index, batch.edge_attr if hasattr(batch,'edge_attr') else None)
            graph_emb = global_mean_pool(emb, batch.batch)

            seq = batch.sequence[0]
            structure = batch.structure[0]
            bpp = batch.bpp[0]  # numpy array

            gc = gc_content(seq)
            mean_bpp, max_bpp, nonzero_bpp_frac = compute_bpp_features(bpp)
            num_pairs = compute_structure_features(structure)

            graph_emb = graph_emb.cpu().numpy().ravel()
            tab_feats = np.array([gc, mean_bpp, max_bpp, nonzero_bpp_frac, num_pairs], dtype=np.float32)
            combined = np.concatenate([graph_emb, tab_feats])

            y = batch.y
            s = seq_scored_tensor[0].item()
            valid = ~torch.isnan(y[:s,:]) 
            y_g = y[:s,:]  # shape [s,5]
            Y_mean = []
            for t_i in range(5):
                t_mask = valid[:,t_i]
                if t_mask.any():
                    Y_mean.append(y_g[t_mask,t_i].mean().item())
                else:
                    Y_mean.append(np.nan)
            Y_mean = np.array(Y_mean, dtype=np.float32)

            X_features.append(combined)
            Y_targets.append(Y_mean)
            IDs.append(batch.id[0])

    X_features = np.array(X_features, dtype=np.float32)
    Y_targets = np.array(Y_targets, dtype=np.float32)
    return X_features, Y_targets, IDs

#######################################
# Train XGBoost
#######################################
def train_xgboost(X_train, Y_train, X_val, Y_val):
    models = []
    eval_results = {}
    target_names = ['reactivity','deg_Mg_pH10','deg_Mg_50C','deg_pH10','deg_50C']

    for i, target_name in enumerate(target_names):
        mask_train = ~np.isnan(Y_train[:,i])
        mask_val = ~np.isnan(Y_val[:,i])

        dtrain = xgb.DMatrix(X_train[mask_train], label=Y_train[mask_train,i])
        dval = xgb.DMatrix(X_val[mask_val], label=Y_val[mask_val,i])

        params = {
            'objective':'reg:squarederror',
            'eta':0.05,
            'max_depth':6,
            'subsample':0.8,
            'colsample_bynode':0.8,
            'verbosity':1,
            'eval_metric':'rmse'
        }
        evals = [(dtrain,'train'),(dval,'eval')]
        bst = xgb.train(params, dtrain, num_boost_round=EPOCHS_XGB, evals=evals, early_stopping_rounds=10, verbose_eval=False)
        models.append(bst)
        pred_val = bst.predict(dval)
        mse = ((pred_val - Y_val[mask_val,i])**2).mean()
        eval_results[target_name] = mse

    print("XGBoost Validation MSE per target:", eval_results)
    return models, eval_results

#######################################
# Main
#######################################
def main():
    full_dataset = RNADataset(TRAIN_JSON, BPP_PATH)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset)-val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Train GNN
    gnn_model = train_gnn(train_dataset, val_dataset)

    # Extract features for XGBoost
    X_train, Y_train, ID_tr = extract_features(gnn_model, train_dataset)
    X_val, Y_val, ID_val = extract_features(gnn_model, val_dataset)

    # Train XGBoost
    xgb_models, xgb_results = train_xgboost(X_train, Y_train, X_val, Y_val)

    print("Done. Final results:")
    print("Best GNN validation loss from training step:", min(xgb_results.values()), "(Compare with other approaches!)")

if __name__ == "__main__":
    main()