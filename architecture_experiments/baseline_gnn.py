#!/usr/bin/env python3
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv

TRAIN_JSON = "raw_data/stanford-covid-vaccine/train.json"
BPP_PATH = "raw_data/stanford-covid-vaccine/bpps"
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3
HIDDEN_DIM = 64
PATIENCE = 5     # Early stopping patience
if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

def one_hot_nucleotide(seq_char):
    mapping = {'A':0,'G':1,'C':2,'U':3}
    vec = np.zeros(4)
    vec[mapping.get(seq_char,0)] = 1
    return vec

def one_hot_loop_type(lc):
    types = ['E','S','H','I','M','B','X']
    vec = np.zeros(len(types))
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

        data = Data(
            x=torch.tensor(node_feats, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(targets, dtype=torch.float)
        )
        data.seq_scored = seq_scored
        return data

class GCNModel(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim=5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.lin = nn.Linear(hid_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.lin(x)
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

def main():
    dataset = RNADataset(TRAIN_JSON, BPP_PATH)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset)-val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)

    model = GCNModel(in_dim=11, hid_dim=HIDDEN_DIM, out_dim=5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss_acc = 0
        steps = 0
        for (batch, seq_scored_tensor) in train_loader:
            batch = batch.to(DEVICE)
            seq_scored_tensor = seq_scored_tensor.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index)
            loss = mask_mse_loss(pred, batch.y, batch.batch, seq_scored_tensor)
            loss.backward()
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
                pred = model(batch.x, batch.edge_index)
                loss = mask_mse_loss(pred, batch.y, batch.batch, seq_scored_tensor)
                val_loss_acc += loss.item()
                val_steps += 1
        val_loss = val_loss_acc/val_steps
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "baseline_gnn_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    main()