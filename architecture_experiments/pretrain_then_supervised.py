#!/usr/bin/env python3
import os
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
from torch_geometric.nn import TransformerConv, LayerNorm, global_mean_pool
from torch_geometric.utils import negative_sampling

#######################################
# Configuration
#######################################
TRAIN_JSON = "raw_data/stanford-covid-vaccine/train.json"
TEST_JSON = "raw_data/stanford-covid-vaccine/test.json"
BPP_PATH = "raw_data/stanford-covid-vaccine/bpps"

BATCH_SIZE = 16
PRETRAIN_EPOCHS = 50    # More epochs for rich embedding learning
SUPERVISED_EPOCHS = 50  # Increase supervised epochs as well
LR = 1e-3
HIDDEN_DIM = 128         # Larger dimension for a more "foundation-like" model
HEADS = 4
LAYERS = 4               # More layers in the transformer
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4
PATIENCE = 10
GRAD_CLIP = 5.0
DEVICE = "cpu"

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
# Unlabeled dataset with MLM + link pred
#######################################
class UnlabeledRNADataset(Dataset):
    def __init__(self, json_path, bpp_path, mlm_mask_ratio=0.1, neg_edge_ratio=0.5):
        self.data = []
        with open(json_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
        self.bpp_path = bpp_path
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

        # Node features: nt(4) + loop(7) + pos(4) = 15
        pos_embed = positional_embedding(seq_length)
        node_feats = []
        for i in range(seq_length):
            nt_feat = one_hot_nucleotide(seq[i])
            lt_feat = one_hot_loop_type(loop_type[i])
            node_feat = np.concatenate([nt_feat, lt_feat, pos_embed[i]])
            node_feats.append(node_feat)
        node_feats = np.array(node_feats, dtype=np.float32)

        # Edges
        edge_index = []
        for i in range(seq_length-1):
            edge_index.append([i,i+1])
            edge_index.append([i+1,i])
        pairs = parse_structure(structure)
        for (p,q) in pairs:
            edge_index.append([p,q])
            edge_index.append([q,p])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Mask some nucleotides for MLM
        # target for MLM: if masked -> original NT class, else -1
        targets_mlm = np.full(seq_length, -1, dtype=np.int64)
        num_mask = max(1, int(seq_length*self.mlm_mask_ratio))
        mask_indices = random.sample(range(seq_length), k=num_mask)
        for mi in mask_indices:
            nt = seq[mi]
            targets_mlm[mi] = NUCS.index(nt)

        # Link prediction setup:
        # We'll randomly remove some edges and produce a batch of edges that should exist
        # and negative edges that should not. The model will be trained to predict link presence.
        # This is a binary classification: 1 if edge exists, 0 if not.
        # We'll use negative_sampling from PYG to generate negative edges.
        # For simplicity, we won't remove edges from data.x since we need GNN input stable,
        # but we will produce separate data structures for link pred.
        original_edge_index = edge_index.clone()
        # no removal for simplicity, just sample negatives:
        num_edges = original_edge_index.size(1)
        neg_samples = int(num_edges * self.neg_edge_ratio)
        neg_edge_index = negative_sampling(original_edge_index, num_nodes=seq_length, num_neg_samples=neg_samples)
        # Combine pos & neg edges:
        # Pos edges labeled 1, neg edges labeled 0
        pos_y = torch.ones(num_edges, dtype=torch.long)
        neg_y = torch.zeros(neg_edge_index.size(1), dtype=torch.long)
        lp_edge_index = torch.cat([original_edge_index, neg_edge_index], dim=1)
        lp_y = torch.cat([pos_y, neg_y], dim=0)

        data = Data(
            x=torch.tensor(node_feats, dtype=torch.float),
            edge_index=original_edge_index, # used for GNN input
            y=torch.tensor(targets_mlm, dtype=torch.long)
        )
        data.seq = seq
        # Add link pred data as separate attributes
        data.lp_edge_index = lp_edge_index
        data.lp_y = lp_y
        return data

class LabeledRNADataset(Dataset):
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

        pos_embed = positional_embedding(seq_length)
        node_feats = []
        for i in range(seq_length):
            nt_feat = one_hot_nucleotide(seq[i])
            lt_feat = one_hot_loop_type(loop_type[i])
            node_feat = np.concatenate([nt_feat, lt_feat, pos_embed[i]])
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

#######################################
# Model
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

class FoundationGNN(nn.Module):
    """A large model for pretraining & supervised fine-tuning"""
    def __init__(self, in_dim=15, hid_dim=128, out_dim=4, layers=4, heads=4, dropout=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.input_lin = nn.Linear(in_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([ResidualTransformerLayer(hid_dim, heads, dropout) for _ in range(layers)])

        # For MLM task:
        self.mlm_head = nn.Linear(hid_dim, 4)  # predict nucleotide classes

        # For Link Prediction head:
        # We'll get pair embeddings by taking endpoints and passing through an MLP
        self.lp_mlp = nn.Sequential(
            nn.Linear(hid_dim*2, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, 2) # binary classification
        )

    def forward_gnn(self, x, edge_index, edge_attr=None):
        x = self.input_lin(x)
        x = self.dropout(x)
        residual = x
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x

    def forward_mlm(self, x, edge_index):
        # after forward_gnn call
        return self.mlm_head(x)

    def forward_lp(self, x, lp_edge_index):
        # x: node embeddings [N, hid_dim]
        # lp_edge_index: [2, E'] edges to classify
        src = lp_edge_index[0]
        dst = lp_edge_index[1]
        src_emb = x[src]
        dst_emb = x[dst]
        pair_emb = torch.cat([src_emb,dst_emb], dim=-1) # [E', 2*hid_dim]
        out = self.lp_mlp(pair_emb)
        return out

def mlm_loss(pred, y):
    # pred: [N,4], y: [N], y=-1 means not masked
    mask = (y!=-1)
    if not torch.any(mask):
        return torch.tensor(0.0, device=pred.device)
    return nn.CrossEntropyLoss()(pred[mask], y[mask])

def lp_loss(pred, y):
    # pred: [E,2], y: [E], binary classification
    return nn.CrossEntropyLoss()(pred, y)

def pretrain_epoch(model, loader, optimizer=None):
    # If optimizer is None: validation mode
    is_train = (optimizer is not None)
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    steps = 0
    with torch.set_grad_enabled(is_train):
        for data in loader:
            data = data.to(DEVICE)
            # Forward GNN
            x = model.forward_gnn(data.x, data.edge_index) # node embeddings
            # MLM loss
            pred_mlm = model.forward_mlm(x, data.edge_index)
            loss_mlm = mlm_loss(pred_mlm, data.y)

            # Link prediction loss
            pred_lp = model.forward_lp(x, data.lp_edge_index)
            loss_lp = lp_loss(pred_lp, data.lp_y)

            loss = loss_mlm + loss_lp
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

def supervised_epoch(model, loader, optimizer=None):
    is_train = (optimizer is not None)
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
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
    return total_loss/steps if steps > 0 else 0.0

#######################################
# After pretraining, we do supervised
#######################################
class FineTuneModel(nn.Module):
    def __init__(self, in_dim=15, hid_dim=128, out_dim=5, layers=4, heads=4, dropout=0.2, pretrained_model=None):
        super().__init__()
        # same architecture minus heads
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.input_lin = nn.Linear(in_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([ResidualTransformerLayer(hid_dim, heads, dropout) for _ in range(layers)])
        self.out_lin = nn.Linear(hid_dim, out_dim)

        if pretrained_model is not None:
            # load transformer layers and input_lin from pretrained model
            self.input_lin.load_state_dict(pretrained_model.input_lin.state_dict())
            for i in range(layers):
                self.layers[i].load_state_dict(pretrained_model.layers[i].state_dict())
            # do not load mlm or lp heads

    def forward_gnn(self, x, edge_index):
        x = self.input_lin(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

    def forward(self, x, edge_index):
        x = self.forward_gnn(x, edge_index)
        return self.out_lin(x)

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

def supervised_run(model, train_loader, val_loader, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss_acc += loss.item()
            steps += 1
        train_loss = train_loss_acc/steps if steps>0 else 0.0

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
        val_loss = val_loss_acc/val_steps if val_steps>0 else 0.0
        print(f"[Supervised Fine-tune] Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "supervised.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered in supervised training.")
                break

    print("Best Supervised val loss:", best_val_loss)

#######################################
# Main
#######################################
def main():
    # Pretrain on test set with MLM+LinkPred
    test_dataset = UnlabeledRNADataset(TEST_JSON, BPP_PATH, mlm_mask_ratio=0.1, neg_edge_ratio=0.5)
    val_size = int(0.1*len(test_dataset))
    train_size = len(test_dataset)-val_size
    pretrain_train_ds, pretrain_val_ds = random_split(test_dataset, [train_size, val_size])

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

    # Large foundation model for pretraining
    foundation_model = FoundationGNN(in_dim=15, hid_dim=HIDDEN_DIM, out_dim=4, layers=LAYERS, heads=HEADS, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(foundation_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    print("Starting advanced pretraining...")
    for epoch in range(PRETRAIN_EPOCHS):
        train_loss = pretrain_epoch(foundation_model, pretrain_train_loader, optimizer)
        val_loss = pretrain_epoch(foundation_model, pretrain_val_loader, optimizer=None)
        print(f"[Pretraining] Epoch {epoch+1}/{PRETRAIN_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(foundation_model.state_dict(), "pretrained.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered in pretraining.")
                break

    print("Best Pretraining val loss:", best_val_loss)
    # Load best pretrain weights
    foundation_model.load_state_dict(torch.load("pretrained.pt", map_location=DEVICE))

    # Now supervised training on train set
    labeled_dataset = LabeledRNADataset(TRAIN_JSON, BPP_PATH)
    val_size = int(0.1*len(labeled_dataset))
    train_size = len(labeled_dataset)-val_size
    finetune_train_ds, finetune_val_ds = random_split(labeled_dataset,[train_size,val_size])
    finetune_train_loader = DataLoader(finetune_train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_labeled)
    finetune_val_loader = DataLoader(finetune_val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_labeled)

    # Initialize a FineTuneModel and load transformer layers from foundation_model
    finetune_model = FineTuneModel(in_dim=15, hid_dim=HIDDEN_DIM, out_dim=5, layers=LAYERS, heads=HEADS, dropout=DROPOUT, pretrained_model=foundation_model).to(DEVICE)

    print("Starting supervised fine-tuning...")
    supervised_run(finetune_model, finetune_train_loader, finetune_val_loader, epochs=SUPERVISED_EPOCHS)

    print("Done. Model trained with advanced pretraining then supervised training.")

if __name__ == "__main__":
    main()