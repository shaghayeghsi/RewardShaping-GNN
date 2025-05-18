import pandas as pd
import networkx as nx
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_sparse import SparseTensor, matmul
from dialogue_config import FAIL, SUCCESS
from node2vec import Node2Vec

# ---------- Load CSV and Build Graph ----------
csv_file = "/content/drive/MyDrive/ArewardShap/ArewardShap/ArewardShap/GO-Bot-DRL/dataset_state_after.csv"
G = nx.Graph()

with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    previous_state = None

    for row in reader:
        try:
            state = [int(float(x)) for x in row]
            if state.count(1) == 3 and state.count(0) == len(state) - 3:
                G.add_node(tuple(state))
                if previous_state is not None:
                    G.remove_edge(previous_state, tuple(state))
            else:
                G.add_node(tuple(state))
                if previous_state is not None:
                    G.add_edge(previous_state, tuple(state))
            previous_state = tuple(state)
        except ValueError:
            print(f"Skipping row with non-numeric value: {row}")

# ---------- Build Node Features ----------
node_list = list(G.nodes())
node_index_map = {node: idx for idx, node in enumerate(node_list)}
node_features = torch.tensor(node_list, dtype=torch.float32)  # shape [N, F]

# ---------- Build Sparse Adjacency ----------
edges = []
for source_node, target_node in G.edges():
    src_idx = node_index_map[source_node]
    tgt_idx = node_index_map[target_node]
    edges.append((src_idx, tgt_idx))
edge_index = torch.tensor(edges, dtype=torch.long).t()  # shape [2, E]
adj_sparse = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(len(node_list), len(node_list)))

# ---------- MinCut Pooling ----------
class MinCutPoolLayerSparse(nn.Module):
    def __init__(self, in_dim, num_clusters):
        super(MinCutPoolLayerSparse, self).__init__()
        self.num_clusters = num_clusters
        self.assign_net = nn.Linear(in_dim, num_clusters)
        self.proj_net = nn.Linear(in_dim, in_dim)

    def forward(self, x, adj_sparse: SparseTensor):
        S = torch.softmax(self.assign_net(x), dim=-1)  # [N, K]
        X_proj = self.proj_net(x)                      # [N, F]
        Z = torch.matmul(S.T, X_proj)                  # [K, F]
        adj_S = matmul(adj_sparse, S)                  # [N, K]
        adj_new = torch.matmul(S.T, adj_S)             # [K, K]

        deg = adj_sparse.sum(dim=1).to_dense()
        D = deg.unsqueeze(-1) * torch.ones_like(S)
        vol = torch.trace(torch.matmul(S.T, D))
        cut = torch.trace(adj_new)
        mincut_loss = -cut / (vol + 1e-9)

        SS = torch.matmul(S.T, S)
        I = torch.eye(self.num_clusters, device=S.device)
        ortho_loss = torch.norm(SS - I, p='fro')

        return Z, adj_new, mincut_loss, ortho_loss, S

# ---------- Train Model ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_clusters = 1500
in_dim = node_features.shape[1]

model = MinCutPoolLayerSparse(in_dim=in_dim, num_clusters=num_clusters).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

x = node_features.to(device)
adj_sparse = adj_sparse.to(device)

for epoch in range(100):
    optimizer.zero_grad()
    Z, adj_new, mincut_loss, ortho_loss, S = model(x, adj_sparse)
    loss = mincut_loss + 0.1 * ortho_loss
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# ---------- Build Reduced Graph (Each cluster is one node) ----------
model.eval()
with torch.no_grad():
    _, _, _, _, S = model(x, adj_sparse)
    S = S.detach().cpu().numpy()
    cluster_assignments = np.argmax(S, axis=1)  # shape [N]

smaller_graph = nx.DiGraph()
for i in range(num_clusters):
    smaller_graph.add_node(i)

for source_node, target_node in G.edges():
    src_idx = node_index_map[source_node]
    tgt_idx = node_index_map[target_node]
    src_cluster = cluster_assignments[src_idx]
    tgt_cluster = cluster_assignments[tgt_idx]
    if src_cluster != tgt_cluster:
        smaller_graph.add_edge(src_cluster, tgt_cluster)

# ---------- Node2Vec on Reduced Graph ----------
node2vec = Node2Vec(smaller_graph, dimensions=1, walk_length=20, num_walks=30, p=1, q=2, workers=1)
model_n2v = node2vec.fit(window=10, min_count=1)
embeddings = {str(node): model_n2v.wv[str(node)] for node in smaller_graph.nodes()}

# ---------- Count Model Parameters ----------
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_size_MB = total_params * 4 / (1024**2)  # assuming float32 (4 bytes per param)
    print(f"📊 Total Parameters: {total_params:,}")
    print(f"🎯 Trainable Parameters: {trainable_params:,}")
    print(f"💾 Approximate Model Size: {total_size_MB:.2f} MB")

count_parameters(model)
# ---------- Utility Functions ----------
def convert_list_to_dict(lst):
    if len(lst) > len(set(lst)):
        raise ValueError('List must be unique!')
    return {k: v for v, k in enumerate(lst)}

def remove_empty_slots(dic):
    for id in list(dic.keys()):
        for key in list(dic[id].keys()):
            if dic[id][key] == '':
                dic[id].pop(key)

# ---------- Reward Function ----------
def reward_function(success, max_round, state):
    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
    partial_tensor = state_tensor[:224]
    assign_probs = torch.softmax(model.assign_net(partial_tensor), dim=-1)  # [K]
    cluster_label = torch.argmax(assign_probs).item()
    rewardd = embeddings[str(cluster_label)][0]
    reward = -1 + rewardd
    if success == FAIL:
        reward += -max_round
    elif success == SUCCESS:
        reward += 2 * max_round
    return reward
