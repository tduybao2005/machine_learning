import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

df = pd.read_csv('semi_supervised_data.csv')
X = df[['x1', 'x2']].values
labels_orig = df['label'].values          

x_torch = torch.tensor(X, dtype=torch.float32)
num_clusters = 3

class RBFLayer(nn.Module):
    def __init__(self, in_features, num_rbfs):
        super().__init__()
        self.centers = nn.Parameter(torch.rand(num_rbfs, in_features))
        self.sigma   = nn.Parameter(torch.ones(num_rbfs))

    def forward(self, x):
        x_expanded = x.unsqueeze(1)                                     
        c_expanded = self.centers.unsqueeze(0)                         
        dist_sq = torch.sum((x_expanded - c_expanded) ** 2, dim=2)  
        phi = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        probs = torch.softmax(phi, dim=1)
        return probs, phi

def clustering_loss(x, probs, centers, sigmas, epoch):
    labels = torch.argmax(probs, dim=1)
    loss = 0
    sigma_loss = 0
    for k in range(centers.size(0)):
        mask = (labels == k)
        if mask.sum() == 0:
            continue
        loss += ((x[mask] - centers[k]) ** 2).sum()
        if epoch > 30:
            cluster_var = torch.mean(torch.sum((x[mask] - centers[k]) ** 2, dim=1))
            sigma_loss += (sigmas[k] ** 2 - cluster_var).pow(2)
    return loss / len(x) + sigma_loss

# ── 4. Khởi tạo tâm ngẫu nhiên trong phạm vi dữ liệu ────────────
x_min, x_max = X.min(axis=0), X.max(axis=0)
init_centers = np.random.uniform(x_min, x_max, size=(num_clusters, 2)).astype(np.float32)

model = RBFLayer(in_features=2, num_rbfs=num_clusters)
with torch.no_grad():
    model.centers.copy_(torch.tensor(init_centers))

optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    probs, phi = model(x_torch)
    loss = clustering_loss(x_torch, probs, model.centers, model.sigma, epoch)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f'Epoch {epoch+1:3d}, Loss: {loss.item():.4f}')

with torch.no_grad():
    probs, _        = model(x_torch)
    labels_cluster  = torch.argmax(probs, dim=1).numpy()
    centers_learned = model.centers.detach().numpy()
    sigma_learned   = model.sigma.detach().numpy()

# Khớp cluster index với label gốc bằng majority vote
from scipy.optimize import linear_sum_assignment
labeled_mask = (labels_orig != -1)
cost_matrix = np.zeros((num_clusters, num_clusters))
for c in range(num_clusters):
    for l in range(num_clusters):
        cost_matrix[c, l] = -np.sum((labels_cluster[labeled_mask] == c) & (labels_orig[labeled_mask] == l))
row_ind, col_ind = linear_sum_assignment(cost_matrix)
mapping = {row_ind[i]: col_ind[i] for i in range(num_clusters)}
labels_cluster = np.array([mapping[l] for l in labels_cluster])
# Sắp xếp lại centers và sigma theo mapping
inv_mapping = {v: k for k, v in mapping.items()}
centers_learned = centers_learned[[inv_mapping[k] for k in range(num_clusters)]]
sigma_learned   = sigma_learned[[inv_mapping[k] for k in range(num_clusters)]]

print(f"\nLearned centers:\n{centers_learned}")
print(f"Learned sigmas: {sigma_learned}")
print(f"Cluster sizes: {[np.sum(labels_cluster == i) for i in range(num_clusters)]}")

cmap_labeled = {0: 'royalblue', 1: 'tomato', 2: 'limegreen'}
cmap_cluster = ['royalblue', 'tomato', 'limegreen']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Hình trái: dữ liệu gốc ---
mask_unlabeled = (labels_orig == -1)
ax1.scatter(X[mask_unlabeled, 0], X[mask_unlabeled, 1],
            c='lightgray', alpha=0.4, s=20, label='Unlabeled (-1)')

for lbl, color in cmap_labeled.items():
    mask = (labels_orig == lbl)
    ax1.scatter(X[mask, 0], X[mask, 1],
                c=color, s=60, edgecolors='black', linewidths=0.5,
                label=f'Label {lbl} ({mask.sum()} pts)')

ax1.set_title('Dữ liệu gốc (giữ nhãn cũ)', fontsize=13)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Hình phải: sau khi clustering RBF ---
for k in range(num_clusters):
    mask = (labels_cluster == k)
    ax2.scatter(X[mask, 0], X[mask, 1],
                c=cmap_cluster[k], alpha=0.5, s=20, label=f'Cluster {k}')

for k in range(num_clusters):
    ax2.scatter(centers_learned[k, 0], centers_learned[k, 1],
                marker='X', s=250, color='black', zorder=5,
                label=f'Center {k} (σ={sigma_learned[k]:.2f})')
    circle = plt.Circle((centers_learned[k, 0], centers_learned[k, 1]),
                        sigma_learned[k], fill=False,
                        linestyle='--', color='black', linewidth=1.5, alpha=0.6)
    ax2.add_artist(circle)

ax2.set_title('Sau khi RBF Clustering', fontsize=13)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal', adjustable='datalim')

plt.suptitle('Semi-supervised Data — Original vs RBF Clustering', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


