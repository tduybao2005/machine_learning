import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TORCH_DISABLE_COMPILE'] = '1'

import torch
import torch.nn as nn

np.set_printoptions(threshold=np.inf, linewidth=300, suppress=True)

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

# Đọc và Scale dữ liệu
df = pd.read_csv('semi_supervised_data.csv')
X_raw = df[['x1', 'x2']].values
labels_orig = df['label'].values          

X_min, X_max = X_raw.min(axis=0), X_raw.max(axis=0)
X = 2 * (X_raw - X_min) / (X_max - X_min) - 1  #[-1, 1]

x_torch = torch.tensor(X, dtype=torch.float32)
num_clusters = 3

# Cài đặt Thuật toán di truyền
pop_size = 100
npar = 9 #3x2 + 3      
ngene = 10     
min_max = [-1, 1] 
mutation_rate = 0.05
generations = 150

def pop_init(pop_size, npar, ngene):
    return np.random.randint(0, 10, (pop_size, npar, ngene))

def decode(p):
    powers = 10 ** np.arange(ngene - 1, -1, -1, dtype=np.float64) 
    decimals = np.sum(p * powers, axis=2) 
    max_val = (10 ** ngene) - 1
    val = min_max[0] + (decimals / max_val) * (min_max[1] - min_max[0])
    return val

def selection(pop, fitness):
    fitness = np.array(fitness)
    fitness = -fitness 
    fitness = fitness - np.min(fitness) + 1e-6 
    probs = fitness / np.sum(fitness) 
    indices = np.random.choice(len(pop), size=len(pop), p=probs)
    return pop[indices]

def crossover(pop):
    new_pop = np.copy(pop)
    flat_pop = new_pop.reshape(len(pop), -1) 
    total_genes = flat_pop.shape[1]
    for i in range(1, len(flat_pop) - 1, 2):
        pt = np.random.randint(1, total_genes) 
        temp1 = np.concatenate((flat_pop[i][:pt], flat_pop[i+1][pt:]))
        temp2 = np.concatenate((flat_pop[i+1][:pt], flat_pop[i][pt:]))
        flat_pop[i] = temp1
        flat_pop[i+1] = temp2
    return flat_pop.reshape(pop.shape)

def mutation(pop, mutation_rate):
    flips = np.random.rand(*(pop.shape)) < mutation_rate
    flips[0, :, :] = False 
    random_genes = np.random.randint(0, 10, size=pop.shape)
    pop[flips] = random_genes[flips] 
    return pop

population = pop_init(pop_size, npar, ngene)
model = RBFLayer(in_features=2, num_rbfs=num_clusters) 

print("Bắt đầu huấn luyện RBF (Không dùng Hoán vị)...")
best_loss = float('inf')
best_chromosome_genes = None

for gen in range(generations):
    decoded_params = decode(population) 
    losses = []
    with torch.no_grad():
        for i in range(pop_size):
            centers_np = decoded_params[i, :6].reshape(3, 2)
            sigmas_np = np.abs(decoded_params[i, 6:]) + 0.05 
            model.centers.data = torch.tensor(centers_np, dtype=torch.float32)
            model.sigma.data = torch.tensor(sigmas_np, dtype=torch.float32)
            probs, phi = model(x_torch)
            loss_t = clustering_loss(x_torch, probs, model.centers, model.sigma, gen)
            loss = loss_t.item()
            labels = torch.argmax(probs, dim=1)
            for k in range(num_clusters):
                if (labels == k).sum() == 0:
                    loss += 1000.0 
            losses.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_chromosome_genes = population[i].copy()
                best_centers = centers_np.copy()
                best_sigmas = sigmas_np.copy()
    if gen % 20 == 0:
        print(f"Generation {gen:3d} | Best Loss: {best_loss:.4f}")
    min_loss_idx = np.argmin(losses)
    population[0] = population[min_loss_idx] 
    population = selection(population, losses)
    population[0] = best_chromosome_genes 
    population = crossover(population)
    population = mutation(population, mutation_rate)

print(f"Hoàn tất! Best Loss: {best_loss:.4f}\n")

with torch.no_grad():
    model.centers.data = torch.tensor(best_centers, dtype=torch.float32)
    model.sigma.data = torch.tensor(best_sigmas, dtype=torch.float32)
    probs, _ = model(x_torch)
    labels_cluster = torch.argmax(probs, dim=1).numpy() #[0, 0, 0, 1, 1, 2, 2, 2]

# labels_origin = [-1, 2, 2, -1, 0, 1, -1, 1]
# k = 0 
# labels_cluster = [true, true, true, false, false, false, false, false]
# labels_origin = [false, true, true, false, true, true, false, true]
# mask_k = [false, true, true, false, false, false, false, false]
# true_labels_in_k = [2, 2]
# most_frequent_label = 2
# mapping = {0: 2}
mapping = {}
for k in range(num_clusters): #k = 0, 1, 2 
    mask_k = (labels_cluster == k) & (labels_orig != -1) 
    
    if mask_k.sum() > 0:
        true_labels_in_k = labels_orig[mask_k]
        most_frequent_label = np.bincount(true_labels_in_k).argmax() 
        mapping[k] = most_frequent_label 
    else:
        mapping[k] = k 

labels_cluster = np.array([mapping[l] for l in labels_cluster])

labeled_mask = (labels_orig != -1)
labels_cluster[labeled_mask] = labels_orig[labeled_mask]

mapped_centers = np.zeros_like(best_centers)
mapped_sigmas = np.zeros_like(best_sigmas)
for machine_k, true_k in mapping.items():
    mapped_centers[true_k] = best_centers[machine_k]
    mapped_sigmas[true_k] = best_sigmas[machine_k]
best_centers = mapped_centers
best_sigmas = mapped_sigmas

# Vẽ biểu đồ
cmap_labeled = {0: 'royalblue', 1: 'tomato', 2: 'limegreen'}
cmap_cluster = ['royalblue', 'tomato', 'limegreen']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Đồ thị 1: Dữ liệu gốc
mask_unlabeled = (labels_orig == -1)
ax1.scatter(X[mask_unlabeled, 0], X[mask_unlabeled, 1], c='lightgray', alpha=0.4, s=20)
for lbl, color in cmap_labeled.items():
    mask = (labels_orig == lbl)
    ax1.scatter(X[mask, 0], X[mask, 1], c=color, s=60, edgecolors='black', linewidths=0.5)
ax1.set_title('Dữ liệu gốc')

# Đồ thị 2: Dữ liệu dự đoán
for k in range(num_clusters):
    mask = (labels_cluster == k)
    ax2.scatter(X[mask, 0], X[mask, 1], c=cmap_cluster[k], alpha=0.5, s=20)
    ax2.scatter(best_centers[k, 0], best_centers[k, 1], marker='X', s=250, color='black', zorder=5)
    circle = plt.Circle((best_centers[k, 0], best_centers[k, 1]), best_sigmas[k], fill=False, linestyle='--', color='black', linewidth=1.5)
    ax2.add_artist(circle)
ax2.set_title('Dự đoán (Majority Voting + Cố định nhãn gốc)')
ax2.set_aspect('equal', adjustable='datalim')

plt.tight_layout()
plt.show()