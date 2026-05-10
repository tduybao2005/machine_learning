import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import copy

np.set_printoptions(threshold=np.inf, linewidth=300, suppress=True)

SEQ_LENGTH = 96  
FEATURES = ['Pressure', 'FlowF']
TIME_COL = 'Datetime' 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Đang sử dụng thiết bị: {device}")

# THAM SỐ CỦA BỘ GEN 
POPULATION_SIZE = 100  
GENERATIONS = 500       
NGENE = 10             
MIN_MAX = [-2.0, 2.0]  
MUTATION_RATE = 0.05
ELITISM = 5
WEIGHTS_FILE = 'weight_linear_ga_8.pth'
USE_SAVED_WEIGHTS = True          


def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], dayfirst=True, format='mixed')
    
    duplicates = df[df.duplicated(subset=[TIME_COL], keep=False)].sort_values(TIME_COL)
    if len(duplicates) > 0:
        print(f"\n⚠️ PHÁT HIỆN DUPLICATE TIMESTAMPS trong {file_path}:")
        print(f"   Tổng duplicate: {len(duplicates)} dòng")
        print("\n   Chi tiết các timestamp bị trùng:")
        for dup_time in duplicates[TIME_COL].unique():
            dup_rows = duplicates[duplicates[TIME_COL] == dup_time]
            print(f"      {dup_time} → {len(dup_rows)} lần trùng")
    
    df.set_index(TIME_COL, inplace=True)
    df_before_dedup = df.copy()
    df = df[~df.index.duplicated(keep='first')]
    
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='15min')
    df_reindex = df.reindex(full_index)
    
    missing_mask = df_reindex[FEATURES[0]].isna()
    missing_times = df_reindex.index[missing_mask]
    
    if len(missing_times) > 0:
        print(f"\n⚠️ PHÁT HIỆN MISSING TIMESTAMPS trong {file_path}:")
        print(f"   Tổng missing: {len(missing_times)} mốc")
        print("\n   Chi tiết các timestamp bị thiếu:")
        
        missing_list = list(missing_times)
        groups = []
        current_group = [missing_list[0]]
        
        for i in range(1, len(missing_list)):
            if (missing_list[i] - missing_list[i-1]).total_seconds() == 900:  # 15 phút
                current_group.append(missing_list[i])
            else:
                groups.append(current_group)
                current_group = [missing_list[i]]
        groups.append(current_group)
        
        for group in groups:
            duration_minutes = (group[-1] - group[0]).total_seconds() / 60
            print(f"      {group[0]} → {group[-1]} ({len(group)} mốc, {duration_minutes:.0f} phút)")
    
    df = df_reindex.copy()
    
    # Nội suy tuyến tính cho khoảng trống ≤ 2 mốc
    df = df.interpolate(method='linear', limit=2)
    
    # Lấy dữ liệu từ hôm trước (24h = 96 mốc)
    # Chia thành 10 đoạn trong ngày để xử lý
    print(f"\n🔄 ĐIỀN DỮ LIỆU từ hôm trước (shift 96 mốc = 24 giờ)...")
    
    for i in range(10):  # 10 đoạn trong ngày
        start_hour = i * 2.4  # Mỗi đoạn = 2.4 giờ
        end_hour = (i + 1) * 2.4
        # Lấy từ 96 mốc trước (24h)
        df = df.fillna(df.shift(96))
    
    # Xử lý edge case (các dòng vẫn còn NaN ở đầu file)
    df = df.bfill().ffill()
    
    return df[FEATURES]

print("Đang xử lý dữ liệu...")
train_df = load_and_preprocess('F98420_2023.csv')
test_df = load_and_preprocess('F98420_2024.csv')
test_df = test_df.loc['2024-01-01':'2024-03-31']

class ManualZScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit_transform(self, data):
        data_np = np.array(data)
        self.mean_ = np.mean(data_np, axis=0)
        self.std_ = np.std(data_np, axis=0)
        self.std_ = np.where(self.std_ == 0, 1e-8, self.std_)
        return (data_np - self.mean_) / self.std_

    def transform(self, data):
        data_np = np.array(data)
        return (data_np - self.mean_) / self.std_

    def inverse_transform(self, data):
        return np.array(data) * self.std_ + self.mean_

scaler = ManualZScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
    def __len__(self): return len(self.data) - self.seq_length
    def __getitem__(self, index):
        x = self.data[index : index + self.seq_length]
        y = self.data[index + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

train_dataset = TimeSeriesDataset(train_scaled, SEQ_LENGTH)


print("\n" + "="*80)
print("THÔNG TIN DỮ LIỆU ĐẦU VÀO")
print("="*80)

print("\n📊 DỮ LIỆU GỐC (Original Data)")
print("-" * 80)
print(f"Training data shape: {train_df.shape} (rows={len(train_df)}, features={len(FEATURES)})")
print(f"Test data shape: {test_df.shape} (rows={len(test_df)}, features={len(FEATURES)})")
print(f"Features: {FEATURES}")
print(f"\nTraining data thời gian: từ {train_df.index[0]} đến {train_df.index[-1]}")
print(f"Test data thời gian: từ {test_df.index[0]} đến {test_df.index[-1]}")

print("\n📈 THỐNG KÊ DỮ LIỆU GỐC (Original Statistics)")
print("-" * 80)
print("Training Data Statistics:")
for feature in FEATURES:
    values = train_df[feature].values
    print(f"  {feature}:")
    print(f"    Mean: {np.mean(values):.6f}, Std: {np.std(values):.6f}")
    print(f"    Min:  {np.min(values):.6f}, Max: {np.max(values):.6f}")

print("\nTest Data Statistics:")
for feature in FEATURES:
    values = test_df[feature].values
    print(f"  {feature}:")
    print(f"    Mean: {np.mean(values):.6f}, Std: {np.std(values):.6f}")
    print(f"    Min:  {np.min(values):.6f}, Max: {np.max(values):.6f}")

print("\n🔄 DỮ LIỆU SAU CHUẨN HÓA (Normalized Data)")
print("-" * 80)
print(f"Training data normalized shape: {train_scaled.shape}")
print(f"Test data normalized shape: {test_scaled.shape}")
print(f"Mean của scaler: {scaler.mean_}")
print(f"Std của scaler: {scaler.std_}")

print("\n📋 THỐNG KÊ DỮ LIỆU CHUẨN HÓA (Normalized Statistics)")
print("-" * 80)
print("Training Data (Normalized):")
for i, feature in enumerate(FEATURES):
    values = train_scaled[:, i]
    print(f"  {feature}:")
    print(f"    Mean: {np.mean(values):8.6f}, Std: {np.std(values):8.6f}")
    print(f"    Min:  {np.min(values):8.6f}, Max: {np.max(values):8.6f}")

print("\nTest Data (Normalized):")
for i, feature in enumerate(FEATURES):
    values = test_scaled[:, i]
    print(f"  {feature}:")
    print(f"    Mean: {np.mean(values):8.6f}, Std: {np.std(values):8.6f}")
    print(f"    Min:  {np.min(values):8.6f}, Max: {np.max(values):8.6f}")

print("\n🎯 DATASET CONFIGURATION")
print("-" * 80)
print(f"Sequence Length (SEQ_LENGTH): {SEQ_LENGTH} time steps")
print(f"Total training samples: {len(train_dataset)} sequences")
print(f"Each sequence shape: (X: {SEQ_LENGTH} × {len(FEATURES)}, Y: {len(FEATURES)})")

print("\n📝 SAMPLE DATA (5 samples từ training dataset)")
print("-" * 80)
print(f"{'Sample':<8} {'Input X Shape':<20} {'Output Y Shape':<20} {'X Range':<25} {'Y Range':<25}")
print("-" * 80)
for i in range(min(5, len(train_dataset))):
    x, y = train_dataset[i]
    x_min, x_max = x.min().item(), x.max().item()
    y_min, y_max = y.min().item(), y.max().item()
    print(f"{i:<8} {str(tuple(x.shape)):<20} {str(tuple(y.shape)):<20} [{x_min:7.4f}, {x_max:7.4f}] [{y_min:7.4f}, {y_max:7.4f}]")

print("\n" + "="*80)
print("Sẵn sàng bắt đầu training!")
print("="*80 + "\n")

input_nodes = SEQ_LENGTH * len(FEATURES) # 96 * 2 = 192 

def create_model(): #3977
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_nodes, 20), # 193x20 = 3860
        nn.Tanh(),
        nn.Linear(20, 5), # 21x5 = 105
        nn.Tanh(),
        nn.Linear(5, len(FEATURES)) # 6x2 = 12 
    ).to(device) 

dummy_model = create_model()
NPAR = sum(p.numel() for p in dummy_model.parameters())
print(f"Tổng số parameter (npar) của mỗi cá thể: {NPAR}")

def set_weights(model, weights_1d_tensor):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data = weights_1d_tensor[idx:idx+n].view_as(p.data).clone()
        idx += n

print("Đang gom toàn bộ dữ liệu 2023 thành 1 Siêu Batch...")
X_all_list = []
Y_all_list = []
for i in range(len(train_dataset)):
    x, y = train_dataset[i]
    X_all_list.append(x)
    Y_all_list.append(y)

X_all = torch.stack(X_all_list).to(device)
Y_all = torch.stack(Y_all_list).to(device)

def evaluate_fitness(model):
    model.eval()
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        # CPU tính toán TẤT CẢ dữ liệu 2023 chỉ bằng 1 phép nhân ma trận
        outputs = model(X_all)
        loss = criterion(outputs, Y_all)
            
    avg_loss = loss.item()
    fitness = 1.0 / (avg_loss + 1e-8) 
    return fitness, avg_loss

# GA
def pop_init(pop_size, npar, ngene):
    return np.random.randint(0, 10, (pop_size, npar, ngene))

def decode(p):
    powers = 10 ** np.arange(NGENE - 1, -1, -1, dtype=np.float64) 
    decimals = np.sum(p * powers, axis=2) 
    max_val = (10 ** NGENE) - 1
    val = MIN_MAX[0] + (decimals / max_val) * (MIN_MAX[1] - MIN_MAX[0])
    return val

def selection(pop, fitness):
    fitness = np.array(fitness)
    fitness = fitness - np.min(fitness) + 1e-6 
    probs = fitness / np.sum(fitness) 
    indices = np.random.choice(len(pop), size=len(pop), p=probs)
    return pop[indices]

def crossover(pop):
    new_pop = np.copy(pop)
    flat_pop = new_pop.reshape(len(pop), -1) 
    total_genes = flat_pop.shape[1]
    
    for i in range(0, len(flat_pop) - 1, 2):
        pt = np.random.randint(1, total_genes) 
        temp1 = np.concatenate((flat_pop[i][:pt], flat_pop[i+1][pt:]))
        temp2 = np.concatenate((flat_pop[i+1][:pt], flat_pop[i][pt:]))
        flat_pop[i] = temp1
        flat_pop[i+1] = temp2
        
    return flat_pop.reshape(pop.shape)

def mutation(pop, mutation_rate):
    flips = np.random.rand(*pop.shape) < mutation_rate
    random_genes = np.random.randint(0, 10, size=pop.shape)
    pop[flips] = random_genes[flips] 
    return pop

eval_model = create_model()

import os
if USE_SAVED_WEIGHTS and os.path.exists(WEIGHTS_FILE):
    print(f"✓ Đang load weights từ file '{WEIGHTS_FILE}'...")
    eval_model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
    print("Đã load weights thành công! Bỏ qua quá trình GA.")
else:
    print(f"Khởi tạo ma trận: {POPULATION_SIZE} cá thể x {NPAR} Trọng số x {NGENE} Chữ số")
    population = pop_init(POPULATION_SIZE, NPAR, NGENE)

    best_global_weights = None
    best_global_fitness = 0
    
    # Tracking losses for plotting
    gen_best_losses = []
    gen_avg_losses = []

    for gen in range(GENERATIONS):
        decoded_pop = decode(population) 
        
        fitness_scores = []
        losses = []
        
        for i in range(POPULATION_SIZE):
            w_tensor = torch.tensor(decoded_pop[i], dtype=torch.float32).to(device)
            set_weights(eval_model, w_tensor)
            
            fit, loss = evaluate_fitness(eval_model) 
            
            fitness_scores.append(fit)
            losses.append(loss)
            
        ranked_indices = np.argsort(fitness_scores)[::-1] 
        gen_best_idx = ranked_indices[0]
        gen_best_fitness = fitness_scores[gen_best_idx]
        gen_best_loss = losses[gen_best_idx]
        
        gen_best_losses.append(gen_best_loss)
        gen_avg_losses.append(np.mean(losses))
        
        if gen % 10 == 0 or gen == GENERATIONS - 1:
            print(f"Thế hệ [{gen:03d}/{GENERATIONS}] | Best Loss: {gen_best_loss:.6f} | Best Fitness: {gen_best_fitness:.6f}")
        
        if gen_best_fitness > best_global_fitness:
            best_global_fitness = gen_best_fitness
            best_global_weights = copy.deepcopy(decoded_pop[gen_best_idx])
            
        elites = population[ranked_indices[:ELITISM]].copy()
        population = selection(population, fitness_scores)
        population = crossover(population)
        population = mutation(population, MUTATION_RATE)
        population[:ELITISM] = elites

    # Lưu mô hình
    best_w_tensor = torch.tensor(best_global_weights, dtype=torch.float32).to(device)
    set_weights(eval_model, best_w_tensor)
    torch.save(eval_model.state_dict(), WEIGHTS_FILE)
    print(f"Đã lưu trọng số hoàn hảo nhất vào '{WEIGHTS_FILE}'")


print("Đang chạy dự báo...")

print("\n" + "="*100)
print("INPUT SEQUENCE CUỐI CÙNG (Last 96 timesteps từ Training 2023 - dùng làm initial input)")
print("="*100)

last_sequence = train_scaled[-SEQ_LENGTH:]
last_sequence_orig = train_df.iloc[-SEQ_LENGTH:].values

print(f"\n📊 THÔNG TIN INPUT SEQUENCE")
print("-" * 100)
print(f"Sequence length: {SEQ_LENGTH} mốc")
print(f"Dữ liệu từ: {train_df.index[-SEQ_LENGTH]} → {train_df.index[-1]}")
print(f"Shape: {last_sequence.shape} (96 timesteps × 2 features)")

print(f"\n📈 THỐNG KÊ INPUT SEQUENCE")
print("-" * 100)
print("Input Sequence (Normalized):")
for i, feature in enumerate(FEATURES):
    values = last_sequence[:, i]
    print(f"  {feature}:")
    print(f"    Mean: {np.mean(values):8.6f}, Std: {np.std(values):8.6f}")
    print(f"    Min:  {np.min(values):8.6f}, Max: {np.max(values):8.6f}")

print("\nInput Sequence (Original Values):")
for i, feature in enumerate(FEATURES):
    values = last_sequence_orig[:, i]
    print(f"  {feature}:")
    print(f"    Mean: {np.mean(values):8.6f}, Std: {np.std(values):8.6f}")
    print(f"    Min:  {np.min(values):8.6f}, Max: {np.max(values):8.6f}")

print(f"\n📋 CHI TIẾT 96 TIMESTEP (All {SEQ_LENGTH} timesteps)")
print("-" * 100)
print(f"{'Index':<8} {'Time':<25} {'Pressure (norm)':<20} {'FlowF (norm)':<20} {'Pressure (orig)':<20} {'FlowF (orig)':<20}")
print("-" * 100)

# In 5 mốc đầu
for i in range(min(5, SEQ_LENGTH)):
    time_str = train_df.index[-SEQ_LENGTH + i].strftime("%Y-%m-%d %H:%M:%S")
    p_norm = last_sequence[i, 0]
    f_norm = last_sequence[i, 1]
    p_orig = last_sequence_orig[i, 0]
    f_orig = last_sequence_orig[i, 1]
    print(f"{i:<8} {time_str:<25} {p_norm:<20.6f} {f_norm:<20.6f} {p_orig:<20.6f} {f_orig:<20.6f}")

# In "..."
if SEQ_LENGTH > 13:
    print("...")

# In 5 mốc cuối
for i in range(max(0, SEQ_LENGTH - 5), SEQ_LENGTH):
    time_str = train_df.index[-SEQ_LENGTH + i].strftime("%Y-%m-%d %H:%M:%S")
    p_norm = last_sequence[i, 0]
    f_norm = last_sequence[i, 1]
    p_orig = last_sequence_orig[i, 0]
    f_orig = last_sequence_orig[i, 1]
    print(f"{i:<8} {time_str:<25} {p_norm:<20.6f} {f_norm:<20.6f} {p_orig:<20.6f} {f_orig:<20.6f}")

print("\n" + "="*100)
print(f"Sequence này sẽ được sử dụng làm initial input để dự báo {len(test_scaled)} bước tiếp theo")
print(f"Khoảng dự báo: {test_df.index[0]} → {test_df.index[-1]} (3 tháng đầu tiên 2024)")
print("="*100 + "\n")

eval_model.eval()
predictions = []

current_seq = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)

num_forecast_steps = len(test_scaled)
with torch.no_grad():
    for _ in range(num_forecast_steps):
        pred = eval_model(current_seq) 
        predictions.append(pred.cpu().numpy()[0])
        pred_expanded = pred.unsqueeze(1) 
        current_seq = torch.cat((current_seq[:, 1:, :], pred_expanded), dim=1)

predictions = np.array(predictions)
actuals = test_scaled 

predictions_inv = scaler.inverse_transform(predictions)
actuals_inv = scaler.inverse_transform(actuals)


print("\n" + "="*100)
print("TÍNH TOÁN MSE - DỰ BÁOFORECAST EVALUATION")
print("="*100)

# MSE = (1/n) * Σ(y_actual - y_pred)²

# Tính MSE cho Pressure
pressure_pred = predictions_inv[:, 0]
pressure_actual = actuals_inv[:, 0]
mse_pressure = np.mean((pressure_actual - pressure_pred) ** 2)

# Tính MSE cho FlowF
flowf_pred = predictions_inv[:, 1]
flowf_actual = actuals_inv[:, 1]
mse_flowf = np.mean((flowf_actual - flowf_pred) ** 2)

# MSE tổng hợp (cả 2 features)
mse_total = np.mean((actuals_inv - predictions_inv) ** 2)

print("\n📊 KẾT QUẢ TÍNH TOÁN MSE:")
print("-" * 100)
print(f"\nPRESSURE:")
print(f"  MSE = (1/{len(pressure_actual)}) × Σ(y_actual - y_pred)²")
print(f"  MSE = {mse_pressure:.10f}")

print(f"\nFLOWF:")
print(f"  MSE = (1/{len(flowf_actual)}) × Σ(y_actual - y_pred)²")
print(f"  MSE = {mse_flowf:.10f}")

print(f"\nTỔNG HỢP (Cả 2 features):")
print(f"  MSE = (1/{len(actuals_inv) * 2}) × Σ(y_actual - y_pred)²")
print(f"  MSE = {mse_total:.10f}")

print("\n📋 BẢNG TÓMLẠI:")
print("-" * 100)
print(f"{'Feature':<15} {'MSE':<25}")
print("-" * 100)
print(f"{'Pressure':<15} {mse_pressure:<25.10f}")
print(f"{'FlowF':<15} {mse_flowf:<25.10f}")
print(f"{'Tổng Hợp':<15} {mse_total:<25.10f}")
print("=" * 100 + "\n")


# ==========================================
# ================== VẼ ĐỒ THỊ ====================
# ==========================================

print("\n" + "="*80)
print("BẮT ĐẦU VẼ TẤT CẢ CÁC BIỂU ĐỒ")
print("="*80)

# ==========================================
# 1. VẼ BIỂU ĐỒ THỰC TẾ DỮ LIỆU
# ==========================================
print("\n📊 Vẽ biểu đồ dữ liệu gốc...")

# Tạo figure cho Pressure
fig_pressure, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

# Train Pressure
ax1.plot(train_df.index, train_df['Pressure'], label='Train Pressure', 
         color='blue', linewidth=1.5, alpha=0.8)
ax1.set_title('Train Data - Pressure (2023)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Pressure (bar)', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Test Pressure
ax2.plot(test_df.index, test_df['Pressure'], label='Test Pressure', 
         color='red', linewidth=1.5, alpha=0.8)
ax2.set_title('Test Data - Pressure (2024 Jan-Mar)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Datetime', fontsize=12)
ax2.set_ylabel('Pressure (bar)', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_visualization_pressure.png', dpi=150)
print("✓ Đã lưu biểu đồ Pressure vào 'data_visualization_pressure.png'")
plt.close()

# Tạo figure cho FlowF
fig_flowf, (ax3, ax4) = plt.subplots(2, 1, figsize=(18, 10))

# Train FlowF
ax3.plot(train_df.index, train_df['FlowF'], label='Train FlowF', 
         color='green', linewidth=1.5, alpha=0.8)
ax3.set_title('Train Data - FlowF (2023)', fontsize=14, fontweight='bold')
ax3.set_ylabel('FlowF (m³/h)', fontsize=12)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# Test FlowF
ax4.plot(test_df.index, test_df['FlowF'], label='Test FlowF', 
         color='orange', linewidth=1.5, alpha=0.8)
ax4.set_title('Test Data - FlowF (2024 Jan-Mar)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Datetime', fontsize=12)
ax4.set_ylabel('FlowF (m³/h)', fontsize=12)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_visualization_flowf.png', dpi=150)
print("✓ Đã lưu biểu đồ FlowF vào 'data_visualization_flowf.png'")
plt.close()

# Tạo figure so sánh Train vs Test (Cùng hệ trục)
fig_comparison, axes = plt.subplots(2, 2, figsize=(20, 12))

# Train Pressure
axes[0, 0].plot(train_df.index, train_df['Pressure'], color='blue', linewidth=1, alpha=0.8)
axes[0, 0].set_title('Train - Pressure', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Pressure (bar)', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=train_df['Pressure'].mean(), color='blue', linestyle='--', 
                    alpha=0.5, label=f'Mean: {train_df["Pressure"].mean():.2f}')
axes[0, 0].legend()

# Train FlowF
axes[0, 1].plot(train_df.index, train_df['FlowF'], color='green', linewidth=1, alpha=0.8)
axes[0, 1].set_title('Train - FlowF', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('FlowF (m³/h)', fontsize=11)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=train_df['FlowF'].mean(), color='green', linestyle='--', 
                   alpha=0.5, label=f'Mean: {train_df["FlowF"].mean():.2f}')
axes[0, 1].legend()

# Test Pressure
axes[1, 0].plot(test_df.index, test_df['Pressure'], color='red', linewidth=1, alpha=0.8)
axes[1, 0].set_title('Test - Pressure', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Datetime', fontsize=11)
axes[1, 0].set_ylabel('Pressure (bar)', fontsize=11)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=test_df['Pressure'].mean(), color='red', linestyle='--', 
                   alpha=0.5, label=f'Mean: {test_df["Pressure"].mean():.2f}')
axes[1, 0].legend()

# Test FlowF
axes[1, 1].plot(test_df.index, test_df['FlowF'], color='orange', linewidth=1, alpha=0.8)
axes[1, 1].set_title('Test - FlowF', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Datetime', fontsize=11)
axes[1, 1].set_ylabel('FlowF (m³/h)', fontsize=11)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=test_df['FlowF'].mean(), color='orange', linestyle='--', 
                   alpha=0.5, label=f'Mean: {test_df["FlowF"].mean():.2f}')
axes[1, 1].legend()

plt.suptitle('Train & Test Data Comparison (After Preprocessing)', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('data_comparison_train_test.png', dpi=150)
print("✓ Đã lưu biểu đồ so sánh vào 'data_comparison_train_test.png'")
plt.close()

print(f"\n📊 THÔNG TIN CÁC BIỂU ĐỒ DỮ LIỆU VỪA VẼ:")
print("-" * 80)
print(f"1. data_visualization_pressure.png")
print(f"   └─ Hiển thị Pressure của Train (2023) và Test (2024 Q1)")
print(f"\n2. data_visualization_flowf.png")
print(f"   └─ Hiển thị FlowF của Train (2023) và Test (2024 Q1)")
print(f"\n3. data_comparison_train_test.png")
print(f"   └─ So sánh 4 subplot: Train Pressure, Train FlowF, Test Pressure, Test FlowF")
print(f"   └─ Mỗi subplot có đường mean để nhìn trend rõ ràng")
print("=" * 80)

# ==========================================
# 2. VẼ BIỂU ĐỒ LOSS GA TRAINING
# ==========================================
if not (USE_SAVED_WEIGHTS and os.path.exists(WEIGHTS_FILE)):
    print("\n📊 Vẽ biểu đồ GA training loss...")
    
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(gen_best_losses, label='Best Loss', linewidth=2, color='blue')
    ax.plot(gen_avg_losses, label='Average Loss', linewidth=2, color='orange', alpha=0.7)
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Loss Progression During GA Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # ==========================================
    # THÊM BẢNG CHÚ THÍCH (ANNOTATION TABLE)
    # ==========================================
    best_loss_value = min(gen_best_losses)
    gen_of_best = np.argmin(gen_best_losses)
    final_loss = gen_best_losses[-1]
    improvement = ((gen_best_losses[0] - final_loss) / gen_best_losses[0]) * 100
    
    # Tạo bảng dữ liệu
    table_data = [
        ['Metric', 'Value'],
        ['Best Loss', f'{best_loss_value:.6f}'],
        ['Final Loss', f'{final_loss:.6f}'],
        ['Gen at Best', f'{gen_of_best}'],
        ['Improvement', f'{improvement:.2f}%'],
        ['Total Gen', f'{GENERATIONS}'],
        ['Avg Loss (init)', f'{gen_avg_losses[0]:.6f}'],
        ['Avg Loss (final)', f'{gen_avg_losses[-1]:.6f}']
    ]
    
    # Vẽ bảng trên plot
    table = ax.table(cellText=table_data, cellLoc='center', loc='upper left',
                     bbox=[0.02, 0.55, 0.18, 0.40],
                     colWidths=[0.55, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Định dạng header row
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Định dạng data rows với màu xen kẽ
    for i in range(1, len(table_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#e8f0f8')
            else:
                table[(i, j)].set_facecolor('#f5f5f5')
            if j == 0:
                table[(i, j)].set_text_props(weight='bold')
    
    plt.tight_layout()
    plt.savefig('loss_ga_training_8.png', dpi=150)
    print("✓ Đã lưu biểu đồ loss vào 'loss_ga_training_8.png'")
    print(f"  ├─ Best Loss: {best_loss_value:.6f} (Gen {gen_of_best})")
    print(f"  ├─ Final Loss: {final_loss:.6f}")
    print(f"  └─ Improvement: {improvement:.2f}%")
    plt.close()

# ==========================================
# 3. VẼ BIỂU ĐỒ DỰ BÁOFORECAST
# ==========================================
print("\n📊 Vẽ biểu đồ dự báo forecast...")

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(actuals_inv[:, 0], label='Thực tế', color='blue', alpha=0.6)
plt.plot(predictions_inv[:, 0], label='Dự báo', color='red', alpha=0.8)
plt.title('Dự báo Multi-step Pressure (Mô hình tối giản tối ưu bằng GA)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(actuals_inv[:, 1], label='Thực tế', color='green', alpha=0.6)
plt.plot(predictions_inv[:, 1], label='Dự báo', color='orange', alpha=0.8)
plt.title('Dự báo Multi-step FlowF (Mô hình tối giản tối ưu bằng GA)')
plt.legend()

plt.tight_layout()
plt.savefig('forecast_results_ga_8.png', dpi=150)
print("✓ Đã lưu biểu đồ vào 'forecast_results_ga_8.png'")
plt.close()

# ==========================================
# 4. VẼ BIỂU ĐỒ ZOOM CHI TIẾT (VÀI CHU KỲ)
# ==========================================
print("\n📊 Vẽ biểu đồ zoom chi tiết...")

start_idx = 500
end_idx = 1000
zoom_range = range(start_idx, min(end_idx, len(actuals_inv)))

plt.figure(figsize=(18, 10))

plt.subplot(2, 1, 1)
plt.plot(zoom_range, actuals_inv[start_idx:end_idx, 0], label='Thực tế', color='blue', linewidth=2, marker='o', markersize=3)
plt.plot(zoom_range, predictions_inv[start_idx:end_idx, 0], label='Dự báo', color='red', linewidth=2, marker='s', markersize=3)
plt.title(f'ZOOM: Dự báo Pressure (Bước {start_idx}-{end_idx})', fontsize=14, fontweight='bold')
plt.xlabel('Số bước (15 phút/bước)')
plt.ylabel('Pressure')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(zoom_range, actuals_inv[start_idx:end_idx, 1], label='Thực tế', color='green', linewidth=2, marker='o', markersize=3)
plt.plot(zoom_range, predictions_inv[start_idx:end_idx, 1], label='Dự báo', color='orange', linewidth=2, marker='s', markersize=3)
plt.title(f'ZOOM: Dự báo FlowF (Bước {start_idx}-{end_idx})', fontsize=14, fontweight='bold')
plt.xlabel('Số bước (15 phút/bước)')
plt.ylabel('FlowF')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('forecast_results_ga_zoom_8.png', dpi=150)
print("✓ Đã lưu biểu đồ zoom vào 'forecast_results_ga_zoom_8.png'")
print(f"  └─ Khoảng zoom: Bước {start_idx} đến {end_idx} ({(end_idx-start_idx)*15} phút ≈ {(end_idx-start_idx)/96:.1f} ngày)")
plt.close()

print("\n" + "="*80)
print("✓ HOÀN THÀNH VẼ TẤT CẢ CÁC BIỂU ĐỒ")
print("="*80)