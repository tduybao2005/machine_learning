import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ==========================================
# 1. CẤU HÌNH THAM SỐ
# ==========================================
SEQ_LENGTH = 96  
EPOCHS = 500
LEARNING_RATE = 0.001
FEATURES = ['Pressure', 'FlowF']
TIME_COL = 'Datetime'
WEIGHTS_FILE = 'weight_mlp_manual_2.pth' #=================================================file trọng số==============
USE_SAVED_WEIGHTS = True  # True: Dùng weights đã lưu, False: Train từ đầu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Đang sử dụng thiết bị: {device}")

# ==========================================
# 2. TIỀN XỬ LÝ DỮ LIỆU
# ==========================================
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], dayfirst=True, format='mixed')
    df.set_index(TIME_COL, inplace=True)
    
    df = df[~df.index.duplicated(keep='first')]
    df = df.resample('15min').interpolate(method='linear')
    df.dropna(inplace=True) 
    return df[FEATURES]

print("Đang tải và tiền xử lý dữ liệu...")
train_df = load_and_preprocess('F98420_2023.csv')
test_df = load_and_preprocess('F98420_2024.csv')
test_df = test_df.loc['2024-01-01':'2024-03-31']

# ==========================================
# TỰ CODE CLASS CHUẨN HÓA Z-SCORE (THAY THẾ SKLEARN)
# ==========================================
class ManualZScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, data):
        # Chuyển sang numpy array để tính toán nhanh
        data_np = np.array(data)
        # Tính trung bình (mean) theo từng cột (axis=0)
        self.mean_ = np.mean(data_np, axis=0)
        # Tính độ lệch chuẩn (std) theo từng cột
        self.std_ = np.std(data_np, axis=0)
        
        # Mẹo nhỏ an toàn: Tránh lỗi chia cho 0 nếu một cột có giá trị cố định không đổi
        self.std_ = np.where(self.std_ == 0, 1e-8, self.std_)

    def transform(self, data):
        data_np = np.array(data)
        # Áp dụng công thức Z = (x - mean) / std
        return (data_np - self.mean_) / self.std_

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        data_np = np.array(data)
        # Áp dụng công thức đảo ngược: x = Z * std + mean
        return data_np * self.std_ + self.mean_

# Sử dụng class thủ công vừa tạo
scaler = ManualZScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

# ==========================================
# 3. TẠO DATASET VÀ DATALOADER
# ==========================================
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data[index : index + self.seq_length]
        y = self.data[index + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

train_dataset = TimeSeriesDataset(train_scaled, SEQ_LENGTH)

# Chuẩn bị "Siêu Batch" giống file GA - gom toàn bộ dữ liệu 2023 thành 1 batch
print("Đang gom toàn bộ dữ liệu 2023 thành 1 Siêu Batch...")
X_all_list = []
Y_all_list = []
for i in range(len(train_dataset)):
    x, y = train_dataset[i]
    X_all_list.append(x)
    Y_all_list.append(y)

X_all = torch.stack(X_all_list).to(device)
Y_all = torch.stack(Y_all_list).to(device)
print(f"Siêu Batch shape: X={X_all.shape}, Y={Y_all.shape}")

# Không cần DataLoader nữa vì dùng Siêu Batch
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 4. XÂY DỰNG MẠNG NƠ-RON BÌNH THƯỜNG (MLP) 
# ==========================================
input_nodes = SEQ_LENGTH * len(FEATURES)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=input_nodes, out_features=128),
    nn.Tanh(),
    nn.Linear(in_features=128, out_features=64),
    nn.Tanh(),
    nn.Linear(in_features=64, out_features=len(FEATURES))
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 5. QUÁ TRÌNH HUẤN LUYỆN (hoặc Load weights từ file)
# ==========================================
import os
if USE_SAVED_WEIGHTS and os.path.exists(WEIGHTS_FILE):
    print(f"✓ Đang load weights từ file '{WEIGHTS_FILE}'...")
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
    print("Đã load weights thành công! Bỏ qua quá trình huấn luyện.")
    losses = None  # Không có losses vì không train
else:
    print("Bắt đầu huấn luyện mạng MLP (Manual Z-score) với Siêu Batch...")
    losses = []
    for epoch in range(EPOCHS):
        model.train()
        
        # Forward pass với toàn bộ dữ liệu
        outputs = model(X_all)
        loss = criterion(outputs, Y_all)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_loss = loss.item()
        losses.append(avg_loss)
            
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}')

    # Vẽ đồ thị Loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, linewidth=2, color='blue')
    plt.title('Training Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss_mlp_manual.png')
    plt.show()
    print("Đã lưu đồ thị loss vào 'training_loss_mlp_manual.png'")

    torch.save(model.state_dict(), WEIGHTS_FILE)
    print(f"Đã lưu trọng số mô hình vào '{WEIGHTS_FILE}'")

# Vẽ đồ thị Loss (chỉ vẽ nếu có dữ liệu losses từ quá trình training)
if losses is not None:
    plt.figure(figsize=(10, 5))
    plt.plot(losses, linewidth=2, color='blue')
    plt.title('Training Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss_mlp_manual.png')
    plt.show()
    print("Đã lưu đồ thị loss vào 'training_loss_mlp_manual.png'")
else:
    print("(Bỏ qua vẽ loss vì load weights từ file - không có dữ liệu training)")

# ==========================================
# 6. ĐÁNH GIÁ (MULTI-STEP FORECASTING)
# ==========================================
model.eval()
predictions = []

last_sequence = train_scaled[-SEQ_LENGTH:]
current_seq = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)

num_forecast_steps = len(test_scaled)

print(f"Đang dự báo tự hồi quy {num_forecast_steps} bước...")
with torch.no_grad():
    for _ in range(num_forecast_steps):
        pred = model(current_seq) 
        predictions.append(pred.cpu().numpy()[0])
        
        pred_expanded = pred.unsqueeze(1) 
        current_seq = torch.cat((current_seq[:, 1:, :], pred_expanded), dim=1)

predictions = np.array(predictions)
actuals = test_scaled 

mse = np.mean((predictions - actuals)**2)
print(f'Test MSE (Z-score scaled): {mse:.6f}')

# Hàm này giờ là hàm thủ công của chúng ta gọi ra
predictions_inv = scaler.inverse_transform(predictions)
actuals_inv = scaler.inverse_transform(actuals)

# ==========================================
# 7. VẼ BIỂU ĐỒ
# ==========================================
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(actuals_inv[:, 0], label='Thực tế (Pressure)', color='blue', alpha=0.6)
plt.plot(predictions_inv[:, 0], label='Dự báo MLP (Pressure)', color='red', alpha=0.8)
plt.title('Dự báo Multi-step Pressure (Manual Z-score) - 2024')
plt.xlabel('Thời gian (15 phút / bước)')
plt.ylabel('Pressure')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(actuals_inv[:, 1], label='Thực tế (FlowF)', color='green', alpha=0.6)
plt.plot(predictions_inv[:, 1], label='Dự báo MLP (FlowF)', color='orange', alpha=0.8)
plt.title('Dự báo Multi-step FlowF (Manual Z-score) - 2024')
plt.xlabel('Thời gian (15 phút / bước)')
plt.ylabel('FlowF')
plt.legend()

plt.tight_layout()
plt.savefig('forecast_results_mlp_manual_1.png') #===================== file hình dự báo toàn bộ 3 tháng đầu 2024
print("Đã lưu biểu đồ vào 'forecast_results_mlp_manual_1.png'")

# ==========================================
# 8. VẼ BIỂU ĐỒ ZOOM CHI TIẾT (VÀI CHU KỲ)
# ==========================================
# Zoom vào 500-1000 bước (7.5 - 15 ngày) để xem rõ chi tiết vài chu kỳ

# ============================= =================SỬA LẠI 2 dòng start/end này ==========================
start_idx = 1500
end_idx = 2000
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
plt.savefig('forecast_results_mlp_manual_zoom.png', dpi=150) #===================== file hình zoom lên
plt.show()
print("Đã lưu biểu đồ zoom vào 'forecast_results_mlp_manual_zoom.png'")
print(f"Khoảng zoom: Bước {start_idx} đến {end_idx} ({(end_idx-start_idx)*15} phút ≈ {(end_idx-start_idx)/96:.1f} ngày)")