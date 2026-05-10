import torch 
import torch.nn as nn 
import torch.optim as optim
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd 
import numpy as np

data = pd.read_csv("dataset_10feature.csv")
x = data[["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]].values
y = data["label"].values

df = pd.DataFrame(x, columns=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"])
df["label"] = y

train_list = []
test_list = []

for cls, group in df.groupby("label"):
    group = group.sample(frac=1, random_state=42).reset_index(drop=True) 
    split_idx = int(0.7 * len(group))
    train_list.append(group.iloc[:split_idx])
    test_list.append(group.iloc[split_idx:])

train_df = pd.concat(train_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)

x_train = train_df.drop("label", axis=1).values
y_train = train_df["label"].values
x_test = test_df.drop("label", axis=1).values
y_test = test_df["label"].values

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


model = nn.Sequential(
    nn.Linear(10, 10),
    nn.Tanh(),
    nn.Linear(10, 20),
    nn.Tanh(),
    nn.Linear(20, 3),   
    nn.Tanh(),
    nn.Linear(3, 2) 
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
for epoch in range(epochs):
    model.train()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("Hoàn thành huấn luyện!\n")


model.eval()
with torch.no_grad():
    #Tính CrossEntropy cả mô hình
    toan_bo_output = model(x_test_tensor)
    loss_toan_bo = criterion(toan_bo_output, y_test_tensor)
    
    #Chỉ tính CrossEntropy ở lớp cuối
    # Trích xuất ngõ vào của lớp cuối (3 đặc trưng)
    ngo_vao_lop_cuoi = model[:-1](x_test_tensor) 
    
    #Chạy qua lớp cuối cùng nn.Linear(3, 2) để lấy ngõ ra
    ngo_ra_lop_cuoi = model[-1](ngo_vao_lop_cuoi)
    
    #Dùng CrossEntropy đánh giá ngõ ra đó với nhãn gốc
    loss_lop_cuoi = criterion(ngo_ra_lop_cuoi, y_test_tensor)

    #print("Weights của lớp cũ:", model[-1].weight)
    #mo_hinh_2 = nn.Linear(3, 2)
    #print("Weights của lớp mới:", mo_hinh_2.weight)

    print("\n--- ĐÁNH GIÁ COST FUNCTION LỚP CUỐI ---")
    print(f"Giá trị Loss khi chạy cả mô hình: {loss_toan_bo.item():.6f}")
    print(f"Giá trị Loss khi cắt riêng lớp cuối: {loss_lop_cuoi.item():.6f}")
    
    if abs(loss_toan_bo.item() - loss_lop_cuoi.item()) < 1e-5:
        print("=> Kết luận: Hai giá trị hoàn toàn khớp nhau!")

    test_outputs = model(x_test_tensor)
    predicted = torch.argmax(ngo_ra_lop_cuoi, dim=1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test)
    print(f"Test outputs: {ngo_ra_lop_cuoi}")
    print(f"Predicted classes: {predicted}")
    print("True classes:   ", y_test_tensor)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
        
    features = model[:6](x_test_tensor).numpy() 
    labels = y_test_tensor.numpy()

feat_class0 = features[labels == 0]
feat_class1 = features[labels == 1]

mean0 = np.mean(feat_class0, axis=0)
mean1 = np.mean(feat_class1, axis=0)

var0 = np.var(feat_class0, axis=0).mean()
var1 = np.var(feat_class1, axis=0).mean()

inter_dist = np.linalg.norm(mean0 - mean1)

print("--- KẾT QUẢ ĐÁNH GIÁ ĐẶC TRƯNG ---")
print(f"Mean Class 0: {mean0.round(4)}")
print(f"Mean Class 1: {mean1.round(4)}")
print(f"Phương sai nội lớp (Class 0): {var0:.4f}")
print(f"Phương sai nội lớp (Class 1): {var1:.4f}")
print(f"Khoảng cách giữa 2 lớp (Inter-class): {inter_dist:.4f}")

# ==========================================
# 5. TRỰC QUAN HÓA 3D (VISUALIZATION)
# ==========================================
fig = plt.figure(figsize=(10, 7))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(feat_class0[:, 0], feat_class0[:, 1], feat_class0[:, 2], 
           c='red', label='Class 0', alpha=0.7, edgecolors='w', s=50)

ax.scatter(feat_class1[:, 0], feat_class1[:, 1], feat_class1[:, 2], 
           c='blue', label='Class 1', alpha=0.7, edgecolors='w', s=50)

ax.scatter(mean0[0], mean0[1], mean0[2], c='black', marker='X', s=200, label='Mean Class 0')
ax.scatter(mean1[0], mean1[1], mean1[2], c='yellow', marker='X', s=200, edgecolors='black', label='Mean Class 1')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('Trực quan hóa 3 đặc trưng không gian (3D)')
plt.legend()
plt.show()