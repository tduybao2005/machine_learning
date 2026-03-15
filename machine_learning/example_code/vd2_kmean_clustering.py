import numpy as np
import matplotlib.pyplot as plt

X = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14]

# Epoch 0: khởi tạo bằng random class
classs = [np.random.randint(1, 3) for _ in X]
print(f"Số điểm class 1: {classs.count(1)}")
print(f"Số điểm class 2: {classs.count(2)}")
epochs = 10
for epoch in range(epochs):
    z1, z2 = 0, 0
    num1, num2 = 0, 0

    for idx, i in enumerate(X):
        if classs[idx] == 1:
            z1 += i
            num1 += 1
        else:
            z2 += i
            num2 += 1

    c1 = z1 / num1
    c2 = z2 / num2

    # Assign theo khoảng cách đến centroid
    classs = [1 if abs(i - c1) < abs(i - c2) else 2 for i in X]

print(classs)
print("c1 =", c1)
print("c2 =", c2)
print(f"Số điểm class 1: {classs.count(1)}")
print(f"Số điểm class 2: {classs.count(2)}")

# Vẽ
cluster1 = [X[i] for i in range(len(X)) if classs[i] == 1]
cluster2 = [X[i] for i in range(len(X)) if classs[i] == 2]

plt.scatter(cluster1, [0] * len(cluster1), color='blue', label='Cluster 1', s=100)
plt.scatter(cluster2, [0] * len(cluster2), color='red', label='Cluster 2', s=100)
plt.scatter([c1], [0], color='blue', marker='x', s=200, linewidths=3, label=f'Centroid 1 ({c1:.1f})')
plt.scatter([c2], [0], color='red', marker='x', s=200, linewidths=3, label=f'Centroid 2 ({c2:.1f})')

plt.xlabel('X')
plt.yticks([])
plt.title('K-means Clustering (k=2)')
plt.legend()
plt.tight_layout()
plt.show()
