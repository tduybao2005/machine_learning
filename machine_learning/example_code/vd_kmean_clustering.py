import numpy as np
import matplotlib.pyplot as plt

X = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14]
c1, c2 = np.random.choice(X, size=2, replace=False)

epochs = 5
for epoch in range(epochs):
    z1, z2 = 0, 0
    class1, class2 = [], []

    for i in X:
        a = abs(c1 - i)
        b = abs(c2 - i)
        if a < b:
            class1.append(i)
            z1 += i
        else:
            class2.append(i)
            z2 += i

    c1 = z1 / len(class1)
    c2 = z2 / len(class2)

print(f"Cluster 1: {class1}")
print(f"Cluster 2: {class2}")
print(f"Tâm cụm 1 = {c1}")
print(f"Tâm cụm 2 = {c2}")

# Vẽ
plt.scatter(class1, [0] * len(class1), color='blue', label=f'Cluster 1', s=100)
plt.scatter(class2, [0] * len(class2), color='red', label=f'Cluster 2', s=100)
plt.scatter([c1], [0], color='blue', marker='x', s=200, linewidths=3, label=f'Centroid 1 ({c1:.1f})')
plt.scatter([c2], [0], color='red', marker='x', s=200, linewidths=3, label=f'Centroid 2 ({c2:.1f})')

plt.xlabel('X')
plt.yticks([])
plt.title('K-means Clustering (k=2)')
plt.legend()
plt.tight_layout()
plt.show()
