import numpy as np 
import matplotlib.pyplot as plt 

X = [i for i in range (10)]
# print(X)
c1, c2, c3 = np.random.choice(X, size= 3, replace= False)
print(f"c1_bandau = {c1}")
print(f"c2_bandau = {c2}")
print(f"c3_bandau = {c3}")

# print(classs)
# print(len(classs))
epochs = 10
for epoch in range(epochs):
    z1, z2, z3 = 0, 0, 0
    class1, class2, class3 = [], [], []
    for i in X:
        a = abs(c1 - i + 1.5)
        b = abs(c2 - i + 0.5)
        c = abs(c3 - i + 0.5)
        if a < b and a < c:
            class1.append(i)
            z1 += i
        elif b < a and b <= c:
            class2.append(i)
            z2 += i
        else:
            class3.append(i)
            z3 += i
    c1 = z1 / len(class1)
    c2 = z2 / len(class2)
    c3 = z3 / len(class3) 


print(f"c1 = {c1}")
print(f"c2 = {c2}")
print(f"c3 = {c3}")


# Vẽ từng điểm của từng cụm
plt.figure(figsize=(8, 2))
plt.scatter(class1, [0]*len(class1), color='blue',  s=200, label=f'Cluster 1 ({len(class1)} điểm)')
plt.scatter(class2, [0]*len(class2), color='green', s=200, label=f'Cluster 2 ({len(class2)} điểm)')
plt.scatter(class3, [0]*len(class3), color='red',   s=200, label=f'Cluster 3 ({len(class3)} điểm)')

# Vẽ tâm cụm
plt.scatter([c1], [0], color='blue',  marker='x', s=300, linewidths=3, zorder=5, label='Centroid 1')
plt.scatter([c2], [0], color='green', marker='x', s=300, linewidths=3, zorder=5, label='Centroid 2')
plt.scatter([c3], [0], color='red',   marker='x', s=300, linewidths=3, zorder=5, label='Centroid 3')
plt.text(c1, 0.08, f'{c1:.2f}', ha='center', color='blue',  fontsize=10, fontweight='bold')
plt.text(c2, 0.08, f'{c2:.2f}', ha='center', color='green', fontsize=10, fontweight='bold')
plt.text(c3, 0.08, f'{c3:.2f}', ha='center', color='red',   fontsize=10, fontweight='bold')

plt.yticks([])
plt.xlabel('Giá trị X')
plt.title('Phân cụm và tâm cụm')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
