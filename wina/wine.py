import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Wczytywanie danych
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

# 2. Standaryzacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Redukcja wymiarów za pomocą PCA
pca = PCA(n_components=2)  # Redukcja do 2 wymiar
X_pca = pca.fit_transform(X_scaled)

# 4. Informacja o wariancji wyjaśnionej przez PCA
print(f'Wariancja wyjaśniona przez 2 główne komponenty: {pca.explained_variance_ratio_.sum():.2f}')
print(f"Suma wyjaśnionej wariancji: {pca.explained_variance_ratio_}")

# 5. Wagi cech w składowych PCA
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=feature_names)
print("\nWagi cech w składowych PCA:")
print(loadings)

# 6. Wykres PCA
plt.figure(figsize=(8, 6))
for class_value, class_name in enumerate(target_names):
    plt.scatter(
        X_pca[y == class_value, 0], 
        X_pca[y == class_value, 1],
        label=class_name,
        alpha=0.7
    )

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Wino')
plt.legend()
plt.grid(True)
plt.show()

# 7. Najważniejsze cechy dla PC1
print(f"\n Najważniejsze cechy dla PC1:")
print(loadings['PC1'].abs().sort_values(ascending=False))