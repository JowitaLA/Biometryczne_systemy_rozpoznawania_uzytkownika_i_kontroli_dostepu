import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generowanie syntetycznego zbioru danych (50 cech, ale tylko 5 istotnych)
X, y = make_regression(
    n_samples=150,
    n_features=50,
    n_informative=5,
    noise=15.0,
    random_state=42
)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Standaryzacja danych (Kluczowa dla metod regularyzacji!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Bazowy: Klasyczna Regresja Liniowa (OLS)
ols_model = LinearRegression()
ols_model.fit(X_train_scaled, y_train)

y_pred_ols = ols_model.predict(X_test_scaled)
mse_ols = mean_squared_error(y_test, y_pred_ols)
r2_ols = r2_score(y_test, y_pred_ols)

# 4. Model Zaawansowany: Regresja Lasso z Walidacją Krzyżową (LassoCV)
# LassoCV automatycznie dobierze optymalny parametr alpha (siłę kary)
lasso_model = LassoCV(cv=5, random_state=42)
lasso_model.fit(X_train_scaled, y_train)

y_pred_lasso = lasso_model.predict(X_test_scaled)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

# 5. Podsumowanie wyników
print("--- WYNIKI MODELU OLS ---")
print(f"MSE: {mse_ols:.2f}")
print(f"R2 Score: {r2_ols:.4f}")

print("\n--- WYNIKI MODELU LASSO ---")
print(f"Optymalne Alpha: {lasso_model.alpha_:.4f}")
print(f"MSE: {mse_lasso:.2f}")
print(f"R2 Score: {r2_lasso:.4f}")

# Sprawdzenie liczby wyzerowanych cech
zeroed_features = np.sum(lasso_model.coef_ == 0)
total_features = X.shape[1]

print(f"\nLasso wyzerowało {zeroed_features} z {total_features} cech.")

# 6. Wizualizacja współczynników
plt.figure(figsize=(10, 5))
plt.stem(ols_model.coef_, linefmt='r-', markerfmt='ro', label='OLS Coefficients')
plt.stem(lasso_model.coef_, linefmt='g-', markerfmt='go', label='Lasso Coefficients')
plt.title("Porównanie wag modelu OLS i Lasso")
plt.xlabel("Indeks cechy")
plt.ylabel("Wartość współczynnika")
plt.legend()
plt.show()