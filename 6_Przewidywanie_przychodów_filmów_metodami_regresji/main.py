import pandas as pd
import numpy as np
import joblib
import ast
from kagglehub import dataset_load, KaggleDatasetAdapter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

import matplotlib.pyplot as plt

# Wczytanie danych

df = dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "tmdb/tmdb-movie-metadata",
    "tmdb_5000_movies.csv"
)

# Wyświetlenie podstawowych informacji o danych
print("===================================")
print("\nPODSTAWOWE INFORMACJE\n")
print("-----------------------------------")
print(df.info())

# Wyświetlenie statystyk opisowych
print("\n===================================")
print("\nSTATYSTYKI OPISOWE\n")
print("-----------------------------------")
print(df.describe())

# Wybór kolumn
df = df[['budget', 'popularity', 'vote_average', 'vote_count', 'revenue']]

# Czyszczenie danych
df = df.dropna()

# Usunięcie rekordów z zerowym budżetem lub przychodem
df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
df = df.reset_index(drop=True)

# Logarytmowanie kolumn 'budget' i 'popularity' dla lepszej dystrybucji danych
df['budget_log'] = np.log1p(df['budget'])
df['popularity_log'] = np.log1p(df['popularity'])

# Przygotowanie danych do modelowania
X = df.drop(columns=['revenue'])
y = np.log1p(df['revenue'])

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Skalowanie danych numerycznych
scaler = StandardScaler()

num_cols = ['budget', 'popularity', 'vote_average',
            'vote_count', 'budget_log', 'popularity_log']

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


# Budowa modelu regresji liniowej
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_model.fit(X_train, y_train)

# Predykcja
lr_y_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)
svr_pred = svr_model.predict(X_test)

# cofnięcie log
y_test_real = np.expm1(y_test)

lr_y_pred_real = np.expm1(lr_y_pred)
rf_pred_real = np.expm1(rf_pred)
ridge_pred_real = np.expm1(ridge_pred)
svr_pred_real = np.expm1(svr_pred)

# Ewaluacja
# Mean Absolute Error (MAE) to miara błędu, 
# która oblicza średnią wartość bezwzględnych różnic 
# między rzeczywistymi a przewidywanymi wartościami. Im niższa wartość MAE, 
# tym lepsze dopasowanie modelu do danych.

# R2 (R-squared) to miara, 
# która wskazuje, jaka część wariancji w danych 
# jest wyjaśniona przez model. 
# Wartość R2 wynosząca 1 oznacza idealne dopasowanie, 
# podczas gdy wartość 0 oznacza, że model nie wyjaśnia żadnej z wariancji.

lr_mae = mean_absolute_error(y_test_real, lr_y_pred_real)
lr_r2 = r2_score(y_test_real, lr_y_pred_real)

rf_mae = mean_absolute_error(y_test_real, rf_pred_real)
rf_r2 = r2_score(y_test_real, rf_pred_real)

ridge_mae = mean_absolute_error(y_test_real, ridge_pred_real)
ridge_r2 = r2_score(y_test_real, ridge_pred_real)

svr_mae = mean_absolute_error(y_test_real, svr_pred_real)
svr_r2 = r2_score(y_test_real, svr_pred_real)

print("\n===================================")
print("\nPORÓWNANIE MODELI\n")
print("-----------------------------------")
print(f"Linear Regression")
print(f"MAE: {lr_mae:.2f}, R2: {lr_r2:.4f}")

print(f"\nRandom Forest")
print(f"MAE: {rf_mae:.2f}, R2: {rf_r2:.4f}")

print(f"\nRidge Regression")
print(f"MAE: {ridge_mae:.2f}, R2: {ridge_r2:.4f}")

print(f"\nSupport Vector Regression")
print(f"MAE: {svr_mae:.2f}, R2: {svr_r2:.4f}")
print("-----------------------------------")

# GENEROWANIE WSZYSTKICH 4 WYKRESÓW W JEDNYM OKNIE / PLIKU
# Tworzymy siatkę 2x2 na 4 wykresy o wymiarach 14x12 cali
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Lista modeli, ich predykcji oraz kolorów dla wykresów
models_data = [
    {"name": "Linear Regression", "pred": lr_y_pred_real, "ax": axes[0, 0], "color": "blue"},
    {"name": "Random Forest", "pred": rf_pred_real, "ax": axes[0, 1], "color": "green"},
    {"name": "Ridge Regression", "pred": ridge_pred_real, "ax": axes[1, 0], "color": "orange"},
    {"name": "Support Vector Regression (SVR)", "pred": svr_pred_real, "ax": axes[1, 1], "color": "purple"}
]

# Automatyczne rysowanie każdego wykresu w pętli
for m in models_data:
    ax = m["ax"]
    
    # Rysowanie punktów (Rzeczywiste vs Przewidywane)
    ax.scatter(y_test_real, m["pred"], alpha=0.5, color=m["color"])
    
    # Rysowanie czerwonej linii idealnej predykcji
    ax.plot([y_test_real.min(), y_test_real.max()],
            [y_test_real.min(), y_test_real.max()], 
            color='red', linestyle='--', linewidth=2)
    
    # Opisy osi i tytuły dla każdego podwykresu
    ax.set_title(m["name"], fontsize=12, fontweight='bold')
    ax.margins(0.05)
    ax.set_xlabel("Rzeczywisty przychód", fontsize=10)
    ax.set_ylabel("Przewidywany przychód", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    

# Dopasowanie układu, aby podpisy na siebie nie nachodziły
plt.subplots_adjust(left=0.1, right=0.95, top=0.93, bottom=0.1, wspace=0.3, hspace=0.35)

# Zapisanie całego zestawu wykresów do jednego pliku graficznego
plt.savefig("porownanie_regresji.png", dpi=300)

# Wyświetlenie okna z wykresami
plt.show()

# Po wynikach wychodzi, że regresja Linear oraz Ridge są bardzo podobne.
# Wybrany więc zostaje Linear Regression, ponieważ jest prostsza w implementacji,
# łatwiejsza do interpretacji oraz nie wymaga dodatkowego strojenia parametrów.

# Zapisanie nauczonego modelu i skalera do plików
try: 
    joblib.dump(lr_model, 'model.pkl')
    joblib.dump(scaler, 'skaler.pkl')
    print("\nPliki z modelem (model.pkl) i skalerem (skaler.pkl) zostały zapisane na dysku!")
except Exception as e:
    print(f"\nWystąpił błąd podczas zapisywania modelu (model.pkl) lub skalera (skaler.pkl): {e}")

# Zapisanie metryk modelu do pliku tekstowego
try:
    with open("metryki.txt", "w") as f:
        f.write("Model: Linear Regression\n")
        f.write(f"MAE: {lr_mae:.2f}\n")
        f.write(f"R2: {lr_r2:.4f}\n")
    print("Metryki modelu zostały zapisane do metryki.txt!")
except Exception as e:
    print(f"Wystąpił błąd podczas zapisywania metryk modelu do metryki.txt: {e}")
print("\n===================================")