import pandas as pd
import numpy as np
import ast
from kagglehub import dataset_load, KaggleDatasetAdapter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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
df = df[['budget', 'popularity', 'genres',
         'vote_average', 'vote_count', 'revenue']]


# Czyszczenie danych
df = df.dropna()

# Usunięcie rekordów z zerowym budżetem lub przychodem
df = df[(df['budget'] > 0) & (df['revenue'] > 0)]


# Parsowanie gatunków filmów (kolumna 'genres' jest w formacie string, który reprezentuje listę słowników)
def parse_genres(x):
    genres = ast.literal_eval(x)
    return [g['name'] for g in genres]

df['genres'] = df['genres'].apply(parse_genres)


# Kodowanie gatunków filmów
mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(
    mlb.fit_transform(df['genres']),
    columns=mlb.classes_
)

df = df.reset_index(drop=True)
genres_encoded = genres_encoded.reset_index(drop=True)

df = pd.concat([df, genres_encoded], axis=1)
df = df.drop(columns=['genres'])


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
model = LinearRegression()

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# Predykcja
y_pred = model.predict(X_test)

# cofnięcie log
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred)

rf_pred = rf_model.predict(X_test)
rf_pred_real = np.expm1(rf_pred)

# Ewaluacja
# Mean Absolute Error (MAE) to miara błędu, która oblicza średnią wartość bezwzględnych różnic między rzeczywistymi a przewidywanymi wartościami. Im niższa wartość MAE, tym lepsze dopasowanie modelu do danych.
mae = mean_absolute_error(y_test_real, y_pred_real)
# R2 (R-squared) to miara, która wskazuje, jaka część wariancji w danych jest wyjaśniona przez model. Wartość R2 wynosząca 1 oznacza idealne dopasowanie, podczas gdy wartość 0 oznacza, że model nie wyjaśnia żadnej z wariancji.
r2 = r2_score(y_test_real, y_pred_real)

rf_mae = mean_absolute_error(y_test_real, rf_pred_real)
rf_r2 = r2_score(y_test_real, rf_pred_real)

print("\n===================================")
print("\nWYNIKI DLA MODELU REGRESJI LINIOWEJ\n")
print("-----------------------------------")
print("MAE:", round(mae, 2))
print("R2:", round(r2, 4))
print("===================================")


print("\n===================================")
print("\nWYNIKI DLA MODELU RANDOM FOREST\n")
print("-----------------------------------")
print("MAE:", round(rf_mae, 2))
print("R2:", round(rf_r2, 4))

print("\n===================================")
print("PORÓWNANIE MODELI")
print("-----------------------------------")
print(f"Linear Regression  -> MAE: {mae:.2f}, R2: {r2:.4f}")
print(f"Random Forest      -> MAE: {rf_mae:.2f}, R2: {rf_r2:.4f}")
print("===================================")

# Przykładowe predykcje
results = pd.DataFrame({
    "Realny_przychod": y_test_real,
    "Przewidywany_przychod": y_pred_real
})

print(results.head())

# Wykres porównania
plt.figure()

plt.scatter(y_test_real, y_pred_real)
plt.xlabel("Rzeczywisty przychód")
plt.ylabel("Przewidywany przychód")
plt.title("Rzeczywisty vs Przewidywany przychód")

# linia idealna
plt.plot([y_test_real.min(), y_test_real.max()],
         [y_test_real.min(), y_test_real.max()])

plt.show()
