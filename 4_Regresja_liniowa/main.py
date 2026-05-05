import numpy as np
import matplotlib.pyplot as plt

# Dane treningowe (x – powierzchnia [m²], y – cena [tys. zł])
X = np.array([30, 50, 70, 90, 110, 130])
y = np.array([180, 250, 310, 370, 450, 480])

# ==============================
# RĘCZNE OBLICZENIE REGRESJI LINIOWEJ (OLS)
#
# Model: y = a * x + b
# Klasyczna metoda najmniejszych kwadratów (OLS), czyli minimalizacja sumy:
# Sum_(y_i − (a * x_i + b))²
# 
# Wzory analityczne:
# a = (n * Sum_(xy) − Sum_x * Sum_y) / (n * Sum_(x²) − (Sum_x)²)
# b = (Sum_y − a * Sum_x) / n
# ============================== 

n = len(X)
sum_x = np.sum(X)
sum_y = np.sum(y)
sum_xy = np.sum(X * y)
sum_x2 = np.sum(X ** 2)

# Obliczenie współczynnika kierunkowego (a) i wyrazu wolnego (b)
a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
b = (sum_y - a * sum_x) / n

# Wyświetlenie współczynników
print("Współczynnik kierunkowy (a):", a)
print("Wyraz wolny (b):", b)

# Predykcja dla domu o powierzchni 100 m^2
x_new = 100
y_pred = a * x_new + b
print(f"Szacowana cena dla 100 m²: {y_pred:.2f} tys. zł")

# Obliczenie wartości przewidywanych dla wszystkich punktów treningowych
y_predicted = a * X + b

# Wizualizacja
plt.scatter(X, y, color='blue', label='Dane uczące')
plt.plot(X, y_predicted, color='red', label='Regresja liniowa')
plt.scatter(x_new, y_pred, color='green', s=100, label='Nowy punkt (100 m²)')
plt.xlabel("Powierzchnia [m²]")
plt.ylabel("Cena [tys. zł]")
plt.title("Prosty model regresji liniowej")
plt.legend()
plt.show()