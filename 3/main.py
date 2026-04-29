import numpy as np
from scipy import stats

while (True):
    print("Wybierz opcję:")
    print("1: Wpisz własne dane")
    print("2: Przykład z wykryciem dryfu")
    print("3: Przykład bez wykrycia dryfu")
    print("4: Zakończ")
    choice = input("> ")

    # ==========================================
    # Dane
    # ==========================================
    # Wpisanie własnych danych przez użytkownika
    if choice == "1":    
        print("\nPodaj dane dla X1 (oddzielone spacją):")    
        X1 = np.array(list(map(float, input().split())))    

        print("Podaj dane dla X2 (oddzielone spacją):")    
        X2 = np.array(list(map(float, input().split())))

    elif choice == "3":
        # Dane na niewykrycie dryfu (X2 podobne do X1)
        X1 = np.array([2, 3, 2, 4, 3]) # dane referencyjne
        X2 = np.array([2, 3, 2, 4, 8]) # dane aktualne

    elif choice == "2":
        # Dane na wykrycie dryfu (X2 różne od X1)
        X1 = np.array([2, 2, 3, 3, 2])
        X2 = np.array([4, 5, 10, 10, 10])

    elif choice == "4":
        print("Koniec programu.")
        break

    else:
        print("Nieprawidłowy wybór. Spróbuj ponownie.")
        continue

    # ==========================================
    # Parametr testu
    # ==========================================
    alpha = 0.05  # poziom istotności 
    # (aby wystąpił dryf w klasycznym i brak dryfu w prostym, należy zmienić parametr alpha na 0.01 i wybrać opcję 2)

    # ==========================================
    # Statystyki
    # ==========================================
    n1, n2 = len(X1), len(X2)

    mu1, mu2 = np.mean(X1), np.mean(X2)
    var1 = np.var(X1, ddof=1)
    var2 = np.var(X2, ddof=1)

    print("\n=== STATYSTYKI ===")
    print(f"Średnia X1: {mu1:.3f}")
    print(f"Średnia X2: {mu2:.3f}")
    print(f"Wariancja X1: {var1:.3f}")
    print(f"Wariancja X2: {var2:.3f}")

    # ==========================================
    # Statystyka t (jak w zadaniu)
    # ==========================================
    while (True):
         print("\nWybierz wersję testu t:")
         print("1: Prosta (nie uwzględnia różnic w rozmiarach próbek i wariancjach)")
         print("2: Klasyczna (uwzględnia wspólną wariancję i rozmiary próbek)")
         version = input("> ")

         if version in ["1", "2"]:
             break
         else:
             print("Nieprawidłowy wybór. Spróbuj ponownie.")

    if version == "1":
        # WERSJA PROSTA
        # (nie uwzględnia różnic w rozmiarach próbek i wariancjach)
        T = abs(mu1 - mu2) / np.sqrt((var1 + var2) / 2)

    else:
        # WERSJA KLASYCZNA
        # wspólna wariancja (sp2) i uwzględnienie rozmiarów próbek
        sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

        # statystyka t
        T = abs(mu1 - mu2) / np.sqrt(sp2 * (1/n1 + 1/n2))

    # stopnie swobody
    df = n1 + n2 - 2

    # próg z rozkładu t (test dwustronny)
    t_crit = stats.t.ppf(1 - alpha/2, df)

    print("\n=== TEST t-STUDENTA ===")
    print(f"T = {T:.3f}")
    print(f"t_crit = {t_crit:.3f} (alpha={alpha}, df={df})")

    # ==========================================
    # Decyzja
    # ==========================================
    print("\n=== DECYZJA ===")
    if T > t_crit:
        print("Dryf wykryty (różnica istotna statystycznie)")
    else:
        print("Brak dryfu")

    # ==========================================
    # (Opcjonalnie) klasyczny test t z SciPy
    # ==========================================
    t_stat, p_value = stats.ttest_ind(X1, X2, equal_var=True)

    print("\n=== WERYFIKACJA (scipy.stats.ttest_ind) ===")
    print(f"t_stat = {t_stat:.3f}")
    print(f"p_value = {p_value:.5f}")

    if p_value < alpha:
        print("Dryf wykryty (p-value < alpha)")
    else:
        print("Brak dryfu")
    print("\n" + "="*40 + "\n")