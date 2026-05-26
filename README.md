# Biometryczne systemy rozpoznawania użytkownika i kontroli dostępu
---
# Jowita Kruk, 339351

### Jak uruchomić zadania:
1. Będąc w folderze głównym w terminalu, należy uruchomić środowisko wirtualne za pomocą instrukcji: `python -m venv .venv`.
2. Następnie użyć instrukcji:
- dla Widows: `.venv\Scripts\activate`,
- dla MacOS\Linux: `source .venv/bin/activate`.
3. Po aktywowaniu środowiska, trzeba zainstalować zależności: `pip install -r requirements.txt`.
3. Następnie przejść do folderu z zadaniem (`cd nazwa`).
4. Na koniec uruchomić program, używając instrukcji: `python main.py`.

## Zadanie 1
### Treść:
Na podstawie programu rozpoznającego cyfry od `0` do `9` należy opracować system rozpoznawania ręcznie pisanych liter od `A` do `F`.

### Opis:
Program ten realizuje rozpoznawanie ręcznie pisanych liter od 'A' do 'F', wykorzystując zbiór EMNIST Letters *(ograniczony do klas A-F)* oraz swojego własnego zbioru napisanych liter, znajdujących się w folderze ,,moje_litery''.

## Zadanie 2
### Treść:
Na podstawie programu realizującego transformację PCA należy wykonać **transformację odwrotną** po redukcji wymiarowości danych poprzez **usunięcie składowej o najmniejszej wariancji** oraz **odtworzyć oryginalną macierz danych**. Następnie należy **obliczyć różnice** pomiędzy wartościami pierwotnymi a zrekonstruowanymi.

### Opis:
Program realizuje transformację PCA dla zadanego zbioru danych, a następnie wykonuje jej transformację odwrotną po redukcji wymiarowości poprzez usunięcie składowej o najmniejszej wariancji. Na podstawie zredukowanych danych odtwarzana jest oryginalna macierz cech, a następnie analizowane są różnice pomiędzy wartościami pierwotnymi i zrekonstruowanymi.

## Zadanie 3
### Treść:
Za pomocą kodu `zadanie_dryf.py` oraz załączonego do zadania pliku *„Dryf danych w zbiorze”* należy przeprowadzić **klasyczny test t‑Studenta** w celu sprawdzenia występowania dryfu danych pomiędzy zbiorem referencyjnym a zbiorem aktualnym.

### Opis:
Program oblicza średnie oraz wariancje dla dwóch zbiorów danych, a następnie wyznacza wartość statystyki testowej t‑Studenta. Na podstawie otrzymanego wyniku oraz zadanego poziomu istotności podejmowana jest decyzja o występowaniu lub braku dryfu danych. Otrzymany rezultat zostaje zweryfikowany przy użyciu funkcji ttest_ind z biblioteki SciPy.
Dodatkowo do programu zostało dodane menu wyboru, jakie dane mają zostać użyte oraz wersja testu t. W celu pokazania różnicy obu tych wersji, należy zmienić parametr alpha z `0.05` (5%) na `0.01` (1%) oraz wybrać opcję `2`: *Przykład z wykryciem dryfu* (brak dryfu dla wersji prostej, wykrycie dryfu dla wersji klasycznej).

## Zadanie 4
### Treść:
Za pomocą kodu `regresja_liniowa.py` oraz załączonego do zadania pliku *„Wyjaśnienia”* należy przeprowadzić **ręczne wyliczenia regresji liniowej** oraz dostać jak najbardziej możliwie podobny wykres z zautomatyzowanego.

### Opis:
Program ręcznie oblicza regresję liniową. Program wydaje się bardziej obszerny oraz minimalnie zmienił się wyraz wolny (b) niż w programie obliczającym automatycznie OLS.

## Zadanie 5
### Treść zadania
Celem zadania jest opracowanie systemu ekspertowego, który na podstawie odpowiedzi studenta ocenia jakość prowadzenia zajęć przez prowadzącego. System powinien wykorzystywać reguły typu IF warunek THEN decyzja

## Zadanie 6
### Treść zadania
Celem projektu jest zbudowanie modelu regresyjnego przewidującego przychód filmu na podstawie wybranych cech opisujących film, tj. budżet, popularność, gatunek, ocena użytkowników, oraz liczba głosów. W projekcie należy wykrozystać rzeczywisty zbiór danych **TMB 5000 Movie Dataset** dostępny pod adresem:
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Należy zbudować model regresyjnego przewidującego przychód filmu. Można wykorzystać wybraną metodę regresji, np. Linear Regression z biblioteki Scikit-learn w Pythonie.