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

### Opis:
Program zgodnie z treścią zadania, posiada 10 reguł oceniania, oraz 6 możliwych ocen dla prowadzących. Na koniec ocenia trzech prowadzących, oraz wypisuje dla każdego wszystkie zalecenia, jeżeli takowe są. Trzeci prowadzący jest stworzony pod to, by użytkownik sam mógł wpisać oceny prowadzącego (można też go pominąć).

## Zadanie 6
### Treść zadania
Celem projektu jest zbudowanie modelu regresyjnego przewidującego przychód filmu na podstawie wybranych cech opisujących film, tj. budżet, popularność, gatunek, ocena użytkowników, oraz liczba głosów. W projekcie należy wykrozystać rzeczywisty zbiór danych **TMB 5000 Movie Dataset** dostępny pod adresem:
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Należy zbudować model regresyjnego przewidującego przychód filmu. Można wykorzystać wybraną metodę regresji, np. Linear Regression z biblioteki Scikit-learn w Pythonie.

### Opis:
Program webowy umożliwia predykcję przychodu filmu na podstawie danych wejściowych. Interfejs znajduje się w pliku `index.html` i wyświetla formularz z czterema polami:

- `Budżet` - kwota w dolarach USD, wpisywana jako liczba całkowita i formatowana spacjami.
- `Popularność` - liczba zmiennoprzecinkowa; wpisany przecinek zostanie automatycznie zamieniony na kropkę.
- `Ocena` - średnia ocena użytkowników w skali 0–10.
- `Liczba głosów` - całkowita liczba głosów oddanych na film.

Po kliknięciu przycisku `Oblicz` formularz wysyła dane do backendu w `app.py` na trasę `/predict`. Backend przetwarza dane, używa wytrenowanego modelu i zwraca przewidywany przychód w odpowiedzi JSON.

Wynik predykcji jest wyświetlany w elemencie `#result` na stronie, razem z formatowaniem walutowym w stylu polskim. W bocznym panelu `#modelInfo` wyświetlane są metryki modelu, takie jak trafność (R²) i średni błąd (MAE) w mln USD.

Dodatkowy przycisk `Użyj przykładowych danych` wypełnia formularz losowymi wartościami i natychmiast wykonuje predykcję.
---
#### Co robi `main.py`?
W folderze `6_Przewidywanie_przychodów_filmów_metodami_regresji` plik `main.py`:
- wczytuje dane z zestawu TMDB (`tmdb_5000_movies.csv`) przy użyciu `kagglehub` i biblioteki `pandas`,
- wybiera tylko kolumny `budget`, `popularity`, `vote_average`, `vote_count` oraz `revenue`,
- usuwa wiersze z brakującymi wartościami i rekordy z zerowym budżetem lub przychodem,
- tworzy nowe cechy `budget_log` i `popularity_log` jako logarytmowane wersje danych, aby lepiej dopasować rozkład cech,
- przekształca zmienną docelową `revenue` za pomocą logarytmu `np.log1p`, aby model uczył się lepiej stabilizowanej wartości,
- dzieli dane na zbiór treningowy i testowy,
- normalizuje cechy numeryczne za pomocą `StandardScaler`,
- trenuje cztery modele: `LinearRegression`, `RandomForestRegressor`, `Ridge` i `SVR`,
- ocenia ich jakość miernikami `MAE` i `R2` po odwróceniu logarytmu do oryginalnej skali przychodu,
- generuje wykresy porównujące rzeczywiste i przewidywane przychody oraz zapisuje je do pliku `porownanie_regresji.png`,
- zapisuje wytrenowany model `LinearRegression` do `model.pkl`, skaler do `skaler.pkl` oraz metryki do pliku `metryki.txt`.

#### Co robi `app.py`?*
Plik `app.py` uruchamia serwer Flask:
- ładuje zapisany model (`model.pkl`) oraz skaler (`skaler.pkl`) przy starcie,
- udostępnia endpoint `/predict`, który akceptuje żądanie `POST` z danymi JSON,
- oblicza dodatkowe cechy `budget_log` i `popularity_log` z przesłanych wartości,
- skaluje cechy przy użyciu wcześniej wyuczonego skalera,
- wykonuje predykcję przychodu za pomocą modelu i odwraca logarytm `np.expm1`, aby zwrócić przychód w oryginalnej skali,
- zwraca wynik jako JSON z polem `prediction`.

`main.py` przygotowuje i zapisuje model, a `app.py` obsługuje zapytania z interfejsu `index.html` i zwraca gotowe przewidywania.

