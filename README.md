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
