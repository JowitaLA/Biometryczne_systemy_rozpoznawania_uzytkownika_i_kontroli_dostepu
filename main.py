
"""
Klasyfikator liter a–f z perceptronem wielowarstwowym (MLP) z interfejsem tekstowym.

Obsługiwane tryby treningu:
  [1] EMNIST + moje litery (katalog 'moje_litery/')
  [2] Tylko moje litery (bez EMNIST)

Struktura katalogu z literami (po polsku, ale bez ogonków):

    moje_litery/
        a/
        b/
        c/
        d/
        e/
        f/

Nazwa folderu => etykieta klasy (a..f).
"""
import warnings
warnings.filterwarnings("ignore")
import os
import gzip
import glob
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. Pobieranie i wczytywanie EMNIST (pliki IDX .gz z mirrora)
# ============================================================

FILES = {
    "train_images": (
        "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip/eMNIST-letters-train-images-idx3-ubyte.gz",
        "eMNIST-letters-train-images-idx3-ubyte.gz",
    ),
    "train_labels": (
        "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip/eMNIST-letters-train-labels-idx1-ubyte.gz",
        "eMNIST-letters-train-labels-idx1-ubyte.gz",
    ),
    "test_images": (
        "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip/eMNIST-letters-test-images-idx3-ubyte.gz",
        "eMNIST-letters-test-images-idx3-ubyte.gz",
    ),
    "test_labels": (
        "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip/eMNIST-letters-test-labels-idx1-ubyte.gz",
        "eMNIST-letters-test-labels-idx1-ubyte.gz",
    ),
}

def download(url: str, filename: str) -> None:
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"Plik już istnieje: {filename} (rozmiar: {os.path.getsize(filename)} B)")
        return

    print(f"Pobieram {url} -> {filename}")
    try:
        urllib.request.urlretrieve(url, filename)
    except Exception as e:
        print(f"Błąd podczas pobierania {url}: {e}")
        raise


def load_emnist(data_dir: str = "emnist_data"):
    """
    Pobiera (jeśli trzeba) i wczytuje dane EMNIST z plików IDX (.gz).

    Zwraca:
        (X_train, y_train), (X_test, y_test)
    """
    os.makedirs(data_dir, exist_ok=True)

    # 1. Pobranie plików (jeśli trzeba)
    for key, (url, fname) in FILES.items():
        path = os.path.join(data_dir, fname)
        download(url, path)

    # 2. Funkcje pomocnicze do odczytu formatu IDX
    
    def read_images(path: str) -> np.ndarray:
        with gzip.open(path, "rb") as f:
            magic = int.from_bytes(f.read(4), "big")
            if magic != 2051:
                raise ValueError(f"Zły magic number w {path}: {magic}")
            n_images = int.from_bytes(f.read(4), "big")
            n_rows = int.from_bytes(f.read(4), "big")
            n_cols = int.from_bytes(f.read(4), "big")
            data = f.read(n_images * n_rows * n_cols)
            images = np.frombuffer(data, dtype=np.uint8)
            images = images.reshape(n_images, n_rows * n_cols)

            images = images.reshape(n_images, 28, 28)
            images = np.rot90(images, k=1, axes=(1, 2))
            images = images.reshape(n_images, 784)

            return images

    def read_labels(path: str) -> np.ndarray:
        with gzip.open(path, "rb") as f:
            magic = int.from_bytes(f.read(4), "big")
            if magic != 2049:
                raise ValueError(f"Zły magic number w {path}: {magic}")
            n_labels = int.from_bytes(f.read(4), "big")
            data = f.read(n_labels)
            labels = np.frombuffer(data, dtype=np.uint8)
            return labels

    train_images_path = os.path.join(data_dir, FILES["train_images"][1])
    train_labels_path = os.path.join(data_dir, FILES["train_labels"][1])
    test_images_path = os.path.join(data_dir, FILES["test_images"][1])
    test_labels_path = os.path.join(data_dir, FILES["test_labels"][1])

    X_train = read_images(train_images_path)
    y_train = read_labels(train_labels_path) - 1  # EMNIST Letters: etykiety 1..26 => przesuwamy do 0..25
    X_test = read_images(test_images_path)
    y_test = read_labels(test_labels_path) - 1  # EMNIST Letters: etykiety 1..26 => przesuwamy do 0..25


    # 3. Filtrujemy tylko litery A–F (0..5), resztę pomijamy
    mask_train = y_train < 6
    mask_test  = y_test < 6

    X_train = X_train[mask_train]
    y_train = y_train[mask_train]

    X_test = X_test[mask_test]
    y_test = y_test[mask_test]
    return (X_train, y_train), (X_test, y_test)


# ============================================================
# 2. Wspólny preprocessing jednego obrazka do formatu 28x28
# ============================================================

def preprocess_image_to_canvas28(image_path: str) -> np.ndarray:
    """
    Wczytuje rysunek litery (np. z Painta), przycina do litery,
    centruje w 28x28 i normalizuje do [0,1], tak jak EMNIST.
    Zwraca tablicę [28,28].
    """
    # 1. Wczytanie obrazu i konwersja do skali szarości
    img = Image.open(image_path).convert("L")  # grayscale

    # 2. Zamiana na tablicę, odwrócenie kolorów (litera jasna, tło ciemne)
    arr = np.array(img)               # [H, W], 0..255, litera CZARNA na białym tle
    arr = 255 - arr                   # litera JASNA, tło ciemne (jak w EMNIST)

    # 3. Progowanie, żeby pozbyć się szumu
    thresh = 80
    bin_arr = (arr > thresh).astype(np.uint8)  # 1 = litera, 0 = tło

    # 4. Bounding box litery
    coords = np.argwhere(bin_arr == 1)
    if coords.size == 0:
        raise ValueError(f"Nie znalazłem żadnych jasnych pikseli w {image_path}")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1   # +1 bo górna granica jest wyłączna

    digit = arr[y0:y1, x0:x1]         # rdzeń litery

    # 5. Przeskalowanie rdzenia do maks. 20x20 z zachowaniem proporcji
    digit_img = Image.fromarray(digit.astype(np.uint8))
    digit_img.thumbnail((20, 20), Image.Resampling.LANCZOS)
    digit_resized = np.array(digit_img)  # [h, w]

    h, w = digit_resized.shape

    # 6. Płótno 28x28 i centrowanie
    canvas = np.zeros((28, 28), dtype=np.float32)
    y_off = (28 - h) // 2
    x_off = (28 - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = digit_resized.astype(np.float32)

    # 7. Normalizacja
    canvas /= 255.0

    return canvas  # [28, 28]


# ============================================================
# 3. Wczytanie własnych liter z katalogu "moje_litery/"
# ============================================================

POLISH_LABEL_MAP = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
    "e": 4,
    "f": 5
}


def load_custom_digits_polish(base_dir: str = "moje_litery"):
    """
    Ładuje litery użytkownika z katalogu base_dir o strukturze:

        base_dir/
            a/
            b/
            c/
            d/
            e/
            f/

    Zwraca:
        X_custom: ndarray [n_próbek, 784] w skali [0,1]
        y_custom: ndarray [n_próbek] z etykietami 0..9
    """
    X_list = []
    y_list = []

    if not os.path.isdir(base_dir):
        print(f"(Informacja) Brak katalogu {base_dir} – pomijam własne litery.")
        return np.empty((0, 784), dtype=np.float32), np.empty((0,), dtype=int)

    for dirname, label in POLISH_LABEL_MAP.items():
        folder = os.path.join(base_dir, dirname)
        if not os.path.isdir(folder):
            continue

        pattern = os.path.join(folder, "*.*")
        for path in glob.glob(pattern):
            if not path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            try:
                canvas = preprocess_image_to_canvas28(path)
                X_list.append(canvas.reshape(-1))  # 28x28 -> 784
                y_list.append(label)
            except Exception as e:
                print(f"Pomijam {path}: {e}")

    if not X_list:
        print(f"(Informacja) Nie znaleziono żadnych obrazów w {base_dir}")
        return np.empty((0, 784), dtype=np.float32), np.empty((0,), dtype=int)

    X_custom = np.stack(X_list).astype(np.float32)
    y_custom = np.array(y_list, dtype=int)
    print(f"Wczytano {len(X_custom)} własnych przykładów liter z {base_dir}")
    return X_custom, y_custom


# ============================================================
# 4. Funkcje pomocnicze: wizualizacja i predykcja
# ============================================================

def show_examples_from_test(X: np.ndarray,
                            y_true: np.ndarray,
                            model: MLPClassifier,
                            n_samples: int = 9) -> None:
    """
    Pokazuje n_samples losowych przykładów z X (EMNIST),
    wraz z prawdziwą etykietą i predykcją modelu.
    """
    if X.size == 0:
        print("Brak danych testowych (EMNIST) w tym trybie.")
        return

    n_samples = min(n_samples, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)

    plt.figure(figsize=(8, 8))
    for i, idx in enumerate(indices, start=1):
        img = X[idx].reshape(28, 28)
        true_label = int(y_true[idx])

        x_single = X[idx].reshape(1, -1)
        pred_label = int(model.predict(x_single)[0])

        plt.subplot(3, 3, i)
        plt.imshow(img, cmap="gray")
        plt.title(f"Prawda: {true_label}\nPred: {pred_label}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def predict_custom_image(model: MLPClassifier, image_path: str):
    """
    Wczytuje własny rysunek litery, przygotowuje go "w stylu EMNIST"
    i zwraca (pred_label, proba, canvas_28x28).

    UWAGA: wektor 'proba' jest w tej samej kolejności, co model.classes_.
    """
    canvas = preprocess_image_to_canvas28(image_path)
    x_flat = canvas.reshape(1, -1)

    pred_label = int(model.predict(x_flat)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_flat)[0]
    else:
        proba = None

    return pred_label, proba, canvas


# ============================================================
# 5. Główna logika: wybór trybu, trening + interfejs tekstowy
# ============================================================

def main():
    print("Wybierz tryb treningu sieci:")
    print("[1] EMNIST + moje litery (katalog 'moje_litery/')")
    print("[2] Tylko moje litery (bez EMNIST)")
    mode = input("Tryb [1/2, domyślnie 1]: ").strip()
    if mode not in {"1", "2"}:
        mode = "1"

    has_emnist_test = False

    # -------------------------------
    # TRYB 1: EMNIST + moje litery
    # -------------------------------
    if mode == "1":
        print("\n== TRYB 1: Trening na EMNIST + własne litery ==")
        print("Wczytywanie danych EMNIST (z mirrora GitHub)...")
        (X_train, y_train), (X_test, y_test) = load_emnist(data_dir="emnist_data")
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        # Normalizacja EMNIST
        X_train = X_train.astype("float32") / 255.0
        X_test = X_test.astype("float32") / 255.0
        has_emnist_test = True

        # Wczytanie własnych liter i doklejenie do treningu
        X_custom, y_custom = load_custom_digits_polish("moje_litery")
        if X_custom.shape[0] > 0:
            print(f"Doklejam {X_custom.shape[0]} własnych przykładów do zbioru treningowego.")
            X_train = np.vstack([X_train, X_custom])
            y_train = np.hstack([y_train, y_custom])
            print(f"Nowy rozmiar X_train: {X_train.shape}, y_train: {y_train.shape}")
        else:
            print("Brak własnych liter – trenuję tylko na EMNIST.")

    # -------------------------------
    # TRYB 2: Tylko moje litery
    # -------------------------------
    else:
        print("\n== TRYB 2: Trening tylko na własnych literach ==")
        X_custom, y_custom = load_custom_digits_polish("moje_litery")
        if X_custom.shape[0] == 0:
            print("Brak danych w katalogu 'moje_litery/'. Nie mogę trenować sieci.")
            return

        # W tym trybie nie korzystamy z EMNIST
        X_train = X_custom
        y_train = y_custom
        X_test = np.empty((0, 784), dtype=np.float32)
        y_test = np.empty((0,), dtype=int)
        print(f"Trening tylko na {X_train.shape[0]} własnych obrazach liter.")

    # 4. Definicja perceptronu wielowarstwowego (MLP)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=20,           # możesz zwiększyć np. do 50
        random_state=42,
        verbose=True,
    )

    # 5. Trening
    print("\n== Trening MLP na wybranym zbiorze treningowym ==")
    mlp.fit(X_train, y_train)

    print("KLASY MODELU:", mlp.classes_)

    # 6. Ewaluacja na zbiorze testowym EMNIST (tylko w trybie 1)
    if has_emnist_test:
        print("\n== Ewaluacja na zbiorze testowym EMNIST ==")
        y_pred = mlp.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy (dokładność) na zbiorze testowym: {acc:.4f}\n")

        print("Raport klasyfikacji:")
        print(classification_report(y_test, y_pred))
        print("Macierz pomyłek:")
        print(confusion_matrix(y_test, y_pred))
    else:
        print("\nBrak zbioru testowego EMNIST w tym trybie – pomijam ewaluację na EMNIST.")

    # 7. Prosty interfejs tekstowy
    while True:
        print("\n=======================================")
        print("MENU:")
        if has_emnist_test:
            print("[1] Pokaż przykładowe litery z EMNIST z predykcjami")
        else:
            print("[1] (Niedostępne w tym trybie – brak danych EMNIST)")
        print("[2] Rozpoznaj literę z własnego pliku graficznego")
        print("[3] Zakończ program")
        print("=======================================")

        choice = input("Wybierz opcję (1/2/3): ").strip()

        if choice == "1":
            if not has_emnist_test:
                print("Opcja niedostępna: ten tryb nie korzysta z EMNIST.")
                continue
            try:
                n_samples_str = input("Ile przykładów pokazać? [domyślnie 9]: ").strip()
                n_samples = int(n_samples_str) if n_samples_str else 9
            except ValueError:
                n_samples = 9

            print(f"Wyświetlam {n_samples} losowych przykładów z EMNIST...")
            show_examples_from_test(X_test, y_test, mlp, n_samples=n_samples)

        elif choice == "2":
            image_path = input(
                "Podaj pełną ścieżkę do pliku z obrazem (np. C:\\...\\moja_litera.png):\n> "
            ).strip()

            if not os.path.isfile(image_path):
                print("Plik nie istnieje. Sprawdź ścieżkę i spróbuj ponownie.")
                continue

            try:
                pred_label, proba, arr = predict_custom_image(mlp, image_path)
                print(f"\nSieć przewiduje literę: {pred_label}")

                if proba is not None:
                    print("Prawdopodobieństwa klas:")
                    # KLUCZOWA POPRAWKA: używamy mlp.classes_, nie enumerate
                    for cls, p in zip(mlp.classes_, proba):
                        print(f"  {cls}: {p:.3f}")

                plt.figure()
                plt.imshow(arr, cmap="gray")
                plt.title(f"Twoja litera – przewidywana: {chr(ord('A') + pred_label)}")
                plt.axis("off")
                plt.show()

            except Exception as e:
                print(f"Wystąpił błąd podczas przetwarzania obrazu: {e}")

        elif choice == "3":
            print("Zamykanie programu.")
            break

        else:
            print("Nieznana opcja. Wpisz 1, 2 lub 3.")


if __name__ == "__main__":
    main()
