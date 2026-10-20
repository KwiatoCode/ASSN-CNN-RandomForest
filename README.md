# ASSN — rozpoznawanie alfabetu ASL w czasie rzeczywistym

Link dataset: https://www.kaggle.com/datasets/usifelshafie/american-sign-language-asl-hand-skeleton-dataset

> Kamera → szkielet dłoni → obraz 400×400 → **CNN trenowana od podstaw** → **Random Forest** → litera A–Z

Aplikacja na iOS i macOS (SwiftUI + Apple Vision + Core ML).
Projekt: `~/Desktop/ASSN/ASSN.xcodeproj`

---

## 1. Przepływ danych — od fotonu do litery

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  ① KAMERA  · AVFoundation                                                      │
│     CVPixelBuffer BGRA  640×480 (iPhone w pionie: 480×640)  · do 60 FPS        │
│     klatka obrócona fizycznie do pionu, NIE odbita lustrzanie                  │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ② FRAME MANAGER                                                               │
│     odrzuca klatki starsze niż 100 ms i klatki spóźnione (alwaysDiscardsLate)  │
│     mierzy FPS (EMA) i opóźnienie kamera → wynik                               │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ③ HAND DETECTOR  · Apple Vision  VNDetectHumanHandPoseRequest                 │
│     → 21 × (x, y, pewność)  w układzie Vision (0…1, początek: lewy DÓŁ)        │
│     → strona dłoni (left / right)                                              │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ④ LANDMARK PROCESSOR                                                          │
│     • odrzuca punkty z pewnością < 0.3 i całą dłoń z pewnością < 0.5           │
│     • wymaga ≥ 14 / 21 dobrych punktów                                         │
│     • y → 1 − y  (układ MediaPipe: początek w lewym GÓRNYM rogu)               │
│     • kolejność punktów Vision → kolejność MediaPipe (0 = nadgarstek…20)       │
│     • braki: poprzednia klatka (≤ 250 ms) → interpolacja w palcu → ekstrapol.  │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ⑤ HAND FILTER  · One Euro Filter  (42 filtry: x i y każdego punktu)           │
│     filtrowanie w pikselach · minCutoff 1.0 Hz · β 0.02 · reset po 300 ms      │
│     wolny ruch → mocne wygładzanie,  szybki ruch → małe opóźnienie             │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ⑥ HAND STATE   21 × CGPoint · pewności · strona dłoni · timestamp · rozmiar   │
│                 (Codable → JSON)                                               │
└──────────────┬───────────────────────────────────────────────┬───────────────┘
               ▼                                               ▼
┌─────────────────────────────┐   ┌────────────────────────────────────────────┐
│  ⑦ RENDERER (podgląd)        │   │  ⑧ SKELETON IMAGE RENDERER  (wejście sieci) │
│  kamera + overlay            │   │  płótno 400×400, białe tło, styl datasetu  │
│  zielone punkty, czerwone    │   │  odbicie lustrzane + kadrowanie            │
│  linie, aspect-fill, lustro  │   └──────────────────────┬─────────────────────┘
└─────────────────────────────┘                          ▼
                                  ┌────────────────────────────────────────────┐
                                  │  ⑨ SIEĆ  · Core ML  (rozdział 3)           │
                                  │  400×400 → domyślnie 224×224 → CNN → 256     │
                                  │  → Random Forest → 26 prawdopodobieństw                │
                                  └──────────────────────┬─────────────────────┘
                                                         ▼
                                  ┌────────────────────────────────────────────┐
                                  │  ⑩ PREDICTION SMOOTHER                     │
                                  │  średnia z 6 ostatnich wyników · próg 0.5  │
                                  │  → litera + top-3                          │
                                  └────────────────────────────────────────────┘
```

### Wątki

| Kolejka | Etapy | Tempo |
|---|---|---|
| `asl.camera.video` | ① – ⑥ | każda klatka, do 60 FPS |
| `asl.recognition` | ⑧ – ⑩ | tyle, ile zdąży sieć; gdy jest zajęta, klatka jest **pomijana**, a nie kolejkowana |
| main | ⑦, UI | odświeżanie SwiftUI |

Dzięki temu podgląd ze szkieletem chodzi płynnie, nawet gdy sieć liczy wolniej niż kamera.

---

## 2. Wejście sieci — obraz szkieletu (krok ⑧)

Model uczył się na datasecie **[ASL Hand Skeleton](https://www.kaggle.com/datasets/usifelshafie/american-sign-language-asl-hand-skeleton-dataset)**.
To 26 000 **obrazków** szkieletu narysowanych przez `mediapipe.drawing_utils`, a nie współrzędne.
Dlatego aplikacja odtwarza ten sam styl rysowania. Parametry zmierzyłem na obrazkach z datasetu:

```
        przykład z datasetu (litera L)          parametry rysowania
     ┌──────────────────────────────┐
     │            ●                 │      płótno   400 × 400, białe tło
     │            │                 │      kości    21 połączeń MediaPipe
     │            ●                 │               czerwone (255,0,0), grubość 2
     │            │                 │      stawy    zielone (0,255,0), r = 2,
     │            ●   ●  ●  ●       │               biała obwódka r = 3
     │  ●         │  ╱│ ╱│ ╱│       │      widok    lustrzany (selfie)
     │   ╲        ● ● ● ● ●  ●      │
     │    ●       │╲ ╲│╱  ╱ ╱       │      zgodność z cv2 po skalowaniu
     │     ╲      │ ╲ │ ╱ ╱         │      do 224×224: średnio 0.3/255
     │      ●     │  ╲│╱╱           │      na piksel
     │       ╲    │   ●             │
     │        ●───┴──╱              │
     │                              │
     └──────────────────────────────┘
```

**Kadrowanie** (przełącznik w aplikacji):

| Tryb | Przekształcenie punktów |
|---|---|
| **Kamera** | `(x, y) × 400`, dokładnie jak w datasecie (kamera 640×480 rysowana na kwadracie) |
| **Normalizacja** (domyślnie) | dłoń przeskalowana tak, by odległość nadgarstek → MCP środkowego palca wynosiła **130 px**; oś X ściśnięta o **480/640** (jak w datasecie); środek dłoni przesunięty do **(188, 209)** — średnia z datasetu |

---

## 3. Architektura modelu

Model korzysta z `ASLBackbone.mlpackage` i `ASLClassifier.json`. Własna sieć CNN jest uczona od losowych wag na obrazach datasetu, bez ResNet, ImageNet i wstępnie wytrenowanych modeli. Random Forest klasyfikuje embedding 256 cech otrzymany z wytrenowanej CNN.

```text
obrazy A–Z (400×400)
        ↓
train / validation / test (przed augmentacją)
        ↓
augmentacja wyłącznie train: obrót ±12°, przesunięcie ±8%, skala 0.85–1.15
        ↓
224×224 RGB (ustawialne przez `--image-size`) → CNN (od zera) → 256 cech → 26 logitów
        ↓
trening CNN: CrossEntropyLoss + AdamW, wybór epoki na validation
        ↓
embedding 256 cech → RandomForestClassifier (160 drzew)
        ↓
test na oryginalnych, niewidzianych obrazach
        ↓
ASLBackbone.mlpackage + ASLClassifier.json
        ↓
SignClassifier → PredictionSmoother → litera A–Z
```

CNN zawiera pięć konwolucji 3×3, BatchNorm, ReLU, cztery operacje MaxPool oraz AdaptiveAvgPool. Warstwa klasyfikacji CNN jest trenowana na 26 literach, a następnie do modelu aplikacji eksportowany jest ekstraktor 256 cech. Klasyfikatorem końcowym pozostaje Random Forest.

Podział datasetu: 25% test, z pozostałych obrazów co najmniej 10% validation, reszta trening. Augmentacja występuje wyłącznie po podziale, na treningowych przykładach. Test i validation nie są augmentowane. Wyniki i historia treningu trafiają do `tools/trained/results.json` i `tools/trained/history.json`.

Dane: [ASL Hand Skeleton Dataset](https://www.kaggle.com/datasets/usifelshafie/american-sign-language-asl-hand-skeleton-dataset). Katalog datasetu powinien zawierać foldery `A`, `B`, …, `Z` z obrazami `.png`, `.jpg`, `.jpeg`, `.bmp` lub `.webp`. Dopuszczalny jest również nadrzędny folder `dataset/`.

Dataset użytkownika `archive (3).zip` zawiera 26 000 obrazów 400×400, po 1000 dla każdej litery A–Z. Nie należy umieszczać obrazów w katalogu `ASSN/` z zasobami aplikacji. Wyniki treningu znajdują się w `tools/trained/results.json` po zakończeniu treningu. Do uruchomienia na iOS potrzebny jest dodatkowo `ASLBackbone.mlpackage` wygenerowany w środowisku z `coremltools`.

## 4. Mapa kodu

| Etap | Plik |
|---|---|
| ① kamera | `ASSN/Camera/CameraManager.swift` |
| ② frame manager | `ASSN/Camera/FrameManager.swift` |
| ③ detektor | `ASSN/Tracking/HandDetector.swift` · `HandJoint.swift` |
| ④ przetwarzanie | `ASSN/Tracking/LandmarkProcessor.swift` |
| ⑤ filtr | `ASSN/Tracking/OneEuroFilter.swift` |
| ⑥ stan | `ASSN/Tracking/HandState.swift` |
| ①–⑩ orkiestracja | `ASSN/Tracking/HandTrackingPipeline.swift` |
| ⑦ podgląd | `ASSN/UI/CameraPreviewView.swift` · `HandOverlayView.swift` |
| ⑧ płótno | `ASSN/Recognition/SkeletonImageRenderer.swift` |
| ⑨ ⑩ sieć + wygładzanie | `ASSN/Recognition/SignClassifier.swift` |
| konwersja / trening | `tools/build_coreml.py` |

---

## 5. Ograniczenia i kierunki rozwoju

- MediaPipe kontra Apple Vision — różnice położenia punktów mogą obniżyć skuteczność klasyfikacji.
- J i Z obejmują ruch, który nie jest bezpośrednio modelowany przez klasyfikację pojedynczego obrazu.
- Obrót, skala i przesunięcie są symulowane w treningu; test na iPhonie nadal wymaga osobnej walidacji.
- Metryki nowej CNN i Random Forest powstają dopiero po treningu na rzeczywistym zbiorze zdjęć.

---

## 6. Uruchomienie (skrót)

```bash
pip install torch torchvision scikit-learn pillow numpy joblib coremltools matplotlib pandas
python tools/build_coreml.py --train "dataset" --count 0 --augmentations 2 --epochs 20 --image-size 224
```

Bez konwersji Core ML, np. na Windowsie:

```bash
python tools/build_coreml.py --train "dataset" --count 0 --augmentations 2 --epochs 20 --image-size 224 --skip-coreml
```

Późniejszy eksport zapisanych wag i lasu losowego, bez ponownego treningu:

```bash
python tools/build_coreml.py --checkpoint tools/trained/cnn.pt --randomforest tools/trained/randomforest.pkl
```

`--count 0` oznacza użycie wszystkich 1000 obrazów każdej litery. Podczas testu szybkości na CPU można wybrać `--count 100 --epochs 4 --image-size 96`, lecz wyniku takiego treningu nie należy przedstawiać jako wyniku dla całego zbioru.

Eksport do Core ML wymaga zgodnego środowiska z pakietem `coremltools`; gotową aplikację iOS buduje się w Xcode na Macu. Skrypt automatycznie tworzy `ASSN/ASLBackbone.mlpackage` i `ASSN/ASLClassifier.json`. Pliki te muszą zostać dołączone do zasobów aplikacji. Struktura katalogów Swift i nazwy `SignClassifier`, `RandomForest`, `HandTrackingPipeline` nie zostały zmienione.


## 7. Pilotaż na otrzymanym datasecie

Archiwum `archive (3).zip` ma 26 000 obrazów JPEG, 1000 na klasę A–Z, o wymiarach 400×400.
W pakiecie znajdują się pliki z ukończonego pilotażu: `tools/trained/cnn.pt`, `tools/trained/randomforest.pkl`, `tools/trained/results.json`, `tools/trained/history.json` i `ASSN/ASLClassifier.json`.

| Parametr | Pilotaż |
|---|---|
| Obrazy użyte w podziale | 2600 (100 na klasę) |
| Train / validation / test | 1755 / 195 / 650 |
| Augmentacja | jedna dodatkowa kopia na obraz treningowy |
| Rozdzielczość CNN | 96×96 |
| Epoki | 4 (najlepsza epoka: 2) |
| CNN accuracy — test | 95,08% |
| Random Forest accuracy — test | 100,00% |
| Random Forest weighted F1 — test | 1,0000 |

To są wyniki wstępne z 10% obrazów, z losowym podziałem zdjęć tego samego źródła. Nie dowodzą skuteczności na osobach niezależnych od zbioru treningowego ani na obrazie z kamery iPhone. Dataset może zawierać bardzo podobne pozy.

Sprawdzenie rozpoznania pojedynczego obrazu JPEG w Pythonie:

```bash
python tools/predict_image.py --image "dataset/L/0.jpg" --checkpoint tools/trained/cnn.pt --randomforest tools/trained/randomforest.pkl
```

Pełne uczenie od zera na wszystkich 26 000 obrazów, np. na komputerze z GPU:

```bash
python tools/build_coreml.py --train "dataset" --count 0 --augmentations 2 --epochs 20 --image-size 224 --batch-size 32 --learning-rate 0.0003 --skip-coreml
```

Po zakończeniu pełnego treningu w środowisku wspierającym konwersję Core ML:

```bash
python tools/build_coreml.py --checkpoint tools/trained/cnn.pt --randomforest tools/trained/randomforest.pkl
```

`ASLBackbone.mlpackage` nie jest dołączony do niniejszego archiwum. Bez konwersji tego modelu i zbudowania aplikacji w Xcode klasyfikacja w iOS nie zadziała.
