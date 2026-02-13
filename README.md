# 🐦 Klasyfikacja Gatunków Ptaków — Transfer Learning

Edukacyjny projekt klasyfikacji obrazów z wykorzystaniem **transfer learningu**
i modelu **EfficientNet-B0** pretrenowanego na ImageNet.

## 📋 Opis projektu

Projekt rozpoznaje **30 gatunków ptaków** na zdjęciach. Wykorzystuje technikę
transfer learningu w dwóch fazach:

1. **Feature Extraction** — zamrożone warstwy bazowe, trening nowego klasyfikatora
2. **Fine-Tuning** — odmrożone warstwy, delikatne dostrojenie całego modelu

## 🗂️ Struktura projektu

```
bird_classification/
├── birds_train.py              # Skrypt treningowy (CLI)
├── notebook.ipynb        # Jupyter Notebook (krok po kroku)
├── requirements.txt      # Zależności Python
├── README.md             # Ten plik
├── data/                 # Dane (tworzone za pomocą skryptu split_dataset.py)
│   ├── train/
│   ├── val/
│   └── test/
└── output/               # Wyniki (generowane automatycznie)
    ├── bird_classifier.pth
    ├── metadata.json
    ├── training_history.png
    └── predictions/              ← Wyniki z predict.py
        ├── confusion_matrix.png
        ├── f1_per_class.png
        ├── error_examples.png
        ├── correct_examples.png
        └── report.json
```

## 🚀 Szybki start

### 1. Instalacja

```bash
pip install -r requirements.txt
```

### 2. Przygotowanie danych

#### Opcja A: Dane demo (do testowania)
Skrypt automatycznie wygeneruje syntetyczne dane demo. Wystarczy uruchomić trening.

#### Opcja B: Prawdziwe dane (zalecane)
Pobierz dataset i umieść w katalogu `data/`:

- **[CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/)** — 200 gatunków, ~12k obrazów

Dane powinny mieć strukturę:
```
data/
├── train/
│   ├── Gatunek_1/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── Gatunek_2/
│   └── ...
├── val/
└── test/
```
#### Skrypty do pomocy
Do projektu zostały dołączone skrypty `select_species.py` `split_dataset.py` pozwalające na łatwiejsze kopiowanie obrazów z datasetu CUD-200-2011 do folderu data w sposób randomowy.
Wystarczy że do projektu wrzucisz folder CUD-200-2011/images z pobranej paczki (patrz Opcja B).

#### Ograniczenie od 1 do 200 gatunków
Jeśli chcesz możesz wybrać dowolne ograniczenie sprawdzanych gatunków np. 50 

```bash
python select_species.py --data_dir ./data --num_species 50
```
### 3. Trening

#### Skrypt Python:
```bash
python birds_train.py
```

#### Z parametrami:
```bash
python birds_train.py --num_epochs 30 --batch_size 64 --learning_rate 0.001
```

#### Jupyter Notebook:
```bash
jupyter notebook notebook.ipynb
```

## ⚙️ Hiperparametry

| Parametr | Domyślna wartość | Opis |
|----------|-----------------|------|
| `--num_epochs` | 20 | Całkowita liczba epok |
| `--freeze_epochs` | 5 | Epoki z zamrożonymi warstwami |
| `--batch_size` | 16 | Rozmiar batcha |
| `--learning_rate` | 0.001 | LR dla feature extraction |
| `--fine_tune_lr` | 0.0001 | LR dla fine-tuningu |
| `--image_size` | 224 | Rozmiar obrazu wejściowego |
| `--patience` | 5 | Early stopping patience |
| `--data_dir` | ./data | Ścieżka do danych |
| `--output_dir` | ./output | Ścieżka do wyników |

## 📊 Techniki zastosowane

- **Transfer Learning** z EfficientNet-B0 (ImageNet)
- **Dwufazowy trening**: Feature Extraction → Fine-Tuning
- **Augmentacja danych**: RandomCrop, Flip, Rotation, ColorJitter
- **Early Stopping** — zapobiega przeuczeniu
- **Learning Rate Scheduling** — ReduceLROnPlateau
- **Differential Learning Rates** — różne LR dla różnych warstw

## 🔮 Predykcja na nowym zdjęciu

```python
from train import predict_image, create_data_transforms
import torch
from torchvision import models
import torch.nn as nn

# Załaduj model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('output/bird_classifier.pth', map_location=device)

model = models.efficientnet_b0(weights=None)
nf = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3), nn.Linear(nf, 512), nn.ReLU(),
    nn.Dropout(0.2), nn.Linear(512, checkpoint['num_classes']))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Predykcja
transforms = create_data_transforms()
predicted, confidence, top5 = predict_image(
    model, 'path/to/bird.jpg', checkpoint['class_names'], transforms, device)

print(f'Gatunek: {predicted} ({confidence:.1%})')
```

## 📖 Zasoby edukacyjne

- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [EfficientNet Paper (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946)
- [CS231n: Transfer Learning](https://cs231n.github.io/transfer-learning/)


