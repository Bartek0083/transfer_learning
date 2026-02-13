import os
import copy
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import matplotlib
matplotlib.use('Agg')  # Backend bez GUI — pozwala zapisywać wykresy do plików
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def get_config():
    """Zwraca domyślną konfigurację projektu."""
    parser = argparse.ArgumentParser(description='🐦 Klasyfikacja ptaków — Transfer Learning')

    # Ścieżki
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Ścieżka do katalogu z danymi (train/val/test)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Ścieżka do zapisu modelu i wyników')

    # Hiperparametry treningu
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='Liczba epok treningu (1 epoka = przejście przez cały zbiór)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Rozmiar batcha — ile obrazów przetwarzamy naraz')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Szybkość uczenia — jak duże kroki robi optymalizator')
    parser.add_argument('--fine_tune_lr', type=float, default=0.0001,
                        help='Szybkość uczenia przy fine-tuningu (mniejsza!)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Rozmiar obrazu wejściowego (224x224 dla EfficientNet)')
    parser.add_argument('--freeze_epochs', type=int, default=5,
                        help='Ile epok trenujemy z zamrożonymi warstwami bazowymi')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Liczba wątków do ładowania danych')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping — ile epok bez poprawy przed zatrzymaniem')

    return parser.parse_args()

def create_data_transforms(image_size=224):

    # Średnia i odchylenie standardowe z ImageNet
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    data_transforms = {
        # TRENING: Stosujemy augmentację, żeby model widział różnorodne dane
        'train': transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),  # Losowe przycięcie
            transforms.RandomHorizontalFlip(p=0.5),       # Losowe odbicie poziome (50% szans)
            transforms.RandomRotation(15),                  # Losowy obrót ±15 stopni
            transforms.ColorJitter(                         # Losowa zmiana kolorów
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),                          # Konwersja PIL Image -> Tensor
            transforms.Normalize(imagenet_mean, imagenet_std)  # Normalizacja ImageNet
        ]),

        # WALIDACJA i TEST: Bez augmentacji! Tylko resize + normalizacja
        'val': transforms.Compose([
            transforms.Resize(image_size + 32),    # Trochę większy niż docelowy
            transforms.CenterCrop(image_size),      # Przycięcie do docelowego rozmiaru
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ]),

        'test': transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ]),
    }

    return data_transforms

def load_data(data_dir, data_transforms, batch_size=32, num_workers=4):

    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            image_datasets[split] = datasets.ImageFolder(
                split_dir,
                transform=data_transforms[split]
            )
            dataloaders[split] = DataLoader(
                image_datasets[split],
                batch_size=batch_size,
                shuffle=(split == 'train'),  # Mieszamy tylko dane treningowe
                num_workers=num_workers,
                pin_memory=True  # Szybszy transfer CPU -> GPU
            )
            dataset_sizes[split] = len(image_datasets[split])
            print(f"  📁 {split}: {dataset_sizes[split]} obrazów")
        else:
            print(f"  ⚠️  Brak katalogu: {split_dir}")

    # Nazwy klas (gatunków ptaków)
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"\n  🐦 Znaleziono {num_classes} gatunków: {', '.join(class_names[:5])}...")

    return dataloaders, dataset_sizes, class_names

def create_model(num_classes, pretrained=True):

    print("\n🔧 Budowanie modelu...")

    # Krok 1: Ładujemy pretrenowany EfficientNet-B0
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        print("  ✅ Załadowano wagi pretrenowane na ImageNet")
    else:
        model = models.efficientnet_b0(weights=None)
        print("  ⚠️  Model bez pretrenowanych wag (trening od zera)")

    # Krok 2: Zamrażamy wszystkie warstwy bazowe
    # "Zamrożenie" = ustawienie requires_grad=False
    # Dzięki temu gradienty nie będą obliczane dla tych warstw
    for param in model.parameters():
        param.requires_grad = False

    # Krok 3: Podmieniamy ostatnią warstwę (classifier) na naszą
    # Oryginalna: Linear(1280, 1000) — 1000 klas ImageNet
    # Nasza: Linear(1280, num_classes) — nasze gatunki ptaków
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),          # Dropout — losowo wyłącza 30% neuronów (regularyzacja)
        nn.Linear(num_features, 512),  # Warstwa ukryta
        nn.ReLU(),                      # Funkcja aktywacji
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)     # Warstwa wyjściowa — tyle neuronów ile gatunków
    )

    # Nowe warstwy mają domyślnie requires_grad=True — będą trenowane
    print(f"  🔄 Nowa warstwa klasyfikacyjna: {num_features} -> 512 -> {num_classes}")

    # Podsumowanie
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  📊 Parametry: {total_params:,} łącznie, {trainable_params:,} trenowalnych")
    print(f"  ❄️  Zamrożono {total_params - trainable_params:,} parametrów")

    return model

def unfreeze_model(model, num_layers_to_unfreeze=None):
    if num_layers_to_unfreeze is None:
        # Odmrażamy WSZYSTKIE warstwy
        for param in model.parameters():
            param.requires_grad = True
        print("  🔥 Odmrożono WSZYSTKIE warstwy modelu")
    else:
        # Odmrażamy tylko ostatnie N warstw features
        layers = list(model.features.children())
        for layer in layers[-num_layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"  🔥 Odmrożono ostatnie {num_layers_to_unfreeze} bloków")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  📊 Teraz trenowalnych: {trainable:,} / {total:,} parametrów")

class EarlyStopping:
   
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"  ⏳ Early Stopping: {self.counter}/{self.patience} (brak poprawy)")
            if self.counter >= self.patience:
                self.should_stop = True
                print("  🛑 Early Stopping aktywowany! Zatrzymuję trening.")
        else:
            self.best_score = val_score
            self.counter = 0

def train_one_epoch(model, dataloader, criterion, optimizer, device, dataset_size):
    """Trenuje model przez jedną epokę."""
    model.train()  # Tryb treningowy (włącza Dropout, BatchNorm w trybie train)

    running_loss = 0.0
    running_corrects = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        # Przenosimy dane na GPU (jeśli dostępne)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zerujemy gradienty z poprzedniego kroku
        optimizer.zero_grad()

        # Forward pass — model generuje predykcje
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        # Backward pass — obliczamy gradienty
        loss.backward()

        # Aktualizujemy wagi
        optimizer.step()

        # Statystyki
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size

    return epoch_loss, epoch_acc.item()


def validate(model, dataloader, criterion, device, dataset_size):
    """Waliduje model (bez gradientów — oszczędzamy pamięć i czas)."""
    model.eval()  # Tryb ewaluacji (wyłącza Dropout)

    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():  # Nie obliczamy gradientów!
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size

    return epoch_loss, epoch_acc.item()

def train_model(model, dataloaders, dataset_sizes, config, device, phase='feature_extraction'):

    # Funkcja straty — CrossEntropyLoss dla klasyfikacji wieloklasowej
    criterion = nn.CrossEntropyLoss()

    # Optymalizator — Adam z odpowiednim learning rate
    if phase == 'feature_extraction':
        lr = config.learning_rate
        num_epochs = config.freeze_epochs
        # Trenujemy TYLKO parametry z requires_grad=True (nowa warstwa)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )
    else:  # fine_tuning
        lr = config.fine_tune_lr
        num_epochs = config.num_epochs - config.freeze_epochs
        # Trenujemy WSZYSTKIE parametry, ale z różnymi learning rate
        # Warstwy bazowe: niski LR (delikatna korekta)
        # Nowa warstwa: wyższy LR (szybsze uczenie)
        optimizer = optim.Adam([
            {'params': model.features.parameters(), 'lr': lr * 0.1},  # Bazowe: 10x mniejszy LR
            {'params': model.classifier.parameters(), 'lr': lr}        # Nowe: normalny LR
        ])

    # Scheduler — zmniejsza learning rate, gdy walidacja się nie poprawia
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience)

    # Historia treningu (do wykresów)
    history = defaultdict(list)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print(f"\n{'='*60}")
    print(f"🚀 Faza: {phase.upper()}")
    print(f"   Epoki: {num_epochs}, LR: {lr}, Batch: {config.batch_size}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\nEpoka {epoch+1}/{num_epochs}")
        print("-" * 40)

        # Trening
        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, device, dataset_sizes['train']
        )

        # Walidacja
        val_loss, val_acc = validate(
            model, dataloaders['val'], criterion, device, dataset_sizes['val']
        )

        # Czas epoki
        elapsed = time.time() - start_time

        # Logowanie
        print(f"  📈 Train — Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  📊 Val   — Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"  ⏱️  Czas: {elapsed:.1f}s")

        # Zapisujemy historię
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Scheduler
        scheduler.step(val_acc)

        # Zapisujemy najlepszy model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  🏆 Nowy najlepszy model! Val Acc: {best_acc:.4f}")

        # Early stopping
        early_stopping(val_acc)
        if early_stopping.should_stop:
            break

    # Przywracamy najlepsze wagi
    model.load_state_dict(best_model_wts)
    print(f"\n✅ Najlepsza walidacja ({phase}): {best_acc:.4f}")

    return model, dict(history)


def evaluate_model(model, dataloader, dataset_size, class_names, device):

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Accuracy ogólna
    accuracy = np.mean(all_preds == all_labels)
    print(f"\n🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    # Metryki per klasa
    print(f"\n{'Gatunek':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 72)

    results = {}
    for i, name in enumerate(class_names):
        tp = np.sum((all_preds == i) & (all_labels == i))
        fp = np.sum((all_preds == i) & (all_labels != i))
        fn = np.sum((all_preds != i) & (all_labels == i))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(all_labels == i)

        results[name] = {
            'precision': precision, 'recall': recall, 'f1': f1, 'support': int(support)
        }
        print(f"  {name:<28} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {support:>10}")

    # Confusion Matrix
    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        confusion_matrix[true][pred] += 1

    return accuracy, results, confusion_matrix


def plot_training_history(history_phase1, history_phase2, output_dir):
    """Rysuje wykresy historii treningu."""
    # Łączymy obie fazy
    train_loss = history_phase1.get('train_loss', []) + history_phase2.get('train_loss', [])
    val_loss = history_phase1.get('val_loss', []) + history_phase2.get('val_loss', [])
    train_acc = history_phase1.get('train_acc', []) + history_phase2.get('train_acc', [])
    val_acc = history_phase1.get('val_acc', []) + history_phase2.get('val_acc', [])

    epochs = range(1, len(train_loss) + 1)
    phase1_end = len(history_phase1.get('train_loss', []))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Wykres straty (Loss)
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    if phase1_end > 0 and phase1_end < len(epochs):
        ax1.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7,
                     label='Fine-tuning start')
    ax1.set_title('Loss w trakcie treningu', fontsize=14)
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Wykres dokładności (Accuracy)
    ax2.plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2)
    if phase1_end > 0 and phase1_end < len(epochs):
        ax2.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7,
                     label='Fine-tuning start')
    ax2.set_title('Accuracy w trakcie treningu', fontsize=14)
    ax2.set_xlabel('Epoka')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Wykres zapisany: {plot_path}")


def plot_confusion_matrix(cm, class_names, output_dir):
    """Rysuje macierz pomyłek (confusion matrix)."""
    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.5),
                                     max(8, len(class_names) * 0.4)))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Macierz pomyłek (Confusion Matrix)',
           ylabel='Prawdziwa klasa',
           xlabel='Przewidziana klasa')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Dodajemy wartości liczbowe
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Macierz pomyłek zapisana: {plot_path}")

def predict_image(model, image_path, class_names, data_transforms, device):
   
    from PIL import Image

    model.eval()

    # Wczytujemy i transformujemy obraz
    image = Image.open(image_path).convert('RGB')
    input_tensor = data_transforms['test'](image).unsqueeze(0).to(device)

    # Predykcja
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top5_probs, top5_indices = torch.topk(probabilities, min(5, len(class_names)))

    # Wyniki
    predicted_class = class_names[top5_indices[0][0].item()]
    confidence = top5_probs[0][0].item()

    top5 = []
    for i in range(top5_probs.size(1)):
        top5.append({
            'class': class_names[top5_indices[0][i].item()],
            'confidence': top5_probs[0][i].item()
        })

    return predicted_class, confidence, top5

def create_demo_data(data_dir, num_classes=25, images_per_class=20):
    from PIL import Image
    import random

    bird_species = [
        "American_Robin", "Blue_Jay", "Cardinal", "Chickadee", "Crow",
        "Eagle", "Falcon", "Goldfinch", "Hawk", "Heron",
        "Hummingbird", "Kingfisher", "Magpie", "Nightingale", "Oriole",
        "Owl", "Parrot", "Pelican", "Penguin", "Robin",
        "Sparrow", "Starling", "Swan", "Woodpecker", "Wren"
    ][:num_classes]

    splits = {
        'train': int(images_per_class * 0.7),
        'val': int(images_per_class * 0.15),
        'test': int(images_per_class * 0.15)
    }

    print(f"\n📦 Generowanie danych demo ({num_classes} gatunków)...")

    for split, count in splits.items():
        count = max(count, 2)  # minimum 2 obrazy
        for species in bird_species:
            species_dir = os.path.join(data_dir, split, species)
            os.makedirs(species_dir, exist_ok=True)

            for i in range(count):
                # Generujemy kolorowy obraz z losowym wzorem
                img = Image.new('RGB', (256, 256))
                pixels = img.load()
                # Każdy gatunek ma unikalny bazowy kolor
                base_r = hash(species) % 200 + 30
                base_g = hash(species + "g") % 200 + 30
                base_b = hash(species + "b") % 200 + 30

                for x in range(256):
                    for y in range(256):
                        r = min(255, max(0, base_r + random.randint(-30, 30)))
                        g = min(255, max(0, base_g + random.randint(-30, 30)))
                        b = min(255, max(0, base_b + random.randint(-30, 30)))
                        pixels[x, y] = (r, g, b)

                img.save(os.path.join(species_dir, f'{species}_{i:04d}.jpg'))

        print(f"  ✅ {split}: {count} obrazów × {num_classes} gatunków = {count * num_classes}")

    print("  ⚠️  UWAGA: To są dane demo! W prawdziwym projekcie użyj zdjęć ptaków.")
    return bird_species

def main():
    """Główna funkcja uruchamiająca cały pipeline."""
    config = get_config()

    print("=" * 60)
    print("🐦 KLASYFIKACJA GATUNKÓW PTAKÓW — Transfer Learning")
    print("=" * 60)

    # Sprawdzamy dostępność GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 Urządzenie: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Pamięć: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Katalog wyjściowy
    os.makedirs(config.output_dir, exist_ok=True)

    # Sprawdzamy dane
    if not os.path.exists(os.path.join(config.data_dir, 'train')):
        print("\n⚠️  Brak danych! Generuję dane demonstracyjne...")
        create_demo_data(config.data_dir)

    # Transformacje danych
    data_transforms = create_data_transforms(config.image_size)

    # Ładowanie danych
    print("\n📂 Ładowanie danych...")
    dataloaders, dataset_sizes, class_names = load_data(
        config.data_dir, data_transforms, config.batch_size, config.num_workers
    )

    # Budowa modelu
    model = create_model(num_classes=len(class_names))
    model = model.to(device)

    # ========================================
    # FAZA 1: Feature Extraction
    # ========================================
    print("\n" + "=" * 60)
    print("❄️  FAZA 1: Feature Extraction (zamrożone warstwy bazowe)")
    print("=" * 60)
    model, history_phase1 = train_model(
        model, dataloaders, dataset_sizes, config, device, phase='feature_extraction'
    )

    # ========================================
    # FAZA 2: Fine-Tuning
    # ========================================
    print("\n" + "=" * 60)
    print("🔥 FAZA 2: Fine-Tuning (odmrożone warstwy)")
    print("=" * 60)
    unfreeze_model(model, num_layers_to_unfreeze=3)  # Odmrażamy 3 ostatnie bloki
    model, history_phase2 = train_model(
        model, dataloaders, dataset_sizes, config, device, phase='fine_tuning'
    )

    # ========================================
    # EWALUACJA
    # ========================================
    if 'test' in dataloaders:
        print("\n" + "=" * 60)
        print("🎯 EWALUACJA NA ZBIORZE TESTOWYM")
        print("=" * 60)
        accuracy, results, cm = evaluate_model(
            model, dataloaders['test'], dataset_sizes['test'], class_names, device
        )

        # Wizualizacja
        plot_training_history(history_phase1, history_phase2, config.output_dir)
        plot_confusion_matrix(cm, class_names, config.output_dir)

    # ========================================
    # ZAPIS MODELU
    # ========================================
    print("\n💾 Zapisywanie modelu...")

    # Zapis pełnego modelu
    model_path = os.path.join(config.output_dir, 'bird_classifier.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': len(class_names),
        'config': vars(config),
        'accuracy': accuracy if 'test' in dataloaders else None,
    }, model_path)
    print(f"  ✅ Model zapisany: {model_path}")

    # Zapis metadanych
    metadata = {
        'class_names': class_names,
        'num_classes': len(class_names),
        'test_accuracy': accuracy if 'test' in dataloaders else None,
        'config': vars(config),
    }
    meta_path = os.path.join(config.output_dir, 'metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Metadane zapisane: {meta_path}")

    print("\n" + "=" * 60)
    print("🎉 GOTOWE! Projekt ukończony.")
    print(f"   Model: {model_path}")
    print(f"   Wykresy: {config.output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()