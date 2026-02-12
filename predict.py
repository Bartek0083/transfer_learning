"""
================================================================================
🐦 PREDYKCJA I EWALUACJA — Klasyfikacja Gatunków Ptaków
================================================================================

Skrypt ładuje wytrenowany model i uruchamia predykcję na zbiorze testowym.
Generuje:
- Raport z metrykami (accuracy, precision, recall, F1)
- Macierz pomyłek (confusion matrix)
- Przykłady poprawnych i błędnych predykcji
- Top-5 accuracy
- Najlepiej i najgorzej rozpoznawane gatunki

Użycie:
    python predict.py
    python predict.py --model ./output/bird_classifier.pth --test_dir ./data/test
    python predict.py --model ./output/bird_classifier.pth --image ./zdjecie_ptaka.jpg
"""

import os
import argparse
import json
import time

import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# KONFIGURACJA
# ==============================================================================

def get_args():
    parser = argparse.ArgumentParser(description='🐦 Predykcja i ewaluacja modelu ptaków')

    parser.add_argument('--model', type=str, default='./output/bird_classifier.pth',
                        help='Ścieżka do zapisanego modelu (.pth)')
    parser.add_argument('--test_dir', type=str, default='./data/test',
                        help='Ścieżka do folderu testowego')
    parser.add_argument('--image', type=str, default=None,
                        help='Ścieżka do pojedynczego zdjęcia (opcjonalne)')
    parser.add_argument('--output_dir', type=str, default='./output/predictions',
                        help='Gdzie zapisać wyniki')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Rozmiar batcha')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Rozmiar obrazu')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Top-K predykcji do wyświetlenia')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Liczba wątków do ładowania danych')
    parser.add_argument('--show_errors', type=int, default=16,
                        help='Ile błędnych predykcji pokazać na wizualizacji')

    return parser.parse_args()


# ==============================================================================
# ŁADOWANIE MODELU
# ==============================================================================

def load_model(model_path, device):
    """Ładuje wytrenowany model z checkpointu."""

    if not os.path.exists(model_path):
        print(f"❌ Nie znaleziono modelu: {model_path}")
        print(f"   Najpierw wytrenuj model za pomocą train.py")
        return None, None, None

    print(f"📦 Ładowanie modelu: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']

    # Odtwarzamy architekturę
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    saved_acc = checkpoint.get('accuracy', None) or checkpoint.get('test_accuracy', None)
    print(f"  ✅ Model załadowany! ({num_classes} gatunków)")
    if saved_acc:
        print(f"  📊 Accuracy z treningu: {saved_acc:.4f}")

    return model, class_names, checkpoint


def get_transform(image_size=224):
    """Transformacja dla danych testowych (bez augmentacji)."""
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# ==============================================================================
# PREDYKCJA NA ZBIORZE TESTOWYM
# ==============================================================================

def evaluate_test_set(model, test_dir, class_names, transform, args, device):
    """Pełna ewaluacja na zbiorze testowym."""

    if not os.path.exists(test_dir):
        print(f"❌ Nie znaleziono folderu testowego: {test_dir}")
        return

    print(f"\n{'='*60}")
    print(f"🎯 EWALUACJA NA ZBIORZE TESTOWYM")
    print(f"{'='*60}")

    # Ładowanie danych
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    num_images = len(test_dataset)
    num_classes = len(class_names)
    print(f"  📂 Obrazów: {num_images}")
    print(f"  🐦 Gatunków: {num_classes}")

    # Predykcja
    print(f"\n⏳ Uruchamiam predykcję...")
    start_time = time.time()

    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

            # Progress
            done = min((batch_idx + 1) * args.batch_size, num_images)
            print(f"\r  Przetworzono: {done}/{num_images} obrazów", end='', flush=True)

    elapsed = time.time() - start_time
    print(f"\n  ⏱️  Czas: {elapsed:.1f}s ({num_images/elapsed:.1f} img/s)")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Ścieżki do plików (do wizualizacji błędów)
    all_paths = [test_dataset.samples[i][0] for i in range(len(test_dataset))]

    # ==========================
    # METRYKI
    # ==========================
    print(f"\n{'='*60}")
    print(f"📊 WYNIKI")
    print(f"{'='*60}")

    # Top-1 Accuracy
    top1_acc = np.mean(all_preds == all_labels)
    print(f"\n  🎯 Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.1f}%)")

    # Top-5 Accuracy
    top5_correct = 0
    for i in range(len(all_labels)):
        top5_indices = np.argsort(all_probs[i])[-5:]
        if all_labels[i] in top5_indices:
            top5_correct += 1
    top5_acc = top5_correct / len(all_labels)
    print(f"  🎯 Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.1f}%)")

    # Metryki per klasa
    print(f"\n  {'Gatunek':<35} {'Prec':>8} {'Recall':>8} {'F1':>8} {'N':>6}")
    print(f"  {'-'*67}")

    class_metrics = []
    for i, name in enumerate(class_names):
        tp = np.sum((all_preds == i) & (all_labels == i))
        fp = np.sum((all_preds == i) & (all_labels != i))
        fn = np.sum((all_preds != i) & (all_labels == i))
        support = np.sum(all_labels == i)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics.append({
            'name': name, 'precision': precision, 'recall': recall,
            'f1': f1, 'support': int(support), 'correct': int(tp)
        })

        print(f"  {name:<35} {precision:>8.3f} {recall:>8.3f} {f1:>8.3f} {support:>6}")

    # Średnie
    avg_precision = np.mean([m['precision'] for m in class_metrics])
    avg_recall = np.mean([m['recall'] for m in class_metrics])
    avg_f1 = np.mean([m['f1'] for m in class_metrics])
    print(f"  {'-'*67}")
    print(f"  {'ŚREDNIA':<35} {avg_precision:>8.3f} {avg_recall:>8.3f} {avg_f1:>8.3f} {len(all_labels):>6}")

    # Top 5 najlepszych i najgorszych
    sorted_by_f1 = sorted(class_metrics, key=lambda x: x['f1'], reverse=True)

    print(f"\n  🏆 TOP 5 — najlepiej rozpoznawane:")
    for m in sorted_by_f1[:5]:
        print(f"     {m['name']:<35} F1: {m['f1']:.3f} ({m['correct']}/{m['support']})")

    print(f"\n  ❌ TOP 5 — najgorzej rozpoznawane:")
    for m in sorted_by_f1[-5:]:
        print(f"     {m['name']:<35} F1: {m['f1']:.3f} ({m['correct']}/{m['support']})")

    # ==========================
    # WIZUALIZACJE
    # ==========================
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Confusion Matrix
    print(f"\n📊 Generowanie wizualizacji...")
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(max(14, num_classes * 0.35), max(12, num_classes * 0.3)))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(num_classes), yticks=np.arange(num_classes),
           xticklabels=class_names, yticklabels=class_names,
           title=f'Macierz pomyłek — Accuracy: {top1_acc:.1%}',
           ylabel='Prawdziwa klasa', xlabel='Przewidziana klasa')
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)

    if num_classes <= 40:
        thresh = cm.max() / 2
        for i in range(num_classes):
            for j in range(num_classes):
                if cm[i, j] > 0:
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            color='white' if cm[i, j] > thresh else 'black', fontsize=5)

    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Macierz pomyłek: {cm_path}")

    # 2. Wykres F1 per klasa
    fig, ax = plt.subplots(figsize=(max(10, num_classes * 0.2), 6))
    f1_scores = [m['f1'] for m in sorted_by_f1]
    names_sorted = [m['name'] for m in sorted_by_f1]
    colors = ['#2ecc71' if f >= 0.8 else '#f39c12' if f >= 0.5 else '#e74c3c' for f in f1_scores]

    ax.barh(range(len(f1_scores)), f1_scores, color=colors)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=6)
    ax.set_xlabel('F1-Score')
    ax.set_title(f'F1-Score per gatunek (średnia: {avg_f1:.3f})')
    ax.axvline(x=avg_f1, color='blue', linestyle='--', alpha=0.5, label=f'Średnia: {avg_f1:.3f}')
    ax.legend()
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    f1_path = os.path.join(args.output_dir, 'f1_per_class.png')
    plt.savefig(f1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ F1 per klasa: {f1_path}")

    # 3. Błędne predykcje
    error_indices = np.where(all_preds != all_labels)[0]
    if len(error_indices) > 0:
        n_show = min(args.show_errors, len(error_indices))
        sample_errors = np.random.choice(error_indices, n_show, replace=False)

        cols = 4
        rows = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        axes_flat = np.array(axes).flatten()

        for idx, ax in enumerate(axes_flat):
            if idx < n_show:
                i = sample_errors[idx]
                img = Image.open(all_paths[i]).convert('RGB')
                true_name = class_names[all_labels[i]]
                pred_name = class_names[all_preds[i]]
                confidence = all_probs[i][all_preds[i]]

                ax.imshow(img)
                ax.set_title(f'Prawda: {true_name}\nPredykcja: {pred_name}\n({confidence:.1%})',
                             fontsize=8, color='red')
            ax.axis('off')

        plt.suptitle(f'Błędne predykcje ({len(error_indices)}/{num_images} łącznie)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        err_path = os.path.join(args.output_dir, 'error_examples.png')
        plt.savefig(err_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Błędne predykcje: {err_path}")

    # 4. Poprawne predykcje (przykłady)
    correct_indices = np.where(all_preds == all_labels)[0]
    if len(correct_indices) > 0:
        n_show = min(16, len(correct_indices))
        sample_correct = np.random.choice(correct_indices, n_show, replace=False)

        cols = 4
        rows = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes_flat = np.array(axes).flatten()

        for idx, ax in enumerate(axes_flat):
            if idx < n_show:
                i = sample_correct[idx]
                img = Image.open(all_paths[i]).convert('RGB')
                pred_name = class_names[all_preds[i]]
                confidence = all_probs[i][all_preds[i]]

                ax.imshow(img)
                ax.set_title(f'{pred_name}\n({confidence:.1%})',
                             fontsize=8, color='green')
            ax.axis('off')

        plt.suptitle(f'Poprawne predykcje ({len(correct_indices)}/{num_images} łącznie)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        ok_path = os.path.join(args.output_dir, 'correct_examples.png')
        plt.savefig(ok_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Poprawne predykcje: {ok_path}")

    # ==========================
    # ZAPIS RAPORTU
    # ==========================
    report = {
        'top1_accuracy': float(top1_acc),
        'top5_accuracy': float(top5_acc),
        'avg_precision': float(avg_precision),
        'avg_recall': float(avg_recall),
        'avg_f1': float(avg_f1),
        'total_images': int(num_images),
        'total_errors': int(len(error_indices)),
        'num_classes': num_classes,
        'class_metrics': class_metrics,
    }

    report_path = os.path.join(args.output_dir, 'report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Raport JSON: {report_path}")

    # Podsumowanie
    print(f"\n{'='*60}")
    print(f"🎉 PODSUMOWANIE")
    print(f"{'='*60}")
    print(f"  Top-1 Accuracy:  {top1_acc:.1%}")
    print(f"  Top-5 Accuracy:  {top5_acc:.1%}")
    print(f"  Średnie F1:      {avg_f1:.3f}")
    print(f"  Błędów:          {len(error_indices)}/{num_images}")
    print(f"  Wyniki w:        {args.output_dir}/")
    print(f"{'='*60}")


# ==============================================================================
# PREDYKCJA NA POJEDYNCZYM ZDJĘCIU
# ==============================================================================

def predict_single_image(model, image_path, class_names, transform, args, device):
    """Predykcja na jednym zdjęciu z wizualizacją."""

    if not os.path.exists(image_path):
        print(f"❌ Nie znaleziono zdjęcia: {image_path}")
        return

    print(f"\n{'='*60}")
    print(f"🔍 PREDYKCJA: {image_path}")
    print(f"{'='*60}")

    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_k_probs, top_k_indices = torch.topk(probs, min(args.top_k, len(class_names)))

    print(f"\n  {'Gatunek':<35} {'Pewność':>10}")
    print(f"  {'-'*47}")
    for i in range(top_k_probs.size(1)):
        name = class_names[top_k_indices[0][i].item()]
        conf = top_k_probs[0][i].item()
        bar = '█' * int(conf * 30)
        marker = ' ← 🏆' if i == 0 else ''
        print(f"  {name:<35} {conf:>8.1%}  {bar}{marker}")

    # Wizualizacja
    os.makedirs(args.output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(img)
    pred_name = class_names[top_k_indices[0][0].item()]
    pred_conf = top_k_probs[0][0].item()
    ax1.set_title(f'Predykcja: {pred_name} ({pred_conf:.1%})', fontsize=12, fontweight='bold')
    ax1.axis('off')

    names = [class_names[top_k_indices[0][i].item()] for i in range(top_k_probs.size(1))]
    confs = [top_k_probs[0][i].item() for i in range(top_k_probs.size(1))]
    colors = ['#2ecc71'] + ['#3498db'] * (len(confs) - 1)

    ax2.barh(names[::-1], confs[::-1], color=colors[::-1])
    ax2.set_xlim(0, 1)
    ax2.set_title(f'Top-{args.top_k} predykcji', fontsize=12)
    ax2.set_xlabel('Pewność')

    for i, (name, conf) in enumerate(zip(names[::-1], confs[::-1])):
        ax2.text(conf + 0.02, i, f'{conf:.1%}', va='center', fontsize=10)

    plt.tight_layout()

    basename = os.path.splitext(os.path.basename(image_path))[0]
    pred_path = os.path.join(args.output_dir, f'prediction_{basename}.png')
    plt.savefig(pred_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 Wizualizacja: {pred_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = get_args()

    print("=" * 60)
    print("🐦 PREDYKCJA — Klasyfikacja Gatunków Ptaków")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Urządzenie: {device}")

    # Ładowanie modelu
    model, class_names, checkpoint = load_model(args.model, device)
    if model is None:
        return

    # Transformacja
    transform = get_transform(args.image_size)

    # Tryb: pojedyncze zdjęcie lub cały zbiór testowy
    if args.image:
        predict_single_image(model, args.image, class_names, transform, args, device)
    else:
        evaluate_test_set(model, args.test_dir, class_names, transform, args, device)


if __name__ == '__main__':
    main()
