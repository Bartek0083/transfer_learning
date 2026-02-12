"""
🐦 Skrypt do podziału datasetu CUB-200-2011 na train/val/test

Użycie:
    python split_dataset.py --source ./CUB_200_2011/images --output ./data
    python split_dataset.py --source ./CUB_200_2011/images --output ./data --train 0.7 --val 0.15 --test 0.15

Skrypt:
1. Skanuje folder źródłowy (każdy podfolder = 1 gatunek)
2. Losowo dzieli zdjęcia na train/val/test
3. Kopiuje (lub przenosi) pliki do nowej struktury

Wynikowa struktura:
    data/
    ├── train/
    │   ├── 001.Black_footed_Albatross/
    │   ├── 002.Laysan_Albatross/
    │   └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...
"""

import os
import shutil
import argparse
import random
from pathlib import Path


def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                  seed=42, move=False, min_images=3):
    """
    Dzieli dataset na train/val/test.

    Args:
        source_dir:   Ścieżka do folderu z gatunkami (np. CUB_200_2011/images/)
        output_dir:   Ścieżka docelowa (np. ./data)
        train_ratio:  Procent danych treningowych (domyślnie 0.7)
        val_ratio:    Procent danych walidacyjnych (domyślnie 0.15)
        test_ratio:   Procent danych testowych (domyślnie 0.15)
        seed:         Ziarno losowości (dla powtarzalności)
        move:         True = przenieś pliki, False = kopiuj (bezpieczniej)
        min_images:   Minimalny wymagany zbiór zdjęć na gatunek
    """
    # Walidacja proporcji
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        print(f"⚠️  Proporcje nie sumują się do 1.0 ({total:.2f}). Normalizuję...")
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total

    random.seed(seed)

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    if not source_path.exists():
        print(f"❌ Nie znaleziono katalogu źródłowego: {source_dir}")
        print(f"   Upewnij się, że ścieżka prowadzi do folderu z gatunkami.")
        return

    # Znajdujemy wszystkie foldery z gatunkami
    species_dirs = sorted([d for d in source_path.iterdir() if d.is_dir()])

    if not species_dirs:
        print(f"❌ Brak podfolderów w {source_dir}")
        print(f"   Każdy gatunek powinien mieć swój folder ze zdjęciami.")
        return

    print(f"{'='*60}")
    print(f"🐦 Podział datasetu na train/val/test")
    print(f"{'='*60}")
    print(f"  Źródło:     {source_dir}")
    print(f"  Cel:        {output_dir}")
    print(f"  Proporcje:  train={train_ratio:.0%} / val={val_ratio:.0%} / test={test_ratio:.0%}")
    print(f"  Gatunki:    {len(species_dirs)}")
    print(f"  Tryb:       {'przenoszenie' if move else 'kopiowanie'}")
    print(f"  Seed:       {seed}")
    print(f"{'='*60}\n")

    # Rozszerzenia obrazów
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    # Statystyki
    total_images = 0
    stats = {'train': 0, 'val': 0, 'test': 0}
    skipped_species = []

    for species_dir in species_dirs:
        species_name = species_dir.name

        # Zbieramy wszystkie obrazy z folderu
        images = [f for f in species_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in image_extensions]

        if len(images) < min_images:
            skipped_species.append((species_name, len(images)))
            continue

        # Losowo mieszamy
        random.shuffle(images)

        # Obliczamy liczbę zdjęć dla każdego zbioru
        n_total = len(images)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))
        n_test = n_total - n_train - n_val  # Reszta idzie do testu

        # Zabezpieczenie — każdy zbiór musi mieć min. 1 obraz
        if n_test < 1:
            n_test = 1
            n_train = n_total - n_val - n_test

        # Podział
        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        # Kopiowanie/przenoszenie
        for split_name, split_images in splits.items():
            dest_dir = output_path / split_name / species_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_images:
                dest_path = dest_dir / img_path.name

                if move:
                    shutil.move(str(img_path), str(dest_path))
                else:
                    shutil.copy2(str(img_path), str(dest_path))

            stats[split_name] += len(split_images)

        total_images += n_total
        print(f"  ✅ {species_name:<40} "
              f"train:{len(splits['train']):>4} | "
              f"val:{len(splits['val']):>4} | "
              f"test:{len(splits['test']):>4}  "
              f"(łącznie: {n_total})")

    # Podsumowanie
    print(f"\n{'='*60}")
    print(f"📊 PODSUMOWANIE")
    print(f"{'='*60}")
    print(f"  Przetworzonych gatunków: {len(species_dirs) - len(skipped_species)}")
    print(f"  Obrazów łącznie:         {total_images}")
    print(f"  ├── train:               {stats['train']} ({stats['train']/total_images*100:.1f}%)")
    print(f"  ├── val:                 {stats['val']} ({stats['val']/total_images*100:.1f}%)")
    print(f"  └── test:                {stats['test']} ({stats['test']/total_images*100:.1f}%)")

    if skipped_species:
        print(f"\n  ⚠️  Pominięto {len(skipped_species)} gatunków (za mało zdjęć):")
        for name, count in skipped_species:
            print(f"     - {name} ({count} zdjęć, wymagane min. {min_images})")

    print(f"\n✅ Gotowe! Dane zapisane w: {output_dir}/")
    print(f"   Możesz teraz uruchomić trening:")
    print(f"   python train.py --data_dir {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='🐦 Podział datasetu ptaków na train/val/test')

    parser.add_argument('--source', type=str, required=True,
                        help='Ścieżka do folderu z gatunkami (np. CUB_200_2011/images/)')
    parser.add_argument('--output', type=str, default='./data',
                        help='Ścieżka docelowa (domyślnie: ./data)')
    parser.add_argument('--train', type=float, default=0.7,
                        help='Proporcja danych treningowych (domyślnie: 0.7)')
    parser.add_argument('--val', type=float, default=0.15,
                        help='Proporcja danych walidacyjnych (domyślnie: 0.15)')
    parser.add_argument('--test', type=float, default=0.15,
                        help='Proporcja danych testowych (domyślnie: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Ziarno losowości (domyślnie: 42)')
    parser.add_argument('--move', action='store_true',
                        help='Przenieś pliki zamiast kopiować (oszczędza dysk)')
    parser.add_argument('--min_images', type=int, default=3,
                        help='Min. zdjęć na gatunek (domyślnie: 3)')

    args = parser.parse_args()

    split_dataset(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
        move=args.move,
        min_images=args.min_images
    )
