"""
🐦 Skrypt do ograniczenia datasetu do wybranych gatunków

Wybiera 25 gatunków z CUB-200-2011 i usuwa pozostałe foldery.
Możesz też podać własną listę gatunków.

Użycie:
    python select_species.py --data_dir ./data
    python select_species.py --data_dir ./data --num_species 30
    python select_species.py --data_dir ./data --species "Cardinal,Eagle,Owl,Parrot"
"""

import os
import sys
import stat
import shutil
import argparse
import random


def force_remove_readonly(func, path, exc_info):
    """Wymusza usunięcie plików tylko do odczytu (problem na Windows)."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


# 25 ciekawych, rozpoznawalnych gatunków z CUB-200-2011
# (nazwy odpowiadają folderom w datasecie)
RECOMMENDED_SPECIES = [
    "001.Black_footed_Albatross",
    "013.Bobolink",
    "014.Indigo_Bunting",
    "017.Cardinal",
    "024.Red_faced_Cormorant",
    "033.Yellow_billed_Cuckoo",
    "047.American_Goldfinch",
    "052.Pied_billed_Grebe",
    "056.Pine_Grosbeak",
    "065.Slaty_backed_Gull",
    "070.Green_Violetear",
    "073.Blue_Jay",
    "080.Green_Kingfisher",
    "087.Mallard",
    "094.White_breasted_Nuthatch",
    "106.Horned_Puffin",
    "112.Great_Grey_Shrike",
    "119.Field_Sparrow",
    "133.White_throated_Sparrow",
    "144.Common_Tern",
    "153.Philadelphia_Vireo",
    "169.Magnolia_Warbler",
    "188.Pileated_Woodpecker",
    "195.Carolina_Wren",
    "200.Common_Yellowthroat",
]


def select_species(data_dir, num_species=25, species_list=None, seed=42):
    """
    Zostawia tylko wybrane gatunki w data/train, data/val, data/test.
    Pozostałe foldery są usuwane.
    """
    random.seed(seed)

    splits = ['train', 'val', 'test']
    existing_splits = [s for s in splits if os.path.exists(os.path.join(data_dir, s))]

    if not existing_splits:
        print(f"❌ Nie znaleziono folderów train/val/test w: {data_dir}")
        return

    # Pobieramy listę wszystkich gatunków
    first_split = existing_splits[0]
    all_species = sorted(os.listdir(os.path.join(data_dir, first_split)))
    all_species = [s for s in all_species if os.path.isdir(os.path.join(data_dir, first_split, s))]

    print(f"{'='*60}")
    print(f"🐦 Ograniczanie datasetu")
    print(f"{'='*60}")
    print(f"  Folder:              {data_dir}")
    print(f"  Gatunków obecnie:    {len(all_species)}")

    # Wybór gatunków
    if species_list:
        # Użytkownik podał własną listę
        selected = []
        for sp in species_list:
            matches = [s for s in all_species if sp.lower() in s.lower()]
            if matches:
                selected.append(matches[0])
                print(f"  ✅ Znaleziono: {sp} → {matches[0]}")
            else:
                print(f"  ⚠️  Nie znaleziono: {sp}")
        selected = list(set(selected))
    else:
        # Używamy rekomendowanej listy lub losujemy
        selected = [s for s in RECOMMENDED_SPECIES if s in all_species]

        if len(selected) < num_species:
            # Jeśli za mało z rekomendowanych, dolosowujemy
            remaining = [s for s in all_species if s not in selected]
            random.shuffle(remaining)
            selected += remaining[:num_species - len(selected)]

        selected = selected[:num_species]

    selected = sorted(selected)
    to_remove = [s for s in all_species if s not in selected]

    print(f"  Wybranych gatunków:  {len(selected)}")
    print(f"  Do usunięcia:        {len(to_remove)}")

    # Wyświetlamy wybrane gatunki
    print(f"\n  📋 Wybrane gatunki:")
    for i, sp in enumerate(selected, 1):
        # Liczymy zdjęcia
        count = 0
        for split in existing_splits:
            sp_dir = os.path.join(data_dir, split, sp)
            if os.path.exists(sp_dir):
                count += len([f for f in os.listdir(sp_dir) if os.path.isfile(os.path.join(sp_dir, f))])
        print(f"     {i:>2}. {sp:<45} ({count} zdjęć)")

    # Potwierdzenie
    print(f"\n⚠️  UWAGA: {len(to_remove)} folderów gatunków zostanie USUNIĘTE!")
    response = input("  Kontynuować? (t/n): ").strip().lower()

    if response not in ['t', 'tak', 'y', 'yes']:
        print("  ❌ Anulowano.")
        return

    # Usuwanie
    removed_count = 0
    for split in existing_splits:
        for species in to_remove:
            species_dir = os.path.join(data_dir, split, species)
            if os.path.exists(species_dir):
                shutil.rmtree(species_dir, onexc=force_remove_readonly)
                removed_count += 1

    # Podsumowanie
    print(f"\n{'='*60}")
    print(f"✅ GOTOWE!")
    print(f"{'='*60}")
    print(f"  Usunięto: {removed_count} folderów")
    print(f"  Zostało:  {len(selected)} gatunków")

    for split in existing_splits:
        split_dir = os.path.join(data_dir, split)
        remaining = [s for s in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, s))]
        total_imgs = sum(
            len([f for f in os.listdir(os.path.join(split_dir, s))
                 if os.path.isfile(os.path.join(split_dir, s, f))])
            for s in remaining
        )
        print(f"  {split}: {len(remaining)} gatunków, {total_imgs} zdjęć")

    print(f"\n  Możesz teraz uruchomić trening:")
    print(f"  python train.py --data_dir {data_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='🐦 Wybór gatunków z datasetu')

    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Ścieżka do folderu z danymi')
    parser.add_argument('--num_species', type=int, default=25,
                        help='Ile gatunków zostawić (domyślnie 25)')
    parser.add_argument('--species', type=str, default=None,
                        help='Własna lista gatunków oddzielona przecinkami (np. "Cardinal,Eagle,Owl")')
    parser.add_argument('--seed', type=int, default=42,
                        help='Ziarno losowości')

    args = parser.parse_args()

    species_list = None
    if args.species:
        species_list = [s.strip() for s in args.species.split(',')]

    select_species(
        data_dir=args.data_dir,
        num_species=args.num_species,
        species_list=species_list,
        seed=args.seed
    )