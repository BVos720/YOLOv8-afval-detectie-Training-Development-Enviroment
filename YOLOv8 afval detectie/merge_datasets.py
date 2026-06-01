"""
merge_datasets.py — Voeg twee YOLOv8 datasets samen.

Gebruik:
    python merge_datasets.py

Wat het doet:
    1. Neemt de bestaande dataset (Garbage Classification)
    2. Neemt de nieuwe metal dataset
    3. Mapt alle metaalklassen → METAL (klasse-index 4)
    4. Combineert train/valid/test splits
    5. Maakt een nieuwe data.yaml aan
"""

import os
import shutil
import glob
import yaml
import random

# ── Paden ──────────────────────────────────────────────────────────────────
BESTAANDE_DATASET = r"C:\Projecten\Periode 4\YOLOv8 afval detectie\Dataset\GARBAGE CLASSIFICATION 3.v2-gc1.yolov8"
METAL_DATASET     = r"C:\Projecten\Periode 4\YOLOv8 afval detectie\Dataset\Metal.yolov8"
UITVOER_DATASET   = r"C:\Projecten\Periode 4\YOLOv8 afval detectie\Dataset\samengevoegd"

# ── Klassen in de samengevoegde dataset ────────────────────────────────────
KLASSEN = ['BIODEGRADABLE', 'CARDBOARD', 'CLOTH', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
METAL_IDX = KLASSEN.index('METAL')  # = 4

# ── Klasse-mapping van de metal dataset naar onze METAL index ──────────────
# Pas dit aan als de metaalklassen andere namen/volgorde hebben
METAL_KLASSEN_MAPPING = {
    0: METAL_IDX,   # Aluminium Foil    → METAL
    1: METAL_IDX,   # Metal Bottle Cap  → METAL
    2: METAL_IDX,   # Metal Can Bottle  → METAL
    3: METAL_IDX,   # Metal Can Container → METAL
    4: METAL_IDX,   # Metal Container Cap → METAL
}


def maak_mappen(uitvoer: str):
    for split in ("train", "valid", "test"):
        os.makedirs(os.path.join(uitvoer, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(uitvoer, split, "labels"), exist_ok=True)
    print(f"[merge] Uitvoermappen aangemaakt in: {uitvoer}")


def kopieer_dataset(bron: str, uitvoer: str, klasse_mapping: dict = None, prefix: str = ""):
    """Kopieer afbeeldingen en labels van bron naar uitvoer.

    klasse_mapping: {oude_idx: nieuwe_idx} — None betekent geen wijziging
    prefix: toegevoegd aan bestandsnaam om duplicaten te voorkomen
    """
    totaal = 0
    for split in ("train", "valid", "test"):
        img_map = os.path.join(bron, split, "images")
        lbl_map = os.path.join(bron, split, "labels")

        if not os.path.exists(img_map):
            print(f"[merge] Map niet gevonden, overgeslagen: {img_map}")
            continue

        afbeeldingen = glob.glob(os.path.join(img_map, "*.jpg")) + \
                       glob.glob(os.path.join(img_map, "*.jpeg")) + \
                       glob.glob(os.path.join(img_map, "*.png"))

        # Als alleen train bestaat: splits 80/20 voor train/valid
        if split == "valid" and not afbeeldingen:
            trn_map = os.path.join(bron, "train", "images")
            alle = glob.glob(os.path.join(trn_map, "*.jpg")) + \
                   glob.glob(os.path.join(trn_map, "*.jpeg")) + \
                   glob.glob(os.path.join(trn_map, "*.png"))
            random.shuffle(alle)
            afbeeldingen = alle[int(len(alle) * 0.8):]  # laatste 20% als valid
        elif split == "train" and os.path.exists(os.path.join(bron, "train", "images")):
            alle = afbeeldingen[:]
            random.shuffle(alle)
            afbeeldingen = alle[:int(len(alle) * 0.8)]  # eerste 80% als train

        for img_pad in afbeeldingen:
            bestandsnaam = os.path.splitext(os.path.basename(img_pad))[0]
            ext          = os.path.splitext(img_pad)[1]
            nieuwe_naam  = f"{prefix}{bestandsnaam}"

            # Kopieer afbeelding
            dst_img = os.path.join(uitvoer, split, "images", f"{nieuwe_naam}{ext}")
            shutil.copy2(img_pad, dst_img)

            # Kopieer en hermap label
            src_lbl = os.path.join(lbl_map, f"{bestandsnaam}.txt")
            dst_lbl = os.path.join(uitvoer, split, "labels", f"{nieuwe_naam}.txt")

            if os.path.exists(src_lbl):
                if klasse_mapping:
                    _hermap_label(src_lbl, dst_lbl, klasse_mapping)
                else:
                    shutil.copy2(src_lbl, dst_lbl)
            else:
                # Leeg labelbestand (geen objecten)
                open(dst_lbl, 'w').close()

            totaal += 1

    print(f"[merge] {prefix or 'bestaand'}: {totaal} afbeeldingen gekopieerd")
    return totaal


def _hermap_label(src: str, dst: str, mapping: dict):
    """Herschrijf label txt met nieuwe klasse-indices."""
    regels_uit = []
    with open(src, 'r') as f:
        for regel in f:
            delen = regel.strip().split()
            if not delen:
                continue
            oude_idx = int(delen[0])
            nieuwe_idx = mapping.get(oude_idx, oude_idx)
            regels_uit.append(f"{nieuwe_idx} {' '.join(delen[1:])}")

    with open(dst, 'w') as f:
        f.write("\n".join(regels_uit))


def schrijf_yaml(uitvoer: str, klassen: list):
    inhoud = {
        "path":  os.path.abspath(uitvoer),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    len(klassen),
        "names": klassen,
    }
    yaml_pad = os.path.join(uitvoer, "data.yaml")
    with open(yaml_pad, 'w') as f:
        yaml.dump(inhoud, f, default_flow_style=False, allow_unicode=True)
    print(f"[merge] data.yaml aangemaakt: {yaml_pad}")
    return yaml_pad


def main():
    print("=" * 50)
    print("  Dataset Samenvoegen")
    print("=" * 50)

    if not os.path.exists(METAL_DATASET):
        print(f"\n[!] Metal dataset niet gevonden op: {METAL_DATASET}")
        print("    Pas METAL_DATASET aan in dit script naar het juiste pad.")
        return

    # Stap 1: mappen aanmaken
    maak_mappen(UITVOER_DATASET)

    # Stap 2: bestaande dataset kopiëren (geen hermap nodig)
    kopieer_dataset(BESTAANDE_DATASET, UITVOER_DATASET, klasse_mapping=None, prefix="gc_")

    # Stap 3: metal dataset kopiëren met hermap naar METAL
    kopieer_dataset(METAL_DATASET, UITVOER_DATASET, klasse_mapping=METAL_KLASSEN_MAPPING, prefix="mt_")

    # Stap 4: data.yaml schrijven
    yaml_pad = schrijf_yaml(UITVOER_DATASET, KLASSEN)

    print("\n" + "=" * 50)
    print("  Klaar! Samengevoegde dataset:")
    print(f"  {UITVOER_DATASET}")
    print(f"  Klassen: {KLASSEN}")
    print("=" * 50)
    print("\nPas nu in config.py aan:")
    print(f'  DATA_YAML = r"{yaml_pad}"')


if __name__ == "__main__":
    main()
