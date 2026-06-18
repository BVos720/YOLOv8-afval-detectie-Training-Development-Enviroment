"""
train_from_api.py — Haal training afbeeldingen op uit de AIAPI en train YOLOv8.

Gebruik:
    python train_from_api.py --epochs 50 --val-split 0.2 --batch-size 16 --img-size 192

De API key moet in een .env bestand staan of als omgevingsvariabele:
    TRAINING_API_KEY=jouw-sleutel
    API_BASE_URL=https://jouw-api.azurewebsites.net
"""

import os
import io
import argparse
import requests
import yaml
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
from trainer import train

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5000")
API_KEY      = os.getenv("TRAINING_API_KEY", "")
HEADERS      = {"X-Api-Key": API_KEY}

LABEL_NAMEN = ["Cardboard", "Garbage", "Glass", "Metal", "Paper", "Plastic", "Trash"]
LABEL_NAAR_ID = {naam: idx for idx, naam in enumerate(LABEL_NAMEN)}


def haal_alle_afbeeldingen_op() -> list[dict]:
    """Haal metadata + bounding boxes op voor alle training afbeeldingen."""
    url = f"{API_BASE_URL}/api/trainingimage"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_afbeelding(image_id: int) -> bytes:
    """Download de ruwe bytes van één afbeelding."""
    url = f"{API_BASE_URL}/api/trainingimage/{image_id}/image"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.content


def bereid_dataset_voor(
    entries: list[dict],
    dataset_map: str,
    val_grootte: float,
    seed: int = 42,
) -> str:
    """
    Sla afbeeldingen + YOLO label-bestanden op en schrijf dataset.yaml.
    Geeft het pad naar dataset.yaml terug.
    """
    dataset_pad = Path(dataset_map)
    if dataset_pad.exists():
        shutil.rmtree(dataset_pad)

    for split in ("train", "val"):
        (dataset_pad / "images" / split).mkdir(parents=True)
        (dataset_pad / "labels" / split).mkdir(parents=True)

    # Splits in train/val
    train_entries, val_entries = train_test_split(
        entries, test_size=val_grootte, random_state=seed
    )
    print(f"[data] {len(train_entries)} train | {len(val_entries)} val")

    for split, subset in [("train", train_entries), ("val", val_entries)]:
        for entry in subset:
            img_bytes = download_afbeelding(entry["id"])
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            breedte, hoogte = img.size

            bestandsnaam = Path(entry["fileName"]).stem
            img.save(dataset_pad / "images" / split / f"{bestandsnaam}.jpg")

            regels = []
            for box in entry.get("boundingBoxes", []):
                label_id = LABEL_NAAR_ID.get(box["label"])
                if label_id is None:
                    print(f"  [!] Onbekend label: {box['label']} — overgeslagen")
                    continue
                regels.append(
                    f"{label_id} {box['centerX']:.6f} {box['centerY']:.6f} "
                    f"{box['width']:.6f} {box['height']:.6f}"
                )

            label_pad = dataset_pad / "labels" / split / f"{bestandsnaam}.txt"
            label_pad.write_text("\n".join(regels))

        print(f"[data] {split}: {len(subset)} afbeeldingen verwerkt")

    # dataset.yaml
    yaml_pad = dataset_pad / "dataset.yaml"
    yaml_inhoud = {
        "path":  str(dataset_pad.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(LABEL_NAMEN),
        "names": LABEL_NAMEN,
    }
    yaml_pad.write_text(yaml.dump(yaml_inhoud, allow_unicode=True))
    print(f"[data] dataset.yaml aangemaakt: {yaml_pad}")
    return str(yaml_pad)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 vanuit de AIAPI dataset")
    parser.add_argument("--epochs",     type=int,   default=50,   help="Aantal epochs (standaard 50)")
    parser.add_argument("--val-split",  type=float, default=0.2,  help="Validatiefractie 0–1 (standaard 0.2)")
    parser.add_argument("--batch-size", type=int,   default=16,   help="Batch-grootte (standaard 16)")
    parser.add_argument("--img-size",   type=int,   default=192,  help="Invoergrootte in pixels (standaard 192)")
    parser.add_argument("--dataset-map", type=str,  default=r"C:\Projecten\Periode 4\YOLOv8 afval detectie\dataset_api",
                        help="Tijdelijke map voor de dataset")
    parser.add_argument("--runs-map",   type=str,   default=r"C:\Projecten\Periode 4\YOLOv8 afval detectie\runs",
                        help="Map voor trainingsresultaten")
    args = parser.parse_args()

    if not API_KEY:
        raise RuntimeError(
            "TRAINING_API_KEY is niet ingesteld.\n"
            "Voeg het toe als omgevingsvariabele of in een .env bestand."
        )

    print(f"[main] API: {API_BASE_URL}")
    print(f"[main] Epochs: {args.epochs} | Val-split: {args.val_split} | "
          f"Batch: {args.batch_size} | Img: {args.img_size}px")

    print("[main] Afbeeldingen ophalen van API...")
    entries = haal_alle_afbeeldingen_op()
    print(f"[main] {len(entries)} afbeeldingen gevonden")

    if len(entries) == 0:
        raise RuntimeError("Geen training afbeeldingen gevonden in de database.")

    yaml_pad = bereid_dataset_voor(entries, args.dataset_map, args.val_split)

    print("[main] Training starten...")
    best_pt = train(
        yaml_pad=yaml_pad,
        runs_map=args.runs_map,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
    )
    print(f"[main] Klaar! Beste model: {best_pt}")


if __name__ == "__main__":
    main()
