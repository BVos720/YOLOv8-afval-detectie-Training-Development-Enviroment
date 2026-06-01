"""
data_prep.py — Zet labels_train.csv om naar YOLO-formaat en maak dataset.yaml aan.

YOLO verwacht per afbeelding een .txt-bestand met op elke regel:
    class_id  cx_norm  cy_norm  w_norm  h_norm
Alle waarden zijn genormaliseerd (0.0–1.0) op de afbeeldingsbreedte/-hoogte.

Uitvoerstructuur:
    dataset_yolo/
    ├── images/
    │   ├── train/   ← 80% van de frames
    │   └── val/     ← 20% van de frames
    ├── labels/
    │   ├── train/
    │   └── val/
    └── dataset.yaml
"""

import os
import shutil
import cv2
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def bereid_dataset_voor(
    images_map: str,
    labels_csv: str,
    dataset_map: str,
    class_namen: dict,
    val_grootte: float = 0.2,
    seed: int = 42,
) -> str:
    """
    Converteer CSV-labels naar YOLO-formaat en schrijf de dataset-mappenstructuur.

    Args:
        images_map:  Map met bronafbeeldingen.
        labels_csv:  Pad naar labels_train.csv.
        dataset_map: Doel-map voor de YOLO-dataset (wordt aangemaakt).
        class_namen: Dict {class_id: naam} — bepaalt de klasse-volgorde.
        val_grootte: Fractie voor validatieset (bijv. 0.2 = 20%).
        seed:        Willekeurige seed voor reproduceerbaarheid.

    Returns:
        Pad naar het gegenereerde dataset.yaml bestand.
    """
    print("[data_prep] Labels laden...")
    df = pd.read_csv(labels_csv)
    _valideer_csv(df)

    # YOLO vereist dat class-IDs beginnen bij 0 en aaneengesloten zijn.
    # We mappen jouw class_ids naar 0-gebaseerde indices.
    gesorteerde_ids  = sorted(class_namen.keys())
    id_naar_yolo_idx = {cid: idx for idx, cid in enumerate(gesorteerde_ids)}
    yolo_klasse_namen = [class_namen[cid] for cid in gesorteerde_ids]

    # Splits frames in train en validatie
    alle_frames = df["frame"].unique().tolist()
    train_frames, val_frames = train_test_split(
        alle_frames, test_size=val_grootte, random_state=seed
    )
    print(f"[data_prep] {len(train_frames)} train-frames | {len(val_frames)} val-frames")

    # Maak mappenstructuur aan
    for split in ("train", "val"):
        os.makedirs(os.path.join(dataset_map, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dataset_map, "labels", split), exist_ok=True)

    # Verwerk elk frame
    frames_verwerkt = 0
    frames_overgeslagen = 0

    for split, frames in [("train", train_frames), ("val", val_frames)]:
        for frame in frames:
            src_pad = os.path.join(images_map, frame)
            if not os.path.exists(src_pad):
                frames_overgeslagen += 1
                continue

            # Lees afbeeldingsafmetingen voor normalisatie
            img = cv2.imread(src_pad)
            if img is None:
                frames_overgeslagen += 1
                continue
            hoogte, breedte = img.shape[:2]

            # Kopieer afbeelding naar juiste split-map
            dst_img = os.path.join(dataset_map, "images", split, frame)
            shutil.copy2(src_pad, dst_img)

            # Schrijf YOLO label-bestand
            frame_labels = df[df["frame"] == frame]
            regels = []
            for _, rij in frame_labels.iterrows():
                yolo_idx = id_naar_yolo_idx.get(rij["class_id"])
                if yolo_idx is None:
                    continue  # onbekende class_id → overslaan

                cx = ((rij["xmin"] + rij["xmax"]) / 2) / breedte
                cy = ((rij["ymin"] + rij["ymax"]) / 2) / hoogte
                w  = (rij["xmax"] - rij["xmin"]) / breedte
                h  = (rij["ymax"] - rij["ymin"]) / hoogte

                # Clip op [0, 1] om buiten-beeld annotaties te voorkomen
                cx, cy, w, h = (
                    max(0.0, min(1.0, cx)),
                    max(0.0, min(1.0, cy)),
                    max(0.0, min(1.0, w)),
                    max(0.0, min(1.0, h)),
                )
                regels.append(f"{yolo_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            naam_zonder_ext = os.path.splitext(frame)[0]
            label_pad = os.path.join(dataset_map, "labels", split, f"{naam_zonder_ext}.txt")
            with open(label_pad, "w") as f:
                f.write("\n".join(regels))

            frames_verwerkt += 1

    print(f"[data_prep] {frames_verwerkt} frames verwerkt | {frames_overgeslagen} overgeslagen")

    # Schrijf dataset.yaml
    yaml_pad = _schrijf_dataset_yaml(dataset_map, yolo_klasse_namen)
    print(f"[data_prep] dataset.yaml aangemaakt: {yaml_pad}")
    return yaml_pad


def _valideer_csv(df: pd.DataFrame) -> None:
    """Controleer of alle vereiste kolommen aanwezig zijn."""
    vereist = {"frame", "xmin", "xmax", "ymin", "ymax", "class_id"}
    ontbrekend = vereist - set(df.columns)
    if ontbrekend:
        raise ValueError(f"CSV mist kolommen: {ontbrekend}")


def _schrijf_dataset_yaml(dataset_map: str, klasse_namen: list[str]) -> str:
    """Genereer het dataset.yaml bestand dat YOLO nodig heeft."""
    inhoud = {
        "path":  os.path.abspath(dataset_map),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(klasse_namen),
        "names": klasse_namen,
    }
    yaml_pad = os.path.join(dataset_map, "dataset.yaml")
    with open(yaml_pad, "w") as f:
        yaml.dump(inhoud, f, default_flow_style=False, allow_unicode=True)
    return yaml_pad
