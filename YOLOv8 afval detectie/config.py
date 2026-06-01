"""
config.py — Alle instellingen op één plek.
Dit is het enige bestand dat je hoeft aan te passen.
"""

import os

# ── Paden ──────────────────────────────────────────────────────────────────
# Map met de ruwe afbeeldingen (niet nodig — dataset is al in YOLO-formaat)
IMAGES_MAP   = r"C:\Projecten\Periode 4\YOLOv8 afval detectie\Dataset\GARBAGE CLASSIFICATION 3.v2-gc1.yolov8\train\images"

# Labels CSV (niet nodig — dataset is al in YOLO-formaat)
LABELS_CSV   = r"C:\Projecten\Periode 4\YOLOv8 afval detectie\pad\naar\labels_train.csv"

# Bestaande Roboflow YOLO-dataset
DATASET_MAP  = r"C:\Projecten\Periode 4\YOLOv8 afval detectie\Dataset\GARBAGE CLASSIFICATION 3.v2-gc1.yolov8"

# Pad naar de data.yaml (direct gebruiken bij --stap train)
DATA_YAML    = r"C:\Projecten\Periode 4\YOLOv8 afval detectie\Dataset\samengevoegd\data.yaml"

# Waar de trainingsresultaten worden opgeslagen (weights, logs)
RUNS_MAP     = r"C:\Projecten\Periode 4\YOLOv8 afval detectie\runs"

# Waar de TFLite en Vela exportbestanden worden opgeslagen
UITVOER_MAP  = r"C:\Projecten\Periode 4\YOLOv8 afval detectie\tflite_export"

# ── Klasse-mapping ─────────────────────────────────────────────────────────
CLASS_NAMEN = {
    0: "BIODEGRADABLE",
    1: "CARDBOARD",
    2: "CLOTH",
    3: "GLASS",
    4: "METAL",
    5: "PAPER",
    6: "PLASTIC",
}

# ── Data-splitsing ─────────────────────────────────────────────────────────
VAL_GROOTTE  = 0.2    # 20% validatie, 80% training
RANDOM_SEED  = 42

# ── Trainingsparameters ────────────────────────────────────────────────────
EPOCHS       = 50     # nano traint snel, 50 epochs voor goede nauwkeurigheid
BATCH_SIZE   = 256    # nano gebruikt weinig VRAM, 256 benut de RTX 4050 beter
IMG_SIZE     = 192    # 192x192 vereist voor Grove Vision AI V2
WORKERS      = 0      # 0 op Windows (anders bevriest de training)

# ── Hugging Face ──────────────────────────────────────────────────────────
# Maak een token aan op https://huggingface.co/settings/tokens (write-rechten)
HF_TOKEN     = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"    # ← aanpassen
HF_USERNAME  = "jouw-gebruikersnaam"                # ← aanpassen
HF_REPO_NAAM = "yolov8n-afval-detector"            # naam van de nieuwe repo
