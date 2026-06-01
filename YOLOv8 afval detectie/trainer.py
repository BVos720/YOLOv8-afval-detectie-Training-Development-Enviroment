"""
trainer.py — Train een YOLOv8n model op de voorbereide dataset.

Gebruikt het voorgetrainde yolov8n.pt als startpunt (transfer learning).
Het nano model is klein genoeg voor de Grove Vision AI V2 microcontroller.
Het beste model wordt opgeslagen als best.pt in de runs-map.
"""

import os
import torch
import torch.backends.cudnn as cudnn
from ultralytics import YOLO

# cuDNN stabiliteitsfixes voor CUDA 13.0
cudnn.benchmark = False
cudnn.deterministic = True
cudnn.enabled = False  # cuDNN uitzetten, PyTorch gebruikt dan eigen CUDA kernels


def train(
    yaml_pad: str,
    runs_map: str,
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 192,
    workers: int = 0,
) -> str:
    """
    Train YOLOv8n (nano) en retourneer het pad naar het beste model (best.pt).

    YOLOv8n is vereist voor de Grove Vision AI V2 microcontroller:
    - Klein genoeg voor het geheugen (nano ~6MB INT8)
    - 192x192 inputgrootte voor de camera module

    Args:
        yaml_pad:   Pad naar dataset.yaml (uitvoer van data_prep).
        runs_map:   Map waar trainingsresultaten worden opgeslagen.
        epochs:     Aantal trainingsepochs.
        batch_size: Batch-grootte (16 past makkelijk in 6GB VRAM voor nano).
        img_size:   Invoergrootte — 192 voor Grove Vision AI V2.
        workers:    Aantal dataloader-threads (0 op Windows).

    Returns:
        Pad naar het beste model-gewichtenbestand (best.pt).
    """
    print("[trainer] YOLOv8n laden (yolov8n.pt) — nano model voor embedded...")
    # Start vanaf het officiële nano voorgetrainde model
    model = YOLO("yolov8n.pt")
    print("[trainer] Starten vanaf yolov8n.pt (transfer learning)")

    print(f"[trainer] Training starten: {epochs} epochs, batch {batch_size}, img {img_size}px")
    model.train(
        data       = yaml_pad,
        epochs     = epochs,
        batch      = batch_size,
        imgsz      = img_size,
        workers    = workers,
        project    = runs_map,
        name       = "yolov8n_training",
        exist_ok   = True,      # overschrijf vorige run
        pretrained = True,      # transfer learning vanaf yolov8n.pt
        verbose    = True,
        amp        = True,      # AMP aan voor minder geheugengebruik (FP16)
        device     = 0,         # GPU 0 expliciet instellen
    )

    best_pt = os.path.join(runs_map, "yolov8n_training", "weights", "best.pt")
    if not os.path.exists(best_pt):
        raise FileNotFoundError(
            f"best.pt niet gevonden op verwachte locatie: {best_pt}\n"
            "Controleer de runs-map op de werkelijke locatie."
        )

    print(f"[trainer] Klaar! Beste model: {best_pt}")
    return best_pt
