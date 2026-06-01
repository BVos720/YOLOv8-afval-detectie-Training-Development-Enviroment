"""
main.py — Start de volledige pipeline vanuit VS Code:

  Stap 1 — Data voorbereiden  : CSV → YOLO-formaat + dataset.yaml
  Stap 2 — Model trainen      : YOLOv8n fine-tunen op jouw dataset
  Stap 3 — Model uploaden     : best.pt → Hugging Face Hub
  Stap 4 — Model exporteren   : best.pt → ONNX → TFLite INT8 → Vela

Gebruik:
    python main.py                      ← volledige pipeline (prep+train+upload)
    python main.py --stap prep          ← alleen data voorbereiden
    python main.py --stap train         ← alleen trainen (data al klaar)
    python main.py --stap upload        ← alleen uploaden (model al getraind)
    python main.py --stap export        ← alleen exporteren naar Vela TFLite
    python main.py --stap alles         ← prep + train + upload + export

Exporteren met bestaand model (geen training nodig):
    python main.py --stap export --model C:\Projecten\Periode 4\YOLOv8 afval detectie\runs\yolov8n_training\weights\best.pt
    python main.py --stap export --tflite pad\naar\model_int8.tflite
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import config
from data_prep    import bereid_dataset_voor
from trainer      import train
from uploader     import upload_naar_huggingface
from export_vela  import exporteer


def stap_prep() -> str:
    """Data voorbereiden en dataset.yaml aanmaken."""
    print("\n" + "─" * 55)
    print("  STAP 1 — Data voorbereiden")
    print("─" * 55)
    yaml_pad = bereid_dataset_voor(
        images_map  = config.IMAGES_MAP,
        labels_csv  = config.LABELS_CSV,
        dataset_map = config.DATASET_MAP,
        class_namen = config.CLASS_NAMEN,
        val_grootte = config.VAL_GROOTTE,
        seed        = config.RANDOM_SEED,
    )
    return yaml_pad


def stap_train(yaml_pad: str) -> str:
    """YOLOv8n trainen op de voorbereide dataset."""
    print("\n" + "─" * 55)
    print("  STAP 2 — Model trainen (YOLOv8n)")
    print("─" * 55)
    best_pt = train(
        yaml_pad   = yaml_pad,
        runs_map   = config.RUNS_MAP,
        epochs     = config.EPOCHS,
        batch_size = config.BATCH_SIZE,
        img_size   = config.IMG_SIZE,
        workers    = config.WORKERS,
    )
    return best_pt


def stap_upload(best_pt: str, yaml_pad: str) -> str:
    """Getraind model uploaden naar Hugging Face Hub."""
    print("\n" + "─" * 55)
    print("  STAP 3 — Uploaden naar Hugging Face Hub")
    print("─" * 55)

    if config.HF_TOKEN.startswith("hf_xxx"):
        print("[!] HF_TOKEN is nog niet ingesteld in config.py — upload overgeslagen.")
        return ""

    repo_url = upload_naar_huggingface(
        model_pad   = best_pt,
        yaml_pad    = yaml_pad,
        hf_token    = config.HF_TOKEN,
        hf_username = config.HF_USERNAME,
        repo_naam   = config.HF_REPO_NAAM,
        class_namen = config.CLASS_NAMEN,
        epochs      = config.EPOCHS,
        img_size    = config.IMG_SIZE,
    )
    return repo_url


def stap_export(model_pad: str = None, tflite_pad: str = None) -> str:
    """Exporteer best.pt naar Vela TFLite voor Grove Vision AI V2."""
    print("\n" + "─" * 55)
    print("  STAP 4 — Exporteren naar Vela TFLite")
    print("─" * 55)
    finale_pad = exporteer(model_pad=model_pad, tflite_pad=tflite_pad)
    return finale_pad or ""


def main(stap: str, model_pad: str = None, tflite_pad: str = None):
    print("=" * 55)
    print("  YOLOv8n — Afval Detectie Pipeline")
    print("=" * 55)

    yaml_pad = config.DATA_YAML
    best_pt  = os.path.join(config.RUNS_MAP, "yolov8n_training", "weights", "best.pt")
    repo_url = ""
    vela_pad = ""

    if stap in ("alles", "prep"):
        yaml_pad = stap_prep()

    if stap in ("alles", "train"):
        best_pt = stap_train(yaml_pad)

    if stap in ("alles", "upload"):
        repo_url = stap_upload(best_pt, yaml_pad)

    if stap in ("alles", "export"):
        # --model en --tflite flags worden doorgegeven als ze meegegeven zijn
        vela_pad = stap_export(model_pad=model_pad, tflite_pad=tflite_pad)

    # ── Samenvatting ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Klaar!")
    print("=" * 55)

    if stap in ("alles", "train") and os.path.exists(best_pt):
        print(f"  Model      : {best_pt}")
    if stap in ("alles", "upload") and repo_url:
        print(f"  HF repo    : {repo_url}")
    if stap in ("alles", "export") and vela_pad:
        print(f"  Vela model : {vela_pad}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8n afval detectie pipeline")
    parser.add_argument(
        "--stap",
        choices=["alles", "prep", "train", "upload", "export"],
        default="train",
        help="Welke stap uitvoeren (standaard: train)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Pad naar een bestaand best.pt bestand. "
            "Alleen van toepassing bij --stap export. "
            "Standaard: automatisch zoeken in RUNS_MAP."
        ),
    )
    parser.add_argument(
        "--tflite",
        type=str,
        default=None,
        help=(
            "Pad naar een bestaand TFLite INT8 bestand. "
            "Alleen van toepassing bij --stap export. "
            "Hiermee sla je de ONNX en TFLite conversie over "
            "en ga je direct naar Vela."
        ),
    )
    args = parser.parse_args()
    main(stap=args.stap, model_pad=args.model, tflite_pad=args.tflite)
