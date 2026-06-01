"""
export_vela.py — Exporteer getraind YOLOv8n model naar Vela TFLite
                 voor de Grove Vision AI V2.

Stappen:
    1. best.pt  →  ONNX
    2. ONNX     →  TFLite INT8  (via onnx2tf)
    3. TFLite   →  Vela TFLite  (geoptimaliseerd voor Ethos-U55 NPU)

Gebruik — standalone (zonder training):
    python export_vela.py
    python export_vela.py --model pad\naar\best.pt

Gebruik — via main.py:
    python main.py --stap export
    python main.py --stap export --model pad\naar\best.pt
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

import config


# ── Vela instellingen voor Grove Vision AI V2 ──────────────────────────────
VELA_ACCELERATOR  = "ethos-u55-64"
VELA_SYSTEM       = "Ethos_U55_High_End_Embedded"
VELA_MEMORY_MODE  = "Shared_Sram"


# ─────────────────────────────────────────────────────────────────────────────
#  HULPFUNCTIES
# ─────────────────────────────────────────────────────────────────────────────

def _print_stap(nummer: int, naam: str):
    print("\n" + "─" * 55)
    print(f"  STAP {nummer} — {naam}")
    print("─" * 55)


def _zoek_best_pt() -> str:
    """Zoek automatisch het meest recente best.pt in de runs-map."""
    zoekpad = os.path.join(config.RUNS_MAP, "**", "best.pt")
    gevonden = glob.glob(zoekpad, recursive=True)
    if not gevonden:
        return ""
    # Meest recent gewijzigd
    return max(gevonden, key=os.path.getmtime)


def _check_dependency(package: str, import_pad: str = None) -> bool:
    """Controleer of een Python package beschikbaar is."""
    import importlib
    try:
        importlib.import_module(import_pad or package)
        return True
    except ImportError:
        return False


def _vela_commando() -> list[str] | None:
    """Geef het vela commando terug (cli of module), of None als niet gevonden."""
    if shutil.which("vela"):
        return ["vela"]
    result = subprocess.run(
        [sys.executable, "-m", "ethosu.vela.vela", "--version"],
        capture_output=True
    )
    if result.returncode == 0:
        return [sys.executable, "-m", "ethosu.vela.vela"]
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  STAP 1 — best.pt → ONNX
# ─────────────────────────────────────────────────────────────────────────────

def stap1_naar_onnx(model_pad: str) -> str:
    _print_stap(1, "Exporteren naar ONNX")

    from ultralytics import YOLO

    print(f"[export] Model laden : {model_pad}")
    print(f"[export] Imgsize     : {config.IMG_SIZE}x{config.IMG_SIZE}")

    os.makedirs(config.UITVOER_MAP, exist_ok=True)

    model = YOLO(model_pad)
    geexporteerd = model.export(
        format   = "onnx",
        imgsz    = config.IMG_SIZE,
        dynamic  = False,
        simplify = True,
    )

    # Kopieer naar uitvoermap zodat alles op één plek staat
    onnx_naam = os.path.basename(str(geexporteerd))
    doel = os.path.join(config.UITVOER_MAP, onnx_naam)
    shutil.copy2(str(geexporteerd), doel)

    grootte_mb = os.path.getsize(doel) / 1024 / 1024
    print(f"[export] ✓ ONNX opgeslagen : {doel}  ({grootte_mb:.1f} MB)")
    return doel


# ─────────────────────────────────────────────────────────────────────────────
#  STAP 2 — ONNX → TFLite INT8
# ─────────────────────────────────────────────────────────────────────────────

def stap2_naar_tflite(onnx_pad: str) -> str | None:
    _print_stap(2, "Converteren naar TFLite INT8")

    tflite_uitvoer = os.path.join(config.UITVOER_MAP, "tflite_output")
    os.makedirs(tflite_uitvoer, exist_ok=True)

    # Probeer onnx2tf (werkt lokaal als TensorFlow beschikbaar is)
    if _check_dependency("onnx2tf"):
        print("[tflite] onnx2tf gevonden — lokaal converteren...")
        result = subprocess.run(
            [
                sys.executable, "-m", "onnx2tf",
                "-i", onnx_pad,
                "-o", tflite_uitvoer,
                "-oiqt",          # output INT8 quantized tflite
                "-qt", "per-tensor",
            ],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            # Zoek het gegenereerde INT8 bestand
            kandidaten = (
                glob.glob(os.path.join(tflite_uitvoer, "*full_integer_quant*.tflite")) +
                glob.glob(os.path.join(tflite_uitvoer, "*int8*.tflite")) +
                glob.glob(os.path.join(tflite_uitvoer, "*.tflite"))
            )
            if kandidaten:
                tflite_pad = kandidaten[0]
                grootte_kb = os.path.getsize(tflite_pad) / 1024
                print(f"[tflite] ✓ TFLite INT8 : {tflite_pad}  ({grootte_kb:.0f} KB)")
                return tflite_pad

        print(f"[tflite] ✗ onnx2tf mislukt:\n{result.stderr}")

    # onnx2tf niet beschikbaar of mislukt → instructies tonen
    _toon_colab_instructies(onnx_pad, tflite_uitvoer)
    return None


def _toon_colab_instructies(onnx_pad: str, tflite_uitvoer: str):
    """Toon duidelijke instructies voor conversie via Google Colab."""
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  TFLite conversie via Google Colab (gratis)     │")
    print("  └─────────────────────────────────────────────────┘")
    print()
    print("  Python 3.12+ ondersteunt TensorFlow nog niet.")
    print("  Gebruik Google Colab voor de TFLite conversie:")
    print()
    print("  1. Ga naar https://colab.research.google.com")
    print("  2. Upload dit ONNX bestand:")
    print(f"     {onnx_pad}")
    print()
    print("  3. Voer in Colab uit:")
    print("     ┌──────────────────────────────────────────────┐")
    print("     │ !pip install onnx2tf                         │")
    print(f"     │ !onnx2tf -i best.onnx \\                    │")
    print(f"     │   -o tflite_output -oiqt -qt per-tensor     │")
    print("     └──────────────────────────────────────────────┘")
    print()
    print("  4. Download: tflite_output/*full_integer_quant*.tflite")
    print(f"  5. Plaats het hier: {tflite_uitvoer}")
    print()
    print("  Daarna opnieuw uitvoeren met --tflite vlag:")
    tflite_voorbeeld = os.path.join(tflite_uitvoer, "best_full_integer_quant.tflite")
    print(f"    python export_vela.py --tflite \"{tflite_voorbeeld}\"")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  STAP 3 — TFLite → Vela
# ─────────────────────────────────────────────────────────────────────────────

def stap3_vela(tflite_pad: str) -> str | None:
    _print_stap(3, "Vela optimalisatie voor Grove Vision AI V2")

    vela_cmd = _vela_commando()
    if vela_cmd is None:
        print("[vela] ✗ ethos-u-vela niet gevonden.")
        print("[vela]   Installeren via:  pip install ethos-u-vela")
        print("[vela]   Daarna opnieuw uitvoeren.")
        return None

    vela_uitvoer = os.path.join(config.UITVOER_MAP, "vela_output")
    os.makedirs(vela_uitvoer, exist_ok=True)

    print(f"[vela] Input  : {tflite_pad}")
    print(f"[vela] Output : {vela_uitvoer}")
    print(f"[vela] NPU    : {VELA_ACCELERATOR}")

    # Probeer eerst met volledige Grove Vision AI V2 configuratie
    cmd = vela_cmd + [
        tflite_pad,
        "--output-dir",       vela_uitvoer,
        "--accelerator-config", VELA_ACCELERATOR,
        "--system-config",    VELA_SYSTEM,
        "--memory-mode",      VELA_MEMORY_MODE,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Fallback: minimale flags (sommige Vela versies kennen de flags niet)
    if result.returncode != 0:
        print("[vela] Volledige config mislukt, fallback naar basis conversie...")
        cmd_basis = vela_cmd + [tflite_pad, "--output-dir", vela_uitvoer]
        result = subprocess.run(cmd_basis, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[vela] ✗ Vela mislukt:\n{result.stderr}")
        return None

    # Zoek het gegenereerde vela bestand
    vela_bestanden = glob.glob(os.path.join(vela_uitvoer, "*_vela.tflite"))
    if not vela_bestanden:
        print(f"[vela] ✗ Geen *_vela.tflite gevonden in {vela_uitvoer}")
        return None

    vela_pad = vela_bestanden[0]
    grootte_kb = os.path.getsize(vela_pad) / 1024
    print(f"[vela] ✓ Vela model : {vela_pad}  ({grootte_kb:.0f} KB)")

    # Toon NPU statistieken als Vela ze rapporteert
    for lijn in result.stdout.splitlines():
        if any(kw in lijn.lower() for kw in ["npu", "cpu", "sram", "mac", "ops"]):
            print(f"[vela]   {lijn.strip()}")

    return vela_pad


# ─────────────────────────────────────────────────────────────────────────────
#  FINALE STAP — kopieer naar uitvoermap met herkenbare naam
# ─────────────────────────────────────────────────────────────────────────────

def _sla_finaal_op(vela_pad: str) -> str:
    finale_naam = f"{config.HF_REPO_NAAM}_vela_int8.tflite"
    finale_pad  = os.path.join(config.UITVOER_MAP, finale_naam)
    shutil.copy2(vela_pad, finale_pad)
    return finale_pad


def _toon_upload_instructies(finale_pad: str):
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  Flashen op Grove Vision AI V2                  │")
    print("  └─────────────────────────────────────────────────┘")
    print()
    print("  1. Sluit Grove Vision AI V2 aan via USB")
    print("  2. Open in je browser:")
    print("     https://seeed-studio.github.io/SenseCraft-Web-Toolkit/#/setup/process")
    print("  3. Selecteer 'Grove Vision AI V2' als apparaat")
    print("  4. Upload dit bestand:")
    print(f"     {finale_pad}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIEKE INTERFACE  (aangeroepen door main.py of standalone)
# ─────────────────────────────────────────────────────────────────────────────

def exporteer(model_pad: str = None, tflite_pad: str = None) -> str | None:
    """
    Volledige export pipeline: best.pt → ONNX → TFLite INT8 → Vela TFLite.

    Args:
        model_pad:   Pad naar best.pt. Als None: automatisch zoeken in RUNS_MAP.
        tflite_pad:  Sla stap 1+2 over en gebruik dit bestaande TFLite bestand.

    Returns:
        Pad naar het finale Vela TFLite bestand, of None als mislukt.
    """
    print("=" * 55)
    print("  TFLite + Vela Export Pipeline")
    print("=" * 55)

    os.makedirs(config.UITVOER_MAP, exist_ok=True)

    # ── Bepaal startpunt ───────────────────────────────────────────────────

    if tflite_pad:
        # Direct naar Vela — stap 1 en 2 overslaan
        if not os.path.exists(tflite_pad):
            print(f"[export] ✗ TFLite bestand niet gevonden: {tflite_pad}")
            return None
        print(f"\n[export] TFLite opgegeven — stap 1 en 2 overgeslagen")
        print(f"[export] TFLite : {tflite_pad}")

    else:
        # Bepaal het model pad
        if not model_pad:
            model_pad = _zoek_best_pt()
            if not model_pad:
                print(f"\n[export] ✗ Geen best.pt gevonden in: {config.RUNS_MAP}")
                print("[export]   Gebruik: python export_vela.py --model pad\\naar\\best.pt")
                return None
            print(f"\n[export] Model automatisch gevonden: {model_pad}")
        else:
            if not os.path.exists(model_pad):
                print(f"[export] ✗ Model niet gevonden: {model_pad}")
                return None
            print(f"\n[export] Model : {model_pad}")

        # Stap 1: naar ONNX
        onnx_pad = stap1_naar_onnx(model_pad)

        # Stap 2: naar TFLite INT8
        tflite_pad = stap2_naar_tflite(onnx_pad)
        if not tflite_pad:
            # Instructies zijn al getoond in stap2
            return None

    # Stap 3: Vela optimalisatie
    vela_pad = stap3_vela(tflite_pad)
    if not vela_pad:
        return None

    # Finaal opslaan met herkenbare naam
    finale_pad = _sla_finaal_op(vela_pad)

    print("\n" + "=" * 55)
    print("  ✅  Export klaar!")
    print("=" * 55)
    print(f"\n  Finaal model : {finale_pad}")
    _toon_upload_instructies(finale_pad)

    return finale_pad


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE GEBRUIK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exporteer YOLOv8n naar Vela TFLite voor Grove Vision AI V2"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Pad naar best.pt. "
            "Standaard: automatisch zoeken in RUNS_MAP uit config.py"
        ),
    )
    parser.add_argument(
        "--tflite",
        type=str,
        default=None,
        help=(
            "Pad naar een bestaand TFLite INT8 bestand. "
            "Hiermee sla je stap 1 (ONNX) en stap 2 (TFLite conversie) over "
            "en ga je direct naar Vela optimalisatie."
        ),
    )
    args = parser.parse_args()
    exporteer(model_pad=args.model, tflite_pad=args.tflite)
