"""
export_tflite.py — Exporteer het getrainde YOLOv8n model naar TFLite INT8
                   en optimaliseer met Vela voor Grove Vision AI V2.

Het nano model (~6MB INT8) past in het geheugen van de Grove Vision AI V2.
INT8 is vereist — FP16/FP32 wordt NIET ondersteund door de hardware.

Stappen:
    1. Exporteer best.pt (nano) → ONNX
    2. Converteer ONNX → TFLite INT8 via Google Colab (onnx2tf)
    3. Optimaliseer met Vela voor de Ethos-U55 NPU
    4. Upload het vela .tflite bestand via SenseCraft Web Toolkit

Gebruik:
    python export_tflite.py
"""

import os
import subprocess
import sys
import glob

# ── Paden ──────────────────────────────────────────────────────────────────
MODEL_PAD         = r"C:\Projecten\Periode 4\YOLOv8 afval detectie\runs\yolov8n_training\weights\best.pt"
CALIBRATION_MAP   = r"C:\Projecten\Periode 4\YOLOv8 afval detectie\Dataset\GARBAGE CLASSIFICATION 3.v2-gc1.yolov8\valid\images"
UITVOER_MAP       = r"C:\Projecten\Periode 4\YOLOv8 afval detectie\tflite_export"

# Grove Vision AI ondersteunt maximaal 192x192 of 320x320
IMG_SIZE          = 192


def stap1_exporteer_onnx():
    """Exporteer best.pt (nano) naar ONNX (werkt op Python 3.14)."""
    print("\n" + "─" * 55)
    print("  STAP 1 — Exporteren naar ONNX")
    print("─" * 55)

    from ultralytics import YOLO

    if not os.path.exists(MODEL_PAD):
        print(f"[!] Model niet gevonden: {MODEL_PAD}")
        return None

    os.makedirs(UITVOER_MAP, exist_ok=True)

    print(f"[export] Model laden: {MODEL_PAD}")
    model = YOLO(MODEL_PAD)

    print(f"[export] Exporteren naar ONNX (imgsize={IMG_SIZE})...")
    geexporteerd = model.export(
        format = "onnx",
        imgsz  = IMG_SIZE,
        dynamic= False,
        simplify=True,
    )

    import shutil
    onnx_naam = os.path.basename(str(geexporteerd))
    doel = os.path.join(UITVOER_MAP, onnx_naam)
    shutil.copy2(str(geexporteerd), doel)
    print(f"[export] ONNX bestand opgeslagen: {doel}")
    return doel


def stap2_installeer_vela():
    """Installeer ethos-u-vela als het nog niet aanwezig is."""
    print("\n" + "─" * 55)
    print("  STAP 2 — Vela installeren")
    print("─" * 55)

    try:
        import ethosu.vela
        print("[vela] Vela is al geïnstalleerd ✅")
        return True
    except ImportError:
        print("[vela] Vela installeren...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "ethos-u-vela"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("[vela] Vela succesvol geïnstalleerd ✅")
            return True
        else:
            print(f"[!] Vela installatie mislukt:\n{result.stderr}")
            return False


def stap3_vela_optimaliseer(tflite_pad: str):
    """Optimaliseer het TFLite model met Vela voor de Grove Vision AI."""
    print("\n" + "─" * 55)
    print("  STAP 3 — Vela optimalisatie voor Grove Vision AI")
    print("─" * 55)

    if not tflite_pad or not os.path.exists(tflite_pad):
        print(f"[!] TFLite bestand niet gevonden: {tflite_pad}")
        return None

    vela_uitvoer_map = os.path.join(UITVOER_MAP, "vela_output")
    os.makedirs(vela_uitvoer_map, exist_ok=True)

    print(f"[vela] Optimaliseren: {tflite_pad}")
    print(f"[vela] Uitvoer naar: {vela_uitvoer_map}")

    result = subprocess.run(
        ["vela", tflite_pad, "--output-dir", vela_uitvoer_map],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        # Zoek het gegenereerde vela bestand
        vela_bestanden = glob.glob(os.path.join(vela_uitvoer_map, "*.tflite"))
        if vela_bestanden:
            vela_pad = vela_bestanden[0]
            print(f"[vela] Geoptimaliseerd model: {vela_pad} ✅")
            return vela_pad
    else:
        print(f"[!] Vela optimalisatie mislukt:\n{result.stderr}")
        print("[!] Probeer handmatig via de terminal:")
        print(f"    vela {tflite_pad} --output-dir {vela_uitvoer_map}")
        return None


def stap4_instructies(vela_pad: str):
    """Toon instructies voor het uploaden naar Grove Vision AI."""
    print("\n" + "─" * 55)
    print("  STAP 4 — Flashen op Grove Vision AI")
    print("─" * 55)
    print()
    print("  1. Sluit de Grove Vision AI aan via USB")
    print("  2. Open in je browser:")
    print("     https://seeed-studio.github.io/SenseCraft-Web-Toolkit/#/setup/process")
    print("  3. Kies 'Grove Vision AI' als apparaat")
    print("  4. Upload het volgende bestand:")
    if vela_pad:
        print(f"     {vela_pad}")
    else:
        print(f"     (vela bestand in: {UITVOER_MAP}\\vela_output\\)")
    print()
    print("  Klaar! Het model draait dan live op de microcontroller.")


def main():
    print("=" * 55)
    print("  TFLite INT8 Export Pipeline")
    print("=" * 55)

    # Stap 1: exporteer naar ONNX
    onnx_pad = stap1_exporteer_onnx()
    if not onnx_pad:
        print("\n[!] Export mislukt. Controleer de paden en probeer opnieuw.")
        return

    print("\n" + "=" * 55)
    print("  ONNX export geslaagd!")
    print(f"  ONNX bestand : {onnx_pad}")
    print()
    print("  VOLGENDE STAP — Converteer ONNX naar TFLite INT8:")
    print("  Python 3.14 ondersteunt TensorFlow nog niet.")
    print("  Gebruik een van deze opties:")
    print()
    print("  Optie A — Online converter:")
    print("    https://convertmodel.com  (upload ONNX, download TFLite)")
    print()
    print("  Optie B — Google Colab (gratis):")
    print("    1. Open https://colab.research.google.com")
    print("    2. Upload het ONNX bestand (nano is klein, ~12MB)")
    print("    3. Voer dit uit:")
    print("       !pip install onnx2tf")
    print(f"       !onnx2tf -i best.onnx -o tflite_output -oiqt")
    print("    4. Download best_full_integer_quant.tflite (~6MB)")
    print("    5. Voer Vela uit in een APARTE Colab sessie:")
    print("       !pip install ethos-u-vela")
    print("       !vela best_full_integer_quant.tflite --output-dir vela_output")
    print("    6. Download het vela bestand uit vela_output/")
    print()
    print("  Optie C — Gebruik Roboflow deploy (automatisch TFLite)")
    print("    https://roboflow.com/deploy")
    print("=" * 55)


if __name__ == "__main__":
    main()
