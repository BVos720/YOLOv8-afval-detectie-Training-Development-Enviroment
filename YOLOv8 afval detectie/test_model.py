"""
test_model.py — Test het getrainde model op afbeeldingen.

Gebruik:
    python test_model.py                        ← test op voorbeeldafbeeldingen uit de dataset
    python test_model.py --afbeelding foto.jpg  ← test op een eigen afbeelding
"""

import argparse
import os
import glob
from ultralytics import YOLO

MODEL_PAD = r"C:\Projecten\Periode 4\YOLOv8 afval detectie\runs\yolov8n_training\weights\best.pt"
TEST_MAP  = r"C:\Projecten\Periode 4\YOLOv8 afval detectie\Dataset\GARBAGE CLASSIFICATION 3.v2-gc1.yolov8\test\images"


def test(afbeelding_pad: str = None):
    print(f"[test] Model laden: {MODEL_PAD}")
    model = YOLO(MODEL_PAD)

    # Gebruik eigen afbeelding of pak 10 voorbeelden uit de testset
    if afbeelding_pad:
        bronnen = [afbeelding_pad]
    else:
        alle = glob.glob(os.path.join(TEST_MAP, "*.jpg"))
        bronnen = alle[:10]
        print(f"[test] Geen afbeelding opgegeven — test op {len(bronnen)} voorbeelden uit testset")

    resultaten = model.predict(
        source     = bronnen,
        conf       = 0.25,   # minimale zekerheidsdrempel
        save       = True,   # sla resultaten op als afbeeldingen
        show       = True,   # toon resultaten in een venster
        line_width = 2,
    )

    print("\n[test] Resultaten:")
    for r in resultaten:
        print(f"  {os.path.basename(r.path)}")
        for box in r.boxes:
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            naam  = model.names[cls]
            print(f"    → {naam}: {conf:.0%} zekerheid")

    print(f"\n[test] Afbeeldingen opgeslagen in: runs/predict/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--afbeelding", type=str, default=None, help="Pad naar een eigen afbeelding")
    args = parser.parse_args()
    test(afbeelding_pad=args.afbeelding)
