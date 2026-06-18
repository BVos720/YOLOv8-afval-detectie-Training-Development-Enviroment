"""
detect.py — Voer detecties uit en post ze naar de AIAPI.

Twee modi:
    python detect.py --modus camera          ← live webcam, spatie = foto + detectie, Esc = stoppen
    python detect.py --modus map --map "C:\\pad\\naar\\fotos"   ← hele map verwerken

Coordinaten:
    - Bij map-modus worden GPS-coordinaten uit de EXIF-metadata van de foto gelezen.
    - Heeft een foto geen GPS, dan worden de HANDMATIGE_LOCATIE waarden gebruikt.
    - De webcam heeft geen GPS, dus daar gelden altijd de handmatige waarden.

De API key (geheim) komt uit een .env bestand:  TRAINING_API_KEY=jouw-sleutel
"""

import os
import io
import glob
import argparse

import cv2
import requests
from dotenv import load_dotenv
from ultralytics import YOLO
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# ── Instellingen die je hier in de code mag aanpassen ──────────────────────
API_BASE_URL = "http://localhost:5000"   # URL van de AIAPI
CAMERA_ID    = "laptop-cam"              # naam van deze camera / bron
MODEL_PATH   = r"C:\Projecten\Periode 4\YOLOv8 afval detectie\runs\yolov8n_training\weights\best.pt"
CONF_DREMPEL = 0.25                      # minimale zekerheid voor een detectie

# ── Geheim: NIET in de code. Komt uit een .env bestand ─────────────────────
load_dotenv()
API_KEY = os.getenv("TRAINING_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "TRAINING_API_KEY ontbreekt. Maak een .env bestand met:\n"
        "    TRAINING_API_KEY=jouw-sleutel"
    )

print(f"[detect] Model laden: {MODEL_PATH}")
model = YOLO(MODEL_PATH)


def lees_gps_uit_exif(pad: str):
    """Lees latitude/longitude uit de EXIF-metadata. Geeft (None, None) als er geen GPS is."""
    try:
        exif = Image.open(pad)._getexif()
    except Exception:
        return None, None

    if not exif:
        return None, None

    gps = {}
    for tag, waarde in exif.items():
        if TAGS.get(tag) == "GPSInfo":
            for t, v in waarde.items():
                gps[GPSTAGS.get(t, t)] = v

    if not gps:
        return None, None

    def naar_graden(waarde, ref):
        graden, minuten, seconden = waarde
        decimaal = float(graden) + float(minuten) / 60 + float(seconden) / 3600
        if ref in ("S", "W"):
            decimaal = -decimaal
        return decimaal

    try:
        lat = naar_graden(gps["GPSLatitude"], gps["GPSLatitudeRef"])
        lon = naar_graden(gps["GPSLongitude"], gps["GPSLongitudeRef"])
        return lat, lon
    except (KeyError, ValueError, TypeError):
        return None, None


def vraag_coordinaten(naam: str):
    """Vraag de gebruiker om handmatig coordinaten in te voeren (leeg = overslaan)."""
    print(f"  Geen GPS gevonden voor {naam}. Voer handmatig in (leeg = overslaan):")
    x = input("    Latitude  (X): ").strip().replace(",", ".")
    y = input("    Longitude (Y): ").strip().replace(",", ".")
    try:
        return (float(x) if x else None, float(y) if y else None)
    except ValueError:
        print("    [warn] Ongeldige invoer, locatie overgeslagen.")
        return None, None


def detecteer(bron, locatie_x, locatie_y):
    """Voer detectie uit op een bron (bestandspad of afbeelding) en post elke detectie."""
    resultaten = model(bron, conf=CONF_DREMPEL)

    aantal = 0
    for resultaat in resultaten:
        for box in resultaat.boxes:
            detectie = {
                "label":      resultaat.names[int(box.cls)],
                "confidence": round(float(box.conf), 4),
                "cameraId":   CAMERA_ID,
                "locatieX":   locatie_x,
                "locatieY":   locatie_y,
                "boundingBoxLB":     round(float(box.xyxy[0][0]), 2),
                "boundingBoxRB":     round(float(box.xyxy[0][2]), 2),
                "boundingBoxCenter": round(float((box.xyxy[0][0] + box.xyxy[0][2]) / 2), 2),
            }
            _post_detectie(detectie)
            aantal += 1
            print(f"    → {detectie['label']}: {detectie['confidence']:.0%}")

    if aantal == 0:
        print("    (geen detecties)")
    return aantal


def _post_detectie(detectie: dict):
    """Stuur een detectie naar de AIAPI."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/detection/ai",
            json=detectie,
            headers={"X-Api-Key": API_KEY},
            timeout=5,
        )
        if not resp.ok:
            print(f"    [warn] API gaf {resp.status_code}")
    except requests.RequestException as e:
        print(f"    [warn] Kon detectie niet posten: {e}")


def camera_modus():
    """Live webcam. Spatie = frame vastleggen + detecteren. Esc = stoppen."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Kon de webcam niet openen.")

    print("[detect] Webcam actief — SPATIE = foto maken, ESC = stoppen")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.imshow("Detectie (spatie = foto, esc = stop)", frame)
        toets = cv2.waitKey(1) & 0xFF

        if toets == 27:        # Esc
            break
        elif toets == 32:      # Spatie
            print("[detect] Foto gemaakt.")
            lat, lon = vraag_coordinaten("webcam-foto")
            detecteer(frame, lat, lon)

    cap.release()
    cv2.destroyAllWindows()


def map_modus(map_pad: str):
    """Verwerk alle afbeeldingen in een map."""
    extensies = ("*.jpg", "*.jpeg", "*.png", "*.heic", "*.heif")
    bestanden = []
    for ext in extensies:
        bestanden.extend(glob.glob(os.path.join(map_pad, ext)))

    if not bestanden:
        print(f"[detect] Geen afbeeldingen gevonden in: {map_pad}")
        return

    print(f"[detect] {len(bestanden)} afbeeldingen gevonden")
    for pad in bestanden:
        naam = os.path.basename(pad)
        lat, lon = lees_gps_uit_exif(pad)
        if lat is None or lon is None:
            print(f"\n[detect] {naam}")
            lat, lon = vraag_coordinaten(naam)
        else:
            print(f"\n[detect] {naam}  (GPS uit EXIF: {lat:.5f}, {lon:.5f})")

        detecteer(pad, lat, lon)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detecties uitvoeren en naar de AIAPI posten")
    parser.add_argument("--modus", choices=["camera", "map"], required=True,
                        help="camera = live webcam, map = hele map verwerken")
    parser.add_argument("--map", type=str, default=None,
                        help="Pad naar de map met foto's (verplicht bij --modus map)")
    args = parser.parse_args()

    if args.modus == "camera":
        camera_modus()
    else:
        if not args.map:
            parser.error("--map is verplicht bij --modus map")
        map_modus(args.map)
