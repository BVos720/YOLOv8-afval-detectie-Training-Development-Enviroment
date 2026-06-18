"""
menu.py — Centraal terminal-menu voor het hele afval-detectie project.

Start met:
    python menu.py

Van hieruit kun je alles besturen en instellen:
    - Instellingen bekijken en wijzigen (worden bewaard in menu_settings.json)
    - Model trainen vanuit de AIAPI database
    - Model testen op een afbeelding of de testset
    - Detecteren via de webcam (foto-knop)
    - Detecteren op een hele map met foto's

De API key (geheim) komt uit een .env bestand:  TRAINING_API_KEY=jouw-sleutel
"""

import os
import io
import glob
import json
import base64
import shutil
from pathlib import Path

import cv2
import requests
import yaml
from dotenv import load_dotenv, find_dotenv
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# ── Vaste gegevens ─────────────────────────────────────────────────────────
LABELS = ["Cardboard", "Garbage", "Glass", "Metal", "Paper", "Plastic", "Trash"]
LABEL_NAAR_ID = {naam: idx for idx, naam in enumerate(LABELS)}

INSTELLINGEN_BESTAND = Path(__file__).with_name("menu_settings.json")

STANDAARD_INSTELLINGEN = {
    "API_BASE_URL": "http://localhost:5000",
    "CAMERA_ID":    "laptop-cam",
    "MODEL_MAP":    r"C:\Projecten\Periode 4\YOLOv8 afval detectie\model",
    "CONF_DREMPEL": 0.25,
    "RUNS_MAP":     r"C:\Projecten\Periode 4\YOLOv8 afval detectie\runs",
    "DATASET_MAP":  r"C:\Projecten\Periode 4\YOLOv8 afval detectie\dataset_api",
    "AFBEELDING_MEESTUREN": True,   # stuur de gedetecteerde afbeelding mee als blob
}

# Het actieve model is altijd dit bestand binnen de modelmap.
MODEL_BESTANDSNAAM = "best.pt"


def actief_model_pad(s: dict) -> str:
    """Pad naar het huidige actieve model (model-map/best.pt)."""
    return os.path.join(s["MODEL_MAP"], MODEL_BESTANDSNAAM)


def basis_url(s: dict) -> str:
    """Geef de API-URL terug, met https:// ervoor als dat ontbreekt."""
    url = s["API_BASE_URL"].strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url

# ── Geheim uit Conf.env (of .env) ─────────────────────────────────────────-
# Zoekt eerst Conf.env, daarna een gewoon .env bestand, vanaf de huidige map omhoog.
_env_pad = find_dotenv("Conf.env", usecwd=True) or find_dotenv(usecwd=True)
load_dotenv(_env_pad)
API_KEY = os.getenv("TRAINING_API_KEY")

# Cache voor het geladen model (zodat het niet elke keer opnieuw laadt)
_model_cache = {"pad": None, "model": None}


# ── Instellingen laden / opslaan ─────────────────────────────────────────--
def laad_instellingen() -> dict:
    instellingen = dict(STANDAARD_INSTELLINGEN)
    if INSTELLINGEN_BESTAND.exists():
        try:
            instellingen.update(json.loads(INSTELLINGEN_BESTAND.read_text()))
        except (json.JSONDecodeError, OSError):
            print("[!] Kon menu_settings.json niet lezen — standaardwaarden gebruikt.")
    return instellingen


def sla_instellingen_op(instellingen: dict):
    INSTELLINGEN_BESTAND.write_text(json.dumps(instellingen, indent=2))
    print(f"[ok] Instellingen opgeslagen in {INSTELLINGEN_BESTAND.name}")


def toon_instellingen(s: dict):
    print("\n── Huidige instellingen ─────────────────────────────")
    for i, (sleutel, waarde) in enumerate(s.items(), start=1):
        print(f"  {i}. {sleutel:<14} = {waarde}")
    key_status = "ingesteld" if API_KEY else "ONTBREEKT (.env)"
    print(f"     TRAINING_API_KEY = {key_status}")
    print("─────────────────────────────────────────────────────")


def wijzig_instellingen(s: dict):
    while True:
        toon_instellingen(s)
        keuze = input("Welk nummer wil je wijzigen? (Enter = terug) ").strip()
        if not keuze:
            break
        try:
            index = int(keuze) - 1
            sleutel = list(s.keys())[index]
        except (ValueError, IndexError):
            print("[!] Ongeldige keuze.")
            continue

        nieuw = input(f"Nieuwe waarde voor {sleutel} (huidig: {s[sleutel]}): ").strip()
        if nieuw == "":
            continue
        # CONF_DREMPEL is een getal
        if sleutel == "CONF_DREMPEL":
            try:
                s[sleutel] = float(nieuw.replace(",", "."))
            except ValueError:
                print("[!] Moet een getal zijn.")
                continue
        elif isinstance(s[sleutel], bool):
            s[sleutel] = nieuw.strip().lower() in ("true", "ja", "j", "1", "aan")
        else:
            s[sleutel] = nieuw
        sla_instellingen_op(s)


# ── Model laden ──────────────────────────────────────────────────────────--
def get_model(pad: str):
    if _model_cache["pad"] != pad or _model_cache["model"] is None:
        if not os.path.exists(pad):
            raise FileNotFoundError(
                f"Geen actief model gevonden: {pad}\n"
                f"Train een model (optie 2) of zet een best.pt in de modelmap (optie 6)."
            )
        print(f"[model] Laden: {pad}")
        _model_cache["model"] = YOLO(pad)
        _model_cache["pad"] = pad
    return _model_cache["model"]


# ── Coordinaten ──────────────────────────────────────────────────────────--
def lees_gps_uit_exif(pad: str):
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
        return -decimaal if ref in ("S", "W") else decimaal

    try:
        lat = naar_graden(gps["GPSLatitude"], gps["GPSLatitudeRef"])
        lon = naar_graden(gps["GPSLongitude"], gps["GPSLongitudeRef"])
        return lat, lon
    except (KeyError, ValueError, TypeError):
        return None, None


def vraag_coordinaten(naam: str):
    print(f"  Geen GPS gevonden voor {naam}. Voer handmatig in (leeg = overslaan):")
    x = input("    Latitude  (X): ").strip().replace(",", ".")
    y = input("    Longitude (Y): ").strip().replace(",", ".")
    try:
        return (float(x) if x else None, float(y) if y else None)
    except ValueError:
        print("    [!] Ongeldige invoer, locatie overgeslagen.")
        return None, None


# ── Detecteren + posten ──────────────────────────────────────────────────--
def post_detectie(s: dict, detectie: dict):
    if not API_KEY:
        print("    [!] Geen API key — niet gepost. Zet TRAINING_API_KEY in .env")
        return
    try:
        resp = requests.post(
            f"{basis_url(s)}/api/detection/ai",
            json=detectie,
            headers={"X-Api-Key": API_KEY},
            timeout=5,
        )
        if not resp.ok:
            print(f"    [!] API gaf {resp.status_code}")
    except requests.RequestException as e:
        print(f"    [!] Kon detectie niet posten: {e}")


def afbeelding_bytes(bron):
    """Geef (bytes, content_type) van de bron. bron = bestandspad of webcam-frame."""
    if isinstance(bron, str):
        ext = os.path.splitext(bron)[1].lower()
        ct = {".png": "image/png", ".heic": "image/heic", ".heif": "image/heif"}.get(ext, "image/jpeg")
        with open(bron, "rb") as f:
            return f.read(), ct
    # Anders: een webcam-frame (numpy array) → encode als JPEG
    ok, buffer = cv2.imencode(".jpg", bron)
    return (buffer.tobytes() if ok else None), "image/jpeg"


def post_resultaat(s: dict, resultaat, lat, lon, image_bytes=None, content_type="image/jpeg"):
    """Post alle boxes van één detectie-resultaat naar de AIAPI."""
    afbeelding_b64 = None
    if s.get("AFBEELDING_MEESTUREN") and image_bytes:
        afbeelding_b64 = base64.b64encode(image_bytes).decode("ascii")

    aantal = 0
    for box in resultaat.boxes:
        detectie = {
            "label":      resultaat.names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "cameraId":   s["CAMERA_ID"],
            "locatieX":   lat,
            "locatieY":   lon,
            "boundingBoxLB":     round(float(box.xyxy[0][0]), 2),
            "boundingBoxRB":     round(float(box.xyxy[0][2]), 2),
            "boundingBoxCenter": round(float((box.xyxy[0][0] + box.xyxy[0][2]) / 2), 2),
        }
        if afbeelding_b64:
            detectie["imageBase64"] = afbeelding_b64
            detectie["imageContentType"] = content_type
        post_detectie(s, detectie)
        aantal += 1
        print(f"    → {detectie['label']}: {detectie['confidence']:.0%}")
    if aantal == 0:
        print("    (geen detecties)")


def detecteer(s: dict, bron, lat, lon):
    model = get_model(actief_model_pad(s))
    img_bytes, ct = afbeelding_bytes(bron)
    for resultaat in model(bron, conf=s["CONF_DREMPEL"]):
        post_resultaat(s, resultaat, lat, lon, img_bytes, ct)


def camera_modus(s: dict):
    model = get_model(actief_model_pad(s))
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Kon de webcam niet openen.")
        return

    print("[camera] Live detectie loopt — SPATIE = opslaan/posten, ESC = stoppen")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Detecteer op elk frame en teken de boxes live op het beeld
        resultaten = model(frame, conf=s["CONF_DREMPEL"], verbose=False)
        beeld = resultaten[0].plot()
        cv2.imshow("Live detectie (spatie = opslaan, esc = stop)", beeld)

        toets = cv2.waitKey(1) & 0xFF
        if toets == 27:        # Esc
            break
        elif toets == 32:      # Spatie → huidige detectie opslaan/posten
            print("[camera] Opslaan...")
            lat, lon = vraag_coordinaten("webcam-foto")
            img_bytes, ct = afbeelding_bytes(frame)
            post_resultaat(s, resultaten[0], lat, lon, img_bytes, ct)

    cap.release()
    cv2.destroyAllWindows()


def map_modus(s: dict):
    map_pad = input("Pad naar de map met foto's: ").strip().strip('"')
    if not os.path.isdir(map_pad):
        print("[!] Map bestaat niet.")
        return

    bestanden = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.heic", "*.heif"):
        bestanden.extend(glob.glob(os.path.join(map_pad, ext)))
    if not bestanden:
        print("[!] Geen afbeeldingen gevonden.")
        return

    print(f"[map] {len(bestanden)} afbeeldingen gevonden")
    for pad in bestanden:
        naam = os.path.basename(pad)
        lat, lon = lees_gps_uit_exif(pad)
        if lat is None or lon is None:
            print(f"\n[map] {naam}")
            lat, lon = vraag_coordinaten(naam)
        else:
            print(f"\n[map] {naam}  (GPS uit EXIF: {lat:.5f}, {lon:.5f})")
        detecteer(s, pad, lat, lon)


# ── Testen ───────────────────────────────────────────────────────────────--
def test_modus(s: dict):
    pad = input("Pad naar een afbeelding (Enter = 10 voorbeelden uit een map): ").strip().strip('"')
    model = get_model(actief_model_pad(s))

    if pad:
        bronnen = [pad]
    else:
        map_pad = input("Pad naar testmap: ").strip().strip('"')
        bronnen = glob.glob(os.path.join(map_pad, "*.jpg"))[:10]
        if not bronnen:
            print("[!] Geen afbeeldingen gevonden.")
            return

    resultaten = model.predict(source=bronnen, conf=s["CONF_DREMPEL"], save=True, show=True, line_width=2)
    for r in resultaten:
        print(f"  {os.path.basename(r.path)}")
        for box in r.boxes:
            print(f"    → {model.names[int(box.cls[0])]}: {float(box.conf[0]):.0%}")
    print("[test] Resultaat-afbeeldingen opgeslagen in runs/predict/")


# ── Trainen vanuit de API ────────────────────────────────────────────────--
def train_modus(s: dict):
    if not API_KEY:
        print("[!] Geen API key — zet TRAINING_API_KEY in .env")
        return

    from trainer import train  # pas hier importeren (laadt torch)

    try:
        epochs     = int(input("Epochs (Enter = 50): ") or 50)
        val_split  = float((input("Val-split 0-1 (Enter = 0.2): ") or "0.2").replace(",", "."))
        batch_size = int(input("Batch-grootte (Enter = 16): ") or 16)
        img_size   = int(input("Img-size (Enter = 192): ") or 192)
    except ValueError:
        print("[!] Ongeldige invoer.")
        return

    headers = {"X-Api-Key": API_KEY}
    print("[train] Afbeeldingen ophalen van de API...")
    resp = requests.get(f"{basis_url(s)}/api/trainingimage", headers=headers, timeout=30)
    resp.raise_for_status()
    entries = resp.json()
    print(f"[train] {len(entries)} afbeeldingen gevonden")
    if not entries:
        print("[!] Geen training-afbeeldingen in de database.")
        return

    dataset_pad = Path(s["DATASET_MAP"])
    if dataset_pad.exists():
        shutil.rmtree(dataset_pad)
    for split in ("train", "val"):
        (dataset_pad / "images" / split).mkdir(parents=True)
        (dataset_pad / "labels" / split).mkdir(parents=True)

    train_entries, val_entries = train_test_split(entries, test_size=val_split, random_state=42)
    print(f"[train] {len(train_entries)} train | {len(val_entries)} val")

    for split, subset in [("train", train_entries), ("val", val_entries)]:
        for entry in subset:
            img_resp = requests.get(f"{basis_url(s)}/api/trainingimage/{entry['id']}/image",
                                    headers=headers, timeout=30)
            img_resp.raise_for_status()
            img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            stam = Path(entry["fileName"]).stem
            img.save(dataset_pad / "images" / split / f"{stam}.jpg")

            regels = []
            for box in entry.get("boundingBoxes", []):
                label_id = LABEL_NAAR_ID.get(box["label"])
                if label_id is None:
                    continue
                regels.append(f"{label_id} {box['centerX']:.6f} {box['centerY']:.6f} "
                              f"{box['width']:.6f} {box['height']:.6f}")
            (dataset_pad / "labels" / split / f"{stam}.txt").write_text("\n".join(regels))

    yaml_pad = dataset_pad / "dataset.yaml"
    yaml_pad.write_text(yaml.dump({
        "path":  str(dataset_pad.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(LABELS),
        "names": LABELS,
    }, allow_unicode=True))

    print("[train] Training starten...")
    best_pt = train(yaml_pad=str(yaml_pad), runs_map=s["RUNS_MAP"],
                    epochs=epochs, batch_size=batch_size, img_size=img_size)
    print(f"[train] Klaar! Beste model: {best_pt}")
    if input("Dit model als actief model instellen? (j/n) ").strip().lower() == "j":
        activeer_model(s, best_pt)


# ── Actief model vervangen ─────────────────────────────────────────────────
def activeer_model(s: dict, bron_pad: str):
    """Kopieer een .pt bestand naar de modelmap als het nieuwe actieve model."""
    if not os.path.isfile(bron_pad):
        print(f"[!] Bestand niet gevonden: {bron_pad}")
        return
    os.makedirs(s["MODEL_MAP"], exist_ok=True)
    doel = actief_model_pad(s)
    shutil.copy2(bron_pad, doel)
    # cache leegmaken zodat het nieuwe model opnieuw geladen wordt
    _model_cache["pad"] = None
    _model_cache["model"] = None
    print(f"[ok] Actief model vervangen: {doel}")


def vervang_model(s: dict):
    huidig = actief_model_pad(s)
    bestaat = "ja" if os.path.exists(huidig) else "nee (nog geen model)"
    print(f"\nModelmap : {s['MODEL_MAP']}")
    print(f"Actief   : {huidig}  (aanwezig: {bestaat})")
    print("Je kunt ook gewoon zelf een best.pt in de modelmap zetten.")
    bron = input("Pad naar het nieuwe .pt model (Enter = annuleren): ").strip().strip('"')
    if bron:
        activeer_model(s, bron)


# ── Hoofdmenu ──────────────────────────────────────────────────────────────
def main():
    instellingen = laad_instellingen()

    while True:
        print("\n╔═══════════════════════════════════════╗")
        print("║       AFVAL DETECTIE — HOOFDMENU       ║")
        print("╠═══════════════════════════════════════╣")
        print("║  1. Instellingen bekijken / wijzigen   ║")
        print("║  2. Model trainen (vanuit API)         ║")
        print("║  3. Model testen                       ║")
        print("║  4. Detecteren via webcam              ║")
        print("║  5. Detecteren via map                 ║")
        print("║  6. Actief model vervangen             ║")
        print("║  0. Afsluiten                          ║")
        print("╚═══════════════════════════════════════╝")
        keuze = input("Keuze: ").strip()

        if keuze == "1":
            wijzig_instellingen(instellingen)
        elif keuze == "2":
            train_modus(instellingen)
        elif keuze == "3":
            test_modus(instellingen)
        elif keuze == "4":
            camera_modus(instellingen)
        elif keuze == "5":
            map_modus(instellingen)
        elif keuze == "6":
            vervang_model(instellingen)
        elif keuze == "0":
            print("Tot ziens!")
            break
        else:
            print("[!] Ongeldige keuze.")


if __name__ == "__main__":
    main()
