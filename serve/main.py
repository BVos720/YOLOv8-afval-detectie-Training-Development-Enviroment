import os
import io
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# ── Instellingen die je hier in de code mag aanpassen ──────────────────────
API_BASE_URL = "http://localhost:5000"   # URL van de AIAPI
CAMERA_ID    = "laptop-cam"              # naam van deze camera / bron
MODEL_PATH   = r"C:\Projecten\Periode 4\YOLOv8 afval detectie\runs\yolov8n_training\weights\best.pt"

# ── Geheim: NIET in de code. Komt uit een .env bestand (user secret) ───────
load_dotenv()
API_KEY = os.getenv("TRAINING_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "TRAINING_API_KEY ontbreekt. Maak een .env bestand met:\n"
        "    TRAINING_API_KEY=jouw-sleutel"
    )

app = FastAPI()
model = YOLO(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/label")
async def label_image(file: UploadFile = File(...)):
    contents = await file.read()
    results = model(io.BytesIO(contents))

    detections = []
    for result in results:
        for box in result.boxes:
            detection = {
                "label":      result.names[int(box.cls)],
                "confidence": round(float(box.conf), 4),
                "cameraId":   CAMERA_ID,
                "boundingBoxLB":     round(float(box.xyxy[0][0]), 2),
                "boundingBoxRB":     round(float(box.xyxy[0][2]), 2),
                "boundingBoxCenter": round(float((box.xyxy[0][0] + box.xyxy[0][2]) / 2), 2),
            }
            detections.append(detection)
            _post_detectie(detection)

    return JSONResponse(content={"detections": detections})


def _post_detectie(detection: dict):
    """Stuur een detectie naar de AIAPI."""
    try:
        requests.post(
            f"{API_BASE_URL}/api/detection/ai",
            json=detection,
            headers={"X-Api-Key": API_KEY},
            timeout=5,
        )
    except requests.RequestException as e:
        print(f"[warn] Kon detectie niet posten: {e}")
