from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import io

app = FastAPI()

# Laad het getrainde afval detectie model (YOLOv8 medium)
model = YOLO("/app/model/best.pt")  # best.pt getraind op yolov8m basis

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
            detections.append({
                "label": result.names[int(box.cls)],
                "confidence": round(float(box.conf), 4),
                "x1": round(float(box.xyxy[0][0]), 2),
                "y1": round(float(box.xyxy[0][1]), 2),
                "x2": round(float(box.xyxy[0][2]), 2),
                "y2": round(float(box.xyxy[0][3]), 2),
            })

    return JSONResponse(content={"detections": detections})
