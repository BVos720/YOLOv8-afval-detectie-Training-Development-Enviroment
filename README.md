# Afval Detectie — Trainingsomgeving (YOLOv8)

Python-scripts rond [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) om een
afval-detectiemodel te trainen op de verzamelde data en detecties uit te voeren. Communiceert via
HTTP met de [AI API](../P4AIAPI). Het centrale startpunt is **`menu.py`** — een interactief
terminal-menu dat alle functies bundelt.

## Snelstart

```bash
pip install python-dotenv opencv-python pillow requests pyyaml scikit-learn ultralytics
python menu.py
```

Maak eerst een **`Conf.env`** aan (staat in `.gitignore`) met je geheime key:

```
TRAINING_API_KEY=jouw-training-key
```

## Het hoofdmenu (`menu.py`)

| Optie | Wat het doet |
|---|---|
| 1. Instellingen | Bekijk/wijzig instellingen (bewaard in `menu_settings.json`) |
| 2. Model trainen (vanuit API) | Haalt trainingsdata uit de AI API, bouwt dataset, traint YOLOv8n |
| 3. Model testen | Detectie op één afbeelding of testmap, met weergave |
| 4. Detecteren via webcam | Live detectie met boxes; **spatie** = opslaan/posten, **Esc** = stoppen |
| 5. Detecteren via map | Verwerkt een map; toont elke afbeelding ter controle, **Enter** = volgende |
| 6. Actief model vervangen | Kopieert een ander `best.pt` naar de modelmap |
| 0. Afsluiten | |

### Het model als vervangbare module

Het actieve model staat altijd op `model/best.pt`. Vervangen kan op drie manieren: handmatig een
ander `best.pt` in de map zetten, na trainen laten activeren, of via menu-optie 6. Het model wordt
lazy geladen en gecachet; bij vervanging wordt de cache geleegd.

### Coordinaten & metadata

Bij map-detectie worden GPS-coordinaten uit de **EXIF-metadata** gelezen. Ontbreekt GPS (of bij de
webcam), dan vraagt het script per afbeelding om de coordinaten in **decimale graden**:
Latitude (X = noord/zuid), Longitude (Y = oost/west). Let op: `51°35'` ≠ `51.35` maar `51.58`.

### Afbeelding optioneel meesturen

Met de instelling `AFBEELDING_MEESTUREN` bepaal je of de gedetecteerde afbeelding als base64-blob
wordt meegestuurd (komt dan in de `DetectionImages`-tabel) of niet.

## Scripts

| Script | Functie |
|---|---|
| `menu.py` | Centraal interactief menu (alles-in-één) |
| `trainer.py` | Traint YOLOv8n (transfer learning vanaf `yolov8n.pt`) |
| `train_from_api.py` | Losse variant: trainen vanuit de API via command-line |
| `detect.py` | Losse variant: webcam/map-detectie via command-line |
| `test_model.py` | Snel testen van het model op afbeeldingen |
| `data_prep.py` | Zet CSV-labels om naar YOLO-formaat + `dataset.yaml` |
| `merge_datasets.py` | Voegt twee YOLO-datasets samen |
| `main.py` | Volledige pipeline: prep → train → upload → export |
| `uploader.py` | Upload getraind model naar Hugging Face Hub |
| `export_tflite.py` / `export_vela.py` | Export naar TFLite/Vela voor Grove Vision AI V2 |
| `serve/main.py` | FastAPI-server die detecties uitvoert en post |
| `config.py` | Centrale instellingen voor de pipeline-scripts |

## Model & doelhardware

Getraind met **YOLOv8n** (nano) op **192×192** pixels — klein genoeg voor de **Grove Vision AI V2**
microcontroller. Via `main.py` exporteerbaar naar TFLite (INT8) en Vela, en optioneel te uploaden
naar Hugging Face.

## Instellingen (`menu_settings.json`)

`API_BASE_URL`, `CAMERA_ID`, `MODEL_MAP`, `CONF_DREMPEL`, `RUNS_MAP`, `DATASET_MAP`,
`AFBEELDING_MEESTUREN`. Aanpasbaar via menu-optie 1. De API-URL krijgt automatisch `https://` als
dat ontbreekt.

## Beveiliging

De Training-API-key staat **niet** in de code maar in `Conf.env` (in `.gitignore`, ingelezen met
`python-dotenv`). Commit nooit je echte key.
