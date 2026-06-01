"""
uploader.py — Upload het getrainde model naar Hugging Face Hub.

Wat er wordt geüpload:
  - best.pt          ← de getrainde modelgewichten
  - dataset.yaml     ← klasse-namen en dataset-configuratie
  - README.md        ← automatisch gegenereerde modelkaart
"""

import os
from huggingface_hub import HfApi, login


def upload_naar_huggingface(
    model_pad: str,
    yaml_pad: str,
    hf_token: str,
    hf_username: str,
    repo_naam: str,
    class_namen: dict,
    epochs: int,
    img_size: int,
) -> str:
    """
    Upload het getrainde model en bijbehorende bestanden naar Hugging Face Hub.
    
    Args:
        model_pad:    Pad naar best.pt.
        yaml_pad:     Pad naar dataset.yaml.
        hf_token:     Hugging Face API-token (write-rechten vereist).
        hf_username:  Je Hugging Face gebruikersnaam.
        repo_naam:    Naam van de repository (wordt aangemaakt als die niet bestaat).
        class_namen:  Dict {class_id: naam} voor de modelkaart.
        epochs:       Aantal getrainde epochs (voor de modelkaart).
        img_size:     Inputgrootte (voor de modelkaart).

    Returns:
        URL van de Hugging Face repository.
    """
    print("[uploader] Inloggen bij Hugging Face...")
    login(token=hf_token)
    api = HfApi()

    repo_id = f"{hf_username}/{repo_naam}"

    # Maak de repository aan als die nog niet bestaat
    print(f"[uploader] Repository controleren/aanmaken: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Genereer een README / modelkaart
    readme_pad = _maak_model_card(
        repo_id     = repo_id,
        class_namen = class_namen,
        epochs      = epochs,
        img_size    = img_size,
    )

    # Upload de bestanden
    te_uploaden = [
        (model_pad, "best.pt",       "Getrainde modelgewichten"),
        (yaml_pad,  "dataset.yaml",  "Dataset-configuratie"),
        (readme_pad, "README.md",    "Modelkaart"),
    ]

    for lokaal_pad, repo_bestandsnaam, omschrijving in te_uploaden:
        if not os.path.exists(lokaal_pad):
            print(f"[uploader] ⚠️  Overgeslagen ({omschrijving}): bestand niet gevonden: {lokaal_pad}")
            continue
        print(f"[uploader] Uploaden: {omschrijving}...")
        api.upload_file(
            path_or_fileobj = lokaal_pad,
            path_in_repo    = repo_bestandsnaam,
            repo_id         = repo_id,
            repo_type       = "model",
        )

    # Opruimen tijdelijk README-bestand
    if os.path.exists(readme_pad):
        os.remove(readme_pad)

    repo_url = f"https://huggingface.co/{repo_id}"
    print(f"[uploader] ✔ Model beschikbaar op: {repo_url}")
    return repo_url


def _maak_model_card(
    repo_id: str,
    class_namen: dict,
    epochs: int,
    img_size: int,
) -> str:
    """Genereer een README.md modelkaart en sla tijdelijk op."""
    klasse_regels = "\n".join(
        f"| {idx} | {naam} |"
        for idx, (cid, naam) in enumerate(sorted(class_namen.items()))
    )

    inhoud = f"""---
license: apache-2.0
tags:
  - object-detection
  - yolov8
  - ultralytics
---

# {repo_id.split("/")[-1]}

YOLOv8x object detectie model, getraind met [Ultralytics](https://github.com/ultralytics/ultralytics).

## Trainingsdetails

| Parameter     | Waarde     |
|---------------|------------|
| Model         | YOLOv8x    |
| Epochs        | {epochs}   |
| Inputgrootte  | {img_size}px |

## Klassen

| Index | Naam |
|-------|------|
{klasse_regels}

## Gebruik

```python
from ultralytics import YOLO

model = YOLO("{repo_id}")
resultaten = model.predict("afbeelding.jpg", conf=0.25)
resultaten[0].show()
```
"""
    readme_pad = "README_upload_temp.md"
    with open(readme_pad, "w", encoding="utf-8") as f:
        f.write(inhoud)
    return readme_pad
