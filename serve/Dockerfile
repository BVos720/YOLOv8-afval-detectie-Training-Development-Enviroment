FROM python:3.11-slim

WORKDIR /app

# Systeem dependencies voor OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY serve/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# API code
COPY serve/main.py .

# Getraind model inbakken in de image
COPY model/best.pt /app/model/best.pt

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
