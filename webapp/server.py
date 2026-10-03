"""FastAPI backend that serves the face-tracker UI and runs the Keras model.

Run from the project root:
    .venv\\Scripts\\python -m uvicorn webapp.server:app --port 8000
"""
import asyncio
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("FACETRACKER_MODEL", ROOT.parent / "facetracker.keras"))
INPUT_SIZE = 120  # the model was built with Input(shape=(120, 120, 3))

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model_lock = threading.Lock()
# Warm-up so the first real request isn't slow.
model(np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), np.float32), training=False)


def detect(image_bytes: bytes) -> dict:
    """Decode an image, run the face tracker, return score + normalized box."""
    bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Same preprocessing as the training notebook: tf.image.resize -> /255
    resized = tf.image.resize(rgb, (INPUT_SIZE, INPUT_SIZE)) / 255.0

    start = time.perf_counter()
    with model_lock:
        cls, coords = model(tf.expand_dims(resized, 0), training=False)
    elapsed_ms = (time.perf_counter() - start) * 1000

    box = np.clip(coords.numpy()[0], 0.0, 1.0)  # [x1, y1, x2, y2], normalized
    return {
        "score": float(cls.numpy()[0][0]),
        "box": [float(v) for v in box],
        "inference_ms": round(elapsed_ms, 1),
        "width": int(bgr.shape[1]),
        "height": int(bgr.shape[0]),
    }


app = FastAPI(title="Face Tracker")


@app.middleware("http")
async def no_cache(request, call_next):
    # Always revalidate, so edits (or another app on the same port) never serve stale files.
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/info")
def info():
    return {
        "model": MODEL_PATH.name,
        "input_size": INPUT_SIZE,
        "parameters": int(model.count_params()),
        "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
        "tensorflow": tf.__version__,
    }


@app.post("/api/detect")
async def detect_upload(file: UploadFile = File(...)):
    data = await file.read()
    try:
        return await asyncio.to_thread(detect, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/ws")
async def detect_stream(ws: WebSocket):
    """Client sends JPEG frames as binary messages; each gets a JSON reply."""
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            try:
                result = await asyncio.to_thread(detect, data)
            except ValueError as e:
                result = {"error": str(e)}
            await ws.send_json(result)
    except WebSocketDisconnect:
        pass
