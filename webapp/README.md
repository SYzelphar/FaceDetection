# Face Tracker UI

A browser frontend for `facetracker.keras`, the VGG16 face tracker from the notebook (120×120 input, outputs a face score and one bounding box).

## Run

Double-click `run_ui.bat` in the project root, or run:

```
.venv\Scripts\python -m uvicorn webapp.server:app --port 8000
```

Then open http://localhost:8000. The first time you run `run_ui.bat`, it creates `.venv` with Python 3.11 and TensorFlow 2.15, which matches the Keras version the model was saved with.

To use a different model file, set the `FACETRACKER_MODEL` environment variable to its path.

## Features

- **Live camera:** frames go to the server over a WebSocket and the box is drawn on top of the video. You get a confidence meter, a confidence history line, FPS and latency, a camera picker and snapshots.
- **Image:** drag and drop or choose an image, see the detection, and save the annotated result.
- **Center-crop to square:** on by default, because the model was trained on 450×450 crops. The dimmed area outside the square isn't sent to the model.

## API

- `POST /api/detect` takes an image file as multipart `file` and returns `{score, box: [x1, y1, x2, y2], inference_ms}`. Box coordinates are normalized to 0–1.
- `WS /ws` takes a JPEG frame as a binary message and replies with the same JSON for each frame.
- `GET /api/info` returns model metadata.
