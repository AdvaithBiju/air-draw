# Air Draw

The computer tells you what to draw, you draw it in the air with your finger in front of the webcam, and it tries to guess it before 50 seconds run out.

- ☝️ **Index finger up** – draw
- 🖐️ **Open palm (all 5 fingers)** – stop / lift the pen
- **E** – erase all, **SPACE** – start / next round

## Web version (`web/`)

Runs entirely in the browser: [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) for hand tracking and a small CNN (plain JS forward pass) for guessing. Run locally:

```
python -m http.server 8000 --bind 127.0.0.1 -d web
```

then open http://localhost:8000. Deployed on Vercel as a static site (`vercel.json` serves `web/`).

## Desktop version

```
python -m venv .venv
.venv\Scripts\pip install mediapipe opencv-python numpy torch
.venv\Scripts\python game.py
```

`export_web.py` regenerates `web/weights.bin` and `web/model.json` from the PyTorch model.

## Credits

Doodle classifier: [nateraw/quickdraw](https://huggingface.co/nateraw/quickdraw) (MIT), trained on Google's [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset).
