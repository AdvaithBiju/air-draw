"""Air Draw: the computer tells you what to draw, you draw it in the air
with your index finger, and it tries to guess it before 50 seconds run out.

Gestures
  Index finger only up   -> draw
  All 5 fingers open     -> stop (pen lifted, nothing is drawn)
  Anything else          -> pen lifted

Keys
  SPACE  start / next round      C  clear drawing      Q / ESC  quit
"""
import math
import os
import random
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from model import TARGETS, load_model, pretty

HERE = os.path.dirname(os.path.abspath(__file__))
ROUND_SECONDS = 50
WIN_CONFIDENCE = 0.55     # top guess must be the target with at least this probability
GUESS_INTERVAL = 0.35     # seconds between classifications
SMOOTHING = 0.45          # 0..1, higher = smoother but laggier cursor

INK = (60, 220, 255)      # BGR
FONT = cv2.FONT_HERSHEY_DUPLEX


# ---------------------------------------------------------------- hand logic

def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def fingers_up(lm):
    """Return [thumb, index, middle, ring, pinky] booleans. Rotation independent:
    a finger counts as extended when its tip is clearly farther from the wrist than its middle joint."""
    wrist = lm[0]
    up = [_dist(lm[4], lm[17]) > _dist(lm[2], lm[17]) * 1.25]
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        up.append(_dist(lm[tip], wrist) > _dist(lm[pip], wrist) * 1.15)
    return up


def gesture(lm):
    thumb, index, middle, ring, pinky = fingers_up(lm)
    if all((thumb, index, middle, ring, pinky)):
        return "stop"
    if index and not (middle or ring or pinky):
        return "draw"
    return "hover"


# ---------------------------------------------------------------- classifier

class Guesser:
    def __init__(self):
        self.model, self.labels = load_model()

    @staticmethod
    def to_bitmap(strokes):
        """Render strokes the way Quick, Draw! bitmaps look: white on black,
        cropped to the drawing, centered, 28x28."""
        pts = [p for s in strokes for p in s]
        if len(pts) < 2:
            return None
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x0, y0 = min(xs), min(ys)
        size = max(max(xs) - x0, max(ys) - y0, 1)
        canvas = np.zeros((256, 256), np.uint8)
        pad, inner = 20, 216
        ox = pad + (inner - (max(xs) - x0) / size * inner) / 2
        oy = pad + (inner - (max(ys) - y0) / size * inner) / 2
        for s in strokes:
            mapped = [(int(ox + (x - x0) / size * inner), int(oy + (y - y0) / size * inner)) for x, y in s]
            if len(mapped) == 1:
                cv2.circle(canvas, mapped[0], 7, 255, -1)
            else:
                cv2.polylines(canvas, [np.array(mapped, np.int32)], False, 255, 14, cv2.LINE_AA)
        return cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)

    def guess(self, strokes):
        bmp = self.to_bitmap(strokes)
        if bmp is None:
            return []
        x = torch.from_numpy(bmp).float().div(255).view(1, 1, 28, 28)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), 1)[0]
        top = torch.topk(probs, 3)
        return [(self.labels[i], p.item()) for p, i in zip(top.values, top.indices)]


# ---------------------------------------------------------------- drawing helpers

def text(img, s, org, scale=1.0, color=(255, 255, 255), thick=2, center=False):
    (w, h), _ = cv2.getTextSize(s, FONT, scale, thick)
    x, y = org
    if center:
        x -= w // 2
    cv2.putText(img, s, (x, y), FONT, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    return w, h


def panel(img, x0, y0, x1, y1, alpha=0.55):
    roi = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = (roi * (1 - alpha)).astype(np.uint8)


def draw_strokes(img, strokes):
    for s in strokes:
        if len(s) == 1:
            cv2.circle(img, s[0], 5, INK, -1, cv2.LINE_AA)
        elif len(s) > 1:
            cv2.polylines(img, [np.array(s, np.int32)], False, INK, 8, cv2.LINE_AA)


# ---------------------------------------------------------------- game

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera.")

    hand_model = os.path.join(HERE, "hand_landmarker.task")
    if not os.path.exists(hand_model):
        import urllib.request
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                                   "hand_landmarker/float16/latest/hand_landmarker.task", hand_model)
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ))
    guesser = Guesser()

    state = "menu"            # menu -> playing -> result -> playing ...
    target, strokes, guesses = None, [], []
    start_time = last_guess_time = result_time = 0.0
    won, final_guess = False, None
    cursor, pen_down = None, False
    score, rounds = 0, 0
    t0 = time.monotonic()

    def new_round():
        nonlocal state, target, strokes, guesses, start_time, won, final_guess, pen_down, rounds
        choices = [c for c in TARGETS if c != target]
        target = random.choice(choices)
        strokes, guesses, won, final_guess, pen_down = [], [], False, None, False
        start_time = time.monotonic()
        rounds += 1
        state = "playing"

    cv2.namedWindow("Air Draw", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.monotonic()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int((now - t0) * 1000))

        g = None
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            g = gesture(lm)
            tip = (lm[8].x * w, lm[8].y * h)
            if cursor is None:
                cursor = tip
            else:
                cursor = (cursor[0] * SMOOTHING + tip[0] * (1 - SMOOTHING),
                          cursor[1] * SMOOTHING + tip[1] * (1 - SMOOTHING))
        else:
            cursor = None

        # --- drawing
        if state == "playing":
            drawing = g == "draw" and cursor is not None
            if drawing:
                pt = (int(cursor[0]), int(cursor[1]))
                if not pen_down:
                    strokes.append([pt])
                elif math.dist(pt, strokes[-1][-1]) > 2:
                    strokes[-1].append(pt)
            pen_down = drawing

            if strokes and now - last_guess_time > GUESS_INTERVAL:
                guesses = guesser.guess(strokes)
                last_guess_time = now
                if guesses and guesses[0][0] == target and guesses[0][1] >= WIN_CONFIDENCE:
                    won, final_guess = True, guesses[0]
                    score += 1
                    state, result_time = "result", now

            if state == "playing" and now - start_time >= ROUND_SECONDS:
                guesses = guesser.guess(strokes) if strokes else []
                final_guess = guesses[0] if guesses else None
                won = False
                state, result_time = "result", now

        # --- render
        view = frame.copy()
        panel(view, 0, 0, w, h, alpha=0.25)
        draw_strokes(view, strokes)

        if cursor is not None:
            c = (int(cursor[0]), int(cursor[1]))
            if g == "draw":
                cv2.circle(view, c, 10, INK, -1, cv2.LINE_AA)
            elif g == "stop":
                cv2.circle(view, c, 16, (80, 80, 255), 3, cv2.LINE_AA)
                text(view, "STOP", (c[0] + 22, c[1] + 8), 0.7, (80, 80, 255))
            else:
                cv2.circle(view, c, 12, (255, 255, 255), 2, cv2.LINE_AA)

        if state == "menu":
            panel(view, 0, 0, w, h, alpha=0.5)
            text(view, "AIR DRAW", (w // 2, h // 2 - 90), 2.2, INK, 4, center=True)
            text(view, "I'll tell you what to draw. You have 50 seconds.", (w // 2, h // 2 - 20), 0.9, center=True)
            text(view, "Index finger = draw     Open palm (5 fingers) = stop", (w // 2, h // 2 + 30), 0.8, center=True)
            text(view, "Press SPACE to start", (w // 2, h // 2 + 110), 1.1, (120, 255, 120), center=True)

        elif state in ("playing", "result"):
            panel(view, 0, 0, w, 90)
            text(view, "Draw:", (24, 58), 1.0, (200, 200, 200))
            text(view, pretty(target).upper(), (120, 60), 1.4, INK, 3)
            remaining = max(0.0, ROUND_SECONDS - (now - start_time)) if state == "playing" else 0
            tcol = (80, 80, 255) if remaining < 10 else (255, 255, 255)
            text(view, f"{math.ceil(remaining):02d}s", (w - 130, 60), 1.4, tcol, 3)
            text(view, f"Score {score}/{rounds}", (w - 360, 58), 0.9, (200, 200, 200))
            bar = int(w * remaining / ROUND_SECONDS)
            cv2.rectangle(view, (0, 86), (bar, 90), tcol, -1)

            if state == "playing":
                panel(view, 0, h - 70, w, h)
                if guesses:
                    name, p = guesses[0]
                    others = ",  ".join(f"{pretty(n)} {q:.0%}" for n, q in guesses[1:])
                    text(view, f"I see... {pretty(name)}?  ({p:.0%})", (24, h - 28), 1.0)
                    text(view, others, (w - 24 - cv2.getTextSize(others, FONT, 0.7, 2)[0][0], h - 28), 0.7, (180, 180, 180))
                else:
                    text(view, "Start drawing with your index finger!", (24, h - 28), 0.9, (200, 200, 200))
                text(view, "C clear   Q quit", (w // 2, h - 80), 0.6, (200, 200, 200), 1, center=True)
            else:
                panel(view, 0, 0, w, h, alpha=0.45)
                if won:
                    text(view, f"Oh I know, it's a {pretty(target)}!", (w // 2, h // 2 - 20), 1.6, (120, 255, 120), 3, center=True)
                    text(view, f"Guessed in {result_time - start_time:.1f}s", (w // 2, h // 2 + 35), 0.9, center=True)
                else:
                    text(view, "Time's up!", (w // 2, h // 2 - 50), 1.6, (80, 80, 255), 3, center=True)
                    msg = (f"I thought it was a {pretty(final_guess[0])} ({final_guess[1]:.0%}), not a {pretty(target)}."
                           if final_guess else f"You didn't draw anything! It was {pretty(target)}.")
                    text(view, msg, (w // 2, h // 2 + 10), 0.9, center=True)
                text(view, "Press SPACE for the next drawing", (w // 2, h // 2 + 100), 1.0, (200, 200, 200), center=True)

        cv2.imshow("Air Draw", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty("Air Draw", cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord(" ") and state in ("menu", "result"):
            new_round()
        if key == ord("c") and state == "playing":
            strokes, guesses, pen_down = [], [], False

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
