import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { loadNet } from "./doodle.js";

const ROUND_SECONDS = 50;
const WIN_CONFIDENCE = 0.55;   // top guess must be the target with at least this probability
const GUESS_INTERVAL = 350;    // ms between classifications
const PEN_UP_GRACE = 3;        // frames a non-draw gesture must last before the pen lifts
const INK = "#ffdc3c";

// One Euro filter: smooths jitter when the finger moves slowly, follows it with
// almost no lag when it moves fast. Raise MIN_CUTOFF for less lag, lower it for less jitter.
const MIN_CUTOFF = 1.5, BETA = 0.012, D_CUTOFF = 1.0;

const $ = (id) => document.getElementById(id);
const video = $("cam"), view = $("view"), ctx = view.getContext("2d");
const pretty = (label) => label.replace(/_/g, " ");

// ---------------------------------------------------------------- hand logic

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

// Rotation independent: a finger counts as extended when its tip is clearly
// farther from the wrist than its middle joint.
function fingersUp(lm) {
  const up = [dist(lm[4], lm[17]) > dist(lm[2], lm[17]) * 1.25];
  for (const [tip, pip] of [[8, 6], [12, 10], [16, 14], [20, 18]])
    up.push(dist(lm[tip], lm[0]) > dist(lm[pip], lm[0]) * 1.15);
  return up;
}

function gesture(lm) {
  const [thumb, index, middle, ring, pinky] = fingersUp(lm);
  if (thumb && index && middle && ring && pinky) return "stop";
  if (index && !middle && !ring && !pinky) return "draw";
  return "hover";
}

class OneEuro {
  constructor() { this.x = null; this.dx = 0; this.t = 0; }
  static alpha(cutoff, dt) { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); }
  reset() { this.x = null; }
  filter(x, t) {
    if (this.x === null) { this.x = x; this.dx = 0; this.t = t; return x; }
    const dt = Math.max((t - this.t) / 1000, 1e-3);
    this.t = t;
    const ad = OneEuro.alpha(D_CUTOFF, dt);
    this.dx = ad * ((x - this.x) / dt) + (1 - ad) * this.dx;
    const a = OneEuro.alpha(MIN_CUTOFF + BETA * Math.abs(this.dx), dt);
    this.x = a * x + (1 - a) * this.x;
    return this.x;
  }
}
const fx = new OneEuro(), fy = new OneEuro();

// ---------------------------------------------------------------- classifier

const bmpCanvas = Object.assign(document.createElement("canvas"), { width: 252, height: 252 });
const bmpCtx = bmpCanvas.getContext("2d", { willReadFrequently: true });

// Render strokes like Quick, Draw! bitmaps: white on black, cropped, centered,
// drawn at 252px then averaged down to 28x28 (9x9 blocks).
function toBitmap(strokes) {
  const pts = strokes.flat();
  if (pts.length < 2) return null;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), y0 = Math.min(...ys);
  const wSpan = Math.max(...xs) - x0, hSpan = Math.max(...ys) - y0;
  const size = Math.max(wSpan, hSpan, 1), pad = 20, inner = 212;
  const ox = pad + (inner - (wSpan / size) * inner) / 2;
  const oy = pad + (inner - (hSpan / size) * inner) / 2;

  bmpCtx.fillStyle = "#000";
  bmpCtx.fillRect(0, 0, 252, 252);
  bmpCtx.strokeStyle = bmpCtx.fillStyle = "#fff";
  bmpCtx.lineWidth = 14;
  bmpCtx.lineCap = bmpCtx.lineJoin = "round";
  for (const s of strokes) {
    const m = s.map(([x, y]) => [ox + ((x - x0) / size) * inner, oy + ((y - y0) / size) * inner]);
    bmpCtx.beginPath();
    if (m.length === 1) {
      bmpCtx.arc(m[0][0], m[0][1], 7, 0, Math.PI * 2);
      bmpCtx.fill();
    } else {
      m.forEach(([x, y], i) => (i ? bmpCtx.lineTo(x, y) : bmpCtx.moveTo(x, y)));
      bmpCtx.stroke();
    }
  }

  const data = bmpCtx.getImageData(0, 0, 252, 252).data;
  const out = new Float32Array(784);
  for (let r = 0; r < 28; r++)
    for (let c = 0; c < 28; c++) {
      let sum = 0;
      for (let dy = 0; dy < 9; dy++)
        for (let dx = 0; dx < 9; dx++) sum += data[((r * 9 + dy) * 252 + c * 9 + dx) * 4];
      out[r * 28 + c] = sum / (81 * 255);
    }
  return out;
}

let predict, labels, targets;

function guess(strokes) {
  const bmp = toBitmap(strokes);
  if (!bmp) return [];
  const probs = predict(bmp);
  return [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, 3).map((i) => [labels[i], probs[i]]);
}

// ---------------------------------------------------------------- game state

let state = "menu";          // menu -> playing -> result -> playing ...
let target = null, strokes = [], guesses = [];
let startTime = 0, lastGuessTime = 0;
let cursor = null, penDown = false, notDrawFrames = 0, score = 0, rounds = 0;

function eraseAll() {
  if (state !== "playing") return;
  strokes = []; guesses = []; penDown = false;
  renderGuess();
}

function newRound() {
  const choices = targets.filter((t) => t !== target);
  target = choices[Math.floor(Math.random() * choices.length)];
  strokes = []; guesses = []; penDown = false;
  startTime = performance.now();
  rounds++;
  state = "playing";
  $("target").textContent = pretty(target);
  $("score").textContent = `Score ${score}/${rounds}`;
  $("menu").hidden = $("result").hidden = true;
  $("top").hidden = $("bottom").hidden = false;
  renderGuess();
}

function endRound(won, now) {
  state = "result";
  if (won) score++;
  $("score").textContent = `Score ${score}/${rounds}`;
  $("bottom").hidden = true;
  $("result").hidden = false;
  const title = $("resultTitle");
  if (won) {
    title.textContent = `Oh I know, it's a ${pretty(target)}!`;
    title.style.color = "var(--good)";
    $("resultMsg").textContent = `Guessed in ${((now - startTime) / 1000).toFixed(1)}s`;
  } else {
    title.textContent = "Time's up!";
    title.style.color = "var(--bad)";
    const g = guesses[0];
    $("resultMsg").textContent = g
      ? `I thought it was a ${pretty(g[0])} (${Math.round(g[1] * 100)}%), not a ${pretty(target)}.`
      : `You didn't draw anything! It was ${pretty(target)}.`;
  }
  $("nextBtn").focus();
}

function renderGuess() {
  if (!guesses.length) {
    $("guess").textContent = "Start drawing with your index finger!";
    $("others").textContent = "";
    return;
  }
  const [name, p] = guesses[0];
  $("guess").textContent = `I see… ${pretty(name)}? (${Math.round(p * 100)}%)`;
  $("others").textContent = guesses.slice(1).map(([n, q]) => `${pretty(n)} ${Math.round(q * 100)}%`).join(",  ");
}

// ---------------------------------------------------------------- main loop

let landmarker, lastVideoTime = -1, fps = 0, lastFrameAt = 0;

// Runs once per new camera frame (requestVideoFrameCallback), so no frame is
// processed twice and none is skipped waiting for the next animation tick.
function frame() {
  scheduleFrame();
  if (video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  const now = performance.now();
  if (lastFrameAt) fps = fps * 0.9 + (1000 / (now - lastFrameAt)) * 0.1;
  lastFrameAt = now;
  const w = view.width, h = view.height;

  let g = null;
  const res = landmarker.detectForVideo(video, now);
  if (res.landmarks && res.landmarks.length) {
    const lm = res.landmarks[0];
    g = gesture(lm);
    // mirrored to match the view
    cursor = [fx.filter((1 - lm[8].x) * w, now), fy.filter(lm[8].y * h, now)];
  } else {
    cursor = null;
    fx.reset(); fy.reset();
  }

  if (state === "playing") {
    // Pen goes down immediately, but only lifts after a few non-draw frames so a
    // one-frame tracking hiccup doesn't break the line. Open palm lifts it at once.
    if (g === "draw" && cursor) notDrawFrames = 0;
    else notDrawFrames++;
    const drawing = g !== "stop" && notDrawFrames < PEN_UP_GRACE && (penDown || g === "draw");
    if (drawing && cursor) {
      const pt = [Math.round(cursor[0]), Math.round(cursor[1])];
      if (!penDown) strokes.push([pt]);
      else {
        const last = strokes.at(-1).at(-1);
        if (Math.hypot(pt[0] - last[0], pt[1] - last[1]) > 1.5) strokes.at(-1).push(pt);
      }
    }
    penDown = drawing;

    if (strokes.length && now - lastGuessTime > GUESS_INTERVAL) {
      guesses = guess(strokes);
      lastGuessTime = now;
      renderGuess();
      if (guesses.length && guesses[0][0] === target && guesses[0][1] >= WIN_CONFIDENCE) endRound(true, now);
    }

    const remaining = Math.max(0, ROUND_SECONDS - (now - startTime) / 1000);
    const low = remaining < 10;
    $("timer").textContent = `${String(Math.ceil(remaining)).padStart(2, "0")}s`;
    $("timer").classList.toggle("low", low);
    $("progress").classList.toggle("low", low);
    $("progress").style.width = `${(remaining / ROUND_SECONDS) * 100}%`;
    if (state === "playing" && remaining <= 0) {
      guesses = strokes.length ? guess(strokes) : [];
      endRound(false, now);
    }
  }

  // mirrored camera, dimmed
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();
  ctx.fillStyle = "rgba(0,0,0,.25)";
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = ctx.fillStyle = INK;
  ctx.lineWidth = 8;
  ctx.lineCap = ctx.lineJoin = "round";
  for (const s of strokes) {
    ctx.beginPath();
    if (s.length === 1) { ctx.arc(s[0][0], s[0][1], 5, 0, Math.PI * 2); ctx.fill(); continue; }
    // quadratic curves through segment midpoints for a smooth line
    ctx.moveTo(s[0][0], s[0][1]);
    for (let i = 1; i < s.length - 1; i++)
      ctx.quadraticCurveTo(s[i][0], s[i][1], (s[i][0] + s[i + 1][0]) / 2, (s[i][1] + s[i + 1][1]) / 2);
    ctx.lineTo(s.at(-1)[0], s.at(-1)[1]);
    ctx.stroke();
  }

  ctx.fillStyle = "rgba(255,255,255,.6)";
  ctx.font = "16px system-ui";
  ctx.fillText(`${Math.round(fps)} fps`, 12, h - (state === "playing" ? 80 : 14));

  if (cursor) {
    const [cx, cy] = cursor;
    ctx.beginPath();
    if (g === "draw") {
      ctx.fillStyle = INK;
      ctx.arc(cx, cy, 10, 0, Math.PI * 2);
      ctx.fill();
    } else if (g === "stop") {
      ctx.strokeStyle = ctx.fillStyle = "#ff5a5a";
      ctx.lineWidth = 3;
      ctx.arc(cx, cy, 16, 0, Math.PI * 2);
      ctx.stroke();
      ctx.font = "bold 24px system-ui";
      ctx.fillText("STOP", cx + 24, cy + 8);
    } else {
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.arc(cx, cy, 12, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}

// ---------------------------------------------------------------- startup

addEventListener("keydown", (e) => {
  if (e.code === "Space" && predict && state !== "playing") { e.preventDefault(); newRound(); }
  if (e.code === "KeyE") eraseAll();
});
$("startBtn").onclick = $("nextBtn").onclick = () => newRound();
$("eraseBtn").onclick = () => { eraseAll(); $("eraseBtn").blur(); };

function scheduleFrame() {
  if ("requestVideoFrameCallback" in video) video.requestVideoFrameCallback(frame);
  else requestAnimationFrame(frame);
}

async function init() {
  const btn = $("startBtn");
  try {
    const [meta, weights, stream, fileset] = await Promise.all([
      fetch("model.json").then((r) => r.json()),
      fetch("weights.bin").then((r) => r.arrayBuffer()),
      navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60 }, facingMode: "user" },
      }),
      FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"),
    ]);
    predict = loadNet(meta.layers, weights);
    labels = meta.labels;
    targets = meta.targets;
    landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: "hand_landmarker.task", delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.6,
      minTrackingConfidence: 0.5,
    });
    video.srcObject = stream;
    await video.play();
    btn.disabled = false;
    btn.textContent = "Start";
    scheduleFrame();
  } catch (err) {
    console.error(err);
    btn.textContent = "Error";
    $("menu").querySelector("p").textContent = `Couldn't start: ${err.message}. Allow camera access and reload.`;
  }
}

init();
