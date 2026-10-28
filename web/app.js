import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";
import { loadNet } from "./doodle.js";

const ROUNDS = 5;
const ROUND_SECONDS = 25;
const BETWEEN_MS = 2500;       // "Got it!" / "Time's up" pause between drawings
const FINAL_RETURN_S = 30;     // score page goes back to the main page by itself
const WIN_CONFIDENCE = 0.55;   // top guess must be the target with at least this probability
const GUESS_INTERVAL = 350;    // ms between classifications
const PEN_UP_GRACE_MS = 300;   // pen stays down through tracking dropouts / pose flickers this short (e.g. finger passing a face)
const BRIDGE_MAX = 0.15;       // after such a gap the line continues if the finger is back within this fraction of the frame width
const PALM_HOLD_MS = 1500;     // open palm this long on the main / score page = start / go home
const FIST_HOLD_MS = 1000;     // fist this long while drawing = erase all
const INK = "#ffdc3c";

// Part of the camera image shown on screen (fractions of the frame, centred
// horizontally). The bottom of the screen is well above the bottom of the camera
// frame, so the palm and wrist stay in shot when the finger reaches the bottom
// edge; the tracker loses a hand whose palm is out of frame.
const VIEW = { x: 0.15, y: 0, w: 0.7, h: 0.7 };

// Crowd handling: several hands are detected, people far behind the player
// (small hands) are ignored, and the closest hand is followed as the player.
const MAX_HANDS = 4;
const MIN_HAND_SIZE = 0.07;    // wrist -> middle knuckle, as a fraction of frame width
const LOCK_MS = 2000;          // during a game the player's hand stays locked this long after it was last seen;
                               // other people's hands are ignored even if they come closer to the camera

// One Euro filter: smooths jitter when the finger moves slowly, follows it with
// almost no lag when it moves fast. Raise MIN_CUTOFF for less lag, lower it for less jitter.
const MIN_CUTOFF = 1.5, BETA = 0.012, D_CUTOFF = 1.0;

const EMOJI = {
  cat: "🐱", apple: "🍎", bicycle: "🚲", bird: "🐦", butterfly: "🦋", car: "🚗", clock: "🕐",
  cloud: "☁️", cup: "☕", donut: "🍩", eye: "👁️", flower: "🌸", hat: "🎩", ice_cream: "🍦",
  key: "🔑", ladder: "🪜", light_bulb: "💡", lightning: "⚡", moon: "🌙", mountain: "⛰️",
  mushroom: "🍄", pizza: "🍕", rainbow: "🌈", smiley_face: "😀", snake: "🐍", spider: "🕷️",
  star: "⭐", sun: "☀️", sword: "🗡️", tent: "⛺", tree: "🌳", umbrella: "☂️", airplane: "✈️",
  candle: "🕯️", envelope: "✉️", "t-shirt": "👕", eyeglasses: "👓", headphones: "🎧",
};

const $ = (id) => document.getElementById(id);
const video = $("cam"), view = $("view"), ctx = view.getContext("2d");
const pretty = (label) => label.replace(/_/g, " ");
const emoji = (label) => EMOJI[label] || "✏️";

// ---------------------------------------------------------------- hand logic

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

// Landmark (normalised camera coords) -> canvas pixels, mirrored and cropped to VIEW.
const toScreenX = (p, w) => ((1 - p.x - VIEW.x) / VIEW.w) * w;
const toScreenY = (p, h) => ((p.y - VIEW.y) / VIEW.h) * h;
const dist3 =(a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

// Degrees a finger bends at joint b (0 = straight).
function bend(a, b, c) {
  const u = [b.x - a.x, b.y - a.y, b.z - a.z], v = [c.x - b.x, c.y - b.y, c.z - b.z];
  const cos = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (Math.hypot(...u) * Math.hypot(...v) || 1);
  return (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
}

// Uses the 3D world landmarks (metres, centred on the hand), so it reads the same
// when the finger points toward the camera, the hand is tilted, or far away.
// true = straight, false = curled, null = somewhere in between.
function fingerState(w, mcp) {
  const total = bend(w[0], w[mcp], w[mcp + 1]) + bend(w[mcp], w[mcp + 1], w[mcp + 2]) + bend(w[mcp + 1], w[mcp + 2], w[mcp + 3]);
  return total < 75 ? true : total > 130 ? false : null;
}

// `drawing`: the pen is already down. Starting a line needs a clean pointing pose;
// keeping it going only needs the index not to curl, so noisy landmarks (e.g. the
// hand passing in front of a face) don't break the line.
function gesture(w, drawing) {
  const thumb = dist3(w[4], w[17]) > dist3(w[2], w[17]) * 1.2;
  const [index, middle, ring, pinky] = [5, 9, 13, 17].map((mcp) => fingerState(w, mcp));
  if (thumb && index && middle && ring && pinky) return "stop";
  if (index === false && middle === false && ring === false && pinky === false) return "fist";
  const othersUp = [middle, ring, pinky].filter((f) => f === true).length;
  if (index && othersUp === 0) return "draw";
  if (drawing && index !== false && othersUp <= 1) return "draw";
  return "hover";
}

let player = null;   // { x, y, size, at } wrist and size of the hand we followed last

// Choose the player's hand among everything detected. `pick` is null when the
// player isn't visible this frame; `jumped` means it's a different hand than before.
function pickPlayerHand(hands, now) {
  const sized = hands
    .map((h) => ({ ...h, size: dist(h.lm[0], h.lm[9]) }))
    .filter((h) => h.size >= MIN_HAND_SIZE);
  const wristTo = (h, p) => Math.hypot(h.lm[0].x - p.x, h.lm[0].y - p.y);
  const lockMs = state === "playing" || state === "between" ? LOCK_MS : 600;
  let pick = null, jumped = false;
  if (player && now - player.at < lockMs) {
    // Locked: only a hand of similar size near where the player's hand was
    // (the longer it's been lost, the farther it may have moved) can be the player.
    const reach = Math.min(0.15 + ((now - player.at) / 1000) * 0.5, 0.45);
    const near = sized.filter((h) => wristTo(h, player) < reach && h.size > player.size * 0.6 && h.size < player.size / 0.6);
    if (near.length) pick = near.reduce((a, b) => (wristTo(a, player) <= wristTo(b, player) ? a : b));
  } else if (sized.length) {
    pick = sized.reduce((a, b) => (a.size >= b.size ? a : b));
    jumped = player !== null;
  }
  if (pick) player = { x: pick.lm[0].x, y: pick.lm[0].y, size: pick.size, at: now };
  return { pick, jumped, others: sized.filter((h) => h !== pick).map((h) => h.lm) };
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

let state = "loading";       // loading -> home -> playing <-> between -> final -> home
let game = [];               // [{ target, won, time, guess }]
let round = 0;
let strokes = [], guesses = [];
let roundStart = 0, lastGuessTime = 0, stateSince = 0;
let cursor = null, penDown = false, lastDrawAt = 0;
let holdGesture = null, holdSince = 0, holdArmed = true;

function show(id, on) { $(id).hidden = !on; }

function setState(s) {
  state = s;
  stateSince = performance.now();
  holdGesture = null;
  holdArmed = false;          // the palm that triggered this must be lowered first
  show("home", s === "home");
  show("between", s === "between");
  show("final", s === "final");
  show("top", s === "playing" || s === "between");
  show("bottom", s === "playing");
  document.body.classList.toggle("playing", s === "playing" || s === "between");
}

function goHome() {
  strokes = []; guesses = [];
  $("homeCta").textContent = "Hold up an open palm 🖐️ to start";
  setState("home");
}

function startGame() {
  const pool = [...targets];
  game = [];
  for (let i = 0; i < ROUNDS; i++) game.push({ target: pool.splice(Math.floor(Math.random() * pool.length), 1)[0] });
  round = 0;
  requestWakeLock();
  startRound();
}

function startRound() {
  const r = game[round];
  strokes = []; guesses = []; penDown = false;
  roundStart = performance.now();
  $("round").textContent = `Drawing ${round + 1} / ${ROUNDS}`;
  $("emoji").textContent = emoji(r.target);
  $("target").textContent = pretty(r.target);
  renderDots();
  renderGuess();
  setState("playing");
}

function endRound(won, now) {
  const r = game[round];
  r.won = won;
  r.time = (now - roundStart) / 1000;
  r.guess = guesses[0] ? guesses[0][0] : null;
  renderDots();
  $("betweenEmoji").textContent = won ? emoji(r.target) : "⏰";
  $("betweenTitle").textContent = won ? `Got it! It's a ${pretty(r.target)}!` : "Time's up!";
  $("betweenTitle").style.color = won ? "var(--good)" : "var(--bad)";
  $("betweenMsg").textContent = won
    ? `Guessed in ${r.time.toFixed(1)}s`
    : r.guess ? `I thought it was a ${pretty(r.guess)}.` : "Nothing was drawn.";
  setState("between");
}

function showFinal() {
  const score = game.filter((r) => r.won).length;
  $("finalScore").textContent = `${score} / ${ROUNDS}`;
  $("finalMsg").textContent = ["Better luck next time!", "Nice try!", "Not bad!", "Good job!", "Great drawing!", "Perfect! You're an artist! 🎨"][score];
  $("results").innerHTML = game.map((r) => `
    <div class="${r.won ? "won" : "lost"}">
      <span class="e">${emoji(r.target)}</span>
      <span class="w">${pretty(r.target)}</span><br>
      <span class="d">${r.won ? `✓ ${r.time.toFixed(1)}s` : r.guess ? `✗ saw ${pretty(r.guess)}` : "✗ no drawing"}</span>
    </div>`).join("");
  setState("final");
}

function renderDots() {
  $("dots").innerHTML = game.map((r, i) =>
    `<span class="${r.won === true ? "won" : r.won === false ? "lost" : ""} ${i === round && r.won === undefined ? "now" : ""}"></span>`).join("");
}

function renderGuess() {
  if (!guesses.length) {
    $("guess").textContent = "Start drawing with your index finger! ☝️";
    $("others").textContent = "";
    return;
  }
  const [name, p] = guesses[0];
  $("guess").textContent = `I see… ${pretty(name)}? (${Math.round(p * 100)}%)`;
  $("others").textContent = guesses.slice(1).map(([n, q]) => `${pretty(n)} ${Math.round(q * 100)}%`).join(",  ");
}

function eraseAll() {
  if (state !== "playing") return;
  strokes = []; guesses = []; penDown = false;
  renderGuess();
}

// Returns 0..1 progress of the current gesture hold, and fires `action` when it completes.
function hold(g, wanted, ms, now, action) {
  if (g !== wanted) { holdGesture = null; holdArmed = true; return 0; }   // released -> can trigger again
  if (!holdArmed) return 0;
  if (holdGesture !== wanted) { holdGesture = wanted; holdSince = now; }
  const p = Math.min(1, (now - holdSince) / ms);
  if (p >= 1) { holdGesture = null; holdArmed = false; action(); }
  return p;
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

  let g = null, others = [];
  const res = landmarker.detectForVideo(video, now);
  const hands = (res.landmarks || []).map((lm, i) => ({ lm, world: res.worldLandmarks[i] }));
  const hand = pickPlayerHand(hands, now);
  others = hand.others;
  if (hand.pick) {
    const lm = hand.pick.lm;
    g = gesture(hand.pick.world, penDown);
    if (hand.jumped) { fx.reset(); fy.reset(); penDown = false; }
    cursor = [fx.filter(toScreenX(lm[8], w), now), fy.filter(toScreenY(lm[8], h), now)];
  } else {
    cursor = null;
    fx.reset(); fy.reset();
  }

  let holdProgress = 0;
  if (state === "home") holdProgress = hold(g, "stop", PALM_HOLD_MS, now, startGame);
  else if (state === "final") {
    holdProgress = hold(g, "stop", PALM_HOLD_MS, now, goHome);
    const left = Math.ceil(FINAL_RETURN_S - (now - stateSince) / 1000);
    $("homeCountdown").textContent = `Open palm 🖐️ or button to go back · returning automatically in ${left}s`;
    if (left <= 0) goHome();
  } else if (state === "between" && now - stateSince > BETWEEN_MS) {
    round++;
    if (round < ROUNDS) startRound(); else showFinal();
  } else if (state === "playing") {
    holdProgress = hold(g, "fist", FIST_HOLD_MS, now, eraseAll);

    // Pen goes down immediately. If the hand is lost or the pose glitches for less
    // than PEN_UP_GRACE_MS, the pen stays down and the line is joined up when the
    // finger comes back nearby. Open palm or fist lifts it at once.
    if (g === "draw" && cursor) {
      const pt = [Math.round(cursor[0]), Math.round(cursor[1])];
      const last = penDown && strokes.length ? strokes.at(-1).at(-1) : null;
      const gap = last ? Math.hypot(pt[0] - last[0], pt[1] - last[1]) : Infinity;
      if (gap > BRIDGE_MAX * w) strokes.push([pt]);
      else if (gap > 1.5) strokes.at(-1).push(pt);
      penDown = true;
      lastDrawAt = now;
    } else if (g === "stop" || g === "fist" || now - lastDrawAt > PEN_UP_GRACE_MS) {
      penDown = false;
    }

    const target = game[round].target;
    if (strokes.length && now - lastGuessTime > GUESS_INTERVAL) {
      guesses = guess(strokes);
      lastGuessTime = now;
      renderGuess();
      if (guesses.length && guesses[0][0] === target && guesses[0][1] >= WIN_CONFIDENCE) endRound(true, now);
    }

    if (state === "playing") {
      const remaining = Math.max(0, ROUND_SECONDS - (now - roundStart) / 1000);
      const low = remaining < 6;
      $("timer").textContent = String(Math.ceil(remaining));
      $("timer").classList.toggle("low", low);
      $("progress").classList.toggle("low", low);
      $("progress").style.width = `${(remaining / ROUND_SECONDS) * 100}%`;
      if (remaining <= 0) {
        guesses = strokes.length ? guess(strokes) : [];
        endRound(false, now);
      }
    }
  }

  render(g, others, holdProgress, w, h);
}

function render(g, others, holdProgress, w, h) {
  // mirrored camera cropped to VIEW, dimmed
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  const vw = video.videoWidth, vh = video.videoHeight;
  ctx.drawImage(video, vw * VIEW.x, vh * VIEW.y, vw * VIEW.w, vh * VIEW.h, 0, 0, w, h);
  ctx.restore();
  ctx.fillStyle = "rgba(0,0,0,.3)";
  ctx.fillRect(0, 0, w, h);

  // ignored hands (other people) shown as faint grey dots so it's clear who is playing
  ctx.fillStyle = "rgba(200,200,200,.35)";
  for (const lm of others) {
    ctx.beginPath();
    ctx.arc(toScreenX(lm[8], w), toScreenY(lm[8], h), 6, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.strokeStyle = ctx.fillStyle = INK;
  ctx.lineWidth = 9;
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

  if (cursor) {
    const [cx, cy] = cursor;
    ctx.beginPath();
    if (g === "draw" && state === "playing") {
      ctx.fillStyle = INK;
      ctx.arc(cx, cy, 11, 0, Math.PI * 2);
      ctx.fill();
    } else if (g === "stop") {
      ctx.strokeStyle = ctx.fillStyle = "#ff5a5a";
      ctx.lineWidth = 3;
      ctx.arc(cx, cy, 16, 0, Math.PI * 2);
      ctx.stroke();
      if (state === "playing") { ctx.font = "bold 24px system-ui"; ctx.fillText("STOP", cx + 24, cy + 8); }
    } else {
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.arc(cx, cy, 12, 0, Math.PI * 2);
      ctx.stroke();
    }
    if (holdProgress > 0) {
      ctx.beginPath();
      ctx.strokeStyle = state === "playing" ? "#ffffff" : "#78ff78";
      ctx.lineWidth = 7;
      ctx.arc(cx, cy, 34, -Math.PI / 2, -Math.PI / 2 + holdProgress * Math.PI * 2);
      ctx.stroke();
    }
  }

  ctx.fillStyle = "rgba(255,255,255,.55)";
  ctx.font = "15px system-ui";
  ctx.fillText(`${Math.round(fps)} fps`, 12, h - (state === "playing" ? 90 : 14));
}

// ---------------------------------------------------------------- kiosk helpers

let wakeLock = null;
async function requestWakeLock() {
  try { if ("wakeLock" in navigator && !wakeLock) wakeLock = await navigator.wakeLock.request("screen"); } catch { /* not critical */ }
  if (wakeLock) wakeLock.addEventListener("release", () => { wakeLock = null; }, { once: true });
}
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") requestWakeLock(); });

addEventListener("keydown", (e) => {
  if (e.code === "Space") {
    e.preventDefault();
    if (state === "home") startGame();
    else if (state === "final") goHome();
  }
  if (e.code === "KeyE") eraseAll();
  if (e.code === "KeyF") {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen().catch(() => {});
  }
  if (e.code === "Escape" && state !== "home" && state !== "loading") goHome();
});
$("startBtn").onclick = () => { $("startBtn").blur(); startGame(); };
$("homeBtn").onclick = () => { $("homeBtn").blur(); goHome(); };

function scheduleFrame() {
  if ("requestVideoFrameCallback" in video) video.requestVideoFrameCallback(frame);
  else requestAnimationFrame(frame);
}

// ---------------------------------------------------------------- startup

async function createLandmarker(fileset, delegate) {
  return HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "hand_landmarker.task", delegate },
    runningMode: "VIDEO",
    numHands: MAX_HANDS,
    // Lower than before so tracking holds when the hand overlaps a face or another
    // person; stray detections are filtered out by the player lock.
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.4,
    minTrackingConfidence: 0.4,
  });
}

async function init() {
  try {
    const [meta, weights, stream, fileset] = await Promise.all([
      fetch("model.json").then((r) => r.json()),
      fetch("weights.bin").then((r) => r.arrayBuffer()),
      navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60 }, facingMode: "user" },
      }),
      FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"),
    ]);
    predict = loadNet(meta.layers, weights);
    labels = meta.labels;
    targets = meta.targets;
    try { landmarker = await createLandmarker(fileset, "GPU"); }
    catch { landmarker = await createLandmarker(fileset, "CPU"); }
    video.srcObject = stream;
    await video.play();
    $("startBtn").disabled = false;
    goHome();
    scheduleFrame();
  } catch (err) {
    console.error(err);
    $("homeCta").textContent = `Couldn't start: ${err.message}. Allow camera access and reload.`;
    $("homeCta").style.color = "var(--bad)";
  }
}

init();
