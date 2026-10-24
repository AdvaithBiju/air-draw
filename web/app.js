import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { loadNet } from "./doodle.js";

const ROUNDS = 5;
const ROUND_SECONDS = 25;
const BETWEEN_MS = 2500;       // "Got it!" / "Time's up" pause between drawings
const FINAL_RETURN_S = 30;     // score page goes back to the main page by itself
const WIN_CONFIDENCE = 0.55;   // top guess must be the target with at least this probability
const GUESS_INTERVAL = 350;    // ms between classifications
const PEN_UP_GRACE = 3;        // frames a non-draw gesture must last before the pen lifts
const PALM_HOLD_MS = 1500;     // open palm this long on the main / score page = start / go home
const FIST_HOLD_MS = 1000;     // fist this long while drawing = erase all
const INK = "#ffdc3c";

// Crowd handling: several hands are detected, people far behind the player
// (small hands) are ignored, and the closest hand is followed as the player.
const MAX_HANDS = 4;
const MIN_HAND_SIZE = 0.07;    // wrist -> middle knuckle, as a fraction of frame width
const RIVAL_RATIO = 0.7;       // hands at least this big relative to the biggest stay candidates

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
  if (!index && !middle && !ring && !pinky) return "fist";
  return "hover";
}

let player = null;   // { x, y, at } wrist of the hand we followed last frame

// Choose the player's hand among everything detected.
function pickPlayerHand(hands, now) {
  const sized = hands
    .map((lm) => ({ lm, size: dist(lm[0], lm[9]) }))
    .filter((h) => h.size >= MIN_HAND_SIZE);
  if (!sized.length) return null;
  const biggest = Math.max(...sized.map((h) => h.size));
  const candidates = sized.filter((h) => h.size >= biggest * RIVAL_RATIO);
  let pick;
  if (player && now - player.at < 600) {
    // stick with the hand closest to where the player's hand just was
    pick = candidates.reduce((a, b) =>
      Math.hypot(a.lm[0].x - player.x, a.lm[0].y - player.y) <= Math.hypot(b.lm[0].x - player.x, b.lm[0].y - player.y) ? a : b);
  } else {
    pick = candidates.reduce((a, b) => (a.size >= b.size ? a : b));
  }
  const jumped = player && Math.hypot(pick.lm[0].x - player.x, pick.lm[0].y - player.y) > 0.25;
  player = { x: pick.lm[0].x, y: pick.lm[0].y, at: now };
  return { lm: pick.lm, jumped, others: sized.filter((h) => h !== pick).map((h) => h.lm) };
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
let cursor = null, penDown = false, notDrawFrames = 0;
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
  const hand = res.landmarks && res.landmarks.length ? pickPlayerHand(res.landmarks, now) : null;
  if (hand) {
    const lm = hand.lm;
    g = gesture(lm);
    others = hand.others;
    if (hand.jumped) { fx.reset(); fy.reset(); penDown = false; }
    cursor = [fx.filter((1 - lm[8].x) * w, now), fy.filter(lm[8].y * h, now)];   // mirrored
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

    // Pen goes down immediately, but only lifts after a few non-draw frames so a
    // one-frame tracking hiccup doesn't break the line. Open palm lifts it at once.
    if (g === "draw" && cursor) notDrawFrames = 0;
    else notDrawFrames++;
    const drawing = g !== "stop" && g !== "fist" && notDrawFrames < PEN_UP_GRACE && (penDown || g === "draw");
    if (drawing && cursor) {
      const pt = [Math.round(cursor[0]), Math.round(cursor[1])];
      if (!penDown) strokes.push([pt]);
      else {
        const last = strokes.at(-1).at(-1);
        if (Math.hypot(pt[0] - last[0], pt[1] - last[1]) > 1.5) strokes.at(-1).push(pt);
      }
    }
    penDown = drawing;

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
  // mirrored camera, dimmed
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();
  ctx.fillStyle = "rgba(0,0,0,.3)";
  ctx.fillRect(0, 0, w, h);

  // ignored hands (other people) shown as faint grey dots so it's clear who is playing
  ctx.fillStyle = "rgba(200,200,200,.35)";
  for (const lm of others) {
    ctx.beginPath();
    ctx.arc((1 - lm[8].x) * w, lm[8].y * h, 6, 0, Math.PI * 2);
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
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.5,
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
      FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"),
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
