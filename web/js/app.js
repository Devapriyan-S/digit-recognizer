/*
 * Digit Recognizer — canvas UI.
 *
 * Draws with pointer events (so pen, touch and mouse all work), samples the
 * canvas into a grayscale buffer, runs it through the MNIST preprocessing, and
 * predicts on every stroke end. The 28x28 preview is shown deliberately: it is
 * what the model actually sees, and it explains most surprising predictions.
 */

import { DigitModel, normalise, SIZE } from "./model.js";

const $ = (s) => document.querySelector(s);
const canvas = $("#pad");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

const STROKE = 18;         // roughly matches MNIST's pen weight after scaling
let model = null;
let drawing = false;
let hasInk = false;
let lastPoint = null;

/* ── Canvas setup ─────────────────────────────────────────── */

function sizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  // Back the canvas at device resolution so strokes are not blurry on retina,
  // then scale the drawing context so coordinates stay in CSS pixels.
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.lineWidth = STROKE;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#ffffff";
  clear(false);
}

function clear(predictAfter = true) {
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.restore();
  hasInk = false;
  if (predictAfter) {
    renderPreview(new Float32Array(SIZE * SIZE));
    showEmpty();
  }
}

const pos = (e) => {
  const r = canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
};

canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId);
  drawing = true;
  hasInk = true;
  lastPoint = pos(e);
  // A dot is a legitimate mark; without this a tap draws nothing.
  ctx.beginPath();
  ctx.arc(lastPoint.x, lastPoint.y, STROKE / 2, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
});

canvas.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  const p = pos(e);
  ctx.beginPath();
  ctx.moveTo(lastPoint.x, lastPoint.y);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  lastPoint = p;
  schedulePredict();
});

const endStroke = () => { if (drawing) { drawing = false; predict(); } };
canvas.addEventListener("pointerup", endStroke);
canvas.addEventListener("pointercancel", endStroke);
canvas.addEventListener("pointerleave", endStroke);
// Stop touch-drag from scrolling the page while drawing.
canvas.addEventListener("touchstart", (e) => e.preventDefault(), { passive: false });
canvas.addEventListener("touchmove", (e) => e.preventDefault(), { passive: false });

$("#clear").addEventListener("click", () => clear());

/* ── Prediction ───────────────────────────────────────────── */

let queued = false;
function schedulePredict() {
  // Predicting on every pointermove would run inference hundreds of times a
  // second; once per frame is imperceptibly different and far cheaper.
  if (queued) return;
  queued = true;
  requestAnimationFrame(() => { queued = false; predict(); });
}

function readInk() {
  const w = canvas.width, h = canvas.height;
  const data = ctx.getImageData(0, 0, w, h).data;
  const out = new Float32Array(w * h);
  // The pad is white-on-black, so the red channel alone is the ink intensity.
  for (let i = 0, p = 0; i < data.length; i += 4, p++) out[p] = data[i] / 255;
  return { pixels: out, w, h };
}

function predict() {
  if (!model || !hasInk) { showEmpty(); return; }

  const { pixels, w, h } = readInk();
  const t0 = performance.now();
  const { input, empty } = normalise(pixels, w, h);
  if (empty) { showEmpty(); return; }

  const result = model.predict(input);
  const ms = performance.now() - t0;

  renderPreview(input);
  showResult(result, ms);
}

function showEmpty() {
  $("#digit").textContent = "—";
  $("#digit").style.color = "var(--text-mute)";
  $("#confidence").textContent = "Draw a digit above";
  $("#bars").replaceChildren();
  $("#timing").textContent = "";
}

function showResult(result, ms) {
  $("#digit").textContent = String(result.digit);
  $("#digit").style.color = "var(--primary)";

  const conf = result.confidence;
  $("#confidence").textContent = `${(conf * 100).toFixed(1)}% confident`;
  // Low confidence is worth saying out loud: the model is guessing, and the
  // preview usually shows why.
  $("#confidence").style.color = conf > 0.9 ? "var(--green)"
    : conf > 0.6 ? "var(--amber)" : "var(--red)";
  $("#timing").textContent = `inference ${ms.toFixed(1)} ms`;

  const bars = document.createDocumentFragment();
  result.probabilities.forEach((p, digit) => {
    const row = document.createElement("div");
    row.className = "prob-row" + (digit === result.digit ? " top" : "");
    const label = document.createElement("span");
    label.className = "prob-digit";
    label.textContent = String(digit);
    const track = document.createElement("span");
    track.className = "prob-track";
    const bar = document.createElement("span");
    bar.className = "prob-bar";
    bar.style.width = `${Math.max(p * 100, 0.4)}%`;
    track.append(bar);
    const val = document.createElement("span");
    val.className = "prob-val";
    val.textContent = p < 0.001 ? "<0.1%" : `${(p * 100).toFixed(1)}%`;
    row.append(label, track, val);
    bars.append(row);
  });
  $("#bars").replaceChildren(bars);
}

/* ── 28x28 preview ────────────────────────────────────────── */

const preview = $("#preview");
const pctx = preview.getContext("2d");
preview.width = SIZE;
preview.height = SIZE;
pctx.imageSmoothingEnabled = false;

function renderPreview(input) {
  const img = pctx.createImageData(SIZE, SIZE);
  for (let i = 0; i < input.length; i++) {
    const v = Math.round(Math.min(1, Math.max(0, input[i])) * 255);
    img.data[i * 4] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  pctx.putImageData(img, 0, 0);
}

/* ── Boot ─────────────────────────────────────────────────── */

(async () => {
  try {
    const t0 = performance.now();
    model = await DigitModel.load();
    const ms = performance.now() - t0;

    const m = model.metrics;
    $("#model-badge").textContent =
      `${model.architecture.join("→")} · ${(m.test_accuracy * 100).toFixed(2)}% test acc`;
    $("#model-note").textContent =
      `Loaded in ${ms.toFixed(0)} ms. ` +
      `${model.layers.reduce((a, l) => a + l.w.length + l.b.length, 0).toLocaleString("en-US")} ` +
      `parameters, int8-quantised to 147 KB. Accuracy is measured on the ` +
      `${m.test_size.toLocaleString("en-US")} held-out MNIST test images.`;

    $("#confusions").replaceChildren(...m.top_confusions.map((c) => {
      const li = document.createElement("li");
      li.textContent = `${c.truth} read as ${c.predicted} — ${c.count} times`;
      return li;
    }));

    // Reveal before sizing: getBoundingClientRect() on a hidden element
    // returns zero, which would back the canvas at 0x0 and make getImageData
    // throw on the first stroke.
    $("#loading").hidden = true;
    $("#app").hidden = false;
    sizeCanvas();
    bootstrapDemo();
  } catch (err) {
    $("#loading").textContent = `Could not load the model: ${err.message}`;
  }
})();

/* ── Automatic demo on arrival ────────────────────────────── */

/* A blank canvas and an em-dash tell a visitor nothing. Drawing a digit on
   arrival means the first thing on screen is the model working: a prediction,
   a confidence, and the 28x28 the network actually receives.
 *
 * Coordinates are fractions of the pad so this is independent of its size. */
const DEMO_STROKES = [
  [[0.28, 0.20], [0.66, 0.19], [0.46, 0.46], [0.70, 0.60], [0.64, 0.82], [0.28, 0.83]],
];

function bootstrapDemo() {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return;

  ctx.strokeStyle = "#ffffff";
  ctx.fillStyle = "#ffffff";
  for (const stroke of DEMO_STROKES) {
    ctx.beginPath();
    stroke.forEach(([fx, fy], i) => {
      const x = fx * rect.width, y = fy * rect.height;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  hasInk = true;
  predict();

  const panel = document.querySelector(".result-panel");
  const note = document.createElement("p");
  note.className = "demo-banner";
  note.style.margin = ".9rem 0 0";
  note.innerHTML =
    '<span class="badge badge-privacy">Live demo</span>' +
    "<span>This digit was drawn for you on load. Hit Clear and draw your own.</span>";
  panel.append(note);
}

let resizeTimer;
window.addEventListener("resize", () => {
  // Resizing clears the canvas, so debounce it rather than wiping the user's
  // drawing on every intermediate resize event.
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(sizeCanvas, 200);
});
