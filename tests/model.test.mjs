/*
 * Does the JavaScript inference match the Python that produced the weights?
 *
 * A re-implemented forward pass is exactly the kind of thing that "looks
 * right" and is off by a transpose. These cases are real MNIST test images
 * with logits computed in NumPy from the same quantised bundle.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { DigitModel, normalise, SIZE } from "../web/js/model.js";

const here = dirname(fileURLToPath(import.meta.url));
const bundle = JSON.parse(readFileSync(join(here, "../web/model/weights.json"), "utf8"));
const cases = JSON.parse(readFileSync(join(here, "fixture.json"), "utf8"));

// Node has atob globally from v16; model.js relies on it.
const model = new DigitModel(bundle);

let fail = 0, pass = 0;
const check = (ok, label, detail = "") => {
  if (ok) { pass++; } else { fail++; console.log(`  [FAIL] ${label} ${detail}`); }
};

console.log("\nModel bundle\n");
console.log(`  architecture      ${bundle.architecture.join(" → ")}`);
console.log(`  parameters        ${model.layers.reduce((a, l) => a + l.w.length + l.b.length, 0).toLocaleString()}`);
console.log(`  reported test acc ${(bundle.metrics.test_accuracy * 100).toFixed(2)}% on ${bundle.metrics.test_size.toLocaleString()} held-out images`);

console.log("\nJavaScript forward pass vs NumPy reference\n");

let worstLogit = 0, worstProb = 0, agree = 0, correct = 0;
for (const c of cases) {
  const bytes = Buffer.from(c.image, "base64");
  const input = new Float32Array(SIZE * SIZE);
  for (let i = 0; i < input.length; i++) input[i] = bytes[i] / 255;

  const logits = model.forward(input);
  const out = model.predict(input);

  for (let i = 0; i < 10; i++) {
    worstLogit = Math.max(worstLogit, Math.abs(logits[i] - c.logits[i]));
    worstProb = Math.max(worstProb, Math.abs(out.probabilities[i] - c.probs[i]));
  }
  if (out.digit === c.predicted) agree++;
  if (out.digit === c.label) correct++;

  check(Math.abs(logits[0] - c.logits[0]) < 1e-3, "logit drift", `label ${c.label}`);
  // Probabilities must sum to 1 — a broken softmax often still argmaxes right.
  const total = out.probabilities.reduce((a, b) => a + b, 0);
  check(Math.abs(total - 1) < 1e-5, "probabilities sum to 1", `got ${total}`);
}

console.log(`  worst logit difference       ${worstLogit.toExponential(2)}`);
console.log(`  worst probability difference ${worstProb.toExponential(2)}`);
console.log(`  predictions agreeing with NumPy  ${agree}/${cases.length}`);
console.log(`  predictions matching the label   ${correct}/${cases.length}`);
check(worstLogit < 1e-3, "logits match NumPy to 1e-3", `worst ${worstLogit}`);
check(agree === cases.length, "every prediction agrees with NumPy");

console.log("\nPreprocessing\n");

// A blank canvas must be reported as empty, not classified as some digit.
const blank = normalise(new Float32Array(280 * 280), 280, 280);
check(blank.empty === true, "blank canvas reported empty");
console.log(`  [${blank.empty ? "ok  " : "FAIL"}] blank canvas is reported empty, not classified`);

// A small mark in the corner must be rescaled and recentred, not fed as-is.
const corner = new Float32Array(280 * 280);
for (let y = 10; y < 40; y++) for (let x = 10; x < 25; x++) corner[y * 280 + x] = 1;
const norm = normalise(corner, 280, 280);
let mass = 0, cx = 0, cy = 0;
for (let y = 0; y < SIZE; y++) {
  for (let x = 0; x < SIZE; x++) {
    const v = norm.input[y * SIZE + x];
    mass += v; cx += x * v; cy += y * v;
  }
}
cx /= mass; cy /= mass;
const centred = Math.abs(cx - SIZE / 2) < 1.5 && Math.abs(cy - SIZE / 2) < 1.5;
check(centred, "corner mark is recentred", `centre of mass (${cx.toFixed(1)}, ${cy.toFixed(1)})`);
console.log(`  [${centred ? "ok  " : "FAIL"}] a mark drawn in the corner lands at centre of mass ` +
            `(${cx.toFixed(1)}, ${cy.toFixed(1)}) — target (14, 14)`);

// Aspect ratio must survive: a tall thin stroke must not be stretched square.
const tall = new Float32Array(280 * 280);
for (let y = 40; y < 240; y++) for (let x = 130; x < 150; x++) tall[y * 280 + x] = 1;
const tn = normalise(tall, 280, 280);
let minX = SIZE, maxX = -1, minY = SIZE, maxY = -1;
for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
  if (tn.input[y * SIZE + x] > 0.1) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
}
const wRatio = (maxX - minX + 1) / (maxY - minY + 1);
const keptAspect = wRatio < 0.35;
check(keptAspect, "aspect ratio preserved", `w/h = ${wRatio.toFixed(2)}`);
console.log(`  [${keptAspect ? "ok  " : "FAIL"}] a tall thin stroke stays thin (w/h ${wRatio.toFixed(2)}, ` +
            `source 0.10) rather than being stretched square`);

// The scaled digit must fit the 20x20 box MNIST used.
const fits = (maxY - minY + 1) <= 20 && (maxX - minX + 1) <= 20;
check(fits, "fits the 20x20 box");
console.log(`  [${fits ? "ok  " : "FAIL"}] scaled digit fits MNIST's 20x20 box ` +
            `(${maxX - minX + 1}x${maxY - minY + 1})`);

console.log("\n" + "=".repeat(60));
console.log(fail ? `${fail} FAILED, ${pass} passed` : `All ${pass} checks passed.`);
process.exit(fail ? 1 : 0);
