/*
 * model.js — int8 MLP inference, plus the preprocessing that makes it work.
 *
 * The network is the easy half: three dense layers and a ReLU. The half that
 * decides whether the demo works at all is `normalise()`.
 *
 * MNIST images are not raw drawings. Every digit was size-normalised to fit a
 * 20x20 box and then placed in a 28x28 field centred by its *centre of mass*.
 * A model trained on that has never seen a digit drawn small in the corner of
 * a 280x280 canvas, so feeding it a naive downscale produces confident
 * nonsense. Reproducing the original preprocessing is the whole trick.
 */

const SIZE = 28;
const BOX = 20;   // MNIST fits the digit's bounding box inside 20x20

/** Decode base64 into an Int8Array. */
function decodeInt8(b64) {
  const bin = atob(b64);
  const out = new Int8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    const c = bin.charCodeAt(i);
    out[i] = c > 127 ? c - 256 : c;   // atob yields 0..255; reinterpret as signed
  }
  return out;
}

export class DigitModel {
  constructor(bundle) {
    this.architecture = bundle.architecture;
    this.metrics = bundle.metrics;
    // Dequantise once at load rather than per inference: 110k multiplies on
    // every keystroke of drawing would be wasteful, and Float32Array keeps the
    // matmul in a fast path.
    this.layers = bundle.layers.map((l) => {
      const q = decodeInt8(l.w);
      const w = new Float32Array(q.length);
      for (let i = 0; i < q.length; i++) w[i] = q[i] * l.scale;
      return { w, b: new Float32Array(l.b), rows: l.shape[0], cols: l.shape[1] };
    });
  }

  static async load(url = "model/weights.json") {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Could not load the model (HTTP ${res.status}).`);
    return new DigitModel(await res.json());
  }

  /** Forward pass over a 784-length Float32Array in [0, 1]. */
  forward(input) {
    let a = input;
    this.layers.forEach((layer, li) => {
      const { w, b, rows, cols } = layer;
      const out = new Float32Array(cols);
      out.set(b);
      // Row-major weights, so iterate rows outermost and stream across cols —
      // the memory access stays sequential.
      for (let r = 0; r < rows; r++) {
        const av = a[r];
        if (av === 0) continue;      // inputs are mostly zero; skip the row
        const base = r * cols;
        for (let c = 0; c < cols; c++) out[c] += av * w[base + c];
      }
      if (li < this.layers.length - 1) {
        for (let i = 0; i < out.length; i++) if (out[i] < 0) out[i] = 0;
      }
      a = out;
    });
    return a;
  }

  /** Class probabilities from logits, via a max-shifted softmax. */
  predict(input) {
    const logits = this.forward(input);
    // Subtracting the max before exp keeps exp() away from overflow; without
    // it a confident logit of ~800 becomes Infinity and every probability NaN.
    let max = -Infinity;
    for (const v of logits) if (v > max) max = v;
    let sum = 0;
    const probs = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      probs[i] = Math.exp(logits[i] - max);
      sum += probs[i];
    }
    for (let i = 0; i < probs.length; i++) probs[i] /= sum;

    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return { digit: best, confidence: probs[best], probabilities: Array.from(probs) };
  }
}

/**
 * Turn raw canvas ink into an MNIST-shaped 28x28 input.
 *
 * @param {Float32Array} pixels  ink intensity in [0,1], row-major, w x h
 * @returns {{input: Float32Array, preview: Float32Array, empty: boolean}}
 */
export function normalise(pixels, w, h) {
  // 1. Bounding box of the ink. A threshold rather than > 0 so that
  //    antialiasing fringes do not inflate the box.
  let minX = w, minY = h, maxX = -1, maxY = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (pixels[y * w + x] > 0.1) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) {
    return { input: new Float32Array(SIZE * SIZE), preview: new Float32Array(SIZE * SIZE), empty: true };
  }

  // 2. Scale the bounding box to fit a 20x20 area, preserving aspect ratio so
  //    a "1" stays narrow instead of being stretched into a "7"-ish blob.
  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;
  const scale = BOX / Math.max(bw, bh);
  const tw = Math.max(1, Math.round(bw * scale));
  const th = Math.max(1, Math.round(bh * scale));

  // Box-filter downsample: averaging over the source region preserves stroke
  // weight, where nearest-neighbour sampling drops thin strokes entirely.
  const scaled = new Float32Array(tw * th);
  for (let y = 0; y < th; y++) {
    const sy0 = minY + (y / th) * bh;
    const sy1 = minY + ((y + 1) / th) * bh;
    for (let x = 0; x < tw; x++) {
      const sx0 = minX + (x / tw) * bw;
      const sx1 = minX + ((x + 1) / tw) * bw;
      let sum = 0, n = 0;
      for (let sy = Math.floor(sy0); sy < Math.ceil(sy1); sy++) {
        for (let sx = Math.floor(sx0); sx < Math.ceil(sx1); sx++) {
          if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
          sum += pixels[sy * w + sx];
          n++;
        }
      }
      scaled[y * tw + x] = n ? sum / n : 0;
    }
  }

  // 3. Centre of mass of the scaled digit.
  let mass = 0, cx = 0, cy = 0;
  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) {
      const v = scaled[y * tw + x];
      mass += v; cx += x * v; cy += y * v;
    }
  }
  cx = mass ? cx / mass : tw / 2;
  cy = mass ? cy / mass : th / 2;

  // 4. Place it so its centre of mass lands at the centre of the 28x28 field —
  //    which is what MNIST did, and is not the same as centring the bounding box.
  const offX = Math.round(SIZE / 2 - cx);
  const offY = Math.round(SIZE / 2 - cy);

  const input = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < th; y++) {
    const dy = y + offY;
    if (dy < 0 || dy >= SIZE) continue;
    for (let x = 0; x < tw; x++) {
      const dx = x + offX;
      if (dx < 0 || dx >= SIZE) continue;
      input[dy * SIZE + dx] = scaled[y * tw + x];
    }
  }
  return { input, preview: input, empty: false };
}

export { SIZE };
