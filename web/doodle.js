// Quick, Draw! CNN forward pass in plain JS (same architecture as model.py):
// 3x [conv3x3 same -> relu -> maxpool2], flatten, linear 1152->256, relu, linear 256->100.

export function loadNet(manifest, buffer) {
  const all = new Float32Array(buffer);
  const get = (k) => all.subarray(manifest[k].offset, manifest[k].offset + manifest[k].length);
  const convs = [0, 3, 6].map((i) => ({ w: get(`${i}.weight`), b: get(`${i}.bias`), out: manifest[`${i}.weight`].shape[0] }));
  const fc1 = { w: get("10.weight"), b: get("10.bias"), shape: manifest["10.weight"].shape };
  const fc2 = { w: get("12.weight"), b: get("12.bias"), shape: manifest["12.weight"].shape };

  return function predict(pixels /* Float32Array(784), 0..1 */) {
    let x = pixels, c = 1, size = 28;
    for (const conv of convs) {
      x = maxpool(relu(conv2d(x, c, size, conv.w, conv.b, conv.out)), conv.out, size);
      c = conv.out;
      size = Math.floor(size / 2);
    }
    x = relu(linear(x, fc1.w, fc1.b, fc1.shape[0], fc1.shape[1]));
    return softmax(linear(x, fc2.w, fc2.b, fc2.shape[0], fc2.shape[1]));
  };
}

function conv2d(x, cin, n, w, b, cout) {
  const y = new Float32Array(cout * n * n);
  for (let o = 0; o < cout; o++) {
    const yo = o * n * n;
    y.fill(b[o], yo, yo + n * n);
    for (let i = 0; i < cin; i++) {
      const xi = i * n * n, wk = (o * cin + i) * 9;
      for (let ky = 0; ky < 3; ky++) {
        for (let kx = 0; kx < 3; kx++) {
          const wv = w[wk + ky * 3 + kx];
          for (let r = 0; r < n; r++) {
            const sr = r + ky - 1;
            if (sr < 0 || sr >= n) continue;
            for (let col = 0; col < n; col++) {
              const sc = col + kx - 1;
              if (sc < 0 || sc >= n) continue;
              y[yo + r * n + col] += wv * x[xi + sr * n + sc];
            }
          }
        }
      }
    }
  }
  return y;
}

function relu(x) {
  for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0;
  return x;
}

function maxpool(x, c, n) {
  const m = Math.floor(n / 2), y = new Float32Array(c * m * m);
  for (let k = 0; k < c; k++)
    for (let r = 0; r < m; r++)
      for (let col = 0; col < m; col++) {
        const base = k * n * n + 2 * r * n + 2 * col;
        y[k * m * m + r * m + col] = Math.max(x[base], x[base + 1], x[base + n], x[base + n + 1]);
      }
  return y;
}

function linear(x, w, b, out, inp) {
  const y = new Float32Array(out);
  for (let o = 0; o < out; o++) {
    let s = b[o];
    for (let i = 0; i < inp; i++) s += w[o * inp + i] * x[i];
    y[o] = s;
  }
  return y;
}

function softmax(x) {
  const max = Math.max(...x);
  let sum = 0;
  const y = x.map((v) => { const e = Math.exp(v - max); sum += e; return e; });
  return y.map((v) => v / sum);
}
