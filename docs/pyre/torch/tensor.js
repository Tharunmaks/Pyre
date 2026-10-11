// Pyre Torch: a PyTorch-style tensor library for the CPU.
// Contiguous float32 storage, NumPy broadcasting and reverse-mode autograd.

let gradEnabled = true;

// ---------- seeded RNG (mulberry32) so experiments are reproducible ----------
let rngState = 0x2f6b4a1d;
export function manualSeed(seed) { rngState = (seed >>> 0) || 1; }
export function random() {
  rngState = (rngState + 0x6d2b79f5) | 0;
  let t = rngState;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
function gauss() {
  let u = 0;
  while (u === 0) u = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

export function noGrad(fn) {
  const prev = gradEnabled;
  gradEnabled = false;
  try { return fn(); } finally { gradEnabled = prev; }
}
export const isGradEnabled = () => gradEnabled;

// ---------- shape helpers ----------
const numel = (shape) => shape.reduce((a, b) => a * b, 1);
function stridesOf(shape) {
  const s = new Array(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) { s[i] = acc; acc *= shape[i]; }
  return s;
}
function sameShape(a, b) { return a.length === b.length && a.every((v, i) => v === b[i]); }
const fmtShape = (s) => `[${s.join(', ')}]`;

export function broadcastShapes(a, b) {
  const n = Math.max(a.length, b.length);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const x = a[a.length - n + i] ?? 1;
    const y = b[b.length - n + i] ?? 1;
    if (x !== y && x !== 1 && y !== 1) throw new Error(`shapes ${fmtShape(a)} and ${fmtShape(b)} cannot be broadcast together`);
    out[i] = x === 1 ? y : x;
  }
  return out;
}

/** Index map from an output shape to a (broadcast) input's flat indices. */
function broadcastIndex(inShape, outShape) {
  const n = outShape.length;
  const total = numel(outShape);
  const idx = new Int32Array(total);
  if (sameShape(inShape, outShape)) { for (let i = 0; i < total; i++) idx[i] = i; return idx; }
  const inStr = stridesOf(inShape);
  const pad = n - inShape.length;
  const eff = outShape.map((d, i) => (i < pad || inShape[i - pad] === 1 ? 0 : inStr[i - pad]));
  const counter = new Array(n).fill(0);
  let off = 0;
  for (let i = 0; i < total; i++) {
    idx[i] = off;
    for (let d = n - 1; d >= 0; d--) {
      counter[d]++;
      off += eff[d];
      if (counter[d] < outShape[d]) break;
      off -= eff[d] * counter[d];
      counter[d] = 0;
    }
  }
  return idx;
}

/** Sum a gradient of shape `from` down to a broadcast input of shape `to`. */
function sumToShape(grad, to) {
  if (sameShape(grad.shape, to)) return grad;
  const idx = broadcastIndex(to, grad.shape);
  const out = new Float32Array(numel(to));
  const g = grad.data;
  for (let i = 0; i < g.length; i++) out[idx[i]] += g[i];
  return new Tensor(out, to);
}

function flatten(nested, shape = [], depth = 0) {
  if (!Array.isArray(nested) && !ArrayBuffer.isView(nested)) return { data: [nested], shape: [] };
  const out = [];
  const walk = (v, d) => {
    if (Array.isArray(v) || ArrayBuffer.isView(v)) {
      if (shape[d] === undefined) shape[d] = v.length;
      else if (shape[d] !== v.length) throw new Error('ragged nested array: every row needs the same length');
      for (const x of v) walk(x, d + 1);
    } else {
      if (d !== shape.length) throw new Error('ragged nested array');
      if (typeof v === 'boolean') out.push(v ? 1 : 0);
      else if (typeof v !== 'number') throw new Error(`tensor values must be numbers, got ${typeof v}`);
      else out.push(v);
    }
  };
  walk(nested, depth);
  return { data: out, shape };
}

// ---------- Tensor ----------
export class Tensor {
  constructor(data, shape, { requiresGrad = false } = {}) {
    this.data = data instanceof Float32Array ? data : Float32Array.from(data);
    this.shape = shape ? [...shape] : [this.data.length];
    if (numel(this.shape) !== this.data.length) throw new Error(`shape ${fmtShape(this.shape)} does not match ${this.data.length} values`);
    this.requiresGrad = requiresGrad;
    this.grad = null;
    this._parents = null;
    this._backward = null;
    this._op = null;
  }

  get ndim() { return this.shape.length; }
  get size() { return this.data.length; }
  numel() { return this.data.length; }
  dim() { return this.shape.length; }

  item() {
    if (this.data.length !== 1) throw new Error(`item() needs a tensor with one value, this one has ${this.data.length}`);
    return this.data[0];
  }

  tolist() {
    if (!this.shape.length) return this.data[0];
    const build = (d, off) => {
      const out = [];
      const stride = numel(this.shape.slice(d + 1));
      for (let i = 0; i < this.shape[d]; i++) out.push(d === this.shape.length - 1 ? this.data[off + i] : build(d + 1, off + i * stride));
      return out;
    };
    return build(0, 0);
  }

  toString() {
    const fmt = (v) => {
      const s = Number.isInteger(v) ? `${v}.` : v.toFixed(4).replace(/0+$/, '');
      return s.padStart(8);
    };
    const rec = (d, off, ind) => {
      if (!this.shape.length) return fmt(this.data[0]).trim();
      const stride = numel(this.shape.slice(d + 1));
      const n = this.shape[d];
      const show = n > 8 ? [0, 1, 2, -1, n - 3, n - 2, n - 1] : [...Array(n).keys()];
      const parts = show.map((i) => (i === -1 ? '...' : d === this.shape.length - 1 ? fmt(this.data[off + i]) : rec(d + 1, off + i * stride, ind + ' ')));
      return d === this.shape.length - 1 ? `[${parts.join(',')}]` : `[${parts.join(`,\n${ind}`)}]`;
    };
    const extra = this.requiresGrad ? `, requires_grad=True${this._op ? `, grad_fn=<${this._op}>` : ''}` : '';
    return `tensor(${rec(0, 0, '       ')}${extra})`;
  }

  [Symbol.for('nodejs.util.inspect.custom')]() { return this.toString(); }

  // ----- autograd -----
  static _result(data, shape, parents, op, backward) {
    const out = new Tensor(data, shape);
    if (gradEnabled && parents.some((p) => p.requiresGrad)) {
      out.requiresGrad = true;
      out._parents = parents;
      out._op = op;
      out._backward = backward;
    }
    return out;
  }

  backward(gradient) {
    if (!this.requiresGrad) throw new Error('backward() on a tensor that does not require grad');
    if (!gradient && this.data.length !== 1) throw new Error('backward() without a gradient needs a scalar (call .sum() or .mean() first)');
    const order = [];
    const seen = new Set();
    const visit = (t) => {
      if (seen.has(t)) return;
      seen.add(t);
      if (t._parents) for (const p of t._parents) visit(p);
      order.push(t);
    };
    visit(this);
    const grads = new Map([[this, gradient ? gradient.data.slice() : new Float32Array([1])]]);
    for (let i = order.length - 1; i >= 0; i--) {
      const t = order[i];
      const g = grads.get(t);
      if (!g) continue;
      if (!t._parents) {
        if (t.grad) { const a = t.grad.data; for (let k = 0; k < a.length; k++) a[k] += g[k]; } else t.grad = new Tensor(g, t.shape);
        continue;
      }
      const parentGrads = t._backward(new Tensor(g, t.shape));
      t._parents.forEach((p, k) => {
        const pg = parentGrads[k];
        if (!pg || !p.requiresGrad) return;
        const reduced = sumToShape(pg, p.shape).data;
        const prev = grads.get(p);
        if (prev) for (let j = 0; j < prev.length; j++) prev[j] += reduced[j];
        else grads.set(p, Float32Array.from(reduced));
      });
      // retain grads only on leaves, like PyTorch
      grads.delete(t);
    }
  }

  detach() { return new Tensor(this.data.slice(), this.shape); }
  clone() { return Tensor._result(this.data.slice(), this.shape, [this], 'CloneBackward', (g) => [g]); }
  requiresGrad_(on = true) { this.requiresGrad = on; return this; }
  zeroGrad() { this.grad = null; return this; }

  // ----- elementwise -----
  _binary(other, op, f, back) {
    const b = other instanceof Tensor ? other : new Tensor([other], []);
    const shape = broadcastShapes(this.shape, b.shape);
    const ia = broadcastIndex(this.shape, shape);
    const ib = broadcastIndex(b.shape, shape);
    const n = numel(shape);
    const out = new Float32Array(n);
    const A = this.data;
    const B = b.data;
    for (let i = 0; i < n; i++) out[i] = f(A[ia[i]], B[ib[i]]);
    const a = this;
    return Tensor._result(out, shape, [a, b], op, (g) => back(g, a, b, ia, ib, out));
  }

  add(o) { return this._binary(o, 'AddBackward', (x, y) => x + y, (g) => [g, g]); }
  sub(o) { return this._binary(o, 'SubBackward', (x, y) => x - y, (g) => [g, g.neg()]); }
  mul(o) {
    return this._binary(o, 'MulBackward', (x, y) => x * y, (g, a, b, ia, ib) => {
      const ga = new Float32Array(g.size);
      const gb = new Float32Array(g.size);
      for (let i = 0; i < g.size; i++) { ga[i] = g.data[i] * b.data[ib[i]]; gb[i] = g.data[i] * a.data[ia[i]]; }
      return [new Tensor(ga, g.shape), new Tensor(gb, g.shape)];
    });
  }
  div(o) {
    return this._binary(o, 'DivBackward', (x, y) => x / y, (g, a, b, ia, ib) => {
      const ga = new Float32Array(g.size);
      const gb = new Float32Array(g.size);
      for (let i = 0; i < g.size; i++) {
        const y = b.data[ib[i]];
        ga[i] = g.data[i] / y;
        gb[i] = (-g.data[i] * a.data[ia[i]]) / (y * y);
      }
      return [new Tensor(ga, g.shape), new Tensor(gb, g.shape)];
    });
  }
  pow(o) {
    return this._binary(o, 'PowBackward', (x, y) => x ** y, (g, a, b, ia, ib, out) => {
      const ga = new Float32Array(g.size);
      const gb = new Float32Array(g.size);
      for (let i = 0; i < g.size; i++) {
        const x = a.data[ia[i]];
        const y = b.data[ib[i]];
        ga[i] = g.data[i] * y * x ** (y - 1);
        gb[i] = x > 0 ? g.data[i] * out[i] * Math.log(x) : 0;
      }
      return [new Tensor(ga, g.shape), new Tensor(gb, g.shape)];
    });
  }
  maximum(o) {
    return this._binary(o, 'MaximumBackward', Math.max, (g, a, b, ia, ib) => {
      const ga = new Float32Array(g.size);
      const gb = new Float32Array(g.size);
      for (let i = 0; i < g.size; i++) { if (a.data[ia[i]] >= b.data[ib[i]]) ga[i] = g.data[i]; else gb[i] = g.data[i]; }
      return [new Tensor(ga, g.shape), new Tensor(gb, g.shape)];
    });
  }
  minimum(o) { return this.neg().maximum(o instanceof Tensor ? o.neg() : -o).neg(); }

  _unary(op, f, df) {
    const n = this.size;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = f(this.data[i]);
    const a = this;
    return Tensor._result(out, this.shape, [a], op, (g) => {
      const ga = new Float32Array(n);
      for (let i = 0; i < n; i++) ga[i] = g.data[i] * df(a.data[i], out[i]);
      return [new Tensor(ga, a.shape)];
    });
  }

  neg() { return this._unary('NegBackward', (x) => -x, () => -1); }
  exp() { return this._unary('ExpBackward', Math.exp, (x, y) => y); }
  log() { return this._unary('LogBackward', Math.log, (x) => 1 / x); }
  sqrt() { return this._unary('SqrtBackward', Math.sqrt, (x, y) => 0.5 / y); }
  abs() { return this._unary('AbsBackward', Math.abs, (x) => Math.sign(x)); }
  sin() { return this._unary('SinBackward', Math.sin, Math.cos); }
  cos() { return this._unary('CosBackward', Math.cos, (x) => -Math.sin(x)); }
  tanh() { return this._unary('TanhBackward', Math.tanh, (x, y) => 1 - y * y); }
  sigmoid() { return this._unary('SigmoidBackward', (x) => 1 / (1 + Math.exp(-x)), (x, y) => y * (1 - y)); }
  relu() { return this._unary('ReluBackward', (x) => (x > 0 ? x : 0), (x) => (x > 0 ? 1 : 0)); }
  leakyRelu(slope = 0.01) { return this._unary('LeakyReluBackward', (x) => (x > 0 ? x : slope * x), (x) => (x > 0 ? 1 : slope)); }
  gelu() {
    const k = Math.sqrt(2 / Math.PI);
    return this._unary('GeluBackward', (x) => 0.5 * x * (1 + Math.tanh(k * (x + 0.044715 * x ** 3))), (x) => {
      const u = k * (x + 0.044715 * x ** 3);
      const t = Math.tanh(u);
      return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * k * (1 + 3 * 0.044715 * x * x);
    });
  }
  square() { return this.mul(this); }
  clamp(lo = -Infinity, hi = Infinity) { return this._unary('ClampBackward', (x) => Math.min(Math.max(x, lo), hi), (x) => (x >= lo && x <= hi ? 1 : 0)); }

  // ----- reductions -----
  _reduceDims(dim) {
    if (dim === undefined || dim === null) return [...this.shape.keys()];
    return (Array.isArray(dim) ? dim : [dim]).map((d) => {
      const r = d < 0 ? d + this.ndim : d;
      if (r < 0 || r >= this.ndim) throw new Error(`dim ${d} out of range for ${this.ndim}-d tensor`);
      return r;
    });
  }

  sum(dim, keepdim = false) {
    const dims = this._reduceDims(dim);
    const keepShape = this.shape.map((s, i) => (dims.includes(i) ? 1 : s));
    const outShape = keepdim ? keepShape : this.shape.filter((_, i) => !dims.includes(i));
    const idx = broadcastIndex(keepShape, this.shape);
    const out = new Float32Array(numel(keepShape));
    for (let i = 0; i < this.size; i++) out[idx[i]] += this.data[i];
    const a = this;
    return Tensor._result(out, outShape, [a], 'SumBackward', (g) => {
      const ga = new Float32Array(a.size);
      for (let i = 0; i < a.size; i++) ga[i] = g.data[idx[i]];
      return [new Tensor(ga, a.shape)];
    });
  }

  mean(dim, keepdim = false) {
    const dims = this._reduceDims(dim);
    const count = dims.reduce((s, d) => s * this.shape[d], 1);
    return this.sum(dim, keepdim).div(count);
  }

  variance(dim, keepdim = false, unbiased = true) {
    const dims = this._reduceDims(dim);
    const count = dims.reduce((s, d) => s * this.shape[d], 1);
    const centered = this.sub(this.mean(dim, true));
    return centered.square().sum(dim, keepdim).div(unbiased ? Math.max(1, count - 1) : count);
  }
  std(dim, keepdim = false, unbiased = true) { return this.variance(dim, keepdim, unbiased).sqrt(); }

  _extreme(dim, keepdim, better, op) {
    const d = dim === undefined ? null : this._reduceDims(dim)[0];
    const src = d === null ? this.reshape([this.size]) : this;
    const axis = d === null ? 0 : d;
    const outer = numel(src.shape.slice(0, axis));
    const n = src.shape[axis];
    const inner = numel(src.shape.slice(axis + 1));
    const values = new Float32Array(outer * inner);
    const index = new Int32Array(outer * inner);
    for (let o = 0; o < outer; o++) {
      for (let i = 0; i < inner; i++) {
        let best = src.data[o * n * inner + i];
        let bi = 0;
        for (let k = 1; k < n; k++) {
          const v = src.data[(o * n + k) * inner + i];
          if (better(v, best)) { best = v; bi = k; }
        }
        values[o * inner + i] = best;
        index[o * inner + i] = bi;
      }
    }
    const shape = d === null ? [] : keepdim ? src.shape.map((s, i) => (i === axis ? 1 : s)) : src.shape.filter((_, i) => i !== axis);
    const a = this;
    const vals = Tensor._result(values, shape, [a], op, (g) => {
      const ga = new Float32Array(a.size);
      for (let o = 0; o < outer; o++) for (let i = 0; i < inner; i++) ga[(o * n + index[o * inner + i]) * inner + i] = g.data[o * inner + i];
      return [new Tensor(ga, a.shape)];
    });
    return { values: vals, indices: new Tensor(Float32Array.from(index), shape) };
  }
  max(dim, keepdim = false) { const r = this._extreme(dim, keepdim, (a, b) => a > b, 'MaxBackward'); return dim === undefined ? r.values : r; }
  min(dim, keepdim = false) { const r = this._extreme(dim, keepdim, (a, b) => a < b, 'MinBackward'); return dim === undefined ? r.values : r; }
  argmax(dim, keepdim = false) { return this._extreme(dim, keepdim, (a, b) => a > b, 'ArgMax').indices; }
  argmin(dim, keepdim = false) { return this._extreme(dim, keepdim, (a, b) => a < b, 'ArgMin').indices; }

  logsumexp(dim = -1, keepdim = false) {
    const m = noGrad(() => this.max(dim, true).values);
    return this.sub(m).exp().sum(dim, true).log().add(m).reshape(keepdim ? this.shape.map((s, i) => (i === this._reduceDims(dim)[0] ? 1 : s)) : this.shape.filter((_, i) => i !== this._reduceDims(dim)[0]));
  }
  softmax(dim = -1) {
    const m = noGrad(() => this.max(dim, true).values);
    const e = this.sub(m).exp();
    return e.div(e.sum(dim, true));
  }
  logSoftmax(dim = -1) {
    const m = noGrad(() => this.max(dim, true).values);
    const shifted = this.sub(m);
    return shifted.sub(shifted.exp().sum(dim, true).log());
  }

  // ----- shape -----
  reshape(...shape) {
    let s = (Array.isArray(shape[0]) ? shape[0] : shape).slice();
    const neg = s.indexOf(-1);
    if (neg >= 0) s[neg] = this.size / numel(s.filter((_, i) => i !== neg));
    if (numel(s) !== this.size) throw new Error(`cannot reshape ${fmtShape(this.shape)} into ${fmtShape(s)}`);
    const a = this;
    return Tensor._result(this.data, s, [a], 'ViewBackward', (g) => [new Tensor(g.data, a.shape)]);
  }
  view(...s) { return this.reshape(...s); }
  flatten(start = 0) {
    const st = start < 0 ? start + this.ndim : start;
    return this.reshape([...this.shape.slice(0, st), numel(this.shape.slice(st))]);
  }
  unsqueeze(d) { const s = this.shape.slice(); s.splice(d < 0 ? d + this.ndim + 1 : d, 0, 1); return this.reshape(s); }
  squeeze(d) {
    if (d === undefined) return this.reshape(this.shape.filter((x) => x !== 1));
    const r = d < 0 ? d + this.ndim : d;
    return this.shape[r] === 1 ? this.reshape(this.shape.filter((_, i) => i !== r)) : this;
  }

  permute(...order) {
    const p = Array.isArray(order[0]) ? order[0] : order;
    if (p.length !== this.ndim) throw new Error('permute needs one index per dimension');
    const shape = p.map((i) => this.shape[i]);
    const inStr = stridesOf(this.shape);
    const src = new Int32Array(this.size);
    const counter = new Array(this.ndim).fill(0);
    for (let i = 0; i < this.size; i++) {
      let off = 0;
      for (let d = 0; d < this.ndim; d++) off += counter[d] * inStr[p[d]];
      src[i] = off;
      for (let d = this.ndim - 1; d >= 0; d--) { if (++counter[d] < shape[d]) break; counter[d] = 0; }
    }
    const out = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) out[i] = this.data[src[i]];
    const a = this;
    return Tensor._result(out, shape, [a], 'PermuteBackward', (g) => {
      const ga = new Float32Array(a.size);
      for (let i = 0; i < a.size; i++) ga[src[i]] = g.data[i];
      return [new Tensor(ga, a.shape)];
    });
  }
  transpose(d0 = 0, d1 = 1) {
    const p = [...this.shape.keys()];
    const a = d0 < 0 ? d0 + this.ndim : d0;
    const b = d1 < 0 ? d1 + this.ndim : d1;
    [p[a], p[b]] = [p[b], p[a]];
    return this.permute(p);
  }
  get T() { return this.ndim < 2 ? this : this.transpose(-2, -1); }
  t() { return this.T; }

  /** Rows/slices along a dimension: x.narrow(0, 2, 3) is x[2:5]. */
  narrow(dim, start, length) {
    const d = dim < 0 ? dim + this.ndim : dim;
    const outer = numel(this.shape.slice(0, d));
    const n = this.shape[d];
    const inner = numel(this.shape.slice(d + 1));
    if (start < 0 || start + length > n) throw new Error('narrow out of range');
    const shape = this.shape.slice();
    shape[d] = length;
    const out = new Float32Array(outer * length * inner);
    for (let o = 0; o < outer; o++) out.set(this.data.subarray((o * n + start) * inner, (o * n + start + length) * inner), o * length * inner);
    const a = this;
    return Tensor._result(out, shape, [a], 'NarrowBackward', (g) => {
      const ga = new Float32Array(a.size);
      for (let o = 0; o < outer; o++) ga.set(g.data.subarray(o * length * inner, (o + 1) * length * inner), (o * n + start) * inner);
      return [new Tensor(ga, a.shape)];
    });
  }
  slice(start, end) { return this.narrow(0, start, Math.min(end ?? this.shape[0], this.shape[0]) - start); }

  /** Select one index along dim 0 (x[i]). */
  get(i) { return this.narrow(0, i < 0 ? i + this.shape[0] : i, 1).squeeze(0); }

  /** Pick values along `dim` with an index tensor of the same rank (torch.gather). */
  gather(dim, index) {
    const d = dim < 0 ? dim + this.ndim : dim;
    const outer = numel(index.shape.slice(0, d));
    const m = index.shape[d];
    const inner = numel(index.shape.slice(d + 1));
    const n = this.shape[d];
    const out = new Float32Array(index.size);
    const src = new Int32Array(index.size);
    for (let o = 0; o < outer; o++) for (let k = 0; k < m; k++) for (let i = 0; i < inner; i++) {
      const j = (o * m + k) * inner + i;
      const c = index.data[j];
      if (c < 0 || c >= n || !Number.isInteger(c)) throw new Error(`gather index ${c} out of range`);
      src[j] = (o * n + c) * inner + i;
      out[j] = this.data[src[j]];
    }
    const a = this;
    return Tensor._result(out, index.shape, [a], 'GatherBackward', (g) => {
      const ga = new Float32Array(a.size);
      for (let j = 0; j < src.length; j++) ga[src[j]] += g.data[j];
      return [new Tensor(ga, a.shape)];
    });
  }

  /** Rows of a tensor by index along dim 0 (embedding lookup). */
  indexSelect(indices) {
    const rows = this.shape[0];
    const inner = this.size / rows;
    const n = indices.size;
    const out = new Float32Array(n * inner);
    for (let i = 0; i < n; i++) {
      const r = indices.data[i];
      if (!Number.isInteger(r) || r < 0 || r >= rows) throw new Error(`index ${r} out of range for ${rows} rows`);
      out.set(this.data.subarray(r * inner, (r + 1) * inner), i * inner);
    }
    const a = this;
    return Tensor._result(out, [n, ...this.shape.slice(1)], [a], 'IndexSelectBackward', (g) => {
      const ga = new Float32Array(a.size);
      for (let i = 0; i < n; i++) {
        const r = indices.data[i];
        for (let j = 0; j < inner; j++) ga[r * inner + j] += g.data[i * inner + j];
      }
      return [new Tensor(ga, a.shape)];
    });
  }

  // ----- linear algebra -----
  matmul(other) {
    const a = this.ndim === 1 ? this.reshape([1, this.size]) : this;
    const b = other.ndim === 1 ? other.reshape([other.size, 1]) : other;
    const [m, k] = a.shape.slice(-2);
    const [k2, n] = b.shape.slice(-2);
    if (k !== k2) throw new Error(`matmul shape mismatch: ${fmtShape(this.shape)} @ ${fmtShape(other.shape)}`);
    const batchA = a.shape.slice(0, -2);
    const batchB = b.shape.slice(0, -2);
    const batch = broadcastShapes(batchA, batchB);
    const nb = numel(batch);
    const ia = broadcastIndex(batchA, batch);
    const ib = broadcastIndex(batchB, batch);
    const out = new Float32Array(nb * m * n);
    const A = a.data;
    const B = b.data;
    for (let bi = 0; bi < nb; bi++) {
      const ao = ia[bi] * m * k;
      const bo = ib[bi] * k * n;
      const oo = bi * m * n;
      for (let i = 0; i < m; i++) {
        for (let p = 0; p < k; p++) {
          const av = A[ao + i * k + p];
          if (av === 0) continue;
          const brow = bo + p * n;
          const orow = oo + i * n;
          for (let j = 0; j < n; j++) out[orow + j] += av * B[brow + j];
        }
      }
    }
    let shape = [...batch, m, n];
    const res = Tensor._result(out, shape, [a, b], 'MmBackward', (g) => [g.matmul(b.transpose(-2, -1)), a.transpose(-2, -1).matmul(g)]);
    if (this.ndim === 1 && other.ndim === 1) return res.reshape([]);
    if (this.ndim === 1) { shape = [...batch, n]; return res.reshape(shape); }
    if (other.ndim === 1) { shape = [...batch, m]; return res.reshape(shape); }
    return res;
  }
  mm(o) { return this.matmul(o); }
  dot(o) { return this.mul(o).sum(); }

  // ----- comparisons (no grad) -----
  _cmp(o, f) {
    const b = o instanceof Tensor ? o : new Tensor([o], []);
    const shape = broadcastShapes(this.shape, b.shape);
    const ia = broadcastIndex(this.shape, shape);
    const ib = broadcastIndex(b.shape, shape);
    const out = new Float32Array(numel(shape));
    for (let i = 0; i < out.length; i++) out[i] = f(this.data[ia[i]], b.data[ib[i]]) ? 1 : 0;
    return new Tensor(out, shape);
  }
  eq(o) { return this._cmp(o, (x, y) => x === y); }
  gt(o) { return this._cmp(o, (x, y) => x > y); }
  lt(o) { return this._cmp(o, (x, y) => x < y); }
  ge(o) { return this._cmp(o, (x, y) => x >= y); }
  le(o) { return this._cmp(o, (x, y) => x <= y); }

  /** Apply a plain JS function elementwise (no gradient). */
  map(f) { return new Tensor(this.data.map(f), this.shape); }
}

// ---------- constructors ----------
export function tensor(values, { requiresGrad = false } = {}) {
  if (values instanceof Tensor) return new Tensor(values.data.slice(), values.shape, { requiresGrad });
  const { data, shape } = flatten(values);
  return new Tensor(data, shape, { requiresGrad });
}
const shapeArg = (args) => (Array.isArray(args[0]) ? args[0] : args.filter((a) => typeof a === 'number'));
const optsArg = (args) => (args.length && typeof args[args.length - 1] === 'object' && !Array.isArray(args[args.length - 1]) ? args[args.length - 1] : {});
export function full(shape, value, opts = {}) { return new Tensor(new Float32Array(numel(shape)).fill(value), shape, opts); }
export function zeros(...args) { return full(shapeArg(args), 0, optsArg(args)); }
export function ones(...args) { return full(shapeArg(args), 1, optsArg(args)); }
export function rand(...args) { const s = shapeArg(args); return new Tensor(Float32Array.from({ length: numel(s) }, random), s, optsArg(args)); }
export function randn(...args) { const s = shapeArg(args); return new Tensor(Float32Array.from({ length: numel(s) }, gauss), s, optsArg(args)); }
export function randint(low, high, shape) { return new Tensor(Float32Array.from({ length: numel(shape) }, () => low + Math.floor(random() * (high - low))), shape); }
export function arange(start, end, step = 1) {
  if (end === undefined) { end = start; start = 0; }
  const n = Math.max(0, Math.ceil((end - start) / step));
  return new Tensor(Float32Array.from({ length: n }, (_, i) => start + i * step), [n]);
}
export function linspace(start, end, steps) {
  return new Tensor(Float32Array.from({ length: steps }, (_, i) => (steps === 1 ? start : start + ((end - start) * i) / (steps - 1))), [steps]);
}
export function eye(n) { const t = zeros(n, n); for (let i = 0; i < n; i++) t.data[i * n + i] = 1; return t; }
export const zerosLike = (t) => zeros(t.shape);
export const onesLike = (t) => ones(t.shape);
export function randperm(n) {
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return new Tensor(Float32Array.from(a), [n]);
}

export function cat(tensors, dim = 0) {
  const d = dim < 0 ? dim + tensors[0].ndim : dim;
  const shape = tensors[0].shape.slice();
  shape[d] = tensors.reduce((s, t) => s + t.shape[d], 0);
  const outer = numel(shape.slice(0, d));
  const inner = numel(shape.slice(d + 1));
  const out = new Float32Array(numel(shape));
  let pos = 0;
  const offsets = [];
  for (const t of tensors) {
    const n = t.shape[d];
    offsets.push(pos);
    for (let o = 0; o < outer; o++) out.set(t.data.subarray(o * n * inner, (o + 1) * n * inner), (o * shape[d] + pos) * inner);
    pos += n;
  }
  return Tensor._result(out, shape, tensors, 'CatBackward', (g) => tensors.map((t, k) => {
    const n = t.shape[d];
    const ga = new Float32Array(t.size);
    for (let o = 0; o < outer; o++) ga.set(g.data.subarray((o * shape[d] + offsets[k]) * inner, (o * shape[d] + offsets[k] + n) * inner), o * n * inner);
    return new Tensor(ga, t.shape);
  }));
}
export function stack(tensors, dim = 0) { return cat(tensors.map((t) => t.unsqueeze(dim)), dim); }
export function where(cond, a, b) {
  const A = a instanceof Tensor ? a : full(cond.shape, a);
  const B = b instanceof Tensor ? b : full(cond.shape, b);
  return A.mul(cond).add(B.mul(cond.map((x) => 1 - x)));
}
