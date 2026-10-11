// CPU runtime for code produced by the JS backend. Vectors are plain arrays,
// matrices are column-major arrays (m[col * n + row]), scalars are numbers.

const isA = Array.isArray;
const map1 = (f) => (x) => (isA(x) ? x.map(f) : f(x));
const map2 = (f) => (a, b) => {
  if (isA(a)) return isA(b) ? a.map((v, i) => f(v, b[i])) : a.map((v) => f(v, b));
  return isA(b) ? b.map((v) => f(a, v)) : f(a, b);
};
const map3 = (f) => (a, b, c) => {
  const n = isA(a) ? a.length : isA(b) ? b.length : isA(c) ? c.length : 0;
  if (!n) return f(a, b, c);
  const g = (x, i) => (isA(x) ? x[i] : x);
  return Array.from({ length: n }, (_, i) => f(g(a, i), g(b, i), g(c, i)));
};

const clamp1 = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
const fract1 = (x) => x - Math.floor(x);
const mod1 = (x, y) => x - y * Math.floor(x / y);
const smooth1 = (e0, e1, x) => {
  const t = clamp1((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
};
const dot = (a, b) => (isA(a) ? a.reduce((s, v, i) => s + v * b[i], 0) : a * b);
const length = (v) => (isA(v) ? Math.sqrt(dot(v, v)) : Math.abs(v));
const add = map2((a, b) => a + b);
const sub = map2((a, b) => a - b);
const mul = map2((a, b) => a * b);

function flatten(args) {
  const out = [];
  for (const a of args) {
    if (isA(a)) out.push(...a);
    else out.push(a);
  }
  return out;
}

export const runtime = Object.freeze({
  PI: Math.PI,
  TAU: Math.PI * 2,
  E: Math.E,
  add,
  sub,
  mul,
  div: map2((a, b) => a / b),
  mod: map2(mod1),
  neg: map1((x) => -x),
  idiv: (a, b) => (b === 0 ? 0 : Math.trunc(a / b) | 0),
  imod: map2((a, b) => (b === 0 ? 0 : (a % b) | 0)),
  toInt: (x) => (typeof x === 'boolean' ? (x ? 1 : 0) : Math.trunc(x) | 0),
  sin: map1(Math.sin), cos: map1(Math.cos), tan: map1(Math.tan),
  asin: map1(Math.asin), acos: map1(Math.acos), atan: map1(Math.atan),
  sinh: map1(Math.sinh), cosh: map1(Math.cosh), tanh: map1(Math.tanh),
  exp: map1(Math.exp), log: map1(Math.log), exp2: map1((x) => 2 ** x), log2: map1(Math.log2),
  sqrt: map1(Math.sqrt), inversesqrt: map1((x) => 1 / Math.sqrt(x)),
  floor: map1(Math.floor), ceil: map1(Math.ceil), fract: map1(fract1),
  round: map1(Math.round), trunc: map1(Math.trunc),
  radians: map1((x) => (x * Math.PI) / 180), degrees: map1((x) => (x * 180) / Math.PI),
  saturate: map1((x) => clamp1(x, 0, 1)),
  abs: map1(Math.abs), sign: map1(Math.sign),
  min: map2(Math.min), max: map2(Math.max), clamp: map3(clamp1),
  pow: map2(Math.pow), atan2: map2(Math.atan2),
  step: map2((e, x) => (x < e ? 0 : 1)),
  smoothstep: map3(smooth1),
  mix: map3((a, b, t) => a + (b - a) * t),
  length,
  distance: (a, b) => length(sub(a, b)),
  dot,
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  normalize: (v) => {
    if (!isA(v)) return Math.sign(v);
    const l = length(v);
    return v.map((x) => x / l);
  },
  reflect: (i, n) => sub(i, mul(n, 2 * dot(n, i))),
  refract: (i, n, eta) => {
    const d = dot(n, i);
    const k = 1 - eta * eta * (1 - d * d);
    if (k < 0) return isA(i) ? i.map(() => 0) : 0;
    return sub(mul(i, eta), mul(n, eta * d + Math.sqrt(k)));
  },
  vec(n, ...args) {
    if (args.length === 1 && !isA(args[0])) return Array(n).fill(args[0]);
    return flatten(args);
  },
  mat(n, ...args) {
    if (args.length === 1 && !isA(args[0])) {
      const m = Array(n * n).fill(0);
      for (let i = 0; i < n; i++) m[i * n + i] = args[0];
      return m;
    }
    return flatten(args);
  },
  mmul(a, b, ta, tb) {
    const matA = ta[0] === 'm';
    const matB = tb[0] === 'm';
    if (!isA(a)) return b.map((x) => a * x);
    if (!isA(b)) return a.map((x) => x * b);
    if (matA && matB) {
      const n = Number(ta[3]);
      const r = Array(n * n).fill(0);
      for (let c = 0; c < n; c++) for (let row = 0; row < n; row++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += a[k * n + row] * b[c * n + k];
        r[c * n + row] = s;
      }
      return r;
    }
    if (matA) {
      const n = Number(ta[3]);
      return Array.from({ length: n }, (_, row) => {
        let s = 0;
        for (let c = 0; c < n; c++) s += a[c * n + row] * b[c];
        return s;
      });
    }
    const n = Number(tb[3]);
    return Array.from({ length: n }, (_, c) => {
      let s = 0;
      for (let row = 0; row < n; row++) s += a[row] * b[c * n + row];
      return s;
    });
  },
  sw: (v, idx) => idx.map((i) => v[i]),
  at: (v, i) => (i >= 0 && i < v.length ? v[i] : 0),
  set: (v, i, x) => { if (i >= 0 && i < v.length) v[i] = x; },
  col: (m, i, n) => (i >= 0 && i < n ? m.slice(i * n, i * n + n) : Array(n).fill(0)),
  ld: (buf, i) => (buf && i >= 0 && i < buf.length ? buf[i] : 0),
  st: (buf, i, v) => { if (buf && i >= 0 && i < buf.length) buf[i] = v; },
  copy: (x) => (isA(x) ? x.slice() : x),
  zero(t) {
    if (t === 'bool') return false;
    if (t === 'int' || t === 'float') return 0;
    const n = Number(t[3]);
    return Array(t[0] === 'm' ? n * n : n).fill(0);
  },
  limit() {
    throw new Error('Pyre: loop budget exceeded (possible infinite loop)');
  },
});
