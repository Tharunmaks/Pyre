// torch.nn: modules, layers and losses.
import { Tensor, randn, zeros, ones, random, noGrad, tensor } from './tensor.js';

/**
 * Base class. Instances are callable like PyTorch modules: model(x) runs
 * forward(x). Parameters and sub-modules are discovered from properties.
 */
export class Module {
  constructor() {
    const self = (...args) => self.forward(...args);
    Object.setPrototypeOf(self, new.target.prototype);
    self.training = true;
    return self;
  }

  forward() { throw new Error(`${this.constructor.name}.forward() is not implemented`); }

  namedParameters(prefix = '') {
    const out = [];
    for (const [k, v] of Object.entries(this)) {
      if (v instanceof Tensor && v.requiresGrad) out.push([prefix + k, v]);
      else if (v instanceof Module) out.push(...v.namedParameters(`${prefix}${k}.`));
      else if (Array.isArray(v)) v.forEach((m, i) => { if (m instanceof Module) out.push(...m.namedParameters(`${prefix}${k}.${i}.`)); });
    }
    return out;
  }
  parameters() { return this.namedParameters().map(([, p]) => p); }
  modules() {
    const out = [this];
    for (const v of Object.values(this)) {
      if (v instanceof Module) out.push(...v.modules());
      else if (Array.isArray(v)) v.forEach((m) => { if (m instanceof Module) out.push(...m.modules()); });
    }
    return out;
  }
  train(on = true) { for (const m of this.modules()) m.training = on; return this; }
  eval() { return this.train(false); }
  zeroGrad() { for (const p of this.parameters()) p.grad = null; }
  numParameters() { return this.parameters().reduce((s, p) => s + p.size, 0); }

  stateDict() { return Object.fromEntries(this.namedParameters().map(([k, p]) => [k, { shape: p.shape, data: Array.from(p.data) }])); }
  loadStateDict(state) {
    for (const [k, p] of this.namedParameters()) {
      const s = state[k];
      if (!s) throw new Error(`missing parameter ${k}`);
      if (s.data.length !== p.size) throw new Error(`parameter ${k} has ${s.data.length} values, expected ${p.size}`);
      p.data.set(s.data);
    }
    return this;
  }

  toString() {
    const kids = Object.entries(this).filter(([, v]) => v instanceof Module || (Array.isArray(v) && v.some((m) => m instanceof Module)));
    if (!kids.length) return `${this.constructor.name}(${this.extraRepr ? this.extraRepr() : ''})`;
    const lines = [];
    for (const [k, v] of kids) {
      if (Array.isArray(v)) v.forEach((m, i) => lines.push(`  (${i}): ${String(m).replace(/\n/g, '\n  ')}`));
      else lines.push(`  (${k}): ${String(v).replace(/\n/g, '\n  ')}`);
    }
    return `${this.constructor.name}(\n${lines.join('\n')}\n)`;
  }
  [Symbol.for('nodejs.util.inspect.custom')]() { return this.toString(); }
}

function uniform(shape, bound) {
  const t = zeros(shape);
  for (let i = 0; i < t.size; i++) t.data[i] = (random() * 2 - 1) * bound;
  return t.requiresGrad_();
}

export class Linear extends Module {
  constructor(inFeatures, outFeatures, { bias = true } = {}) {
    super();
    const bound = 1 / Math.sqrt(inFeatures);
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.weight = uniform([outFeatures, inFeatures], bound);
    this.bias = bias ? uniform([outFeatures], bound) : null;
  }
  forward(x) {
    const y = x.matmul(this.weight.T);
    return this.bias ? y.add(this.bias) : y;
  }
  extraRepr() { return `in_features=${this.inFeatures}, out_features=${this.outFeatures}, bias=${!!this.bias}`; }
}

export class Embedding extends Module {
  constructor(num, dim) {
    super();
    this.num = num;
    this.dimension = dim;
    this.weight = randn(num, dim).requiresGrad_();
  }
  forward(indices) {
    return this.weight.indexSelect(indices.reshape([indices.size])).reshape([...indices.shape, this.dimension]);
  }
  extraRepr() { return `${this.num}, ${this.dimension}`; }
}

const named = (cls, name) => Object.defineProperty(cls, 'name', { value: name });
const activation = (name, fn) => named(class extends Module { forward(x) { return fn(x); } }, name);
export const ReLU = activation('ReLU', (x) => x.relu());
export const Tanh = activation('Tanh', (x) => x.tanh());
export const Sigmoid = activation('Sigmoid', (x) => x.sigmoid());
export const GELU = activation('GELU', (x) => x.gelu());
export class LeakyReLU extends Module {
  constructor(slope = 0.01) { super(); this.slope = slope; }
  forward(x) { return x.leakyRelu(this.slope); }
}
export class Softmax extends Module {
  constructor(dim = -1) { super(); this.axis = dim; }
  forward(x) { return x.softmax(this.axis); }
}
export class Flatten extends Module { forward(x) { return x.flatten(1); } }
export class Identity extends Module { forward(x) { return x; } }

export class Dropout extends Module {
  constructor(p = 0.5) { super(); this.p = p; }
  forward(x) {
    if (!this.training || this.p === 0) return x;
    const keep = 1 - this.p;
    const mask = new Tensor(Float32Array.from({ length: x.size }, () => (random() < keep ? 1 / keep : 0)), x.shape);
    return x.mul(mask);
  }
}

export class LayerNorm extends Module {
  constructor(features, { eps = 1e-5 } = {}) {
    super();
    this.eps = eps;
    this.weight = ones(features).requiresGrad_();
    this.bias = zeros(features).requiresGrad_();
  }
  forward(x) {
    const mean = x.mean(-1, true);
    const v = x.sub(mean).square().mean(-1, true);
    return x.sub(mean).div(v.add(this.eps).sqrt()).mul(this.weight).add(this.bias);
  }
}

export class Sequential extends Module {
  constructor(...layers) {
    super();
    this.layers = layers.flat();
  }
  forward(x) { return this.layers.reduce((h, l) => l(h), x); }
}

// ---------- losses ----------
export function mseLoss(input, target, reduction = 'mean') {
  const d = input.sub(target).square();
  return reduction === 'sum' ? d.sum() : reduction === 'none' ? d : d.mean();
}
export function l1Loss(input, target) { return input.sub(target).abs().mean(); }
/** logits [N, C], targets [N] of class indices. */
export function crossEntropy(logits, targets) {
  const lp = logits.logSoftmax(-1);
  return lp.gather(1, targets.reshape([targets.size, 1])).neg().mean();
}
export function nllLoss(logProbs, targets) { return logProbs.gather(1, targets.reshape([targets.size, 1])).neg().mean(); }
export function binaryCrossEntropy(probs, targets, eps = 1e-7) {
  const p = probs.clamp(eps, 1 - eps);
  return targets.mul(p.log()).add(targets.neg().add(1).mul(p.neg().add(1).log())).neg().mean();
}
export function bceWithLogits(logits, targets) {
  // log(1 + e^x) - t*x, written stably
  const relu = logits.relu();
  return relu.sub(logits.mul(targets)).add(logits.abs().neg().exp().add(1).log()).mean();
}

const lossModule = (fn, name) => named(class extends Module { forward(a, b) { return fn(a, b); } }, name);
export const MSELoss = lossModule(mseLoss, 'MSELoss');
export const L1Loss = lossModule(l1Loss, 'L1Loss');
export const CrossEntropyLoss = lossModule(crossEntropy, 'CrossEntropyLoss');
export const NLLLoss = lossModule(nllLoss, 'NLLLoss');
export const BCELoss = lossModule(binaryCrossEntropy, 'BCELoss');
export const BCEWithLogitsLoss = lossModule(bceWithLogits, 'BCEWithLogitsLoss');

export const init = {
  zeros_(t) { noGrad(() => t.data.fill(0)); return t; },
  ones_(t) { t.data.fill(1); return t; },
  normal_(t, mean = 0, std = 1) { const r = randn(t.shape); for (let i = 0; i < t.size; i++) t.data[i] = mean + std * r.data[i]; return t; },
  uniform_(t, a = 0, b = 1) { for (let i = 0; i < t.size; i++) t.data[i] = a + (b - a) * random(); return t; },
  xavierUniform_(t) { const [o, i] = t.shape; const b = Math.sqrt(6 / (i + o)); return init.uniform_(t, -b, b); },
  kaimingUniform_(t) { const b = Math.sqrt(6 / t.shape[1]); return init.uniform_(t, -b, b); },
};

export const functional = {
  relu: (x) => x.relu(), gelu: (x) => x.gelu(), tanh: (x) => x.tanh(), sigmoid: (x) => x.sigmoid(),
  softmax: (x, dim = -1) => x.softmax(dim), logSoftmax: (x, dim = -1) => x.logSoftmax(dim),
  mseLoss, l1Loss, crossEntropy, nllLoss, binaryCrossEntropy, bceWithLogits,
  oneHot(t, classes) {
    const out = zeros(t.size, classes);
    for (let i = 0; i < t.size; i++) out.data[i * classes + t.data[i]] = 1;
    return out;
  },
  linear: (x, w, b) => (b ? x.matmul(w.T).add(b) : x.matmul(w.T)),
};

export { tensor };
