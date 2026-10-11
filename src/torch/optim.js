// torch.optim: optimizers and learning-rate schedulers. Updates happen in
// place on parameter storage, outside the autograd graph.

class Optimizer {
  constructor(params, defaults) {
    this.params = [...params];
    if (!this.params.length) throw new Error('optimizer got an empty parameter list');
    this.lr = defaults.lr;
    this.state = new Map();
  }
  zeroGrad() { for (const p of this.params) p.grad = null; }
}

export class SGD extends Optimizer {
  constructor(params, { lr = 0.01, momentum = 0, weightDecay = 0, nesterov = false } = {}) {
    super(params, { lr });
    Object.assign(this, { momentum, weightDecay, nesterov });
  }
  step() {
    for (const p of this.params) {
      if (!p.grad) continue;
      const g = p.grad.data;
      const w = p.data;
      let buf = this.state.get(p);
      if (this.momentum && !buf) { buf = new Float32Array(w.length); this.state.set(p, buf); }
      for (let i = 0; i < w.length; i++) {
        let d = g[i] + this.weightDecay * w[i];
        if (buf) {
          buf[i] = this.momentum * buf[i] + d;
          d = this.nesterov ? d + this.momentum * buf[i] : buf[i];
        }
        w[i] -= this.lr * d;
      }
    }
  }
}

export class Adam extends Optimizer {
  constructor(params, { lr = 1e-3, betas = [0.9, 0.999], eps = 1e-8, weightDecay = 0, decoupled = false } = {}) {
    super(params, { lr });
    Object.assign(this, { betas, eps, weightDecay, decoupled, t: 0 });
  }
  step() {
    this.t++;
    const [b1, b2] = this.betas;
    const c1 = 1 - b1 ** this.t;
    const c2 = 1 - b2 ** this.t;
    for (const p of this.params) {
      if (!p.grad) continue;
      let s = this.state.get(p);
      if (!s) { s = { m: new Float32Array(p.size), v: new Float32Array(p.size) }; this.state.set(p, s); }
      const g = p.grad.data;
      const w = p.data;
      for (let i = 0; i < w.length; i++) {
        let gi = g[i];
        if (this.weightDecay && !this.decoupled) gi += this.weightDecay * w[i];
        s.m[i] = b1 * s.m[i] + (1 - b1) * gi;
        s.v[i] = b2 * s.v[i] + (1 - b2) * gi * gi;
        if (this.weightDecay && this.decoupled) w[i] -= this.lr * this.weightDecay * w[i];
        w[i] -= (this.lr * (s.m[i] / c1)) / (Math.sqrt(s.v[i] / c2) + this.eps);
      }
    }
  }
}

export class AdamW extends Adam {
  constructor(params, opts = {}) { super(params, { weightDecay: 0.01, ...opts, decoupled: true }); }
}

export class RMSprop extends Optimizer {
  constructor(params, { lr = 0.01, alpha = 0.99, eps = 1e-8 } = {}) {
    super(params, { lr });
    Object.assign(this, { alpha, eps });
  }
  step() {
    for (const p of this.params) {
      if (!p.grad) continue;
      let v = this.state.get(p);
      if (!v) { v = new Float32Array(p.size); this.state.set(p, v); }
      const g = p.grad.data;
      for (let i = 0; i < g.length; i++) {
        v[i] = this.alpha * v[i] + (1 - this.alpha) * g[i] * g[i];
        p.data[i] -= (this.lr * g[i]) / (Math.sqrt(v[i]) + this.eps);
      }
    }
  }
}

export const lrScheduler = {
  StepLR: class {
    constructor(opt, { stepSize, gamma = 0.1 }) { Object.assign(this, { opt, stepSize, gamma, epoch: 0 }); }
    step() { if (++this.epoch % this.stepSize === 0) this.opt.lr *= this.gamma; }
  },
  CosineAnnealingLR: class {
    constructor(opt, { tMax, etaMin = 0 }) { Object.assign(this, { opt, tMax, etaMin, base: opt.lr, epoch: 0 }); }
    step() {
      this.epoch++;
      this.opt.lr = this.etaMin + ((this.base - this.etaMin) * (1 + Math.cos((Math.PI * this.epoch) / this.tMax))) / 2;
    }
  },
};
