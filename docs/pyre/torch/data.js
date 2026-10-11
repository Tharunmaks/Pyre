// torch.utils.data: datasets and a batching loader.
import { Tensor, stack, randperm } from './tensor.js';

export class TensorDataset {
  constructor(...tensors) {
    const n = tensors[0].shape[0];
    if (tensors.some((t) => t.shape[0] !== n)) throw new Error('all tensors need the same first dimension');
    this.tensors = tensors;
    this.length = n;
  }
  get(i) { return this.tensors.map((t) => t.get(i)); }
}

export class DataLoader {
  constructor(dataset, { batchSize = 32, shuffle = false, dropLast = false } = {}) {
    Object.assign(this, { dataset, batchSize, shuffle, dropLast });
  }
  get length() {
    const n = this.dataset.length / this.batchSize;
    return this.dropLast ? Math.floor(n) : Math.ceil(n);
  }
  *[Symbol.iterator]() {
    const n = this.dataset.length;
    const order = this.shuffle ? Array.from(randperm(n).data) : Array.from({ length: n }, (_, i) => i);
    for (let s = 0; s < n; s += this.batchSize) {
      const idx = order.slice(s, s + this.batchSize);
      if (this.dropLast && idx.length < this.batchSize) break;
      if (this.dataset.tensors) {
        const t = new Tensor(Float32Array.from(idx), [idx.length]);
        yield this.dataset.tensors.map((x) => x.indexSelect(t));
      } else {
        const items = idx.map((i) => this.dataset.get(i));
        yield items[0].map((_, k) => stack(items.map((it) => it[k])));
      }
    }
  }
}

/** Two interleaved spirals: a classic non-linear classification toy dataset. */
export function spirals(n = 200, { noise = 0.15, classes = 2 } = {}, rand = Math.random) {
  const xs = [];
  const ys = [];
  for (let c = 0; c < classes; c++) {
    for (let i = 0; i < n; i++) {
      const r = i / n;
      const t = (c * 2 * Math.PI) / classes + r * 4 * Math.PI;
      xs.push([r * Math.cos(t) + (rand() - 0.5) * noise, r * Math.sin(t) + (rand() - 0.5) * noise]);
      ys.push(c);
    }
  }
  return { x: xs, y: ys };
}
