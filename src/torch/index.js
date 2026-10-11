// Pyre Torch — PyTorch-style deep learning on the CPU, in plain JavaScript.
//
//   import torch from 'pyre-lang/torch';
//   const model = new torch.nn.Sequential(new torch.nn.Linear(2, 16), new torch.nn.ReLU(), new torch.nn.Linear(16, 1));
//   const opt = new torch.optim.Adam(model.parameters(), { lr: 0.01 });
//   const loss = torch.nn.functional.mseLoss(model(x), y);
//   opt.zeroGrad(); loss.backward(); opt.step();
//
// Pyre @kernel programs plug in as tensor ops with torch.kernel(source).
import * as T from './tensor.js';
import * as nn from './nn.js';
import * as optim from './optim.js';
import * as data from './data.js';
import { createProgram } from '../index.js';

/**
 * Wrap a Pyre @kernel as a tensor op. Buffers are bound by name to tensors;
 * results come back as new tensors (inputs are not modified).
 *   const scale = torch.kernel(src);
 *   const { out } = scale({ x, out: torch.zeros(4) }, { uniforms: { k: 2 } });
 */
export function kernel(source, { loopLimit } = {}) {
  return (buffers, { uniforms = {}, count } = {}) => {
    const plain = {};
    for (const [k, v] of Object.entries(buffers)) plain[k] = v instanceof T.Tensor ? v.data.slice() : Float32Array.from(v);
    const prog = createProgram(source, { uniforms, buffers: plain, loopLimit });
    prog.run(count);
    const out = {};
    for (const [k, v] of Object.entries(prog.buffers)) {
      const shape = buffers[k] instanceof T.Tensor ? buffers[k].shape : [v.length];
      out[k] = new T.Tensor(Float32Array.from(v), shape);
    }
    return out;
  };
}

export function save(module) { return JSON.stringify(module.stateDict()); }
export function load(module, json) { return module.loadStateDict(typeof json === 'string' ? JSON.parse(json) : json); }

/** Mean accuracy of argmax predictions against class targets. */
export function accuracy(logits, targets) {
  const pred = logits.argmax(-1);
  let ok = 0;
  for (let i = 0; i < pred.size; i++) if (pred.data[i] === targets.data[i]) ok++;
  return ok / pred.size;
}

export const { Tensor, tensor, zeros, ones, full, rand, randn, randint, arange, linspace, eye, zerosLike, onesLike, randperm, cat, stack, where, noGrad, manualSeed, broadcastShapes, isGradEnabled } = T;
export { nn, optim, data };
export const utils = { data };
export const version = '1.0.0';

const torch = {
  Tensor: T.Tensor, tensor: T.tensor, zeros: T.zeros, ones: T.ones, full: T.full, rand: T.rand, randn: T.randn, randint: T.randint,
  arange: T.arange, linspace: T.linspace, eye: T.eye, zerosLike: T.zerosLike, onesLike: T.onesLike, randperm: T.randperm,
  cat: T.cat, stack: T.stack, where: T.where, noGrad: T.noGrad, manualSeed: T.manualSeed, isGradEnabled: T.isGradEnabled, broadcastShapes: T.broadcastShapes,
  nn, optim, data, utils, kernel, save, load, accuracy, version,
};
export default torch;
