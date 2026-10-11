import { test } from 'node:test';
import assert from 'node:assert/strict';
import torch from '../src/torch/index.js';

const { nn, optim } = torch;
const close = (a, b, tol = 1e-3, msg = '') => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${msg} ${a} != ${b}`);

/** Compare autograd gradients with central finite differences (float64 math on float32 data). */
function gradcheck(fn, ...shapes) {
  torch.manualSeed(7);
  const inputs = shapes.map((s) => torch.randn(s).map((v) => v * 0.8 + 0.1).requiresGrad_());
  fn(...inputs).sum().backward();
  for (const [k, x] of inputs.entries()) {
    for (let i = 0; i < x.size; i++) {
      const h = 1e-2;
      const orig = x.data[i];
      x.data[i] = orig + h;
      const up = torch.noGrad(() => fn(...inputs).sum().item());
      x.data[i] = orig - h;
      const down = torch.noGrad(() => fn(...inputs).sum().item());
      x.data[i] = orig;
      close(x.grad.data[i], (up - down) / (2 * h), 2e-2, `input ${k}[${i}]`);
    }
  }
}

test('tensors: creation, shapes, printing, tolist', () => {
  const t = torch.tensor([[1, 2, 3], [4, 5, 6]]);
  assert.deepEqual(t.shape, [2, 3]);
  assert.deepEqual(t.T.tolist(), [[1, 4], [2, 5], [3, 6]]);
  assert.deepEqual(t.reshape(3, -1).shape, [3, 2]);
  assert.deepEqual(torch.zeros(2, 2).tolist(), [[0, 0], [0, 0]]);
  assert.deepEqual(torch.arange(0, 1, 0.25).tolist(), [0, 0.25, 0.5, 0.75]);
  assert.match(String(t), /^tensor\(\[\[/);
  assert.throws(() => torch.tensor([[1, 2], [3]]), /ragged/);
  assert.throws(() => t.reshape(4, 2), /cannot reshape/);
});

test('broadcasting follows NumPy rules', () => {
  const a = torch.tensor([[1], [2], [3]]);
  const b = torch.tensor([10, 20]);
  assert.deepEqual(a.add(b).tolist(), [[11, 21], [12, 22], [13, 23]]);
  assert.deepEqual(torch.broadcastShapes([4, 1, 3], [5, 1]), [4, 5, 3]);
  assert.throws(() => torch.ones(2, 3).add(torch.ones(4)), /cannot be broadcast/);
});

test('reductions and indexing', () => {
  const t = torch.tensor([[1, 5, 3], [4, 2, 6]]);
  assert.deepEqual(t.sum(0).tolist(), [5, 7, 9]);
  assert.deepEqual(t.mean(1, true).tolist(), [[3], [4]]);
  assert.deepEqual(t.max(1).values.tolist(), [5, 6]);
  assert.deepEqual(t.argmax(1).tolist(), [1, 2]);
  assert.equal(t.max().item(), 6);
  assert.deepEqual(t.gather(1, torch.tensor([[2], [0]])).tolist(), [[3], [4]]);
  assert.deepEqual(t.narrow(1, 1, 2).tolist(), [[5, 3], [2, 6]]);
  assert.deepEqual(torch.cat([t, t], 0).shape, [4, 3]);
  assert.deepEqual(torch.stack([t, t], 0).shape, [2, 2, 3]);
  close(torch.tensor([1, 2, 3, 4]).std().item(), 1.2910, 1e-3);
  t.softmax(1).sum(1).tolist().forEach((v) => close(v, 1, 1e-6));
});

test('matmul: 2D, batched, vector cases', () => {
  const a = torch.tensor([[1, 2], [3, 4]]);
  assert.deepEqual(a.matmul(torch.eye(2)).tolist(), [[1, 2], [3, 4]]);
  assert.deepEqual(a.matmul(torch.tensor([1, 1])).tolist(), [3, 7]);
  assert.equal(torch.tensor([1, 2]).matmul(torch.tensor([3, 4])).item(), 11);
  assert.deepEqual(torch.ones(5, 2, 3).matmul(torch.ones(3, 4)).shape, [5, 2, 4]);
  assert.throws(() => a.matmul(torch.ones(3, 2)), /mismatch/);
});

test('autograd matches finite differences for every differentiable op', () => {
  gradcheck((a, b) => a.add(b).mul(b), [2, 3], [3]);
  gradcheck((a, b) => a.div(b.abs().add(1)), [2, 3], [2, 3]);
  gradcheck((a) => a.pow(2).sub(a.mul(3)), [4]);
  gradcheck((a) => a.exp().add(a.abs().add(1).log()).add(a.abs().add(1).sqrt()), [5]);
  gradcheck((a) => a.sin().add(a.cos()).add(a.tanh()).add(a.sigmoid()), [5]);
  gradcheck((a) => a.gelu().add(a.leakyRelu(0.1)), [6]);
  gradcheck((a, b) => a.matmul(b), [3, 4], [4, 2]);
  gradcheck((a, b) => a.matmul(b), [2, 3, 4], [4, 2]);
  gradcheck((a) => a.softmax(-1).mul(torch.arange(3)), [2, 3]);
  gradcheck((a) => a.logSoftmax(1).mul(torch.arange(4)), [3, 4]);
  gradcheck((a) => a.mean(0).add(a.sum(1, true).mean()), [3, 4]);
  gradcheck((a) => a.T.reshape(-1).mul(torch.arange(6)), [2, 3]);
  gradcheck((a) => a.permute(2, 0, 1).mul(torch.arange(24).reshape(4, 2, 3)), [2, 3, 4]);
  gradcheck((a) => a.max(1).values.add(a.min(0).values.sum()), [3, 4]);
  gradcheck((a) => a.gather(1, torch.tensor([[0], [2]])), [2, 3]);
  gradcheck((a, b) => torch.cat([a, b], 1).mul(torch.arange(5)), [2, 2], [2, 3]);
  gradcheck((a) => a.narrow(0, 1, 2).mul(2), [4, 2]);
  gradcheck((a) => a.std(1), [2, 5]);
  gradcheck((a) => a.indexSelect(torch.tensor([2, 0, 2])).mul(torch.arange(6).reshape(3, 2)), [3, 2]);
});

test('gradients accumulate on leaves and noGrad disables tracking', () => {
  const w = torch.tensor([2, 3], { requiresGrad: true });
  w.mul(w).sum().backward();
  w.mul(w).sum().backward();
  assert.deepEqual(w.grad.tolist(), [8, 12]);
  const y = torch.noGrad(() => w.mul(2));
  assert.equal(y.requiresGrad, false);
  assert.throws(() => torch.ones(2).backward(), /does not require grad/);
  assert.throws(() => w.mul(2).backward(), /scalar/);
});

test('losses: cross entropy, MSE, BCE agree with closed forms', () => {
  const logits = torch.tensor([[2, 1, 0.1]]);
  const ce = nn.crossEntropy(logits, torch.tensor([0])).item();
  const p = Math.exp(2) / (Math.exp(2) + Math.exp(1) + Math.exp(0.1));
  close(ce, -Math.log(p), 1e-5);
  close(nn.mseLoss(torch.tensor([1, 2]), torch.tensor([3, 2])).item(), 2, 1e-6);
  const x = torch.tensor([0.3, -1.2]);
  const t = torch.tensor([1, 0]);
  close(nn.bceWithLogits(x, t).item(), nn.binaryCrossEntropy(x.sigmoid(), t).item(), 1e-4);
});

test('modules: callable, parameters, state dict round-trip, printing', () => {
  torch.manualSeed(3);
  const m = new nn.Sequential(new nn.Linear(4, 8), new nn.ReLU(), new nn.Dropout(0.5), new nn.LayerNorm(8), new nn.Linear(8, 2));
  assert.equal(typeof m, 'function');
  assert.deepEqual(m(torch.randn(5, 4)).shape, [5, 2]);
  assert.equal(m.numParameters(), 4 * 8 + 8 + 16 + 8 * 2 + 2);
  assert.match(String(m), /Linear\(in_features=4, out_features=8/);
  assert.match(String(m), /ReLU\(\)/);
  m.eval();
  const x = torch.randn(3, 4);
  const before = m(x).tolist();
  const copy = new nn.Sequential(new nn.Linear(4, 8), new nn.ReLU(), new nn.Dropout(0.5), new nn.LayerNorm(8), new nn.Linear(8, 2)).eval();
  torch.load(copy, torch.save(m));
  copy(x).tolist().flat().forEach((v, i) => close(v, before.flat()[i], 1e-6));
  const emb = new nn.Embedding(10, 3);
  assert.deepEqual(emb(torch.tensor([[1, 2], [3, 4]])).shape, [2, 2, 3]);
});

test('training: an MLP learns XOR and two spirals', () => {
  torch.manualSeed(1);
  const x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]);
  const y = torch.tensor([[0], [1], [1], [0]]);
  const model = new nn.Sequential(new nn.Linear(2, 8), new nn.Tanh(), new nn.Linear(8, 1));
  const opt = new optim.Adam(model.parameters(), { lr: 0.05 });
  for (let e = 0; e < 400; e++) {
    const loss = nn.bceWithLogits(model(x), y);
    opt.zeroGrad();
    loss.backward();
    opt.step();
  }
  assert.deepEqual(model(x).sigmoid().tolist().flat().map(Math.round), [0, 1, 1, 0]);

  let seed = 5;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const d = torch.data.spirals(80, {}, rnd);
  const X = torch.tensor(d.x);
  const Y = torch.tensor(d.y);
  const clf = new nn.Sequential(new nn.Linear(2, 32), new nn.ReLU(), new nn.Linear(32, 32), new nn.ReLU(), new nn.Linear(32, 2));
  const o2 = new optim.AdamW(clf.parameters(), { lr: 0.02 });
  const loader = new torch.data.DataLoader(new torch.data.TensorDataset(X, Y), { batchSize: 32, shuffle: true });
  for (let e = 0; e < 120; e++) {
    for (const [xb, yb] of loader) {
      const l = nn.crossEntropy(clf(xb), yb);
      o2.zeroGrad();
      l.backward();
      o2.step();
    }
  }
  assert.ok(torch.accuracy(clf(X), Y) > 0.85, 'spiral accuracy');
});

test('SGD with momentum and schedulers update parameters', () => {
  const w = torch.tensor([5], { requiresGrad: true });
  const opt = new optim.SGD([w], { lr: 0.1, momentum: 0.9 });
  const sched = new optim.lrScheduler.StepLR(opt, { stepSize: 50, gamma: 0.5 });
  for (let i = 0; i < 200; i++) {
    const loss = w.sub(2).square().sum();
    opt.zeroGrad();
    loss.backward();
    opt.step();
    sched.step();
  }
  close(w.item(), 2, 1e-3);
  close(opt.lr, 0.1 / 16, 1e-9);
});

test('Pyre kernels run as tensor ops', () => {
  const k = torch.kernel('uniform s: float\nbuffer v: [float]\nbuffer o: [float]\n@kernel\nfn k(id: int):\n    o[id] = v[id] * s + 1.0\n');
  const x = torch.arange(6).reshape(2, 3);
  const { o } = k({ v: x, o: torch.zeros(2, 3) }, { uniforms: { s: 2 } });
  assert.deepEqual(o.tolist(), [[1, 3, 5], [7, 9, 11]]);
  assert.deepEqual(x.tolist(), [[0, 1, 2], [3, 4, 5]], 'inputs untouched');
});
