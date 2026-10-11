// Train a small MLP to separate two interleaved spirals, on the CPU.
//   node examples/torch/spirals.mjs
import torch from '../../src/torch/index.js';

const { nn, optim, data } = torch;
torch.manualSeed(0);

const d = data.spirals(150);
const X = torch.tensor(d.x);
const Y = torch.tensor(d.y);
const loader = new data.DataLoader(new data.TensorDataset(X, Y), { batchSize: 32, shuffle: true });

const model = new nn.Sequential(
  new nn.Linear(2, 64), new nn.ReLU(),
  new nn.Linear(64, 64), new nn.ReLU(),
  new nn.Linear(64, 2),
);
console.log(String(model), `\n${model.numParameters()} parameters\n`);

const opt = new optim.AdamW(model.parameters(), { lr: 0.01 });
const sched = new optim.lrScheduler.CosineAnnealingLR(opt, { tMax: 200 });
for (let epoch = 1; epoch <= 200; epoch++) {
  let total = 0;
  for (const [xb, yb] of loader) {
    const loss = nn.crossEntropy(model(xb), yb);
    opt.zeroGrad();
    loss.backward();
    opt.step();
    total += loss.item();
  }
  sched.step();
  if (epoch % 25 === 0) {
    const acc = torch.noGrad(() => torch.accuracy(model(X), Y));
    console.log(`epoch ${String(epoch).padStart(3)}  loss ${(total / loader.length).toFixed(4)}  accuracy ${(acc * 100).toFixed(1)}%`);
  }
}
