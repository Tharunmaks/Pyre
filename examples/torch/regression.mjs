// Fit y = sin(x) with a tiny network and print predictions.
//   node examples/torch/regression.mjs
import torch from '../../src/torch/index.js';

const { nn, optim } = torch;
torch.manualSeed(42);
const x = torch.linspace(-Math.PI, Math.PI, 128).reshape(-1, 1);
const y = x.sin();
const model = new nn.Sequential(new nn.Linear(1, 32), new nn.Tanh(), new nn.Linear(32, 32), new nn.Tanh(), new nn.Linear(32, 1));
const opt = new optim.Adam(model.parameters(), { lr: 0.01 });
for (let step = 0; step <= 1500; step++) {
  const loss = nn.mseLoss(model(x), y);
  opt.zeroGrad();
  loss.backward();
  opt.step();
  if (step % 300 === 0) console.log(`step ${step}  mse ${loss.item().toExponential(3)}`);
}
const probe = torch.tensor([[-2], [0], [1], [3]]);
console.log('x        ', probe.tolist().flat());
console.log('model(x)', torch.noGrad(() => model(probe)).tolist().flat().map((v) => +v.toFixed(3)));
console.log('sin(x)  ', probe.sin().tolist().flat().map((v) => +v.toFixed(3)));
