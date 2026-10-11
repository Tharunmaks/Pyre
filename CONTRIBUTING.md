# Contributing to Pyre

```sh
git clone https://github.com/Tharunmaks/Pyre && cd Pyre
npm test                 # compiler + torch gradient checks, zero dependencies
npm run gpu-check        # optional: real WebGL2/WebGPU via Playwright
node scripts/build-docs.mjs   # refresh the GitHub Pages playground after changing src/
```

* **Compiler:** `src/lexer.js` → `parser.js` → `checker.js` → `gen-glsl.js` / `gen-wgsl.js` / `gen-js.js`.
  New built-ins go in `types.js` (signature) and each generator; add a test and, if visual, an example
  that `gpu-check` can verify.
* **Standard library:** written in Pyre in `src/stdlib.js`.
* **Torch:** `src/torch/`. Every new differentiable op needs a `gradcheck` case in `test/torch.test.js`.
* Generated JS must stay injection-proof: only emit identifiers that passed the checker and numbers.

Small, focused pull requests with tests are easiest to review. MIT licensed.
