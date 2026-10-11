# Security

Pyre is built to compile and run **untrusted** programs, because the Chroma platform accepts
`.pyre` files from any user and executes `@kernel` functions on its servers.

## Guarantees

| Risk | How Pyre handles it |
| --- | --- |
| Code injection into generated JS/GLSL/WGSL | The language has no strings. Output is built only from checked AST nodes: identifiers match `[A-Za-z_][A-Za-z0-9_]*`, numbers are printed with `String(number)`. Names that collide with JS, GLSL or WGSL keywords or prototype members (`constructor`, `__proto__`, `eval`, `texture`, …) are renamed. Built-in lookup tables have no prototype. |
| Escaping the sandbox from JS output | Generated code only references its own locals and the `$R` (frozen runtime), `$U` (uniforms) and `$B` (typed buffers) parameters. There is no way to name a global. |
| Infinite loops | Every loop iteration on the CPU ticks a budget (`loopLimit`, default 50M); exceeding it throws. `for` loops need a constant non-zero step. Recursion is rejected at compile time. |
| Out-of-bounds memory | CPU buffer and vector access is bounds-checked: reads return 0, writes are dropped. WebGPU applies robust buffer access on the GPU. |
| Compiler denial of service | Source is capped at 256 KB and expression/block nesting at 64 levels. |

Hosts should still run untrusted kernels off the main thread with a wall-clock timeout (Chroma
uses `worker_threads` with a hard kill) and cap buffer sizes.

## Reporting

Please open a private security advisory on GitHub
(Security → Advisories → Report a vulnerability) rather than a public issue.
