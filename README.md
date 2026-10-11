# 🔥 Pyre

**A typed GPU language with Python-style syntax.** One `.pyre` file compiles to

| Target | Runs on | Use it for |
| --- | --- | --- |
| **GLSL ES 3.00** | WebGL2 (every browser, Three.js, Helix) | fragment shaders / materials |
| **WGSL** | WebGPU | fragment shaders **and compute kernels** |
| **JavaScript** | Node.js or any browser, no GPU needed | tests, servers, CI, CPU fallback |

`npm run gpu-check` compiles every example in [`examples/`](examples) with Chrome's real WebGL2
and WebGPU drivers, runs it, and compares the GPU output with the CPU backend pixel by pixel (and
buffer by buffer for kernels). They match.

```python
# plasma.pyre
uniform time: float

@fragment
fn main(uv: vec2) -> vec4:
    let p = (uv - 0.5) * 6.0
    var v = sin(p.x + time) + sin((p.y + time) * 0.5)
    v += sin(length(p + vec2(sin(time / 3.0), cos(time / 2.0)) * 2.0) + time)
    return vec4(palette(v * 0.25, vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.0, 0.33, 0.67)), 1.0)
```

```python
# saxpy.pyre — a WebGPU compute kernel (also runs on the CPU)
uniform a: float
uniform n: int
buffer x: [float]
buffer y: [float]

@kernel(64)
fn saxpy(id: int):
    if id < n:
        y[id] = a * x[id] + y[id]
```

## Install

```sh
npm install github:Tharunmaks/Pyre     # library + `pyre` CLI
npx pyre --help
```

No dependencies. Node.js 18+.

## CLI

```sh
pyre init hello.pyre                                  # starter shader
pyre check examples/*.pyre                            # type-check with friendly errors
pyre compile plasma.pyre --target wgsl -o plasma.wgsl # glsl | wgsl | js
pyre render raymarch.pyre --time 1 --size 512x512     # CPU render to PNG
pyre run saxpy.pyre -u a=2 -u n=4 -b x=1,2,3,4 -b y=4 # run a kernel on the CPU
```

Errors point at the exact column and suggest fixes:

```
bad.pyre:2:12: error: unknown function 'lenght' (did you mean 'length'?)
   2 |     return lenght(x) + y
                  ^
```

## JavaScript API

```js
import { compile, compileAll, createProgram, runKernel, renderFragment, GLSL_VERTEX } from 'pyre-lang';

const { code, meta } = compile(src, { target: 'glsl' });   // or 'wgsl' / 'js'
meta.uniforms        // [{ name: 'time', type: 'float' }]
meta.uniformLayout   // WGSL uniform-buffer offsets, ready for writeBuffer
meta.buffers         // [{ name: 'x', elem: 'float', binding: 1 }]

const out = runKernel(saxpySrc, { uniforms: { a: 2, n: 4 }, buffers: { x: [1, 2, 3, 4], y: 4 } });
const img = renderFragment(plasmaSrc, { width: 256, height: 256, uniforms: { time: 1 } }); // RGBA8
```

Use the GLSL output with Three.js:

```js
new THREE.RawShaderMaterial({
  vertexShader: GLSL_VERTEX.replace('#version 300 es\n', ''),
  fragmentShader: compile(src, { target: 'glsl', version: false }).code,
  glslVersion: THREE.GLSL3,
  uniforms: { time: { value: 0 } },
});
```

## The language

* **Types**: `bool int float vec2 vec3 vec4 mat2 mat3 mat4`; buffers `[float]`, `[int]`.
* **Declarations**: `uniform name: type`, `buffer name: [float]`, `const NAME = expr`,
  `let` (immutable), `var` (mutable). Types are inferred; annotate with `let x: float = 1`.
* **Ints become floats automatically** where a float is expected (`vec3(1, 0, 0)` is fine).
* **Operators**: `+ - * / % **`, comparisons, `and or not`, int bitwise `& | ^ << >>`,
  `a if cond else b`, swizzles `v.xyz` / `c.rgb`, `v.x = 1.0`, `+= -= *= /=`.
* **Control flow**: `if / elif / else`, `for i in range(start, end, step)`, `while`, `break`,
  `continue`, `return`. No recursion (GPUs can't), and every path must return.
* **Entry points**: `@fragment fn main(uv: vec2, frag: vec2, normal: vec3, position: vec3) -> vec4`
  (take any subset of those inputs) and `@kernel(workgroup_size) fn k(id: int)`.
* **Built-ins**: `sin cos tan asin acos atan atan2 sinh cosh tanh exp log exp2 log2 sqrt inversesqrt
  pow abs sign floor ceil fract round trunc mod min max clamp saturate mix step smoothstep length
  distance dot cross normalize reflect refract radians degrees`, constants `PI TAU E`.
* **Standard library** (written in Pyre, linked only when used, overridable): `ihash hash21 hash22
  hash31 noise fbm rotate palette hsv2rgb sdSphere sdBox sdTorus smin`. Hashing is integer-based,
  so noise is bit-identical on every GPU and the CPU.

`frag` is the pixel coordinate. Its y axis grows upward in GLSL and on the CPU and downward in
WGSL; this is the platform convention, so prefer `uv` when orientation matters.

## Security

Pyre is designed to compile **untrusted** code safely (the Chroma platform runs user kernels on
its servers):

* The language has no strings, pointers, I/O or recursion. Generated JS can only touch its own
  locals and the runtime, uniform and buffer objects passed in; identifiers that collide with
  JS/GLSL/WGSL keywords (`constructor`, `__proto__`, `eval`, …) are renamed.
* Every loop iteration on the CPU ticks a budget (`loopLimit`), so `while true:` throws instead
  of hanging the host. Buffer reads/writes are bounds-checked; out-of-range access reads 0 and
  writes nothing (matching WebGPU's robust buffer access).
* Source size (256 KB) and nesting depth (64) are capped.

See [SECURITY.md](SECURITY.md).

## Development

```sh
npm test                 # 23 unit tests, no dependencies
npm run gpu-check        # real WebGL2 + WebGPU in headless Chromium (needs playwright)
```

The playground in [`docs/index.html`](docs/index.html) runs entirely in the browser and works on
GitHub Pages (Settings → Pages → folder `/docs`).

Pyre powers the **Chroma** platform and its **Helix** 3D engine:
[Tharunmaks/Chroma-node](https://github.com/Tharunmaks/Chroma-node).

MIT licensed.
