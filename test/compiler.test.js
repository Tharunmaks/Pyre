import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  compile, compileAll, diagnose, createProgram, runKernel, renderFragment, PyreError,
} from '../src/index.js';
import { tokenize } from '../src/lexer.js';
import { encodePNG } from '../src/png.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const examples = path.join(here, '..', 'examples');

const errorOf = (src) => {
  try { compile(src, { target: 'js' }); } catch (e) { if (e instanceof PyreError) return e; throw e; }
  return assert.fail('expected a compile error');
};
const fn = (body, sig = 'fn f() -> float') => `${sig}:\n${body.split('\n').map((l) => '    ' + l).join('\n')}\n`;
const call = (src, name, ...args) => createProgram(src).functions[name](...args);

test('lexer produces INDENT/DEDENT and ignores newlines inside brackets', () => {
  const types = tokenize('fn f():\n    let x = (1 +\n        2)\n    pass\n').map((t) => t.type);
  assert.deepEqual(types.filter((t) => t === 'INDENT' || t === 'DEDENT'), ['INDENT', 'DEDENT']);
  assert.equal(types.filter((t) => t === 'NEWLINE').length, 3);
});

test('number literals: ints, floats, hex, exponents, underscores', () => {
  const t = tokenize('1 1.5 .5 2. 1e3 0xff 1_000');
  assert.deepEqual(t.slice(0, 7).map((x) => [x.type, x.value]), [
    ['INT', 1], ['FLOAT', 1.5], ['FLOAT', 0.5], ['FLOAT', 2], ['FLOAT', 1000], ['INT', 255], ['INT', 1000],
  ]);
  assert.throws(() => tokenize('99999999999'), /32 bits/);
});

test('every example compiles for every target it supports', () => {
  for (const f of fs.readdirSync(examples).filter((n) => n.endsWith('.pyre'))) {
    const r = compileAll(fs.readFileSync(path.join(examples, f), 'utf8'), { filename: f });
    assert.ok(r.wgsl && r.js, `${f} should compile to wgsl and js`);
    if (r.meta.entry === 'fragment') assert.ok(r.glsl.startsWith('#version 300 es'), f);
    else assert.match(r.errors.glsl, /WGSL or JS/);
  }
});

test('int literals are promoted to float where needed', () => {
  const src = fn('let x = 1\nreturn x + 2.5');
  assert.equal(call(src, 'f'), 3.5);
  assert.match(compile('@fragment\nfn main(uv: vec2) -> vec4:\n    return vec4(1, 0, 0, 1)\n').code, /vec4\(1\.0, 0\.0, 0\.0, 1\.0\)/);
});

test('integer arithmetic truncates and wraps like the GPU', () => {
  const src = fn('return 7 / 2', 'fn f() -> int') + fn('return 2147483647 + 1', 'fn g() -> int') + fn('return -7 % 3', 'fn h() -> int');
  assert.equal(call(src, 'f'), 3);
  assert.equal(call(src, 'g'), -2147483648);
  assert.equal(call(src, 'h'), -1);
});

test('float % follows GLSL mod() semantics', () => {
  assert.ok(Math.abs(call(fn('return -1.0 % 3.0'), 'f') - 2) < 1e-12);
});

test('swizzles read, assign single components and reject invalid sets', () => {
  const src = fn('var v = vec4(1.0, 2.0, 3.0, 4.0)\nv.y = 9.0\nreturn v.wzy', 'fn f() -> vec3');
  assert.deepEqual(call(src, 'f'), [4, 3, 9]);
  assert.match(errorOf(fn('let v = vec2(1.0)\nreturn v.z')).message, /out of range/);
  assert.match(errorOf(fn('let v = vec3(1.0)\nreturn v.xg.x')).message, /invalid swizzle/);
  assert.match(errorOf(fn('var v = vec3(1.0)\nv.xy = vec2(0.0)\nreturn v.x')).message, /single components/);
});

test('vectors are values: copies never alias', () => {
  const src = fn('var a = vec2(1.0, 2.0)\nlet b = a\nvar c = a\na.x = 5.0\nc.y = 7.0\nreturn vec4(b, c)', 'fn f() -> vec4');
  assert.deepEqual(call(src, 'f'), [1, 2, 1, 7]);
});

test('matrices multiply column-major like GLSL', () => {
  const src = fn('let m = mat2(1.0, 2.0, 3.0, 4.0)\nreturn m * vec2(1.0, 1.0)', 'fn f() -> vec2')
    + fn('let m = mat2(1.0, 2.0, 3.0, 4.0)\nreturn vec2(1.0, 1.0) * m', 'fn g() -> vec2')
    + fn('return rotate(1.5707963267948966) * vec2(1.0, 0.0)', 'fn h() -> vec2');
  assert.deepEqual(call(src, 'f'), [4, 6]);
  assert.deepEqual(call(src, 'g'), [3, 7]);
  const r = call(src, 'h');
  assert.ok(Math.abs(r[0]) < 1e-9 && Math.abs(r[1] - 1) < 1e-9);
});

test('built-ins broadcast scalars and check vector sizes', () => {
  assert.deepEqual(call(fn('return clamp(vec3(-1.0, 0.5, 2.0), 0.0, 1.0)', 'fn f() -> vec3'), 'f'), [0, 0.5, 1]);
  assert.equal(call(fn('return max(3, 9)', 'fn f() -> int'), 'f'), 9);
  assert.match(errorOf(fn('return dot(vec2(1.0), vec3(1.0))')).message, /mix vec2 and vec3/);
  assert.match(compile('@fragment\nfn main(uv: vec2) -> vec4:\n    return vec4(atan2(uv.y, uv.x))\n').code, /vec4\(atan\(uv\.y, uv\.x\)\)/);
});

test('control flow: elif, for with steps, while, break, continue, ternary', () => {
  const src = fn([
    'var s = 0',
    'for i in range(10, 0, -2):',
    '    if i == 4:',
    '        continue',
    '    elif i == 2:',
    '        break',
    '    s += i',
    'var k = 0',
    'while true:',
    '    k += 1',
    '    if k >= 3:',
    '        break',
    'return s * 10 + k + (100 if s > 20 else 0)',
  ].join('\n'), 'fn f() -> int');
  assert.equal(call(src, 'f'), (10 + 8 + 6) * 10 + 3 + 100);
});

test('bitwise operators work on ints only', () => {
  assert.equal(call(fn('return (0xf0 | 0x0f) ^ (1 << 4) & 0xff >> 1', 'fn f() -> int'), 'f'), (0xf0 | 0x0f) ^ ((1 << 4) & (0xff >> 1)));
  assert.match(errorOf(fn('return 1.0 & 2.0')).message, /works on ints/);
});

test('helpful errors with positions and suggestions', () => {
  const e = errorOf(fn('let radius = 1.0\nreturn raduis * 2.0'));
  assert.match(e.message, /unknown name 'raduis' \(did you mean 'radius'\?\)/);
  assert.equal(e.line, 3);
  assert.match(e.format(), /\^/);
  assert.match(errorOf(fn('let x = 1.0\nx = 2.0\nreturn x')).message, /declare it with 'var'/);
  assert.match(errorOf(fn('if true:\n    return 1.0')).message, /without returning/);
  assert.match(errorOf('fn a() -> float:\n    return b()\nfn b() -> float:\n    return a()\n').message, /recursion/);
  assert.match(errorOf(fn('return 1.0 < 2.0 < 3.0')).message, /chained/);
  assert.match(errorOf('@fragment\nfn main(color: vec3) -> vec4:\n    return vec4(1.0)\n').message, /not a known input/);
  assert.match(errorOf(fn('break')).message, /outside a loop/);
  assert.match(errorOf('fn f():\n  let x = 1\n    let y = 2\n').message, /expected end of line|indent/);
});

test('diagnose returns structured errors instead of throwing', () => {
  const r = diagnose('fn f() -> float:\n    return nope\n');
  assert.equal(r.ok, false);
  assert.equal(r.errors[0].line, 2);
  assert.equal(diagnose(fs.readFileSync(path.join(examples, 'plasma.pyre'), 'utf8')).ok, true);
});

test('reserved words are renamed so generated code stays valid', () => {
  const src = 'fn class(input: float) -> float:\n    let texture = input * 2.0\n    return texture\n'
    + '@fragment\nfn main(uv: vec2) -> vec4:\n    return vec4(class(uv.x))\n';
  const glsl = compile(src).code;
  assert.doesNotMatch(glsl, /\btexture\b|\binput\b|float class\(/);
  assert.equal(createProgram(src).functions.class(2), 4);
  assert.match(errorOf('uniform texture: float\n').message, /reserved/);
});

test('JS backend cannot be steered into arbitrary code', () => {
  // dangerous-looking names become plain, renamed locals
  const src = 'fn constructor(__proto__: float) -> float:\n    let eval = __proto__ + 1.0\n    return eval\n';
  assert.equal(createProgram(src).functions.constructor(1), 2);
  assert.throws(() => compile('fn f():\n    let x = "a"\n', { target: 'js' }), /unexpected character/);
});

test('runaway loops hit the loop budget instead of hanging', () => {
  const src = '@kernel\nfn spin(id: int):\n    var i = 0\n    while i >= 0:\n        i = 1\n';
  assert.throws(() => runKernel(src, { count: 1, loopLimit: 10_000 }), /loop budget/);
});

test('kernels run on the CPU with typed buffers and bounds-checked access', () => {
  const src = fs.readFileSync(path.join(examples, 'saxpy.pyre'), 'utf8');
  const out = runKernel(src, { uniforms: { a: 2, n: 3 }, buffers: { x: [1, 2, 3, 4], y: [10, 10, 10, 10] } });
  assert.deepEqual(Array.from(out.y), [12, 14, 16, 10]);
  const oob = runKernel('buffer b: [int]\n@kernel\nfn k(id: int):\n    b[id + 100] = b[id - 100] + 1\n', { buffers: { b: 2 } });
  assert.deepEqual(Array.from(oob.b), [0, 0]);
  const primes = runKernel(fs.readFileSync(path.join(examples, 'primes.pyre'), 'utf8'), { buffers: { result: 20 } });
  assert.deepEqual([...primes.result.keys()].filter((i) => primes.result[i]), [2, 3, 5, 7, 11, 13, 17, 19]);
});

test('fragment shaders render on the CPU, deterministic, uv origin bottom-left', () => {
  const src = '@fragment\nfn main(uv: vec2) -> vec4:\n    return vec4(uv.x, uv.y, 0.0, 1.0)\n';
  const img = renderFragment(src, { width: 2, height: 2 });
  // top-left pixel: x small, y large
  assert.deepEqual(Array.from(img.data.slice(0, 4)), [64, 191, 0, 255]);
  const plasma = fs.readFileSync(path.join(examples, 'plasma.pyre'), 'utf8');
  const a = renderFragment(plasma, { width: 8, height: 8, uniforms: { time: 2 } }).data;
  const b = renderFragment(plasma, { width: 8, height: 8, uniforms: { time: 2 } }).data;
  assert.deepEqual(a, b);
});

test('WGSL output: uniforms in a struct with std layout, buffers bound in order', () => {
  const r = compile(fs.readFileSync(path.join(examples, 'particles.pyre'), 'utf8'), { target: 'wgsl' });
  assert.match(r.code, /var<uniform> pyre_u: PyreUniforms/);
  assert.match(r.code, /@compute @workgroup_size\(128\)/);
  assert.deepEqual(r.meta.buffers.map((b) => b.binding), [1, 2, 3, 4]);
  const layout = compile('uniform a: float\nuniform b: vec3\nuniform c: float\n@kernel\nfn k(id: int):\n    pass\n', { target: 'wgsl' }).meta.uniformLayout;
  assert.deepEqual(layout.fields.map((f) => f.offset), [0, 16, 28]);
  assert.equal(layout.size, 32);
});

test('standard library functions are linked only when used and can be overridden', () => {
  const plain = compile('@fragment\nfn main(uv: vec2) -> vec4:\n    return vec4(1.0)\n').code;
  assert.doesNotMatch(plain, /hash21|noise/);
  const own = 'fn noise(p: vec2) -> float:\n    return 0.25\n@fragment\nfn main(uv: vec2) -> vec4:\n    return vec4(noise(uv))\n';
  assert.deepEqual(renderFragment(own, { width: 1, height: 1 }).data[0], 64);
});

test('PNG encoder writes a valid signature and IHDR', () => {
  const png = encodePNG({ width: 3, height: 2, data: new Uint8ClampedArray(24) });
  assert.deepEqual([...png.subarray(0, 8)], [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  assert.equal(png.readUInt32BE(16), 3);
  assert.equal(png.readUInt32BE(20), 2);
});

test('source size and nesting limits protect the compiler', () => {
  assert.throws(() => compile('#'.repeat(300 * 1024)), /larger than/);
  const deep = fn(`return ${'('.repeat(200)}1.0${')'.repeat(200)}`);
  assert.throws(() => compile(deep, { target: 'js' }), /nested too deeply/);
});
