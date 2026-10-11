#!/usr/bin/env node
// Compiles every example and runs it on real GPU drivers (WebGL2 + WebGPU in
// headless Chromium via Playwright), then compares the GPU output with the
// CPU (JS) backend. Usage: node scripts/gpu-check.mjs [files...]
//   PLAYWRIGHT_MODULE=/path/to/playwright CHROMIUM=/path/to/chrome to override.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { renderFragment, runKernel, compile } from '../src/index.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const files = process.argv.slice(2).length
  ? process.argv.slice(2)
  : fs.readdirSync(path.join(root, 'examples')).filter((f) => f.endsWith('.pyre')).map((f) => path.join(root, 'examples', f));

let playwright;
try {
  playwright = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
} catch {
  console.error('playwright is not installed; run `npm i -D playwright` to use the GPU check');
  process.exit(2);
}
const { chromium } = playwright.default || playwright;

const SIZE = 64; // WebGPU copies need 256-byte rows (64 px * 4 bytes)
const TIME = 1.3;

const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  if (url.pathname === '/') {
    res.setHeader('content-type', 'text/html');
    return res.end('<!doctype html><title>gpu-check</title>');
  }
  const file = path.join(root, path.normalize(url.pathname));
  if (!file.startsWith(root + path.sep) || !fs.existsSync(file)) { res.statusCode = 404; return res.end(); }
  res.setHeader('content-type', 'text/javascript');
  res.end(fs.readFileSync(file));
}).listen(0);
const port = server.address().port;

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM || undefined,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=swiftshader', '--use-webgpu-adapter=swiftshader'],
});
const page = await browser.newPage();
await page.goto(`http://localhost:${port}/`);

function compare(gpu, cpu) {
  let bad = 0;
  let sum = 0;
  for (let i = 0; i < cpu.length; i++) {
    const d = Math.abs(gpu[i] - cpu[i]);
    sum += d;
    if (d > 3) bad++;
  }
  return { mean: sum / cpu.length, badPct: (100 * bad) / cpu.length };
}

let failures = 0;
for (const file of files) {
  const src = fs.readFileSync(file, 'utf8');
  const name = path.basename(file);
  const meta = compile(src, { target: 'js' }).meta;
  const uniforms = {};
  for (const u of meta.uniforms) uniforms[u.name] = u.type === 'int' ? 1000 : u.type === 'float' ? (u.name === 'dt' ? 0.016 : u.name === 'a' ? 2.5 : TIME) : 0;
  const result = await page.evaluate(async ({ src, size, uniforms, entry }) => {
    const pyre = await import('/src/index.js');
    const out = {};
    if (entry === 'fragment') {
      // ---- WebGL2 / GLSL ----
      const canvas = new OffscreenCanvas(size, size);
      const gl = canvas.getContext('webgl2');
      const glsl = pyre.compile(src, { target: 'glsl' }).code;
      const sh = (type, code) => {
        const s = gl.createShader(type);
        gl.shaderSource(s, code);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('GLSL: ' + gl.getShaderInfoLog(s) + '\n' + code);
        return s;
      };
      const prog = gl.createProgram();
      gl.attachShader(prog, sh(gl.VERTEX_SHADER, pyre.GLSL_VERTEX));
      gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, glsl));
      gl.bindAttribLocation(prog, 0, 'position');
      gl.bindAttribLocation(prog, 1, 'uv');
      gl.bindAttribLocation(prog, 2, 'normal');
      gl.vertexAttrib3f(2, 0, 0, 1);
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link: ' + gl.getProgramInfoLog(prog));
      gl.useProgram(prog);
      const quad = new Float32Array([-1, -1, 0, 1, -1, 0, -1, 1, 0, 1, 1, 0]);
      const uvs = new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]);
      for (const [loc, data, n] of [[0, quad, 3], [1, uvs, 2]]) {
        gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, n, gl.FLOAT, false, 0, 0);
      }
      const ident4 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
      gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'modelViewMatrix'), false, ident4);
      gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'projectionMatrix'), false, ident4);
      gl.uniformMatrix3fv(gl.getUniformLocation(prog, 'normalMatrix'), false, new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));
      for (const [k, v] of Object.entries(uniforms)) {
        const loc = gl.getUniformLocation(prog, k);
        if (loc) gl.uniform1f(loc, v);
      }
      gl.viewport(0, 0, size, size);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      const px = new Uint8Array(size * size * 4);
      gl.readPixels(0, 0, size, size, gl.RGBA, gl.UNSIGNED_BYTE, px);
      // flip to top-row-first
      const flipped = new Uint8Array(px.length);
      for (let y = 0; y < size; y++) flipped.set(px.subarray((size - 1 - y) * size * 4, (size - y) * size * 4), y * size * 4);
      out.glsl = Array.from(flipped);
    }
    // ---- WebGPU / WGSL ----
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const { code, meta } = pyre.compile(src, { target: 'wgsl' });
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === 'error');
    if (errs.length) throw new Error('WGSL: ' + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n') + '\n' + code);
    const entries = [];
    if (meta.uniforms.length) {
      const data = new ArrayBuffer(meta.uniformLayout.size);
      const view = new DataView(data);
      for (const f of meta.uniformLayout.fields) {
        if (f.type === 'int') view.setInt32(f.offset, uniforms[f.name], true);
        else view.setFloat32(f.offset, uniforms[f.name], true);
      }
      const buf = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buf, 0, data);
      entries.push({ binding: 0, resource: { buffer: buf } });
    }
    if (entry === 'fragment') {
      const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module, entryPoint: 'vs_fullscreen' },
        fragment: { module, entryPoint: 'fs_main', targets: [{ format: 'rgba8unorm' }] },
      });
      const tex = device.createTexture({ size: [size, size], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      const read = device.createBuffer({ size: size * size * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = device.createCommandEncoder();
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }] });
      pass.setPipeline(pipeline);
      if (entries.length) pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
      pass.draw(3);
      pass.end();
      enc.copyTextureToBuffer({ texture: tex }, { buffer: read, bytesPerRow: size * 4 }, [size, size]);
      device.queue.submit([enc.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      // WebGPU framebuffer row 0 is the top row
      out.wgsl = Array.from(new Uint8Array(read.getMappedRange()));
    } else {
      const n = 1000;
      const bufs = [];
      for (const b of meta.buffers) {
        const init = b.elem === 'int' ? new Int32Array(n) : new Float32Array(n).map((_, i) => Math.sin(i) * 3);
        const gb = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(gb, 0, init);
        entries.push({ binding: b.binding, resource: { buffer: gb } });
        bufs.push([b, gb, Array.from(init)]);
      }
      const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'cs_main' } });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
      pass.dispatchWorkgroups(Math.ceil(n / meta.workgroupSize));
      pass.end();
      const reads = bufs.map(([b, gb]) => {
        const r = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        enc.copyBufferToBuffer(gb, 0, r, 0, n * 4);
        return [b, r];
      });
      device.queue.submit([enc.finish()]);
      out.buffers = {};
      out.inputs = {};
      for (const [b, r] of reads) {
        await r.mapAsync(GPUMapMode.READ);
        out.buffers[b.name] = Array.from(b.elem === 'int' ? new Int32Array(r.getMappedRange()) : new Float32Array(r.getMappedRange()));
      }
      for (const [b, , init] of bufs) out.inputs[b.name] = init;
    }
    return out;
  }, { src, size: SIZE, uniforms, entry: meta.entry }).catch((e) => ({ error: e.message }));

  if (result.error) {
    failures++;
    console.log(`FAIL ${name}\n${result.error}`);
    continue;
  }
  if (meta.entry === 'fragment') {
    const cpu = renderFragment(src, { width: SIZE, height: SIZE, uniforms }).data;
    const g = compare(result.glsl, cpu);
    const w = compare(result.wgsl, cpu);
    if (process.env.DEBUG) console.log(result.wgsl.slice(0, 12).join(','), '|', result.glsl.slice(0, 12).join(','), '|', Array.from(cpu.slice(0, 12)).join(','));
    const ok = g.badPct < 2 && w.badPct < 2;
    if (!ok) failures++;
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}  WebGL2 vs CPU: ${g.badPct.toFixed(2)}% pixels differ  |  WebGPU vs CPU: ${w.badPct.toFixed(2)}% pixels differ`);
  } else {
    const cpuBufs = runKernel(src, {
      uniforms,
      buffers: Object.fromEntries(Object.entries(result.inputs).map(([k, v]) => [k, meta.buffers.find((b) => b.name === k).elem === 'int' ? Int32Array.from(v) : Float32Array.from(v)])),
      count: 1000,
    });
    let maxErr = 0;
    for (const [k, v] of Object.entries(result.buffers)) {
      v.forEach((x, i) => { maxErr = Math.max(maxErr, Math.abs(x - cpuBufs[k][i]) / Math.max(1, Math.abs(x))); });
    }
    const ok = maxErr < 1e-4;
    if (!ok) failures++;
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}  WebGPU compute vs CPU: max relative error ${maxErr.toExponential(2)}`);
  }
}
await browser.close();
server.close();
process.exit(failures ? 1 : 0);
