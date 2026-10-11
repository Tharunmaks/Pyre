#!/usr/bin/env node
// pyre — command-line compiler and runner for the Pyre GPU language.
import fs from 'node:fs';
import path from 'node:path';
import { compile, diagnose, renderFragment, runKernel, PyreError, VERSION, TARGETS } from '../src/index.js';
import { encodePNG } from '../src/png.js';

const HELP = `pyre ${VERSION} — a typed GPU language with Python-style syntax

Usage:
  pyre compile <file.pyre> [--target glsl|wgsl|js] [-o out]   compile to shader / JS source
  pyre check <files...>                                       type-check, print diagnostics
  pyre render <file.pyre> [-o out.png] [--size 512x512]       render a @fragment on the CPU
               [--time 1.5] [--uniform name=value ...]
  pyre run <file.pyre> --buffer name=1,2,3|name=<length> ...  run a @kernel on the CPU
               [--count N] [--uniform name=value ...]
  pyre init [file.pyre]                                       write a starter shader
  pyre --version | --help

Examples:
  pyre compile plasma.pyre --target wgsl -o plasma.wgsl
  pyre render plasma.pyre --time 2 -o plasma.png
  pyre run saxpy.pyre --uniform a=2 --uniform n=4 --buffer x=1,2,3,4 --buffer y=4`;

const STARTER = `# My first Pyre shader. Try: pyre render hello.pyre -o hello.png
uniform time: float

@fragment
fn main(uv: vec2) -> vec4:
    let d = length(uv - 0.5)
    let ring = smoothstep(0.02, 0.0, abs(d - 0.3 - 0.05 * sin(time * 2.0)))
    let col = palette(d + time * 0.2, vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.0, 0.33, 0.67))
    return vec4(col * (0.35 + ring), 1.0)
`;

function parseArgs(argv) {
  const args = { _: [], uniform: [], buffer: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const val = () => {
      if (i + 1 >= argv.length) die(`missing value for ${a}`);
      return argv[++i];
    };
    if (a === '-o' || a === '--out') args.out = val();
    else if (a === '--target' || a === '-t') args.target = val();
    else if (a === '--size') args.size = val();
    else if (a === '--time') args.time = val();
    else if (a === '--count') args.count = val();
    else if (a === '--uniform' || a === '-u') args.uniform.push(val());
    else if (a === '--buffer' || a === '-b') args.buffer.push(val());
    else if (a === '--help' || a === '-h') args.help = true;
    else if (a === '--version' || a === '-v') args.version = true;
    else if (a.startsWith('-')) die(`unknown option ${a}`);
    else args._.push(a);
  }
  return args;
}

function die(msg, code = 1) {
  console.error(`pyre: ${msg}`);
  process.exit(code);
}

function read(file) {
  if (!file) die('missing input file');
  try {
    return fs.readFileSync(file, 'utf8');
  } catch (err) {
    return die(`cannot read ${file}: ${err.code || err.message}`);
  }
}

function parseValue(text) {
  const parts = text.split(',').map((s) => s.trim()).filter(Boolean).map(Number);
  if (!parts.length || parts.some((n) => !Number.isFinite(n))) die(`invalid number list '${text}'`);
  return parts.length === 1 ? parts[0] : parts;
}

function uniforms(args) {
  const out = {};
  if (args.time !== undefined) out.time = parseValue(args.time);
  for (const u of args.uniform) {
    const eq = u.indexOf('=');
    if (eq < 1) die(`--uniform expects name=value, got '${u}'`);
    out[u.slice(0, eq)] = parseValue(u.slice(eq + 1));
  }
  return out;
}

function report(err) {
  if (err instanceof PyreError) {
    console.error(err.format());
    process.exit(1);
  }
  die(err.message);
}

const args = parseArgs(process.argv.slice(2));
const [cmd, ...rest] = args._;
if (args.version) { console.log(VERSION); process.exit(0); }
if (args.help || !cmd || cmd === 'help') { console.log(HELP); process.exit(cmd || args.help ? 0 : 1); }

try {
  switch (cmd) {
    case 'compile': {
      const file = rest[0];
      const target = args.target || (args.out ? path.extname(args.out).slice(1) : 'glsl');
      if (!TARGETS.includes(target)) die(`unknown target '${target}' (use ${TARGETS.join(', ')})`);
      const { code } = compile(read(file), { target, filename: file });
      if (args.out) {
        fs.writeFileSync(args.out, code);
        console.error(`wrote ${args.out} (${target}, ${code.length} bytes)`);
      } else process.stdout.write(code);
      break;
    }
    case 'check': {
      if (!rest.length) die('missing input file');
      let failed = 0;
      for (const file of rest) {
        const r = diagnose(read(file), { filename: file });
        if (r.ok) console.log(`ok   ${file}  (${r.meta.entry || 'library'}${r.meta.uniforms.length ? `, uniforms: ${r.meta.uniforms.map((u) => u.name).join(', ')}` : ''})`);
        else { failed++; console.log(r.errors.map((e) => e.text).join('\n')); }
      }
      process.exit(failed ? 1 : 0);
      break;
    }
    case 'render': {
      const file = rest[0];
      const [w, h] = (args.size || '512x512').split('x').map(Number);
      if (!(w > 0 && h > 0 && w <= 8192 && h <= 8192)) die('--size must look like 512x512 (max 8192)');
      const started = Date.now();
      const img = renderFragment(read(file), { width: w, height: h, uniforms: uniforms(args) });
      const out = args.out || file.replace(/\.pyre$/, '') + '.png';
      fs.writeFileSync(out, encodePNG(img));
      console.error(`wrote ${out} (${w}x${h}, ${Date.now() - started} ms on the CPU)`);
      break;
    }
    case 'run': {
      const file = rest[0];
      const buffers = {};
      for (const b of args.buffer) {
        const eq = b.indexOf('=');
        if (eq < 1) die(`--buffer expects name=1,2,3 or name=<length>, got '${b}'`);
        const v = parseValue(b.slice(eq + 1));
        if (!Array.isArray(v) && (!Number.isInteger(v) || v < 0 || v > 1e8)) die(`buffer length must be a whole number up to 100000000`);
        buffers[b.slice(0, eq)] = v;
      }
      const count = args.count !== undefined ? Number(args.count) : undefined;
      if (count !== undefined && !(Number.isInteger(count) && count >= 0)) die('--count must be a whole number');
      const result = runKernel(read(file), { buffers, uniforms: uniforms(args), count });
      for (const [k, v] of Object.entries(result)) {
        console.log(`${k} = [${Array.from(v, (x) => Math.round(x * 1e6) / 1e6).join(', ')}]`);
      }
      break;
    }
    case 'init': {
      const file = rest[0] || 'hello.pyre';
      if (fs.existsSync(file)) die(`${file} already exists`);
      fs.writeFileSync(file, STARTER);
      console.log(`created ${file}\nnext: pyre render ${file} --time 1 -o ${file.replace(/\.pyre$/, '')}.png`);
      break;
    }
    default:
      die(`unknown command '${cmd}' (try pyre --help)`);
  }
} catch (err) {
  report(err);
}
