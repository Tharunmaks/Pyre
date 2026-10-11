// Pyre semantic analysis: name resolution, type inference, implicit
// int->float promotion, swizzle validation, entry-point rules, recursion
// and missing-return detection. Annotates the AST in place for the code
// generators (every expression gets `.type`, every declaration `.out`).
import { fail } from './errors.js';
import {
  BUILTINS, BUILTIN_CONSTS, BUFFER_ELEMS, FRAGMENT_INPUTS, RESERVED, VALUE_TYPES,
  dim, isMat, isNumeric, isVec, vecOf,
} from './types.js';

const CONSTRUCTORS = new Set(['float', 'int', 'bool', 'vec2', 'vec3', 'vec4', 'mat2', 'mat3', 'mat4']);
const UNIFORM_TYPES = ['float', 'int', 'vec2', 'vec3', 'vec4', 'mat4'];
const SWIZZLE_SETS = ['xyzw', 'rgba'];

function editDistance(a, b) {
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
  }
  return d[a.length][b.length];
}
function suggest(name, candidates) {
  let best = null;
  let bestD = Infinity;
  for (const c of candidates) {
    const dd = editDistance(name, c);
    if (dd < bestD) { bestD = dd; best = c; }
  }
  return best && bestD <= Math.max(1, Math.floor(name.length / 3)) ? ` (did you mean '${best}'?)` : '';
}

class Scope {
  constructor(parent) { this.parent = parent; this.vars = new Map(); }
  lookup(name) { return this.vars.get(name) || (this.parent && this.parent.lookup(name)) || null; }
  names() { return [...this.vars.keys(), ...(this.parent ? this.parent.names() : [])]; }
}

export function check(prog, allNames = new Set()) {
  const taken = new Set([...allNames, ...RESERVED]);
  const outNames = new Map();
  // stable renaming for identifiers that collide with GLSL/WGSL/JS words
  function rename(name) {
    let out = name.replace(/__+/g, '_');
    while (taken.has(out) || out.startsWith('gl_') || out.startsWith('pyre_')) out = 'p_' + out + '_';
    taken.add(out);
    return out;
  }
  function mangle(name) {
    if (!RESERVED.has(name) && !name.startsWith('gl_') && !name.includes('__') && !name.startsWith('pyre_')) return name;
    if (outNames.has(name)) return outNames.get(name);
    let out = name.replace(/__+/g, '_');
    while (taken.has(out) || out.startsWith('gl_') || out.startsWith('pyre_')) out = 'p_' + out + '_';
    taken.add(out);
    outNames.set(name, out);
    return out;
  }

  const globals = new Scope(null);
  for (const [name, value] of Object.entries(BUILTIN_CONSTS)) {
    globals.vars.set(name, { kind: 'const', type: 'float', out: name, builtin: true, value });
  }
  const declareGlobal = (name, info, pos) => {
    if (globals.vars.has(name)) fail(`'${name}' is already declared`, pos);
    if (BUILTINS[name] || CONSTRUCTORS.has(name)) fail(`'${name}' is a built-in function name`, pos);
    globals.vars.set(name, info);
  };

  for (const u of prog.uniforms) {
    if (u.type.array || !UNIFORM_TYPES.includes(u.type.name)) {
      fail(`uniform '${u.name}' must be one of ${UNIFORM_TYPES.join(', ')}`, u.type.pos);
    }
    if (RESERVED.has(u.name) || u.name.startsWith('gl_') || u.name.includes('__')) fail(`'${u.name}' is a reserved word and cannot name a uniform`, u.pos);
    u.out = u.name;
    declareGlobal(u.name, { kind: 'uniform', type: u.type.name, out: u.name, node: u }, u.pos);
  }
  for (const b of prog.buffers) {
    if (!BUFFER_ELEMS.includes(b.elem)) fail(`buffer elements must be float or int, not '${b.elem}'`, b.pos);
    // hosts bind buffers by name (JS) or binding index (WGSL), so the
    // generated variable may be renamed freely
    b.out = mangle(b.name);
    declareGlobal(b.name, { kind: 'buffer', elem: b.elem, out: b.out, node: b }, b.pos);
  }

  const fns = new Map();
  for (const f of prog.functions) {
    if (fns.has(f.name)) fail(`function '${f.name}' is already defined`, f.pos);
    if (BUILTINS[f.name] || CONSTRUCTORS.has(f.name) || f.name === 'range') fail(`'${f.name}' is a built-in function name`, f.pos);
    if (globals.vars.has(f.name)) fail(`'${f.name}' is already declared`, f.pos);
    fns.set(f.name, f);
  }

  // ---------- type helpers ----------
  function coerce(e, to, what) {
    if (e.type === to) return e;
    if (e.type === 'int' && (to === 'float' || isVec(to))) {
      const f = e.kind === 'num' ? { ...e, isFloat: true, type: 'float' } : { kind: 'construct', ctor: 'float', args: [e], type: 'float', pos: e.pos };
      return to === 'float' ? f : coerce(f, to, what);
    }
    if (e.type === 'float' && isVec(to)) return { kind: 'construct', ctor: to, args: [e], type: to, pos: e.pos };
    return fail(`${what || 'value'} should be ${to} but is ${e.type}`, e.pos);
  }
  function unify(a, b, pos) {
    if (a === b) return a;
    const rank = (t) => (t === 'int' ? 0 : t === 'float' ? 1 : isVec(t) ? 2 : -1);
    if (rank(a) < 0 || rank(b) < 0) fail(`mismatched types ${a} and ${b}`, pos);
    if (isVec(a) && isVec(b)) fail(`mismatched vector sizes ${a} and ${b}`, pos);
    return rank(a) >= rank(b) ? a : b;
  }
  function typeName(t, allowVoid = false) {
    if (!t) return allowVoid ? 'void' : fail('missing type', null);
    if (t.array) fail('array types are only allowed for buffers', t.pos);
    if (!VALUE_TYPES.includes(t.name)) fail(`unknown type '${t.name}'${suggest(t.name, VALUE_TYPES)}`, t.pos);
    return t.name;
  }

  // ---------- constants ----------
  function isConstExpr(e) {
    switch (e.kind) {
      case 'num': case 'bool': return true;
      case 'ident': { const v = globals.lookup(e.name); return !!v && v.kind === 'const'; }
      case 'unary': return isConstExpr(e.arg);
      case 'binary': return isConstExpr(e.left) && isConstExpr(e.right);
      case 'construct': return e.args.every(isConstExpr);
      default: return false;
    }
  }
  for (const c of prog.consts) {
    let e = expr(c.init, globals, null);
    if (!isConstExpr(e)) fail(`const '${c.name}' must be a constant expression (numbers, other consts and constructors)`, c.init.pos);
    if (c.annot) e = coerce(e, typeName(c.annot), `const '${c.name}'`);
    c.init = e;
    c.type = e.type;
    c.out = mangle(c.name);
    declareGlobal(c.name, { kind: 'const', type: c.type, out: c.out, node: c }, c.pos);
  }
  function constInt(e) {
    if (e.kind === 'num' && !e.isFloat) return e.value;
    if (e.kind === 'unary' && e.op === '-') { const v = constInt(e.arg); return v === null ? null : -v; }
    if (e.kind === 'ident') {
      const v = globals.lookup(e.name);
      if (v && v.kind === 'const' && v.node && v.type === 'int') return constInt(v.node.init);
    }
    return null;
  }

  // ---------- signatures ----------
  for (const f of fns.values()) {
    f.out = mangle(f.name);
    f.retType = typeName(f.ret, true);
    const seen = new Set();
    f.paramTypes = f.params.map((p) => {
      if (seen.has(p.name)) fail(`duplicate parameter '${p.name}'`, p.pos);
      seen.add(p.name);
      return typeName(p.type);
    });
    f.calls = new Set();
    if (f.decorator) {
      const d = f.decorator;
      if (d.name === 'fragment') {
        if (d.args.length) fail('@fragment takes no arguments', d.pos);
        if (f.retType !== 'vec4') fail('a @fragment function must return vec4 (the pixel colour)', f.pos);
        f.params.forEach((p, i) => {
          const want = FRAGMENT_INPUTS[p.name];
          if (!want) fail(`@fragment parameter '${p.name}' is not a known input; use ${Object.keys(FRAGMENT_INPUTS).join(', ')}`, p.pos);
          if (want !== f.paramTypes[i]) fail(`@fragment input '${p.name}' has type ${want}`, p.pos);
        });
      } else if (d.name === 'kernel') {
        if (d.args.length > 1) fail('@kernel takes one argument: the workgroup size', d.pos);
        const size = d.args.length ? d.args[0] : 64;
        if (size < 1 || size > 256) fail('@kernel workgroup size must be between 1 and 256', d.pos);
        f.workgroupSize = size;
        if (f.retType !== 'void') fail('a @kernel function cannot return a value; write results into a buffer', f.pos);
        if (f.params.length !== 1 || f.paramTypes[0] !== 'int') fail('a @kernel function takes exactly one parameter: the thread index, e.g. (id: int)', f.pos);
      } else {
        fail(`unknown decorator '@${d.name}'; use @fragment or @kernel`, d.pos);
      }
    }
  }
  const fragments = prog.functions.filter((f) => f.decorator && f.decorator.name === 'fragment');
  const kernels = prog.functions.filter((f) => f.decorator && f.decorator.name === 'kernel');
  if (fragments.length > 1) fail('only one @fragment function is allowed per file', fragments[1].pos);
  if (kernels.length > 1) fail('only one @kernel function is allowed per file', kernels[1].pos);

  // ---------- expressions ----------
  function expr(e, scope, fn) {
    switch (e.kind) {
      case 'num': e.type = e.isFloat ? 'float' : 'int'; return e;
      case 'bool': e.type = 'bool'; return e;
      case 'ident': {
        const v = scope.lookup(e.name);
        if (!v) {
          if (fns.has(e.name) || BUILTINS[e.name]) fail(`'${e.name}' is a function; call it with ()`, e.pos);
          fail(`unknown name '${e.name}'${suggest(e.name, scope.names())}`, e.pos);
        }
        if (v.kind === 'buffer') fail(`buffer '${e.name}' must be indexed, e.g. ${e.name}[id]`, e.pos);
        e.ref = v;
        e.type = v.type;
        return e;
      }
      case 'unary': {
        e.arg = expr(e.arg, scope, fn);
        if (e.op === 'not') {
          if (e.arg.type !== 'bool') fail(`'not' needs a bool, got ${e.arg.type}`, e.pos);
          e.type = 'bool';
        } else {
          if (!isNumeric(e.arg.type)) fail(`cannot negate a ${e.arg.type}`, e.pos);
          e.type = e.arg.type;
        }
        return e;
      }
      case 'binary': return binary(e, scope, fn);
      case 'ternary': {
        e.cond = expr(e.cond, scope, fn);
        if (e.cond.type !== 'bool') fail(`condition must be bool, got ${e.cond.type}`, e.cond.pos);
        e.a = expr(e.a, scope, fn);
        e.b = expr(e.b, scope, fn);
        const t = unify(e.a.type, e.b.type, e.pos);
        e.a = coerce(e.a, t); e.b = coerce(e.b, t);
        e.type = t;
        return e;
      }
      case 'member': {
        e.object = expr(e.object, scope, fn);
        const t = e.object.type;
        if (!isVec(t)) fail(`type ${t} has no field '${e.prop}'`, e.pos);
        const set = SWIZZLE_SETS.find((s) => s.includes(e.prop[0]));
        if (!set || e.prop.length > 4 || [...e.prop].some((c) => !set.includes(c))) fail(`invalid swizzle '.${e.prop}' (use xyzw or rgba)`, e.pos);
        e.swizzle = [...e.prop].map((c) => set.indexOf(c));
        if (e.swizzle.some((i) => i >= dim(t))) fail(`swizzle '.${e.prop}' is out of range for ${t}`, e.pos);
        e.prop = e.swizzle.map((i) => 'xyzw'[i]).join('');
        e.type = vecOf(e.swizzle.length);
        return e;
      }
      case 'index': {
        if (e.object.kind === 'ident') {
          const v = scope.lookup(e.object.name);
          if (v && v.kind === 'buffer') {
            e.object.ref = v;
            e.object.type = `[${v.elem}]`;
            e.indexKind = 'buffer';
            e.type = v.elem;
          }
        }
        if (!e.indexKind) {
          e.object = expr(e.object, scope, fn);
          const t = e.object.type;
          if (isVec(t)) { e.indexKind = 'vec'; e.type = 'float'; }
          else if (isMat(t)) { e.indexKind = 'mat'; e.type = vecOf(dim(t)); }
          else fail(`type ${t} cannot be indexed`, e.pos);
        }
        e.index = expr(e.index, scope, fn);
        if (e.index.type !== 'int') fail(`index must be an int, got ${e.index.type}`, e.index.pos);
        if (e.indexKind !== 'buffer' && e.index.kind === 'num' && (e.index.value < 0 || e.index.value >= dim(e.object.type))) {
          fail(`index ${e.index.value} is out of range for ${e.object.type}`, e.index.pos);
        }
        return e;
      }
      case 'call': return call(e, scope, fn);
      default: return fail(`unexpected expression`, e.pos);
    }
  }

  function binary(e, scope, fn) {
    e.left = expr(e.left, scope, fn);
    e.right = expr(e.right, scope, fn);
    return binaryTyped(e);
  }

  function binaryTyped(e) {
    const op = e.op;
    let l = e.left.type;
    let r = e.right.type;
    if (op === 'and' || op === 'or') {
      if (l !== 'bool' || r !== 'bool') fail(`'${op}' needs bool operands, got ${l} and ${r}`, e.pos);
      e.type = 'bool';
      return e;
    }
    if (['==', '!=', '<', '>', '<=', '>='].includes(op)) {
      if ((op === '==' || op === '!=') && l === 'bool' && r === 'bool') { e.type = 'bool'; return e; }
      if (!['int', 'float'].includes(l) || !['int', 'float'].includes(r)) {
        fail(`'${op}' compares numbers (int or float), got ${l} and ${r}`, e.pos);
      }
      const t = unify(l, r, e.pos);
      e.left = coerce(e.left, t); e.right = coerce(e.right, t);
      e.type = 'bool';
      return e;
    }
    if (['&', '|', '^', '<<', '>>'].includes(op)) {
      if (l !== 'int' || r !== 'int') fail(`'${op}' works on ints, got ${l} and ${r}`, e.pos);
      e.type = 'int';
      return e;
    }
    // arithmetic
    if (!isNumeric(l) || !isNumeric(r)) fail(`'${op}' needs numbers, got ${l} and ${r}`, e.pos);
    if (l === 'int' && r === 'int') { e.type = 'int'; return e; }
    if (l === 'int') { e.left = coerce(e.left, 'float'); l = 'float'; }
    if (r === 'int') { e.right = coerce(e.right, 'float'); r = 'float'; }
    if (isMat(l) || isMat(r)) {
      if (op === '%') fail(`'%' is not defined for matrices`, e.pos);
      if (op === '*') {
        if (l === r) e.type = l;
        else if (isMat(l) && r === 'float') e.type = l;
        else if (l === 'float' && isMat(r)) e.type = r;
        else if (isMat(l) && isVec(r) && dim(l) === dim(r)) e.type = r;
        else if (isVec(l) && isMat(r) && dim(l) === dim(r)) e.type = l;
        else fail(`cannot multiply ${l} by ${r}`, e.pos);
      } else if ((op === '+' || op === '-') && l === r) e.type = l;
      else if (op === '/' && isMat(l) && r === 'float') e.type = l;
      else fail(`'${op}' is not defined for ${l} and ${r}`, e.pos);
      return e;
    }
    if (l === r) e.type = l;
    else if (l === 'float' && isVec(r)) e.type = r;
    else if (isVec(l) && r === 'float') e.type = l;
    else fail(`'${op}' needs matching types, got ${l} and ${r}`, e.pos);
    if (op === '%' && e.left.type !== e.right.type) {
      // mod() helpers take two values of the same type
      e.left = coerce(e.left, e.type); e.right = coerce(e.right, e.type);
    }
    return e;
  }

  function call(e, scope, fn) {
    const name = e.callee;
    e.args = e.args.map((a) => expr(a, scope, fn));
    const types = e.args.map((a) => a.type);
    if (fns.has(name)) {
      const f = fns.get(name);
      if (f.decorator) fail(`'${name}' is an entry point (@${f.decorator.name}) and cannot be called`, e.pos);
      if (e.args.length !== f.params.length) fail(`'${name}' takes ${f.params.length} argument(s), got ${e.args.length}`, e.pos);
      e.args = e.args.map((a, i) => coerce(a, f.paramTypes[i], `argument ${i + 1} of '${name}'`));
      if (fn) fn.calls.add(name);
      e.kind = 'ucall';
      e.fn = f;
      e.type = f.retType;
      return e;
    }
    if (CONSTRUCTORS.has(name)) return construct(e, name, types);
    const b = BUILTINS[name];
    if (!b) fail(`unknown function '${name}'${suggest(name, [...fns.keys(), ...Object.keys(BUILTINS), ...CONSTRUCTORS])}`, e.pos);
    if (e.args.length !== b.params.length) fail(`${name}() takes ${b.params.length} argument(s), got ${e.args.length}`, e.pos);
    // integer-only call (min/max/abs/sign/clamp/mod on ints)
    const tIdx = b.params.map((p, i) => (p === 'T' ? i : -1)).filter((i) => i >= 0);
    if (b.ints && tIdx.every((i) => types[i] === 'int')) {
      e.kind = 'bcall';
      e.type = b.ret === 'T' ? 'int' : b.ret;
      e.generic = 'int';
      return e;
    }
    let T = 'float';
    for (const i of tIdx) {
      const t = types[i];
      if (!['int', 'float'].includes(t) && !isVec(t)) fail(`${name}() argument ${i + 1} must be a float or vector, got ${t}`, e.args[i].pos);
      if (isVec(t)) {
        if (isVec(T) && T !== t) fail(`${name}() arguments mix ${T} and ${t}`, e.args[i].pos);
        T = t;
      }
    }
    e.args = e.args.map((a, i) => {
      const p = b.params[i];
      const want = p === 'T' ? T : p === 'f' ? 'float' : p;
      return coerce(a, want, `argument ${i + 1} of ${name}()`);
    });
    e.kind = 'bcall';
    e.generic = T;
    e.type = b.ret === 'T' ? T : b.ret;
    return e;
  }

  function construct(e, name, types) {
    e.kind = 'construct';
    e.ctor = name;
    e.type = name;
    if (name === 'float' || name === 'int' || name === 'bool') {
      if (e.args.length !== 1 || !['int', 'float', 'bool'].includes(types[0])) fail(`${name}() takes one int, float or bool`, e.pos);
      if (types[0] === name) return e.args[0];
      return e;
    }
    if (!e.args.length) fail(`${name}() needs at least one argument`, e.pos);
    for (let i = 0; i < e.args.length; i++) {
      if (e.args[i].type === 'int') e.args[i] = coerce(e.args[i], 'float');
      const t = e.args[i].type;
      if (!(t === 'float' || isVec(t))) fail(`${name}() arguments must be floats or vectors, got ${t}`, e.args[i].pos);
    }
    const n = dim(name);
    const total = e.args.reduce((s, a) => s + dim(a.type), 0);
    if (isVec(name)) {
      if (e.args.length === 1 && e.args[0].type === 'float') return e;
      if (total !== n) fail(`${name}() needs ${n} components in total, got ${total}`, e.pos);
      return e;
    }
    // matrices: one float (diagonal), n*n floats, or n column vectors
    if (e.args.length === 1 && e.args[0].type === 'float') return e;
    const cols = e.args.every((a) => a.type === vecOf(n)) && e.args.length === n;
    const flat = e.args.every((a) => a.type === 'float') && e.args.length === n * n;
    if (!cols && !flat) fail(`${name}() takes one float, ${n * n} floats or ${n} ${vecOf(n)} columns`, e.pos);
    return e;
  }

  // ---------- statements ----------
  function lvalue(t, scope, fn) {
    if (t.kind === 'ident') {
      const v = scope.lookup(t.name);
      if (!v) fail(`unknown name '${t.name}'${suggest(t.name, scope.names())}`, t.pos);
      if (v.kind === 'buffer') fail(`buffer '${t.name}' must be indexed, e.g. ${t.name}[id] = ...`, t.pos);
      if (!v.mutable) fail(`cannot assign to '${t.name}': ${v.kind === 'local' ? "declare it with 'var' instead of 'let'" : `it is a ${v.kind === 'param' ? 'parameter' : v.kind === 'loop' ? 'loop variable' : v.kind}`}`, t.pos);
      return expr(t, scope, fn);
    }
    if (t.kind === 'member') {
      const target = expr(t, scope, fn);
      if (target.swizzle.length !== 1) fail('only single components can be assigned (e.g. v.x = 1.0)', t.pos);
      lvalue(t.object, scope, fn);
      return target;
    }
    if (t.kind === 'index') {
      const target = expr(t, scope, fn);
      if (target.indexKind === 'mat') fail('matrix columns are read-only; build a new matrix instead', t.pos);
      if (target.indexKind !== 'buffer') lvalue(t.object, scope, fn);
      return target;
    }
    return fail('can only assign to variables, vector components and buffer elements', t.pos);
  }

  function declareLocal(scope, name, info, pos) {
    if (scope.vars.has(name)) fail(`'${name}' is already declared in this block`, pos);
    if (name in BUILTIN_CONSTS) fail(`'${name}' is a built-in constant`, pos);
    // a local named like a function would hide that function in GLSL
    info.out = fns.has(name) || BUILTINS[name] || CONSTRUCTORS.has(name) ? rename(name) : mangle(name);
    scope.vars.set(name, info);
    return info.out;
  }

  function block(stmts, scope, fn, loops) {
    for (const s of stmts) stmt(s, scope, fn, loops);
  }

  function stmt(s, scope, fn, loops) {
    switch (s.kind) {
      case 'let':
      case 'var': {
        let t = s.annot ? typeName(s.annot) : null;
        if (s.init) {
          s.init = expr(s.init, scope, fn);
          if (s.init.type === 'void') fail(`'${s.init.callee}' does not return a value`, s.init.pos);
          if (t) s.init = coerce(s.init, t, `'${s.name}'`);
          else t = s.init.type;
        }
        s.type = t;
        s.out = declareLocal(scope, s.name, { kind: 'local', type: t, mutable: s.kind === 'var' }, s.pos);
        return;
      }
      case 'assign': {
        s.target = lvalue(s.target, scope, fn);
        s.value = expr(s.value, scope, fn);
        const tt = s.target.type;
        if (s.op === '=') {
          s.value = coerce(s.value, tt, 'assigned value');
        } else {
          const probe = binaryTyped({ kind: 'binary', op: s.op[0], left: s.target, right: s.value, pos: s.pos });
          if (probe.type !== tt) fail(`'${s.op}' would turn ${tt} into ${probe.type}`, s.pos);
          s.value = probe.right;
        }
        return;
      }
      case 'expr':
        if (s.expr.kind !== 'call') fail('this expression does nothing; did you mean to assign it?', s.pos);
        s.expr = expr(s.expr, scope, fn);
        return;
      case 'if':
        for (const br of s.branches) {
          br.cond = expr(br.cond, scope, fn);
          if (br.cond.type !== 'bool') fail(`if condition must be bool, got ${br.cond.type}`, br.cond.pos);
          block(br.body, new Scope(scope), fn, loops);
        }
        if (s.otherwise) block(s.otherwise, new Scope(scope), fn, loops);
        return;
      case 'for': {
        s.start = expr(s.start, scope, fn);
        s.end = expr(s.end, scope, fn);
        if (s.start.type !== 'int' || s.end.type !== 'int') fail('range() bounds must be ints', s.pos);
        s.stepValue = 1;
        if (s.step) {
          const v = constInt(s.step);
          if (v === null) fail('range() step must be a constant int', s.step.pos);
          if (v === 0) fail('range() step cannot be 0', s.step.pos);
          s.stepValue = v;
        }
        const inner = new Scope(scope);
        s.out = declareLocal(inner, s.name, { kind: 'loop', type: 'int', mutable: false }, s.pos);
        block(s.body, inner, fn, loops + 1);
        return;
      }
      case 'while':
        s.cond = expr(s.cond, scope, fn);
        if (s.cond.type !== 'bool') fail(`while condition must be bool, got ${s.cond.type}`, s.cond.pos);
        block(s.body, new Scope(scope), fn, loops + 1);
        return;
      case 'return':
        if (fn.retType === 'void') {
          if (s.value) fail(`'${fn.name}' does not return a value`, s.pos);
        } else {
          if (!s.value) fail(`'${fn.name}' must return a ${fn.retType}`, s.pos);
          s.value = coerce(expr(s.value, scope, fn), fn.retType, 'return value');
        }
        return;
      case 'break':
      case 'continue':
        if (!loops) fail(`'${s.kind}' outside a loop`, s.pos);
        return;
      case 'pass':
        return;
      default:
        fail('unknown statement', s.pos);
    }
  }

  const alwaysReturns = (stmts) => stmts.some((s) => s.kind === 'return'
    || (s.kind === 'if' && s.otherwise && s.branches.every((b) => alwaysReturns(b.body)) && alwaysReturns(s.otherwise)));

  for (const f of fns.values()) {
    const scope = new Scope(globals);
    f.paramOut = f.params.map((p, i) => declareLocal(scope, p.name, { kind: 'param', type: f.paramTypes[i], mutable: false }, p.pos));
    block(f.body, scope, f, 0);
    if (f.retType !== 'void' && !alwaysReturns(f.body)) fail(`'${f.name}' can reach the end without returning a ${f.retType}`, f.pos);
  }

  // ---------- call graph ----------
  const state = new Map();
  const order = [];
  function visit(f, stack) {
    if (state.get(f.name) === 'done') return;
    if (state.get(f.name) === 'active') {
      const cycle = [...stack.slice(stack.indexOf(f.name)), f.name].join(' -> ');
      fail(`recursion is not supported on the GPU: ${cycle}`, f.pos);
    }
    state.set(f.name, 'active');
    for (const c of f.calls) visit(fns.get(c), [...stack, f.name]);
    state.set(f.name, 'done');
    order.push(f);
  }
  // check user functions for recursion even when unreachable
  for (const f of fns.values()) if (!f.std) visit(f, []);
  for (const f of fns.values()) visit(f, []);

  const reachable = (roots) => {
    const seen = new Set();
    const walk = (f) => { if (seen.has(f.name)) return; seen.add(f.name); for (const c of f.calls) walk(fns.get(c)); };
    roots.forEach(walk);
    return order.filter((f) => seen.has(f.name));
  };

  return {
    program: prog,
    fragment: fragments[0] || null,
    kernel: kernels[0] || null,
    order,
    reachable,
    uniforms: prog.uniforms,
    buffers: prog.buffers,
    consts: prog.consts,
  };
}

