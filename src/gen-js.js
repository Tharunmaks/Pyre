// JavaScript generator: runs Pyre on the CPU (Node.js or any browser) so
// kernels and shaders work without a GPU and can be unit-tested.
//
// The output is only ever built from checked AST nodes: identifiers match
// [A-Za-z_][A-Za-z0-9_]* and are renamed away from JS keywords, numbers are
// printed with String(number), and the language has no strings. Generated
// code can therefore only reference its own locals and the `$R`, `$U`, `$B`
// parameters. Every loop iteration also ticks a budget so a runaway `while`
// cannot hang the host.
import { fail } from './errors.js';
import { BUILTIN_CONSTS, dim, isMat } from './types.js';

const SCALAR = new Set(['int', 'float', 'bool']);

export function genJS(info) {
  const out = ['"use strict";', 'let $n = 0;'];
  const tick = 'if (++$n > $LIMIT) $R.limit();';

  for (const k of Object.keys(BUILTIN_CONSTS)) out.push(`const ${k} = $R.${k};`);
  for (const c of info.consts) out.push(`const ${c.out} = ${expr(c.init)};`);

  // entry points plus every user function (so hosts can call helpers), and
  // whatever library functions those reach, callee before caller
  const fns = info.reachable(info.order.filter((f) => !f.std));

  for (const f of fns) {
    out.push('', `function ${f.out}(${f.paramOut.join(', ')}) {`);
    block(f.body, 1);
    out.push('}');
  }

  const exportsList = [];
  if (info.fragment) {
    const args = info.fragment.params.map((p) => `$in.${p.name}`).join(', ');
    exportsList.push(`fragment: function ($in) { $n = 0; return ${info.fragment.out}(${args}); }`);
  } else exportsList.push('fragment: null');
  if (info.kernel) exportsList.push(`kernel: function (id) { $n = 0; ${info.kernel.out}(id); }`);
  else exportsList.push('kernel: null');
  const callable = fns.filter((f) => !f.decorator && !f.std).map((f) => `${f.name}: function (...a) { $n = 0; return ${f.out}(...a); }`);
  exportsList.push(`functions: { ${callable.join(', ')} }`);
  out.push('', `return { ${exportsList.join(', ')} };`, '');
  return out.join('\n');

  function block(stmts, depth) {
    const ind = '  '.repeat(depth);
    for (const s of stmts) {
      switch (s.kind) {
        case 'let':
          // values that may alias a mutable vector are copied
          out.push(`${ind}const ${s.out} = ${['ident', 'ternary', 'ucall'].includes(s.init.kind) ? copy(expr(s.init), s.type) : expr(s.init)};`);
          break;
        case 'var':
          out.push(`${ind}let ${s.out} = ${s.init ? copy(expr(s.init), s.type) : `$R.zero('${s.type}')`};`);
          break;
        case 'assign':
          out.push(`${ind}${assign(s)};`);
          break;
        case 'expr':
          out.push(`${ind}${expr(s.expr)};`);
          break;
        case 'if':
          s.branches.forEach((b, i) => {
            out.push(`${ind}${i ? '} else if' : 'if'} (${expr(b.cond)}) {`);
            block(b.body, depth + 1);
          });
          if (s.otherwise) { out.push(`${ind}} else {`); block(s.otherwise, depth + 1); }
          out.push(`${ind}}`);
          break;
        case 'for': {
          const cmp = s.stepValue > 0 ? '<' : '>';
          out.push(`${ind}for (let ${s.out} = ${expr(s.start)}; ${s.out} ${cmp} ${expr(s.end)}; ${s.out} += ${s.stepValue}) {`);
          out.push(`${ind}  ${tick}`);
          block(s.body, depth + 1);
          out.push(`${ind}}`);
          break;
        }
        case 'while':
          out.push(`${ind}while (${expr(s.cond)}) {`);
          out.push(`${ind}  ${tick}`);
          block(s.body, depth + 1);
          out.push(`${ind}}`);
          break;
        case 'return':
          out.push(`${ind}return${s.value ? ' ' + expr(s.value) : ''};`);
          break;
        case 'break':
        case 'continue':
          out.push(`${ind}${s.kind};`);
          break;
        default:
          break;
      }
    }
  }

  function copy(code, type) {
    return SCALAR.has(type) ? code : `$R.copy(${code})`;
  }

  function assign(s) {
    const t = s.target;
    const value = s.op === '='
      ? copy(expr(s.value), t.type)
      : binop(s.op[0], expr(t), t.type, expr(s.value), s.value.type, t.type);
    if (t.kind === 'index' && t.indexKind === 'buffer') return `$R.st($B.${t.object.ref.node.name}, ${expr(t.index)}, ${value})`;
    if (t.kind === 'index') return `$R.set(${expr(t.object)}, ${expr(t.index)}, ${value})`;
    if (t.kind === 'member') return `${expr(t.object)}[${t.swizzle[0]}] = ${value}`;
    return `${expr(t)} = ${value}`;
  }

  function binop(op, a, ta, b, tb, type) {
    if (op === 'and') return `(${a} && ${b})`;
    if (op === 'or') return `(${a} || ${b})`;
    if (op === '==') return `(${a} === ${b})`;
    if (op === '!=') return `(${a} !== ${b})`;
    if (['<', '>', '<=', '>='].includes(op)) return `(${a} ${op} ${b})`;
    if (['&', '|', '^', '<<', '>>'].includes(op)) return `(${a} ${op} ${b})`;
    if (type === 'int') {
      if (op === '*') return `Math.imul(${a}, ${b})`;
      if (op === '/') return `$R.idiv(${a}, ${b})`;
      if (op === '%') return `$R.imod(${a}, ${b})`;
      return `((${a} ${op} ${b}) | 0)`;
    }
    if (type === 'float' && ta === 'float' && tb === 'float') {
      if (op === '%') return `$R.mod(${a}, ${b})`;
      return `(${a} ${op} ${b})`;
    }
    if (op === '*' && (isMat(ta) || isMat(tb))) return `$R.mmul(${a}, ${b}, '${ta}', '${tb}')`;
    const fn = { '+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '%': 'mod' }[op];
    return `$R.${fn}(${a}, ${b})`;
  }

  function expr(e) {
    switch (e.kind) {
      case 'num': return Object.is(e.value, -0) ? '-0' : String(e.value);
      case 'bool': return String(e.value);
      case 'ident':
        if (e.ref.kind === 'uniform') return `$U.${e.ref.node.name}`;
        return e.ref.out;
      case 'unary':
        if (e.op === 'not') return `(!${expr(e.arg)})`;
        if (e.type === 'int') return `((- ${expr(e.arg)}) | 0)`;
        if (e.type === 'float') return `(- ${expr(e.arg)})`;
        return `$R.neg(${expr(e.arg)})`;
      case 'binary': return binop(e.op, expr(e.left), e.left.type, expr(e.right), e.right.type, e.type);
      case 'ternary': return `(${expr(e.cond)} ? ${expr(e.a)} : ${expr(e.b)})`;
      case 'member':
        if (e.swizzle.length === 1) return `${expr(e.object)}[${e.swizzle[0]}]`;
        return `$R.sw(${expr(e.object)}, [${e.swizzle.join(', ')}])`;
      case 'index':
        if (e.indexKind === 'buffer') return `$R.ld($B.${e.object.ref.node.name}, ${expr(e.index)})`;
        if (e.indexKind === 'mat') return `$R.col(${expr(e.object)}, ${expr(e.index)}, ${dim(e.object.type)})`;
        return `$R.at(${expr(e.object)}, ${expr(e.index)})`;
      case 'ucall': return `${e.fn.out}(${e.args.map(expr).join(', ')})`;
      case 'bcall': {
        const args = e.args.map(expr).join(', ');
        if (e.callee === 'mod' && e.generic === 'int') return `$R.imod(${args})`;
        return `$R.${e.callee}(${args})`;
      }
      case 'construct': {
        const args = e.args.map(expr);
        const from = e.args[0].type;
        if (e.ctor === 'float') return from === 'bool' ? `(${args[0]} ? 1 : 0)` : args[0];
        if (e.ctor === 'int') return `$R.toInt(${args[0]})`;
        if (e.ctor === 'bool') return `(${args[0]} != 0)`;
        if (isMat(e.ctor)) return `$R.mat(${dim(e.ctor)}, ${args.join(', ')})`;
        return `$R.vec(${dim(e.ctor)}, ${args.join(', ')})`;
      }
      default: return fail('cannot generate JS for this expression', e.pos);
    }
  }
}
