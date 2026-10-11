// Pyre parser: tokens -> AST (recursive descent).
import { tokenize } from './lexer.js';
import { fail } from './errors.js';

export const MAX_NESTING = 64;

export function parse(src) {
  const toks = tokenize(src);
  let p = 0;
  let nesting = 0;

  const peek = (o = 0) => toks[p + o];
  const at = (type) => toks[p].type === type;
  const pos = (t = toks[p]) => ({ line: t.line, col: t.col });
  const next = () => toks[p++];
  const describe = (t) => (t.type === 'IDENT' ? `'${t.value}'` : t.type === 'INT' || t.type === 'FLOAT' ? `number ${t.value}` : t.type === 'NEWLINE' ? 'end of line' : t.type === 'INDENT' ? 'indented block' : t.type === 'DEDENT' ? 'end of block' : t.type === 'EOF' ? 'end of file' : `'${t.type}'`);
  const expect = (type, what) => {
    if (!at(type)) fail(`expected ${what || `'${type}'`} but found ${describe(peek())}`, pos());
    return next();
  };
  const accept = (type) => (at(type) ? next() : null);
  const ident = (what = 'a name') => expect('IDENT', what);
  const enter = () => { if (++nesting > MAX_NESTING) fail('expression or block nested too deeply', pos()); };
  const leave = () => { nesting--; };

  function parseType() {
    const t = peek();
    if (accept('[')) {
      const elem = ident('an element type').value;
      expect(']');
      return { array: elem, pos: pos(t) };
    }
    return { name: ident('a type').value, pos: pos(t) };
  }

  function program() {
    const prog = { uniforms: [], buffers: [], consts: [], functions: [] };
    while (!at('EOF')) {
      if (accept('NEWLINE')) continue;
      const start = peek();
      if (accept('uniform')) {
        const name = ident();
        expect(':');
        prog.uniforms.push({ kind: 'uniform', name: name.value, type: parseType(), pos: pos(name) });
        expect('NEWLINE', 'end of line');
      } else if (accept('buffer')) {
        const name = ident();
        expect(':');
        const type = parseType();
        if (!type.array) fail(`buffer '${name.value}' must have an array type like [float]`, type.pos);
        prog.buffers.push({ kind: 'buffer', name: name.value, elem: type.array, pos: pos(name) });
        expect('NEWLINE', 'end of line');
      } else if (accept('const')) {
        const name = ident();
        let type = null;
        if (accept(':')) type = parseType();
        expect('=');
        prog.consts.push({ kind: 'const', name: name.value, annot: type, init: expr(), pos: pos(name) });
        expect('NEWLINE', 'end of line');
      } else if (at('@') || at('fn')) {
        prog.functions.push(func());
      } else {
        fail(`expected 'fn', 'uniform', 'buffer' or 'const' at top level but found ${describe(start)}`, pos(start));
      }
    }
    return prog;
  }

  function func() {
    let decorator = null;
    if (accept('@')) {
      const d = ident('a decorator name');
      decorator = { name: d.value, args: [], pos: pos(d) };
      if (accept('(')) {
        if (!at(')')) {
          do { decorator.args.push(expect('INT', 'an integer').value); } while (accept(','));
        }
        expect(')');
      }
      expect('NEWLINE', 'end of line after decorator');
    }
    const kw = expect('fn', "'fn'");
    const name = ident('a function name');
    expect('(');
    const params = [];
    if (!at(')')) {
      do {
        const pn = ident('a parameter name');
        expect(':', `':' and a type after parameter '${pn.value}'`);
        params.push({ name: pn.value, type: parseType(), pos: pos(pn) });
      } while (accept(','));
    }
    expect(')');
    let ret = null;
    if (accept('->')) ret = parseType();
    expect(':', "':' at the end of the function header");
    return { kind: 'fn', name: name.value, params, ret, decorator, body: block(), pos: pos(kw) };
  }

  function block() {
    expect('NEWLINE', 'end of line');
    expect('INDENT', 'an indented block');
    enter();
    const body = [];
    while (!at('DEDENT') && !at('EOF')) {
      if (accept('NEWLINE')) continue;
      body.push(statement());
    }
    accept('DEDENT');
    leave();
    return body;
  }

  function statement() {
    const t = peek();
    const sp = pos(t);
    switch (t.type) {
      case 'let':
      case 'var': {
        next();
        const name = ident();
        let annot = null;
        if (accept(':')) annot = parseType();
        let init = null;
        if (accept('=')) init = expr();
        else if (t.type === 'let') fail(`'let ${name.value}' needs a value`, pos());
        else if (!annot) fail(`'var ${name.value}' needs a type or a value`, pos());
        expect('NEWLINE', 'end of line');
        return { kind: t.type, name: name.value, annot, init, pos: pos(name) };
      }
      case 'if': {
        next();
        const branches = [{ cond: expr(), body: (expect(':'), block()) }];
        let otherwise = null;
        while (at('elif')) { next(); branches.push({ cond: expr(), body: (expect(':'), block()) }); }
        if (accept('else')) { expect(':'); otherwise = block(); }
        return { kind: 'if', branches, otherwise, pos: sp };
      }
      case 'for': {
        next();
        const v = ident('a loop variable');
        expect('in');
        const r = ident("'range'");
        if (r.value !== 'range') fail("for loops iterate over range(start, end[, step])", pos(r));
        expect('(');
        const args = [expr()];
        while (accept(',')) args.push(expr());
        expect(')');
        if (args.length > 3) fail('range() takes at most 3 arguments', pos(r));
        const [a, b, c] = args.length === 1 ? [{ kind: 'num', value: 0, isFloat: false, pos: pos(r) }, args[0]] : args;
        expect(':');
        return { kind: 'for', name: v.value, start: a, end: b, step: c || null, body: block(), pos: pos(v) };
      }
      case 'while': {
        next();
        const cond = expr();
        expect(':');
        return { kind: 'while', cond, body: block(), pos: sp };
      }
      case 'return': {
        next();
        const value = at('NEWLINE') ? null : expr();
        expect('NEWLINE', 'end of line');
        return { kind: 'return', value, pos: sp };
      }
      case 'break':
      case 'continue':
      case 'pass':
        next();
        expect('NEWLINE', 'end of line');
        return { kind: t.type, pos: sp };
      default: {
        const e = expr();
        const op = peek().type;
        if (['=', '+=', '-=', '*=', '/='].includes(op)) {
          next();
          const value = expr();
          expect('NEWLINE', 'end of line');
          return { kind: 'assign', target: e, op, value, pos: sp };
        }
        expect('NEWLINE', 'end of line');
        return { kind: 'expr', expr: e, pos: sp };
      }
    }
  }

  // ---- expressions, lowest precedence first ----
  function expr() {
    enter();
    const sp = pos();
    const e = orExpr();
    let result = e;
    if (at('if')) {
      next();
      const cond = orExpr();
      expect('else', "'else' in conditional expression");
      result = { kind: 'ternary', cond, a: e, b: expr(), pos: sp };
    }
    leave();
    return result;
  }
  function binaryLevel(sub, ops) {
    return function level() {
      let left = sub();
      while (ops.includes(peek().type)) {
        const op = next();
        left = { kind: 'binary', op: op.type, left, right: sub(), pos: pos(op) };
      }
      return left;
    };
  }
  function notExpr() {
    if (at('not')) {
      const t = next();
      enter();
      const arg = notExpr();
      leave();
      return { kind: 'unary', op: 'not', arg, pos: pos(t) };
    }
    return comparison();
  }
  function comparison() {
    const left = bitOr();
    if (['==', '!=', '<', '>', '<=', '>='].includes(peek().type)) {
      const op = next();
      const right = bitOr();
      if (['==', '!=', '<', '>', '<=', '>='].includes(peek().type)) fail('comparisons cannot be chained; use "and"', pos());
      return { kind: 'binary', op: op.type, left, right, pos: pos(op) };
    }
    return left;
  }
  function unary() {
    if (at('-') || at('+')) {
      const t = next();
      enter();
      const arg = unary();
      leave();
      if (t.type === '+') return arg;
      if (arg.kind === 'num') return { ...arg, value: -arg.value, pos: pos(t) };
      return { kind: 'unary', op: '-', arg, pos: pos(t) };
    }
    return power();
  }
  function power() {
    const base = postfix();
    if (at('**')) {
      const t = next();
      enter();
      const exp = unary();
      leave();
      return { kind: 'call', callee: 'pow', args: [base, exp], pos: pos(t) };
    }
    return base;
  }
  function postfix() {
    let e = primary();
    for (;;) {
      if (at('(')) {
        const t = next();
        if (e.kind !== 'ident') fail('only named functions can be called', pos(t));
        const args = [];
        if (!at(')')) {
          do { args.push(expr()); } while (accept(','));
        }
        expect(')');
        e = { kind: 'call', callee: e.name, args, pos: e.pos };
      } else if (at('.')) {
        next();
        const m = ident('a field name');
        e = { kind: 'member', object: e, prop: m.value, pos: pos(m) };
      } else if (at('[')) {
        const t = next();
        const index = expr();
        expect(']');
        e = { kind: 'index', object: e, index, pos: pos(t) };
      } else {
        return e;
      }
    }
  }
  function primary() {
    const t = peek();
    switch (t.type) {
      case 'INT': next(); return { kind: 'num', value: t.value, isFloat: false, pos: pos(t) };
      case 'FLOAT': next(); return { kind: 'num', value: t.value, isFloat: true, pos: pos(t) };
      case 'true': case 'false': next(); return { kind: 'bool', value: t.type === 'true', pos: pos(t) };
      case 'IDENT': next(); return { kind: 'ident', name: t.value, pos: pos(t) };
      case '(': {
        next();
        const e = expr();
        expect(')');
        return e;
      }
      default:
        return fail(`expected an expression but found ${describe(t)}`, pos(t));
    }
  }

  const sum = binaryLevel(binaryLevel(unary, ['*', '/', '%']), ['+', '-']);
  const bitOr = binaryLevel(binaryLevel(binaryLevel(binaryLevel(sum, ['<<', '>>']), ['&']), ['^']), ['|']);
  const andExpr = binaryLevel(notExpr, ['and']);
  const orExpr = binaryLevel(andExpr, ['or']);

  return program();
}
