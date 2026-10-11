// Pyre lexer: Python-style significant indentation, producing INDENT/DEDENT
// tokens. Newlines inside (), [] are ignored so long expressions can wrap.
import { fail } from './errors.js';

export const KEYWORDS = new Set([
  'fn', 'let', 'var', 'const', 'if', 'elif', 'else', 'for', 'in', 'while',
  'return', 'break', 'continue', 'pass', 'and', 'or', 'not', 'true', 'false',
  'uniform', 'buffer',
]);

const OPERATORS = [
  '**', '->', '==', '!=', '<=', '>=', '+=', '-=', '*=', '/=', '<<', '>>',
  '+', '-', '*', '/', '%', '<', '>', '=', '&', '|', '^', '(', ')', '[', ']', ',', ':', '.', '@',
];

export const MAX_SOURCE_BYTES = 256 * 1024;

export function tokenize(src) {
  if (typeof src !== 'string') fail('source must be a string', null);
  if (src.length > MAX_SOURCE_BYTES) fail(`source is larger than ${MAX_SOURCE_BYTES} bytes`, { line: 1, col: 1 });
  const tokens = [];
  const indents = [0];
  let i = 0;
  let line = 1;
  let col = 1;
  let depth = 0; // bracket nesting
  let atLineStart = true;

  const push = (type, value, l, c) => tokens.push({ type, value, line: l, col: c });

  while (i < src.length) {
    if (atLineStart && depth === 0) {
      // measure indentation of this logical line
      let width = 0;
      let j = i;
      while (j < src.length && (src[j] === ' ' || src[j] === '\t')) {
        width += src[j] === '\t' ? 4 : 1;
        j++;
      }
      // blank or comment-only lines don't affect indentation
      if (j >= src.length || src[j] === '\n' || src[j] === '\r' || src[j] === '#') {
        while (j < src.length && src[j] !== '\n') j++;
        col += j - i;
        i = j;
        if (i < src.length) { i++; line++; col = 1; }
        continue;
      }
      col += j - i;
      i = j;
      const top = indents[indents.length - 1];
      if (width > top) {
        indents.push(width);
        push('INDENT', null, line, col);
      } else {
        while (width < indents[indents.length - 1]) {
          indents.pop();
          push('DEDENT', null, line, col);
        }
        if (width !== indents[indents.length - 1]) fail('inconsistent indentation', { line, col });
      }
      atLineStart = false;
    }

    const ch = src[i];
    if (ch === '\n') {
      if (depth === 0) push('NEWLINE', null, line, col);
      i++; line++; col = 1;
      atLineStart = depth === 0; // lines inside brackets continue the current one
      continue;
    }
    if (ch === ' ' || ch === '\t' || ch === '\r') { i++; col++; continue; }
    if (ch === '#') {
      while (i < src.length && src[i] !== '\n') { i++; col++; }
      continue;
    }
    if (ch === '\\' && src[i + 1] === '\n') { i += 2; line++; col = 1; continue; }

    const startCol = col;
    // numbers: 12, 1.5, .5, 1e-3, 0x1F
    if (/[0-9]/.test(ch) || (ch === '.' && /[0-9]/.test(src[i + 1] || ''))) {
      let j = i;
      let isFloat = false;
      if (ch === '0' && (src[i + 1] === 'x' || src[i + 1] === 'X')) {
        j += 2;
        while (j < src.length && /[0-9a-fA-F_]/.test(src[j])) j++;
      } else {
        while (j < src.length && /[0-9_]/.test(src[j])) j++;
        if (src[j] === '.' && /[0-9]/.test(src[j + 1] || '')) {
          isFloat = true; j++;
          while (j < src.length && /[0-9_]/.test(src[j])) j++;
        } else if (src[j] === '.' && !/[a-zA-Z_]/.test(src[j + 1] || '')) {
          isFloat = true; j++; // "1." is a float
        }
        if (src[j] === 'e' || src[j] === 'E') {
          let k = j + 1;
          if (src[k] === '+' || src[k] === '-') k++;
          if (/[0-9]/.test(src[k] || '')) {
            isFloat = true; j = k;
            while (j < src.length && /[0-9]/.test(src[j])) j++;
          }
        }
      }
      const text = src.slice(i, j).replace(/_/g, '');
      const value = Number(text);
      if (!Number.isFinite(value)) fail(`invalid number '${text}'`, { line, col: startCol });
      if (!isFloat && (value > 2147483647)) fail(`integer literal ${text} does not fit in 32 bits`, { line, col: startCol });
      push(isFloat ? 'FLOAT' : 'INT', value, line, startCol);
      col += j - i; i = j;
      continue;
    }
    if (/[a-zA-Z_]/.test(ch)) {
      let j = i;
      while (j < src.length && /[a-zA-Z0-9_]/.test(src[j])) j++;
      const word = src.slice(i, j);
      push(KEYWORDS.has(word) ? word : 'IDENT', word, line, startCol);
      col += j - i; i = j;
      continue;
    }
    const op = OPERATORS.find((o) => src.startsWith(o, i));
    if (op) {
      if (op === '(' || op === '[') depth++;
      if (op === ')' || op === ']') depth = Math.max(0, depth - 1);
      push(op, op, line, startCol);
      i += op.length; col += op.length;
      continue;
    }
    fail(`unexpected character '${ch}'`, { line, col });
  }
  if (tokens.length && tokens[tokens.length - 1].type !== 'NEWLINE') push('NEWLINE', null, line, col);
  while (indents.length > 1) { indents.pop(); push('DEDENT', null, line, col); }
  push('EOF', null, line, col);
  return tokens;
}
