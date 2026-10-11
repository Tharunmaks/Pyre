// Pyre diagnostics: every compiler error carries a source position and can
// render itself with a caret under the offending column.

export class PyreError extends Error {
  constructor(message, pos, source, filename) {
    super(message);
    this.name = 'PyreError';
    this.line = pos ? pos.line : 0;
    this.col = pos ? pos.col : 0;
    this.source = source || '';
    this.filename = filename || '<input>';
  }

  format() {
    const lines = this.source.split('\n');
    const text = lines[this.line - 1];
    let out = `${this.filename}:${this.line}:${this.col}: error: ${this.message}`;
    if (text !== undefined && this.line > 0) {
      const gutter = String(this.line).padStart(4) + ' | ';
      out += '\n' + gutter + text + '\n' + ' '.repeat(gutter.length + Math.max(0, this.col - 1)) + '^';
    }
    return out;
  }

  toJSON() {
    return { message: this.message, line: this.line, col: this.col };
  }
}

export function fail(message, pos) {
  throw new PyreError(message, pos);
}
