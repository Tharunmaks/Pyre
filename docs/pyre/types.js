// Pyre type system and built-in function signatures.

export const SCALARS = ['bool', 'int', 'float'];
export const VECTORS = ['vec2', 'vec3', 'vec4'];
export const MATRICES = ['mat2', 'mat3', 'mat4'];
export const VALUE_TYPES = [...SCALARS, ...VECTORS, ...MATRICES];
export const BUFFER_ELEMS = ['float', 'int'];

export const isVec = (t) => VECTORS.includes(t);
export const isMat = (t) => MATRICES.includes(t);
export const isNumeric = (t) => t === 'int' || t === 'float' || isVec(t) || isMat(t);
export const isFloaty = (t) => t === 'float' || isVec(t);
export const dim = (t) => (isVec(t) || isMat(t) ? Number(t[3]) : 1);
export const vecOf = (n) => (n === 1 ? 'float' : `vec${n}`);

export const BUILTIN_CONSTS = Object.assign(Object.create(null), {
  PI: Math.PI,
  TAU: Math.PI * 2,
  E: Math.E,
});

// Fragment entry parameters the host provides.
export const FRAGMENT_INPUTS = Object.assign(Object.create(null), {
  uv: 'vec2', // 0..1 texture coordinate (y up)
  frag: 'vec2', // pixel coordinate; y grows up in GLSL/CPU, down in WGSL (platform convention)
  normal: 'vec3', // view-space normal
  position: 'vec3', // object-space position
});

// Generic built-ins. Each parameter is 'T' (generic float type shared by all
// T parameters), 'f' (float), or a concrete type. `ints` lets min/max/abs/
// sign/clamp/mod stay integer when every argument is an int.
const B = (params, ret, ints = false) => ({ params, ret, ints });
const unaryT = B(['T'], 'T');
export const BUILTINS = Object.assign(Object.create(null), {
  sin: unaryT, cos: unaryT, tan: unaryT, asin: unaryT, acos: unaryT, atan: unaryT,
  sinh: unaryT, cosh: unaryT, tanh: unaryT,
  exp: unaryT, log: unaryT, exp2: unaryT, log2: unaryT, sqrt: unaryT, inversesqrt: unaryT,
  floor: unaryT, ceil: unaryT, fract: unaryT, round: unaryT, trunc: unaryT,
  radians: unaryT, degrees: unaryT, normalize: unaryT, saturate: unaryT,
  abs: B(['T'], 'T', true), sign: B(['T'], 'T', true),
  min: B(['T', 'T'], 'T', true), max: B(['T', 'T'], 'T', true),
  clamp: B(['T', 'T', 'T'], 'T', true), mod: B(['T', 'T'], 'T', true),
  pow: B(['T', 'T'], 'T'), atan2: B(['T', 'T'], 'T'),
  step: B(['T', 'T'], 'T'), smoothstep: B(['T', 'T', 'T'], 'T'), mix: B(['T', 'T', 'T'], 'T'),
  length: B(['T'], 'float'), distance: B(['T', 'T'], 'float'), dot: B(['T', 'T'], 'float'),
  reflect: B(['T', 'T'], 'T'), refract: B(['T', 'T', 'f'], 'T'),
  cross: B(['vec3', 'vec3'], 'vec3'),
});

// Every identifier reserved by GLSL ES 3.00, WGSL or JavaScript. User names
// that collide are renamed in generated code; uniforms/buffers may not use them.
export const RESERVED = new Set(`
main attribute const uniform varying layout centroid flat smooth break continue do for while switch case default
if else in out inout float int void bool true false invariant discard return mat2 mat3 mat4 mat2x2 mat2x3 mat2x4
mat3x2 mat3x3 mat3x4 mat4x2 mat4x3 mat4x4 vec2 vec3 vec4 ivec2 ivec3 ivec4 bvec2 bvec3 bvec4 uint uvec2 uvec3 uvec4
lowp mediump highp precision sampler2D sampler3D samplerCube sampler2DShadow struct texture input output sample
common partition active asm class union enum typedef template this goto inline noinline volatile public static
extern external interface long short double half fixed unsigned superp filter image sizeof cast namespace using
noise1 noise2 noise3 noise4 fn let var loop array ptr ref f32 f16 i32 u32 vec mat atomic bitcast discard enable
override private storage function workgroup alias module require self super type unsafe where with yield await
async new delete typeof instanceof function arguments eval undefined null NaN Infinity Math Object Array Number
export import package protected implements let static catch try throw finally debugger void var
fs_main cs_main vs_fullscreen fragColor vUv vNormal vPosition
`.trim().split(/\s+/));
