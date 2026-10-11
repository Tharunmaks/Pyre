// The Pyre standard library, written in Pyre. Functions are only emitted
// when a program actually calls them. Defining a function with the same
// name in your own file replaces the library version.
export const STDLIB = `
# Integer hashing is bit-exact on every target (GLSL, WGSL and JS all wrap
# 32-bit integer multiplication), so noise looks identical on GPU and CPU.
fn ihash(n: int) -> int:
    var x = n
    x = (x ^ (x >> 16)) * 0x7feb352d
    x = (x ^ (x >> 15)) * 0x5bd1e995
    return x ^ (x >> 16)

# Random value in [0, 1) for the integer cell that contains p.
fn hash21(p: vec2) -> float:
    let h = ihash(int(floor(p.x)) + ihash(int(floor(p.y)) + 0x9e37))
    return float(h & 0xffffff) / 16777216.0

fn hash22(p: vec2) -> vec2:
    return vec2(hash21(p), hash21(p + vec2(7919.0, 104729.0)))

fn hash31(p: vec3) -> float:
    let h = ihash(int(floor(p.x)) + ihash(int(floor(p.y)) + ihash(int(floor(p.z)) + 0x51ed)))
    return float(h & 0xffffff) / 16777216.0

fn noise(p: vec2) -> float:
    let i = floor(p)
    let f = fract(p)
    let u = f * f * (3.0 - 2.0 * f)
    let a = hash21(i)
    let b = hash21(i + vec2(1.0, 0.0))
    let c = hash21(i + vec2(0.0, 1.0))
    let d = hash21(i + vec2(1.0, 1.0))
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y)

fn fbm(p: vec2) -> float:
    var v = 0.0
    var amp = 0.5
    var q = p
    for i in range(5):
        v += amp * noise(q)
        q = q * 2.03 + vec2(1.7, 9.2)
        amp *= 0.5
    return v

fn rotate(a: float) -> mat2:
    let c = cos(a)
    let s = sin(a)
    return mat2(c, s, -s, c)

fn palette(t: float, a: vec3, b: vec3, c: vec3, d: vec3) -> vec3:
    return a + b * cos(TAU * (c * t + d))

fn hsv2rgb(c: vec3) -> vec3:
    let k = vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)
    let p = abs(fract(c.xxx + k) * 6.0 - 3.0)
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y)

fn sdSphere(p: vec3, r: float) -> float:
    return length(p) - r

fn sdBox(p: vec3, b: vec3) -> float:
    let q = abs(p) - b
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0)

fn sdTorus(p: vec3, t: vec2) -> float:
    let q = vec2(length(p.xz) - t.x, p.y)
    return length(q) - t.y

fn smin(a: float, b: float, k: float) -> float:
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return mix(b, a, h) - k * h * (1.0 - h)
`;
