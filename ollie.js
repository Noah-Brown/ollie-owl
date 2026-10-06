// Ollie: a northern saw-whet owl mascot that doubles as a chat loading indicator.
//
//   import { OllieMascot } from './ollie.js';
//   const ollie = new OllieMascot(canvas, { mode: 'bar' });   // or 'perch'
//   ollie.setState('thinking');   // bar: flies laps over the branch; perch: curious head tilts
//   ollie.setProgress(0.4);       // bar: hops along the branch, leaves sprout up to 40%
//   ollie.setState('done');       // fluffs up and stretches its wings
//   ollie.dispose();
//
// Needs an import map with 'three' and 'three/addons/' (r163+).

import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const TAU = Math.PI * 2;
const { lerp, clamp, degToRad } = THREE.MathUtils;
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const damp = (cur, target, rate, dt) => cur + (target - cur) * (1 - Math.exp(-rate * dt));
const rand = (a, b) => a + Math.random() * (b - a);
const easeOutBack = (t) => { const c = 1.9; return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2; };

// ---------------------------------------------------------------------------
// Body shape: smooth union of a big head and a pear body, as a signed distance field.

const HEAD_C = [0, 0.42, 0.02];
const BODY_C = [0, -0.32, -0.04];
const PIVOT = new THREE.Vector3(0, 0.05, 0);
const EYE = { x: 0.225, y: 0.46, z: 0.415, r: 0.15 };

function sdEllipsoid(x, y, z, rx, ry, rz) {
  const k0 = Math.hypot(x / rx, y / ry, z / rz);
  const k1 = Math.hypot(x / (rx * rx), y / (ry * ry), z / (rz * rz)) || 1e-6;
  return (k0 * (k0 - 1)) / k1;
}
function smin(a, b, k) { const h = Math.max(k - Math.abs(a - b), 0) / k; return Math.min(a, b) - h * h * k * 0.25; }
function sdf(x, y, z) {
  const hz = z - HEAD_C[2];
  const h = sdEllipsoid(x - HEAD_C[0], y - HEAD_C[1], hz, 0.64, 0.56, hz > 0 ? 0.5 : 0.58);
  const b = sdEllipsoid(x - BODY_C[0], y - BODY_C[1], z - BODY_C[2], 0.6, 0.66, 0.56);
  return smin(h, b, 0.22);
}

let bodyGeoCache = null;
function bodyGeometry() {
  if (bodyGeoCache) return bodyGeoCache;
  const g = new THREE.IcosahedronGeometry(1, 26);
  const p = g.attributes.position;
  const nrm = new Float32Array(p.count * 3);
  const d = new THREE.Vector3();
  const e = 1e-3;
  for (let i = 0; i < p.count; i++) {
    d.fromBufferAttribute(p, i).normalize();
    let lo = 0, hi = 1.6;
    for (let k = 0; k < 24; k++) {
      const m = (lo + hi) / 2;
      if (sdf(d.x * m, d.y * m, d.z * m) < 0) lo = m; else hi = m;
    }
    const x = d.x * lo, y = d.y * lo, z = d.z * lo;
    p.setXYZ(i, x, y, z);
    const n = new THREE.Vector3(
      sdf(x + e, y, z) - sdf(x - e, y, z),
      sdf(x, y + e, z) - sdf(x, y - e, z),
      sdf(x, y, z + e) - sdf(x, y, z - e),
    ).normalize();
    nrm.set([n.x, n.y, n.z], i * 3);
  }
  g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  g.computeBoundingSphere();
  return (bodyGeoCache = g);
}

// ---------------------------------------------------------------------------
// Plumage shader: shell-rendered feathers with saw-whet markings painted procedurally.

const GLSL_COMMON = /* glsl */ `
uniform float uFurLen;
uniform vec3 uHeadAng;
uniform float uDensity;
uniform float uBreath;
const vec2 EYE_P = vec2(${EYE.x}, ${EYE.y});
const vec3 PIVOT_P = vec3(${PIVOT.x.toFixed(3)}, ${PIVOT.y.toFixed(3)}, ${PIVOT.z.toFixed(3)});

float ss(float a, float b, float x) { return a < b ? smoothstep(a, b, x) : 1.0 - smoothstep(b, a, x); }
float hash13(vec3 p) { p = fract(p * 0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); }
vec3 hash33(vec3 p) { p = fract(p * vec3(0.1031, 0.1030, 0.0973)); p += dot(p, p.yxz + 33.33); return fract((p.xxy + p.yxx) * p.zyx); }
float vnoise(vec3 p) {
  vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash13(i), hash13(i + vec3(1,0,0)), f.x), mix(hash13(i + vec3(0,1,0)), hash13(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(hash13(i + vec3(0,0,1)), hash13(i + vec3(1,0,1)), f.x), mix(hash13(i + vec3(0,1,1)), hash13(i + vec3(1,1,1)), f.x), f.y), f.z);
}
float fbm(vec3 p) { return (0.5 * vnoise(p) + 0.25 * vnoise(p * 2.03) + 0.125 * vnoise(p * 4.01)) / 0.875; }

void owlMasks(vec3 p, vec3 n, out float face, out float eye, out vec2 rel) {
  rel = vec2(abs(p.x), p.y) - EYE_P;
  float d = length(rel);
  float fwd = ss(0.1, 0.45, n.z) * step(0.0, p.z);
  face = ss(0.35, 0.29, d) * fwd;
  eye = ss(0.175, 0.16, d) * fwd;
}
`;

const GLSL_ROT = /* glsl */ `
mat3 rotX(float a) { float c = cos(a), s = sin(a); return mat3(1.,0.,0., 0.,c,s, 0.,-s,c); }
mat3 rotY(float a) { float c = cos(a), s = sin(a); return mat3(c,0.,-s, 0.,1.,0., s,0.,c); }
mat3 rotZ(float a) { float c = cos(a), s = sin(a); return mat3(c,s,0., -s,c,0., 0.,0.,1.); }
`;

const GLSL_PLUMAGE = /* glsl */ `
vec3 srgb(vec3 c) { return pow(c, vec3(2.2)); }
vec3 plumage(vec3 p, vec3 n, float faceM, vec2 rel) {
  vec3 BROWN = srgb(vec3(0.43, 0.28, 0.18));
  vec3 DARK  = srgb(vec3(0.25, 0.16, 0.10));
  vec3 RUF   = srgb(vec3(0.58, 0.35, 0.21));
  vec3 CREAM = srgb(vec3(0.95, 0.92, 0.86));
  vec3 BUFF  = srgb(vec3(0.88, 0.80, 0.69));
  float head = ss(0.02, 0.2, p.y);
  float front = ss(0.05, 0.5, n.z);

  vec3 col = mix(DARK, BROWN, fbm(p * 6.0));
  float streak = ss(0.74, 0.86, vnoise(vec3(p.x * 62.0, p.y * 20.0, p.z * 62.0)));
  col = mix(col, CREAM, streak * head * (1.0 - faceM) * 0.9);
  float spot = ss(0.80, 0.87, vnoise(p * 18.0));
  col = mix(col, CREAM, spot * (1.0 - head) * (1.0 - front) * 0.8);

  float chest = front * (1.0 - head * 0.85) * ss(0.48, 0.26, abs(p.x)) * ss(0.3, 0.12, p.y);
  float cs = ss(0.46, 0.66, vnoise(vec3(p.x * 19.0, p.y * 3.6 + 3.0, p.z * 19.0))) * ss(0.12, 0.3, abs(p.x) + 0.08 * vnoise(p * 9.0));
  col = mix(col, mix(CREAM, RUF, cs * 0.95), chest);

  float throat = front * ss(0.32, 0.14, abs(p.x)) * ss(0.02, 0.12, p.y) * ss(0.34, 0.24, p.y);
  col = mix(col, CREAM, throat);

  float d = length(rel);
  float ang = atan(rel.y, rel.x);
  float rays = ss(0.5, 1.0, sin(ang * 26.0 + vnoise(p * 22.0) * 3.0)) * ss(0.17, 0.3, d);
  vec3 faceCol = mix(BUFF, RUF, rays * 0.6 + ss(0.2, 0.3, d) * 0.25);
  faceCol = mix(faceCol, BROWN, ss(0.25, 0.33, d) * 0.8);
  float brow = ss(0.16, 0.18, d) * ss(0.24, 0.21, d) * ss(-0.01, 0.07, rel.y) * ss(0.12, -0.04, rel.x);
  float stripe = ss(0.075, 0.03, abs(p.x)) * ss(0.30, 0.38, p.y) * ss(0.76, 0.62, p.y);
  faceCol = mix(faceCol, CREAM * 1.04, max(brow, stripe));
  faceCol = mix(faceCol, DARK, ss(0.18, 0.16, d) * 0.85);
  col = mix(col, faceCol, faceM);

  // rows of overlapping feathers, tips pointing down
  vec2 fu = vec2(atan(p.x, p.z) * 2.6, p.y * 9.5);
  float row = floor(fu.y);
  fu.x += row * 0.5 + hash13(vec3(row, 1.0, 2.0)) * 0.3;
  vec2 ff = fract(fu) - vec2(0.5, 0.0);
  float arcY = 0.78 - 1.5 * ff.x * ff.x;
  float under = ss(arcY - 0.35, arcY, ff.y) * (1.0 - ss(arcY, arcY + 0.04, ff.y));
  float rows = 1.0 - 0.16 * under * (0.6 + 0.8 * hash13(vec3(floor(fu), 5.0))) + 0.04 * ss(arcY, arcY + 0.04, ff.y);
  col *= mix(rows, 1.0, max(faceM * 0.8, pow(abs(n.y), 4.0)));

  float bristle = front * ss(0.11, 0.04, abs(p.x)) * ss(0.18, 0.26, p.y) * ss(0.42, 0.33, p.y);
  col = mix(col, CREAM, bristle);
  return col;
}
`;

function furMaterial(U) {
  const m = new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0 });
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, U);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', `#include <common>
${GLSL_COMMON}
${GLSL_ROT}
attribute float aShell;
varying vec3 vP0; varying vec3 vN0; varying float vShell;`)
      .replace('#include <beginnormal_vertex>', `
float faceM, eyeM; vec2 rel;
owlMasks(position, normal, faceM, eyeM, rel);
float hw = smoothstep(-0.05, 0.25, position.y);
vec3 ha = uHeadAng * hw;
mat3 HR = rotY(ha.y) * rotX(ha.x) * rotZ(ha.z);
vec3 objectNormal = HR * normal;`)
      .replace('#include <begin_vertex>', `
float shell = aShell;
float len = uFurLen * mix(1.0, 0.42, faceM) * (1.0 - eyeM);
vec3 radial = vec3(sign(position.x) * rel.x, rel.y, 0.0);
radial = radial / max(length(radial), 1e-4);
vec3 comb = normalize(mix(vec3(0.0, -1.0, -0.35), radial, faceM));
vec3 pos = position + normal * len * shell + comb * len * shell * shell * 0.55;
pos.xz *= 1.0 + uBreath * (1.0 - hw);
vec3 transformed = PIVOT_P + HR * (pos - PIVOT_P);
vP0 = position; vN0 = normal; vShell = shell;`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
${GLSL_COMMON}
${GLSL_PLUMAGE}
varying vec3 vP0; varying vec3 vN0; varying float vShell;`)
      .replace('#include <map_fragment>', `
float faceM, eyeM; vec2 rel;
owlMasks(vP0, vN0, faceM, eyeM, rel);
if (vShell > 0.0) {
  if (eyeM > 0.5) discard;
  vec3 q = vP0 * uDensity * mix(1.0, 1.5, faceM);
  vec3 h = hash33(floor(q));
  float strandLen = 0.45 + 0.55 * h.z;
  float s = vShell / strandLen;
  if (s > 1.0) discard;
  float d = length(fract(q) - (0.25 + 0.5 * h));
  if (d > 0.9 * (1.0 - s * 0.8)) discard;
}
vec3 plum = plumage(vP0, vN0, faceM, rel);
plum *= mix(0.32, 1.0, pow(vShell, 0.65));
plum = mix(plum, plum * 1.18 + 0.01, smoothstep(0.7, 1.0, vShell) * 0.6);
diffuseColor.rgb = plum;`);
  };
  m.customProgramCacheKey = () => 'ollie-fur';
  return m;
}

function eyeMaterial(U) {
  const m = new THREE.MeshPhysicalMaterial({ roughness: 0.4, clearcoat: 1, clearcoatRoughness: 0.03 });
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uPupil = U.uPupil;
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vEyeP;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvEyeP = normalize(position);');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
${GLSL_COMMON}
uniform float uPupil;
varying vec3 vEyeP;`)
      .replace('#include <map_fragment>', `
vec3 p = vEyeP;
float a = acos(clamp(p.z, -1.0, 1.0));
float az = atan(p.y, p.x);
float rIris = 0.98;
float t = clamp((a - uPupil) / (rIris - uPupil), 0.0, 1.0);
vec3 inner = pow(vec3(0.86, 0.50, 0.06), vec3(2.2));
vec3 outer = pow(vec3(0.98, 0.80, 0.12), vec3(2.2));
vec3 iris = mix(inner, outer, smoothstep(0.0, 0.55, t));
float fib = vnoise(vec3(cos(az) * 14.0, sin(az) * 14.0, t * 7.0)) * 0.6 + vnoise(vec3(cos(az) * 40.0, sin(az) * 40.0, t * 3.0)) * 0.4;
iris *= 0.7 + 0.6 * fib;
iris *= mix(0.55, 1.0, smoothstep(0.0, 0.14, t));
iris = mix(iris, pow(vec3(0.42, 0.22, 0.04), vec3(2.2)), smoothstep(0.88, 1.0, t));
vec3 col = mix(vec3(0.004, 0.004, 0.006), iris, smoothstep(uPupil - 0.012, uPupil + 0.012, a));
col = mix(col, pow(vec3(0.16, 0.10, 0.06), vec3(2.2)), smoothstep(rIris - 0.01, rIris + 0.05, a));
col *= mix(1.0, 0.3, smoothstep(0.35, 0.8, p.y));
diffuseColor.rgb = col;`);
  };
  m.customProgramCacheKey = () => 'ollie-eye';
  return m;
}

// ---------------------------------------------------------------------------
// Painted textures (feathers, leaves, bark), generated once.

const texCache = {};
function canvasTex(key, w, h, draw, opts = {}) {
  if (texCache[key]) return texCache[key];
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  if (opts.repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; }
  return (texCache[key] = t);
}

const FEATHERS = {
  primary:   { rx: 0.40, outer: 0.30, inner: 0.58, tip: 0.80, main: '#58392a', dark: '#3b2519', bar: '#e6d6ba', mode: 'outerSpots' },
  secondary: { rx: 0.45, outer: 0.42, inner: 0.52, tip: 0.76, main: '#5e3d2a', dark: '#3f281b', bar: '#dcc8a8', mode: 'bars' },
  covert:    { rx: 0.50, outer: 0.48, inner: 0.48, tip: 0.55, main: '#6b4631', dark: '#4a2f20', bar: '#f1e9da', mode: 'tipSpot' },
  covertPlain: { rx: 0.50, outer: 0.48, inner: 0.48, tip: 0.55, main: '#694430', dark: '#462c1e', bar: '#f1e9da', mode: 'none' },
  tail:      { rx: 0.50, outer: 0.46, inner: 0.46, tip: 0.72, main: '#5a3b29', dark: '#3b2519', bar: '#e2d1b4', mode: 'bars' },
};

function featherTexture(kind) {
  const f = FEATHERS[kind];
  return canvasTex('feather-' + kind, 128, 512, (g, W, H) => {
    const width = (t, frac) => {
      let w = frac * Math.min(1, t / 0.09) ** 0.5;
      if (t > f.tip) w *= Math.sqrt(Math.max(0, 1 - ((t - f.tip) / (1 - f.tip)) ** 2));
      return w * W;
    };
    const rx = f.rx * W;
    const outline = new Path2D();
    for (let i = 0; i <= 80; i++) { const t = i / 80; outline.lineTo(rx - width(t, f.outer), t * H); }
    for (let i = 80; i >= 0; i--) { const t = i / 80; outline.lineTo(rx + width(t, f.inner), t * H); }
    outline.closePath();
    g.save();
    g.clip(outline);
    const grd = g.createLinearGradient(0, 0, 0, H);
    grd.addColorStop(0, '#9b8470');
    grd.addColorStop(0.12, f.main);
    grd.addColorStop(0.85, f.main);
    grd.addColorStop(1, f.dark);
    g.fillStyle = grd;
    g.fillRect(0, 0, W, H);
    // mottling
    for (let i = 0; i < 260; i++) {
      g.fillStyle = `rgba(${Math.random() < 0.5 ? '30,18,10' : '150,110,80'},${rand(0.04, 0.12)})`;
      g.beginPath();
      g.ellipse(rand(0, W), rand(0, H), rand(2, 7), rand(4, 14), 0, 0, TAU);
      g.fill();
    }
    // markings
    g.fillStyle = f.bar;
    if (f.mode === 'outerSpots') {
      for (let k = 0; k < 6; k++) {
        const y = (0.2 + k * 0.105) * H;
        g.globalAlpha = 0.85;
        g.beginPath();
        g.ellipse(rx - width(y / H, f.outer) * 0.5, y, width(y / H, f.outer) * 0.45, 13, 0, 0, TAU);
        g.fill();
        g.globalAlpha = 0.18;
        g.fillRect(rx, y - 8, W, 16);
      }
    } else if (f.mode === 'bars') {
      for (let k = 0; k < 6; k++) {
        const y = (0.2 + k * 0.11) * H;
        g.globalAlpha = 0.32;
        g.beginPath();
        g.ellipse(rx, y, W * 0.6, 11, -0.08, 0, TAU);
        g.fill();
      }
    } else if (f.mode === 'tipSpot') {
      g.globalAlpha = 0.95;
      g.beginPath();
      g.ellipse(rx + W * 0.06, H * 0.74, W * 0.11, H * 0.045, 0, 0, TAU);
      g.fill();
    }
    g.globalAlpha = 1;
    // barbs
    g.lineWidth = 1;
    for (let y = 0; y < H; y += 3) {
      const t = y / H;
      for (const side of [-1, 1]) {
        const wv = width(t, side < 0 ? f.outer : f.inner);
        g.strokeStyle = (y / 3) % 2 ? 'rgba(20,10,5,0.16)' : 'rgba(255,240,220,0.07)';
        g.beginPath();
        g.moveTo(rx, y);
        g.lineTo(rx + side * wv, y + wv * 0.55);
        g.stroke();
      }
    }
    // downy base
    for (let i = 0; i < 90; i++) {
      g.strokeStyle = `rgba(200,180,160,${rand(0.2, 0.5)})`;
      const y = rand(0, H * 0.13);
      g.beginPath();
      g.moveTo(rx, y);
      g.lineTo(rx + rand(-W * 0.5, W * 0.5), y + rand(4, 20));
      g.stroke();
    }
    // edge shading
    g.lineWidth = 5;
    g.strokeStyle = 'rgba(20,10,4,0.28)';
    g.stroke(outline);
    g.restore();
    // shaft
    const sh = g.createLinearGradient(0, 0, 0, H * f.tip);
    sh.addColorStop(0, 'rgba(225,205,180,0.95)');
    sh.addColorStop(1, 'rgba(225,205,180,0.2)');
    g.strokeStyle = sh;
    g.lineWidth = 3;
    g.beginPath();
    g.moveTo(rx, 0);
    g.quadraticCurveTo(rx + 2, H * 0.5, rx + 1, H * (f.tip + 0.12));
    g.stroke();
  });
}

const leafTexture = () => canvasTex('leaf', 128, 256, (g, W, H) => {
  const grd = g.createLinearGradient(0, 0, W, H);
  grd.addColorStop(0, '#7fa357');
  grd.addColorStop(1, '#4d7434');
  g.fillStyle = grd;
  g.fillRect(0, 0, W, H);
  g.strokeStyle = 'rgba(225,240,190,0.55)';
  g.lineWidth = 3;
  g.beginPath(); g.moveTo(W / 2, H); g.lineTo(W / 2, 0); g.stroke();
  g.lineWidth = 1.3;
  for (let i = 1; i < 9; i++) {
    const y = H - i * (H / 9.5);
    for (const s of [-1, 1]) {
      g.beginPath(); g.moveTo(W / 2, y); g.quadraticCurveTo(W / 2 + s * W * 0.2, y - 12, W / 2 + s * W * 0.45, y - 30); g.stroke();
    }
  }
});

const barkTexture = () => canvasTex('bark', 128, 512, (g, W, H) => {
  g.fillStyle = '#5b4535';
  g.fillRect(0, 0, W, H);
  for (let i = 0; i < 900; i++) {
    const x = rand(0, W), y = rand(0, H);
    const shade = Math.random() < 0.55;
    g.strokeStyle = shade ? `rgba(30,20,12,${rand(0.15, 0.45)})` : `rgba(160,135,110,${rand(0.1, 0.3)})`;
    g.lineWidth = rand(0.8, 2.6);
    g.beginPath(); g.moveTo(x, y); g.lineTo(x + rand(-3, 3), y + rand(10, 40)); g.stroke();
  }
  for (let i = 0; i < 40; i++) {
    g.fillStyle = `rgba(170,185,150,${rand(0.15, 0.35)})`;
    g.beginPath(); g.ellipse(rand(0, W), rand(0, H), rand(2, 8), rand(2, 6), 0, 0, TAU); g.fill();
  }
}, { repeat: true });

function featherGeometry() {
  const g = new THREE.PlaneGeometry(1, 1, 1, 8);
  g.translate(0, -0.5, 0);
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) { const t = -p.getY(i); p.setZ(i, -0.07 * t * t); }
  g.computeVertexNormals();
  return g;
}

function leafGeometry() {
  const s = new THREE.Shape();
  s.moveTo(0, 0);
  s.bezierCurveTo(0.09, 0.06, 0.08, 0.2, 0, 0.3);
  s.bezierCurveTo(-0.08, 0.2, -0.09, 0.06, 0, 0);
  const g = new THREE.ShapeGeometry(s, 10);
  const p = g.attributes.position, uv = g.attributes.uv;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i);
    p.setZ(i, -Math.abs(x) * 0.5 + Math.sin(y * 8) * 0.004);
    uv.setXY(i, x / 0.2 + 0.5, y / 0.3);
  }
  g.computeVertexNormals();
  return g;
}

// ---------------------------------------------------------------------------
// The owl.

const WING_SPREAD = { S: [0, 0], E: [0.42, 0.1], W: [0.84, 0.14], T: [1.18, 0.04] };
function chainPoint(c) {
  const segs = [[WING_SPREAD.S, WING_SPREAD.E], [WING_SPREAD.E, WING_SPREAD.W], [WING_SPREAD.W, WING_SPREAD.T]];
  const x = clamp(c, 0, 1) * 3;
  const i = Math.min(2, Math.floor(x));
  const [a, b] = segs[i];
  const t = x - i;
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t)];
}

class Owl {
  constructor({ shells, density }) {
    this.U = {
      uFurLen: { value: 0.056 },
      uHeadAng: { value: new THREE.Vector3() },
      uDensity: { value: density },
      uBreath: { value: 0 },
      uPupil: { value: 0.42 },
    };
    this.root = new THREE.Group();
    this.disposables = [];

    // plumage shells
    const geo = bodyGeometry().clone();
    const shellAttr = new Float32Array(shells);
    for (let i = 0; i < shells; i++) shellAttr[i] = i / (shells - 1);
    geo.setAttribute('aShell', new THREE.InstancedBufferAttribute(shellAttr, 1));
    this.disposables.push(geo);
    this.body = new THREE.InstancedMesh(geo, furMaterial(this.U), shells);
    const I = new THREE.Matrix4();
    for (let i = 0; i < shells; i++) this.body.setMatrixAt(i, I);
    this.body.frustumCulled = false;
    this.root.add(this.body);

    // head parts follow the same rotation as the head plumage
    this.head = new THREE.Group();
    this.head.rotation.order = 'YXZ';
    this.head.position.copy(PIVOT);
    this.root.add(this.head);
    const at = (x, y, z) => new THREE.Vector3(x, y, z).sub(PIVOT);

    const eyeMat = eyeMaterial(this.U);
    const lidMat = new THREE.MeshStandardMaterial({ color: '#a38b74', roughness: 0.85, side: THREE.DoubleSide });
    const rimMat = new THREE.MeshStandardMaterial({ color: '#1c120c', roughness: 0.5 });
    const eyeGeo = new THREE.SphereGeometry(EYE.r, 56, 40);
    const lidGeo = new THREE.SphereGeometry(EYE.r + 0.007, 48, 16, 0, TAU, 0, Math.PI / 2);
    const rimGeo = new THREE.TorusGeometry(EYE.r + 0.007, 0.006, 8, 64);
    rimGeo.rotateX(Math.PI / 2);
    this.lids = [];
    for (const sx of [-1, 1]) {
      const g = new THREE.Group();
      g.position.copy(at(sx * EYE.x, EYE.y, EYE.z));
      g.rotation.set(-0.04, sx * 0.12, 0);
      g.add(new THREE.Mesh(eyeGeo, eyeMat));
      const upper = new THREE.Group();
      upper.add(new THREE.Mesh(lidGeo, lidMat), new THREE.Mesh(rimGeo, rimMat));
      const lowerOuter = new THREE.Group();
      lowerOuter.rotation.z = Math.PI;
      const lower = new THREE.Group();
      lower.add(new THREE.Mesh(lidGeo, lidMat), new THREE.Mesh(rimGeo, rimMat));
      lowerOuter.add(lower);
      g.add(upper, lowerOuter);
      this.head.add(g);
      this.lids.push({ upper, lower });
    }

    // hooked beak
    const beakGeo = new THREE.ConeGeometry(0.046, 0.15, 28, 10);
    beakGeo.rotateX(Math.PI / 2);
    beakGeo.scale(0.78, 1, 1);
    {
      const p = beakGeo.attributes.position;
      for (let i = 0; i < p.count; i++) {
        const t = (p.getZ(i) + 0.075) / 0.15;
        p.setY(i, p.getY(i) - 0.075 * t * t);
        p.setZ(i, p.getZ(i) - 0.025 * t * t);
      }
      beakGeo.computeVertexNormals();
    }
    const beak = new THREE.Mesh(beakGeo, new THREE.MeshPhysicalMaterial({ color: '#26272b', roughness: 0.35, clearcoat: 0.7, clearcoatRoughness: 0.2 }));
    beak.position.copy(at(0, 0.32, 0.47));
    beak.rotation.x = 0.5;
    this.head.add(beak);

    // feathered toes and talons gripping the perch
    this.feet = new THREE.Group();
    const toeMat = new THREE.MeshStandardMaterial({ color: '#c4ad90', roughness: 0.95 });
    const clawMat = new THREE.MeshPhysicalMaterial({ color: '#1b1714', roughness: 0.3, clearcoat: 0.6 });
    const toeGeo = new THREE.CapsuleGeometry(0.027, 0.06, 6, 12);
    toeGeo.rotateX(Math.PI / 2);
    const clawGeo = new THREE.ConeGeometry(0.012, 0.055, 12);
    for (const sx of [-1, 1]) {
      for (const a of [-0.38, 0, 0.38]) {
        const toe = new THREE.Group();
        toe.position.set(sx * 0.16, -0.955, 0.17);
        toe.rotation.set(0.35, a + sx * 0.12, 0);
        const m = new THREE.Mesh(toeGeo, toeMat);
        m.position.z = 0.05;
        const claw = new THREE.Mesh(clawGeo, clawMat);
        claw.position.set(0, -0.025, 0.1);
        claw.rotation.x = Math.PI + 0.5;
        toe.add(m, claw);
        this.feet.add(toe);
      }
    }
    this.root.add(this.feet);

    // feathers
    const fGeo = featherGeometry();
    this.disposables.push(fGeo);
    const fMat = (kind) => new THREE.MeshStandardMaterial({ map: featherTexture(kind), alphaTest: 0.45, side: THREE.DoubleSide, roughness: 0.82 });
    const mats = { primary: fMat('primary'), secondary: fMat('secondary'), covert: fMat('covert'), covertPlain: fMat('covertPlain'), tail: fMat('tail') };

    // tail
    const tail = new THREE.Group();
    tail.position.set(0, -0.62, -0.44);
    for (let i = 0; i < 6; i++) {
      const m = new THREE.Mesh(fGeo, mats.tail);
      m.scale.set(0.1, 0.3, 0.3);
      m.position.set((i - 2.5) * 0.035, 0, -i * 0.001);
      m.rotation.set(0.75, 0, (i - 2.5) * 0.09);
      tail.add(m);
    }
    this.root.add(tail);

    this.wings = [1, -1].map((side) => this._wing(side, fGeo, mats));
  }

  _wing(side, fGeo, mats) {
    const outer = new THREE.Group();
    outer.scale.x = side;
    outer.position.set(side * 0.53, -0.01, -0.08);
    const flap = new THREE.Group();
    flap.rotation.order = 'ZYX';
    outer.add(flap);
    const feathers = [];
    const add = (kind, w, L, spread, fold, z) => {
      const m = new THREE.Mesh(fGeo, mats[kind]);
      m.scale.set(w, L, L);
      flap.add(m);
      feathers.push({ m, spread, fold, z });
    };
    // primaries: fan out from the wrist to the tip
    for (let j = 0; j < 10; j++) {
      const u = j / 9;
      const [x, y] = chainPoint(2 / 3 + u / 3);
      add('primary', 0.13, 0.56 + 0.22 * u - (j === 9 ? 0.08 : 0),
        [x, y, lerp(-88, -14, u ** 1.1)], [0.12 + 0.06 * u, -0.08 - 0.02 * u, -72 + 3 * u], 0.0 + j * 0.0014);
    }
    // secondaries: along the arm, hanging straight back
    for (let i = 0; i < 9; i++) {
      const c = ((i + 0.5) / 9) * (2 / 3);
      const [x, y] = chainPoint(c);
      add('secondary', 0.15, 0.6, [x, y, -97 + i * 0.9], [0.02 + 0.09 * c, -0.03 - 0.05 * c, -80 + i * 0.4], 0.016 + i * 0.0014);
    }
    // greater coverts
    for (let k = 0; k < 12; k++) {
      const c = (k + 0.5) / 12;
      const [x, y] = chainPoint(c);
      add(k % 3 === 1 ? 'covert' : 'covertPlain', 0.12, 0.22, [x, y + 0.01, c < 0.66 ? -95 : lerp(-95, -45, (c - 0.66) / 0.34)],
        [0.02 + 0.16 * c, -0.03 - 0.05 * c, -82], 0.034 + k * 0.0012);
    }
    // lesser and marginal coverts along the leading edge
    for (let m = 0; m < 14; m++) {
      const c = (m / 13) * 0.9;
      const [x, y] = chainPoint(c);
      add(m % 4 === 2 ? 'covert' : 'covertPlain', 0.1, 0.12, [x, y + 0.04, -100 + 45 * c * c], [0.02 + 0.15 * c, 0.0 - 0.04 * c, -85], 0.05 + m * 0.0012);
    }
    for (let q = 0; q < 8; q++) {
      const c = (q / 7) * 0.85;
      const [x, y] = chainPoint(c);
      add('covertPlain', 0.08, 0.08, [x, y + 0.07, -80], [0.02 + 0.14 * c, 0.04 - 0.03 * c, -88], 0.066 + q * 0.0012);
    }
    this.root.add(outer);
    return { outer, flap, feathers };
  }

  poseWing(w, s, yaw, raise) {
    const e = s * s * (3 - 2 * s);
    for (const f of w.feathers) {
      f.m.position.set(lerp(f.fold[0], f.spread[0], e), lerp(f.fold[1], f.spread[1], e), f.z);
      f.m.rotation.z = degToRad(lerp(f.fold[2], f.spread[2], e) + 90);
    }
    w.flap.rotation.set(0, yaw, raise);
  }

  pose(P, phase) {
    const U = this.U;
    U.uHeadAng.value.set(P.pitch, P.yaw, P.roll);
    this.head.rotation.set(P.pitch, P.yaw, P.roll);
    U.uFurLen.value = P.fluff;
    U.uPupil.value = P.pupil;
    for (const l of this.lids) {
      l.upper.rotation.x = P.lidU;
      l.lower.rotation.x = P.lidL;
    }
    const spread = clamp(P.flap * 1.5, 0, 1);
    const flutter = P.flap;
    const s = spread * (1 - 0.28 * Math.max(0, Math.cos(phase)) * flutter);
    const raise = lerp(0.1, 0.22, spread) + flutter * 0.85 * Math.sin(phase);
    const yaw = lerp(1.5, 0.22, spread);
    for (const w of this.wings) this.poseWing(w, s, yaw, raise);
    this.feet.position.y = P.hover * 0.06;
  }

  dispose() {
    this.root.traverse((o) => { if (o.material) [].concat(o.material).forEach((m) => m.dispose()); });
    this.disposables.forEach((d) => d.dispose());
  }
}

// ---------------------------------------------------------------------------
// Branch with contact shading under the owl.

function branchMaterial(U) {
  const tex = barkTexture();
  const m = new THREE.MeshStandardMaterial({ map: tex, bumpMap: tex, bumpScale: 2.5, roughness: 0.95 });
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, U);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vBW;')
      .replace('#include <project_vertex>', '#include <project_vertex>\nvBW = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform float uOwlX; uniform float uOwlW; uniform float uLift;\nvarying vec3 vBW;')
      .replace('#include <map_fragment>', `#include <map_fragment>
float cs = exp(-pow((vBW.x - uOwlX) / uOwlW, 2.0)) * (1.0 - uLift);
diffuseColor.rgb *= 1.0 - 0.55 * cs;`);
  };
  m.customProgramCacheKey = () => 'ollie-branch';
  return m;
}

function makeBranch(length, radius, U) {
  const g = new THREE.CylinderGeometry(radius, radius, length, 28, 60, false);
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
    const r = Math.hypot(x, z);
    if (r < 1e-5) continue;
    const k = 1 + 0.06 * Math.sin(y * 9.0 + Math.atan2(z, x) * 3) + 0.05 * Math.sin(y * 23.0) + 0.12 * Math.exp(-((y - length * 0.18) ** 2) * 60);
    p.setX(i, x * k); p.setZ(i, z * k);
  }
  g.computeVertexNormals();
  g.rotateZ(Math.PI / 2);
  const tex = barkTexture();
  tex.repeat.set(1, length * 1.5);
  return new THREE.Mesh(g, branchMaterial(U));
}

function makeLeaf(mat, geo) {
  const pivot = new THREE.Group();
  const m = new THREE.Mesh(geo, mat);
  pivot.add(m);
  return pivot;
}

// ---------------------------------------------------------------------------
// Public mascot.

export class OllieMascot {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.mode = opts.mode === 'bar' ? 'bar' : 'perch';
    this.followPointer = opts.followPointer ?? this.mode === 'perch';
    this.state = opts.state ?? 'idle';
    this.progress = opts.progress ?? null;
    this.reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

    const r = (this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true }));
    r.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.05;
    r.outputColorSpace = THREE.SRGBColorSpace;

    this.scene = new THREE.Scene();
    const pm = new THREE.PMREMGenerator(r);
    this.envRT = pm.fromScene(new RoomEnvironment(), 0.04);
    pm.dispose();
    this.scene.environment = this.envRT.texture;
    this.scene.environmentIntensity = 0.5;

    const key = new THREE.DirectionalLight('#ffe9cf', 2.6);
    key.position.set(-2.4, 3.2, 4);
    const rim = new THREE.DirectionalLight('#cfe0ff', 3.0);
    rim.position.set(2.6, 2.4, -3.2);
    const rim2 = new THREE.DirectionalLight('#ffd9b0', 1.4);
    rim2.position.set(-3, 0.5, -2.5);
    this.scene.add(key, rim, rim2);

    const small = (canvas.clientHeight || 200) < 160;
    this.camera = new THREE.PerspectiveCamera(this.mode === 'bar' ? 18 : 24, 1, 0.1, 80);
    this.owl = new Owl({ shells: small ? 16 : 24, density: small ? 40 : 58 });
    this.mover = new THREE.Group();
    this.mover.add(this.owl.root);
    this.scene.add(this.mover);

    this.BU = { uOwlX: { value: 0 }, uOwlW: { value: 0.3 }, uLift: { value: 0 } };
    if (this.mode === 'bar') this._buildBar(); else this._buildPerch();

    this.P = { hover: 0, flap: 0, lidU: -0.8, lidL: -1.15, yaw: 0, pitch: 0.05, roll: 0, pupil: 0.42, fluff: 0.056, bodyYaw: 0, fill: 0 };
    this.t = 0;
    this.phase = 0;
    this.blink = { next: rand(1.5, 4), t: -1 };
    this.look = { next: 0, yaw: 0, pitch: 0.05, roll: 0 };
    this.celebrate = 0;
    this.happyHold = 0;
    this.celebrated = false;
    this.curX = null;
    this.hopPhase = 0;
    this.dir = 1;
    this.pointer = { x: 0, y: 0, at: -1e9 };

    this._onPointer = (e) => { this.pointer = { x: e.clientX, y: e.clientY, at: performance.now() }; };
    addEventListener('pointermove', this._onPointer, { passive: true });
    this._visible = true;
    this._io = new IntersectionObserver((es) => { this._visible = es[0].isIntersecting; });
    this._io.observe(canvas);
    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(canvas);
    this._resize();

    this._last = performance.now();
    this._tick = this._tick.bind(this);
    this._raf = requestAnimationFrame(this._tick);
  }

  setState(state) {
    if (state === this.state) return;
    this.state = state;
    if (state === 'done') {
      this.celebrated = false;
      if (this.mode === 'perch') this._startCelebrate();
    }
    if (state === 'working' && this.progress == null && this.mode === 'bar') this.progress = 0;
  }

  setProgress(p) {
    this.progress = p == null ? null : clamp(p, 0, 1);
    if (p != null && this.state !== 'done') this.state = 'working';
  }

  dispose() {
    cancelAnimationFrame(this._raf);
    removeEventListener('pointermove', this._onPointer);
    this._io.disconnect();
    this._ro.disconnect();
    this.owl.dispose();
    this.scene.traverse((o) => { if (o.geometry && o.geometry !== bodyGeoCache) o.geometry.dispose(); });
    this.envRT.dispose();
    this.renderer.dispose();
  }

  // ----- scene pieces -----

  _buildPerch() {
    const branch = makeBranch(9, 0.075, this.BU);
    branch.position.set(0, -1.065, 0.05);
    branch.rotation.z = 0.035;
    this.scene.add(branch);
    const leafMat = new THREE.MeshStandardMaterial({ map: leafTexture(), side: THREE.DoubleSide, roughness: 0.55 });
    const leafGeo = leafGeometry();
    const twig = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.03, 0.5, 10), branch.material);
    twig.position.set(1.05, -0.86, 0.02);
    twig.rotation.z = -0.75;
    this.scene.add(twig);
    for (const [x, y, rz, ry] of [[1.2, -0.7, -0.5, 0.4], [1.05, -0.78, 0.6, -0.5], [1.32, -0.64, -1.3, 0.2]]) {
      const l = makeLeaf(leafMat, leafGeo);
      l.position.set(x, y, 0.03);
      l.rotation.set(0.3, ry, rz);
      this.scene.add(l);
    }
    this.BU.uOwlW.value = 0.38;
  }

  _buildBar() {
    const branch = makeBranch(24, 0.045, this.BU);
    branch.position.set(0, 0, 0);
    this.scene.add(branch);
    this.leafMat = new THREE.MeshStandardMaterial({ map: leafTexture(), side: THREE.DoubleSide, roughness: 0.55 });
    this.leafGeo = leafGeometry();
    this.budMat = new THREE.MeshStandardMaterial({ color: '#7b6a3d', roughness: 0.7 });
    this.budGeo = new THREE.SphereGeometry(0.014, 12, 8);
    this.leaves = [];
    this.leafGroup = new THREE.Group();
    this.scene.add(this.leafGroup);
    this.BU.uOwlW.value = 0.17;
  }

  _layoutLeaves() {
    for (const l of this.leaves) this.leafGroup.remove(l.root);
    this.leaves = [];
    const half = this.barHalf;
    const n = Math.max(4, Math.round((2 * half) / 0.2));
    for (let i = 0; i < n; i++) {
      const u = (i + 0.5) / n;
      const x = lerp(-half, half, u);
      const root = new THREE.Group();
      root.position.set(x, 0.035, 0.02);
      const bud = new THREE.Mesh(this.budGeo, this.budMat);
      const leaf = makeLeaf(this.leafMat, this.leafGeo);
      const up = i % 2 === 0;
      leaf.rotation.set(up ? 0.5 : 1.15, rand(-0.3, 0.3), up ? -0.55 : 0.6);
      leaf.scale.setScalar(0.0001);
      root.add(bud, leaf);
      this.leafGroup.add(root);
      this.leaves.push({ root, leaf, u, v: 0, size: rand(0.42, 0.55) });
    }
  }

  _resize() {
    const w = this.canvas.clientWidth || 1, h = this.canvas.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (this.mode === 'bar') {
      const H = 1.5;
      const d = H / 2 / Math.tan(degToRad(this.camera.fov / 2));
      this.camera.position.set(0, 0.62, d);
      this.camera.lookAt(0, 0.5, 0);
      const half = Math.max(0.6, (H * this.camera.aspect) / 2 - 0.22);
      if (Math.abs(half - (this.barHalf ?? 0)) > 0.05) { this.barHalf = half; this._layoutLeaves(); }
      if (this.curX == null) this.curX = -half + 0.35;
    }
  }

  _startCelebrate() {
    this.celebrate = this.reduced ? 1.0 : 1.7;
    this.happyHold = 3.2;
    this.celebrated = true;
  }

  // ----- behaviour -----

  _targets(dt) {
    const T = { hover: 0, flap: 0, lidU: -0.8, lidL: -1.15, yaw: 0, pitch: 0.05, roll: 0, pupil: 0.42, fluff: 0.056 };
    const L = this.look;
    L.next -= dt;

    if (this.state === 'thinking' && this.mode === 'perch') {
      // curious head-cocking, the classic owl move
      if (L.next <= 0) {
        const r = [0.42, -0.36, 0.18, -0.12][Math.floor(Math.random() * 4)];
        this.look = { next: rand(0.7, 1.4), yaw: rand(-0.25, 0.25), pitch: rand(-0.12, 0.05), roll: r };
      }
    } else if (L.next <= 0) {
      const turn = Math.random() < 0.3;
      this.look = { next: rand(1.6, 4), yaw: turn ? rand(-0.9, 0.9) : rand(-0.3, 0.3), pitch: rand(-0.06, 0.16), roll: Math.random() < 0.25 ? rand(-0.2, 0.2) : 0 };
    }
    T.yaw = this.look.yaw; T.pitch = this.look.pitch; T.roll = this.look.roll;

    if (this.followPointer && this.state === 'idle' && performance.now() - this.pointer.at < 4000) {
      const rc = this.canvas.getBoundingClientRect();
      const dx = (this.pointer.x - (rc.left + rc.width / 2)) / Math.max(260, rc.width * 1.4);
      const dy = (this.pointer.y - (rc.top + rc.height * 0.35)) / Math.max(260, rc.height * 1.4);
      T.yaw = clamp(dx, -1, 1) * 0.95;
      T.pitch = clamp(dy, -1, 1) * 0.45;
      T.roll = -T.yaw * 0.15;
    }

    switch (this.state) {
      case 'thinking':
        T.pupil = 0.54; T.lidU = -1.05;
        if (this.mode === 'bar') { T.hover = 1; T.flap = 1; T.yaw = this.dir * 0.4; T.pitch = 0.1; T.roll = 0; }
        break;
      case 'working':
        T.pupil = 0.48; T.pitch = 0.32;
        if (this.mode === 'perch') { T.flap = 0.42; T.yaw = Math.sin(this.t * 1.3) * 0.35; T.roll = Math.sin(this.t * 0.9) * 0.12; }
        else { T.yaw = this.dir * 0.35; T.roll = 0; }
        break;
      case 'done':
        T.pupil = 0.38;
        break;
    }
    if (this.celebrate > 0) { T.flap = 1; T.hover = this.mode === 'bar' ? 0.5 : 0.35; T.fluff = 0.085; T.yaw = 0; T.roll = 0; T.pitch = -0.05; }
    if (this.happyHold > 0) { T.lidL = -0.3; T.lidU = -0.62; T.fluff = Math.max(T.fluff, 0.07); T.pupil = 0.38; }
    return T;
  }

  _tick(now) {
    this._raf = requestAnimationFrame(this._tick);
    const dt = Math.min(0.05, (now - this._last) / 1000);
    this._last = now;
    if (!this._visible) return;
    this._update(dt);
    this.renderer.render(this.scene, this.camera);
  }

  _update(dt) {
    const P = this.P;
    this.t += dt;
    if (this.celebrate > 0) this.celebrate -= dt;
    if (this.happyHold > 0) this.happyHold -= dt;

    const T = this._targets(dt);
    P.hover = damp(P.hover, T.hover, 4, dt);
    P.flap = damp(P.flap, T.flap, 4, dt);
    P.yaw = damp(P.yaw, T.yaw, 9, dt);       // owls snap their heads round, then hold
    P.pitch = damp(P.pitch, T.pitch, 7, dt);
    P.roll = damp(P.roll, T.roll, 6, dt);
    P.pupil = damp(P.pupil, T.pupil, 3, dt);
    P.fluff = damp(P.fluff, T.fluff, 5, dt);
    P.lidL = damp(P.lidL, T.lidL, 6, dt);
    const lidU = damp(P.lidU, T.lidU, 6, dt);
    P.lidU = lidU;

    // slow owl blink
    const b = this.blink;
    b.next -= dt;
    if (b.next <= 0 && b.t < 0) b.t = 0;
    let blink = 0;
    if (b.t >= 0) {
      b.t += dt;
      blink = Math.sin(Math.min(1, b.t / 0.32) * Math.PI);
      if (b.t >= 0.32) { b.t = -1; b.next = rand(2.8, 6); }
    }

    const hz = lerp(1.0, 3.2, P.flap) * (this.reduced ? 0.55 : 1);
    this.phase += dt * TAU * hz * Math.min(1, P.flap * 6);

    const pose = { ...P, lidU: lerp(P.lidU, 1.5, blink) };
    this.owl.pose(pose, this.phase);
    this.owl.U.uBreath.value = 0.012 * Math.sin(this.t * 2.1);
    const bob = P.hover * ((this.reduced ? 0.2 : 0.34) - 0.05 * Math.cos(this.phase));
    this.owl.root.position.y = bob;

    if (this.mode === 'bar') this._updateBar(dt, P, bob);
    else this._updatePerch(P, bob);
  }

  _updatePerch(P, bob) {
    const spread = clamp(P.flap * 1.5, 0, 1);
    const fh = lerp(2.55, 3.6, spread) + P.hover * 0.4;
    const fw = lerp(2.1, 5.0, spread);
    const F = Math.max(fh, fw / this.camera.aspect);
    const cy = lerp(-0.02, 0.15, spread) + P.hover * 0.18;
    const d = F / 2 / Math.tan(degToRad(this.camera.fov / 2));
    this.camera.position.set(0, cy + 0.3, d);
    this.camera.lookAt(0, cy, 0);
    this.BU.uOwlX.value = 0;
    this.BU.uLift.value = clamp(bob / 0.3, 0, 1);
  }

  _updateBar(dt, P, bob) {
    const half = this.barHalf, s = 0.45;
    const lo = -half + 0.3, hi = half - 0.3;
    const flying = this.state === 'thinking' || (this.state === 'working' && this.progress == null);
    let moving = false;

    if (flying) {
      const range = Math.max(0, hi - lo) / 2;
      const target = (lo + hi) / 2 + range * Math.sin(this.t * (this.reduced ? 0.3 : 0.55));
      const nx = damp(this.curX, target, 5, dt);
      if (Math.abs(nx - this.curX) > 2e-4) this.dir = Math.sign(nx - this.curX);
      this.curX = nx;
      P.fill = damp(P.fill, 0, 3, dt);
    } else {
      const p = this.state === 'done' ? 1 : this.progress ?? 0;
      P.fill = damp(P.fill, p, 3, dt);
      const target = clamp(lerp(-half, half, p), lo, hi);
      const gap = target - this.curX;
      if (Math.abs(gap) > 0.01) {
        moving = true;
        this.dir = Math.sign(gap);
        this.curX += this.dir * Math.min(Math.abs(gap), dt * (this.reduced ? 0.6 : 1.1));
      }
      if (this.state === 'done' && !moving && !this.celebrated) this._startCelebrate();
    }

    // little sideways hops along the branch
    if (moving || this.hopPhase % 1 > 0.04) this.hopPhase += dt * (this.reduced ? 1.6 : 3.2);
    else this.hopPhase = Math.round(this.hopPhase);
    const hop = Math.abs(Math.sin(this.hopPhase * Math.PI)) * (this.reduced ? 0.06 : 0.12);
    if (moving) P.flap = Math.max(P.flap, 0.3 * Math.abs(Math.sin(this.hopPhase * Math.PI)));

    P.bodyYaw = damp(P.bodyYaw, moving || flying ? this.dir * 0.6 : 0, 5, dt);
    this.mover.rotation.y = P.bodyYaw;
    this.mover.position.set(this.curX, 0.04 + 0.975 * s + hop, 0);
    this.mover.scale.setScalar(s);

    this.BU.uOwlX.value = this.curX;
    this.BU.uLift.value = clamp((hop + bob * s) / 0.12, 0, 1);

    for (const l of this.leaves) {
      const on = P.fill >= l.u - 0.001 && P.fill > 0.01 ? 1 : 0;
      l.v = clamp(l.v + (on ? dt * 2.2 : -dt * 3), 0, 1);
      const k = l.v > 0 ? easeOutBack(l.v) * l.size : 0.0001;
      l.leaf.scale.setScalar(Math.max(k, 0.0001));
    }
  }
}
