/* 不可压缩 Navier-Stokes 投影法求解器（fluid/solver.py 的 JS 移植）。
 *
 * 与 Python 版同一套离散格式：半拉格朗日平流、隐式粘性、中心差分散度、
 * 出流压力 Dirichlet、固体内无滑移、限幅涡度约束。唯一区别是压力迭代用
 * 逐点 Gauss-Seidel 而不是 masked-Jacobi——JS 里逐格循环不贵，同一方程
 * 收敛得更快，所以少量迭代就能把散度残差压下去。
 *
 * 普通脚本（非 ES module），file:// 直接打开也能用。
 */
(function (global) {
  'use strict';

  function buildMask(nx, ny, shapes) {
    const s = new Uint8Array(nx * ny);
    for (const o of shapes) {
      const rad = o.angle * Math.PI / 180, c = Math.cos(rad), sn = Math.sin(rad);
      const a = o.size, b = o.shape === 'ellipse' ? o.size * o.ratio : o.size;
      const reach = Math.max(a, b);
      const i0 = Math.max(0, Math.floor(o.cx - reach - 1)), i1 = Math.min(nx - 1, Math.ceil(o.cx + reach + 1));
      const j0 = Math.max(0, Math.floor(o.cy - reach - 1)), j1 = Math.min(ny - 1, Math.ceil(o.cy + reach + 1));
      for (let j = j0; j <= j1; j++) {
        for (let i = i0; i <= i1; i++) {
          const dx = i - o.cx, dy = j - o.cy;
          let inside;
          if (o.shape === 'square') inside = Math.abs(dx) <= a && Math.abs(dy) <= a;
          else if (o.shape === 'ellipse') {
            const xi = c * dx + sn * dy, eta = -sn * dx + c * dy;
            inside = (xi * xi) / (a * a) + (eta * eta) / (b * b) <= 1;
          } else inside = dx * dx + dy * dy <= a * a;
          if (inside) s[j * nx + i] = 1;
        }
      }
    }
    return s;
  }

  function sample(f, nx, ny, x, y) {
    const xc = x < 0 ? 0 : (x > nx - 1.0001 ? nx - 1.0001 : x);
    const yc = y < 0 ? 0 : (y > ny - 1.0001 ? ny - 1.0001 : y);
    const i0 = xc | 0, j0 = yc | 0;
    const i1 = i0 + 1 < nx ? i0 + 1 : i0, j1 = j0 + 1 < ny ? j0 + 1 : j0;
    const sx = xc - i0, sy = yc - j0;
    const r0 = j0 * nx, r1 = j1 * nx;
    return (1 - sy) * ((1 - sx) * f[r0 + i0] + sx * f[r0 + i1])
      + sy * ((1 - sx) * f[r1 + i0] + sx * f[r1 + i1]);
  }

  class Fluid {
    constructor(cfg) {
      this.nx = cfg.nx; this.ny = cfg.ny;
      this.dt = cfg.dt; this.uInf = cfg.uInf;
      this.iters = cfg.iters; this.eps = cfg.vcg;
      this.nu = cfg.uInf * cfg.D / cfg.re;
      this.shapes = cfg.shapes.slice();
      const n = this.nx * this.ny;
      this.solid = buildMask(this.nx, this.ny, this.shapes);
      this.u = new Float32Array(n).fill(cfg.uInf);
      this.v = new Float32Array(n);
      this.p = new Float32Array(n);
      this.u0 = new Float32Array(n); this.v0 = new Float32Array(n);
      this.div = new Float32Array(n); this.w = new Float32Array(n);
      this.time = 0;
      this.divBefore = 0; this.divAfter = 0;
      this.applyBC();
    }

    /* 换掩膜后：固体内归零(无滑移)，刚从固体里解放出来的格子补上来流速度，
       否则它们会停在 0 上，要靠扩散慢慢填，画面里留下一个假的速度洞。 */
    setShapes(shapes) {
      const prev = this.solid;
      this.shapes = shapes.slice();
      this.solid = buildMask(this.nx, this.ny, shapes);
      for (let k = 0, n = this.nx * this.ny; k < n; k++) {
        if (this.solid[k]) { this.u[k] = 0; this.v[k] = 0; }
        else if (prev[k]) { this.u[k] = this.uInf; this.v[k] = 0; this.p[k] = 0; }
      }
      this.applyBC();
    }

    applyBC() {
      const { nx, ny, u, v, solid, uInf } = this;
      for (let j = 0; j < ny; j++) {
        const k = j * nx;
        u[k] = uInf; v[k] = 0;                       // 入流
        u[k + nx - 1] = u[k + nx - 2];                // 出流零梯度
        v[k + nx - 1] = v[k + nx - 2];
      }
      for (let i = 0; i < nx; i++) {
        u[i] = u[i + nx]; v[i] = 0;                   // 上壁自由滑移
        const kb = (ny - 1) * nx;
        u[kb + i] = u[kb - nx + i]; v[kb + i] = v[kb - nx + i];
      }
      for (let k = 0; k < nx * ny; k++) if (solid[k]) { u[k] = 0; v[k] = 0; }
    }

    advect() {
      const { nx, ny, u, v, u0, v0, solid, dt } = this;
      u0.set(u); v0.set(v);
      for (let k = 0, n = nx * ny; k < n; k++) if (solid[k]) { u0[k] = 0; v0[k] = 0; }
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const k = j * nx + i;
          if (solid[k]) { u[k] = 0; v[k] = 0; continue; }
          const xs = i - dt * u0[k], ys = j - dt * v0[k];
          u[k] = sample(u0, nx, ny, xs, ys);
          v[k] = sample(v0, nx, ny, xs, ys);
        }
      }
    }

    diffuse(f, iters) {
      const a = this.nu * this.dt;
      if (a <= 0) return;
      const { nx, ny, solid } = this;
      const rhs = this.u0;                       // 复用暂存：调用处保证此时不用于平流
      rhs.set(f);
      for (let it = 0; it < iters; it++) {
        for (let j = 0; j < ny; j++) {
          for (let i = 0; i < nx; i++) {
            const k = j * nx + i;
            if (solid[k]) continue;
            let sum = 0, cnt = 0;
            if (i > 0 && !solid[k - 1]) { sum += f[k - 1]; cnt++; }
            if (i < nx - 1 && !solid[k + 1]) { sum += f[k + 1]; cnt++; }
            if (j > 0 && !solid[k - nx]) { sum += f[k - nx]; cnt++; }
            if (j < ny - 1 && !solid[k + nx]) { sum += f[k + nx]; cnt++; }
            f[k] = (rhs[k] + a * sum) / (1 + a * cnt);
          }
        }
      }
    }

    vorticity() {
      const { nx, ny, u, v, w, solid } = this;
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const k = j * nx + i;
          if (solid[k]) { w[k] = 0; continue; }
          const uL = i > 0 ? u[k - 1] : u[k], uR = i < nx - 1 ? u[k + 1] : u[k];
          const vU = j > 0 ? v[k - nx] : v[k], vD = j < ny - 1 ? v[k + nx] : v[k];
          w[k] = 0.5 * (vD - vU) - 0.5 * (uR - uL);   // dv/dx - du/dy
        }
      }
      return w;
    }

    confinement() {
      if (this.eps <= 0) return;
      const { nx, ny, w, u, v, solid, dt, eps, uInf } = this;
      this.vorticity();
      const fx = this.u0, fy = this.v0;
      let peak = 0;
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const k = j * nx + i;
          if (solid[k]) { fx[k] = 0; fy[k] = 0; continue; }
          const aL = i > 0 ? Math.abs(w[k - 1]) : Math.abs(w[k]);
          const aR = i < nx - 1 ? Math.abs(w[k + 1]) : Math.abs(w[k]);
          const aU = j > 0 ? Math.abs(w[k - nx]) : Math.abs(w[k]);
          const aD = j < ny - 1 ? Math.abs(w[k + nx]) : Math.abs(w[k]);
          const gx = 0.5 * (aR - aL), gy = 0.5 * (aD - aU);
          const nn = Math.hypot(gx, gy) + 1e-12;
          fx[k] = (gy / nn) * w[k];
          fy[k] = -(gx / nn) * w[k];
          const m = Math.hypot(fx[k], fy[k]);
          if (m > peak) peak = m;
        }
      }
      if (peak <= 1e-12) return;
      const gain = eps * uInf / (dt * peak);
      for (let k = 0, n = nx * ny; k < n; k++) {
        if (solid[k]) continue;
        u[k] += dt * gain * fx[k];
        v[k] += dt * gain * fy[k];
      }
    }

    project() {
      const { nx, ny, u, v, p, div, solid, iters } = this;
      const last = nx - 1;
      let before = 0, cntb = 0;
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const k = j * nx + i;
          if (solid[k]) { div[k] = 0; continue; }
          const uL = i > 0 ? u[k - 1] : u[k], uR = i < nx - 1 ? u[k + 1] : u[k];
          const vU = j > 0 ? v[k - nx] : v[k], vD = j < ny - 1 ? v[k + nx] : v[k];
          div[k] = 0.5 * (uR - uL) + 0.5 * (vD - vU);
          before += div[k] * div[k]; cntb++;
        }
      }
      for (let it = 0; it < iters; it++) {
        for (let j = 0; j < ny; j++) {
          for (let i = 0; i < nx; i++) {
            const k = j * nx + i;
            if (solid[k]) continue;
            if (i === last) { p[k] = 0; continue; }      // 出流 p=0，否则纯 Neumann 不相容
            let sum = 0, cnt = 0;
            if (i > 0 && !solid[k - 1]) { sum += p[k - 1]; cnt++; }
            if (i < last && !solid[k + 1]) { sum += p[k + 1]; cnt++; }
            if (j > 0 && !solid[k - nx]) { sum += p[k - nx]; cnt++; }
            if (j < ny - 1 && !solid[k + nx]) { sum += p[k + nx]; cnt++; }
            if (cnt === 0) { p[k] = 0; continue; }
            p[k] = (sum - div[k]) / cnt;
          }
        }
      }
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const k = j * nx + i;
          if (solid[k]) continue;
          const pr = i === last ? 0 : (solid[k + 1] ? p[k] : p[k + 1]);
          const pl = i === 0 ? p[k] : (solid[k - 1] ? p[k] : p[k - 1]);
          const pd = j === ny - 1 ? p[k] : (solid[k + nx] ? p[k] : p[k + nx]);
          const pu = j === 0 ? p[k] : (solid[k - nx] ? p[k] : p[k - nx]);
          u[k] -= 0.5 * (pr - pl);
          v[k] -= 0.5 * (pd - pu);
        }
      }
      for (let k = 0, n = nx * ny; k < n; k++) if (solid[k]) { u[k] = 0; v[k] = 0; }
      let after = 0, cnta = 0;
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const k = j * nx + i;
          if (solid[k]) continue;
          const uL = i > 0 ? u[k - 1] : u[k], uR = i < nx - 1 ? u[k + 1] : u[k];
          const vU = j > 0 ? v[k - nx] : v[k], vD = j < ny - 1 ? v[k + nx] : v[k];
          const d = 0.5 * (uR - uL) + 0.5 * (vD - vU);
          after += d * d; cnta++;
        }
      }
      this.divBefore = Math.sqrt(before / Math.max(1, cntb));
      this.divAfter = Math.sqrt(after / Math.max(1, cnta));
    }

    step() {
      this.advect();
      const d = Math.max(4, this.iters >> 2);
      this.diffuse(this.u, d);
      this.diffuse(this.v, d);
      this.confinement();
      this.applyBC();
      this.project();
      this.time += this.dt;
      for (let k = 0, n = this.nx * this.ny; k < n; k++) {
        if (isFinite(this.u[k]) && isFinite(this.v[k])) continue;
        this.reinitialize();
        throw new Error('速度场在 t=' + this.time.toFixed(1) + ' 发散：调小 vcg 或 dt，或增大 iters');
      }
    }

    /* 发散时把场拉回均匀来流，让页面还能改参数继续跑，而不是一直抛 NaN。 */
    reinitialize() {
      this.u.fill(this.uInf); this.v.fill(0); this.p.fill(0);
      this.time = 0;
      this.applyBC();
    }

    speed() {
      const { u, v } = this, s = new Float32Array(u.length);
      for (let k = 0; k < u.length; k++) s[k] = Math.hypot(u[k], v[k]);
      return s;
    }
  }

  class Tracers {
    constructor(fluid, count, trailLen) {
      this.f = fluid; this.n = count; this.trail = trailLen;
      this.x = new Float32Array(count); this.y = new Float32Array(count);
      this.spd = new Float32Array(count); this.age = new Float32Array(count);
      this.cnt = new Int32Array(count);
      this.buf = new Float32Array(count * trailLen * 3);
      this.seedAll(true);
    }

    seedAll(initial) {
      for (let k = 0; k < this.n; k++) this.respawn(k, initial);
    }

    respawn(k, scatter) {
      const f = this.f;
      if (scatter) {
        for (let tries = 0; tries < 40; tries++) {
          const x = 0.5 + Math.random() * (f.nx - 2), y = 0.5 + Math.random() * (f.ny - 2);
          if (!f.solid[y * f.nx + x | 0]) { this.x[k] = x; this.y[k] = y; break; }
          this.x[k] = x; this.y[k] = y;
        }
      } else {
        this.x[k] = 0.5;
        this.y[k] = (k + Math.random()) / this.n * (f.ny - 1);
      }
      this.cnt[k] = 0; this.age[k] = 0;
      this.spd[k] = f.uInf;
    }

    step() {
      const f = this.f, dt = f.dt, nx = f.nx, ny = f.ny;
      for (let k = 0; k < this.n; k++) {
        const x = this.x[k], y = this.y[k];
        const ux = sample(f.u, nx, ny, x, y), uy = sample(f.v, nx, ny, x, y);
        const mx = x + 0.5 * dt * ux, my = y + 0.5 * dt * uy;
        const vx = sample(f.u, nx, ny, mx, my), vy = sample(f.v, nx, ny, mx, my);
        const nxp = x + dt * vx, nyp = y + dt * vy;
        this.spd[k] = Math.hypot(vx, vy);
        this.age[k] += 1;
        const ci = Math.max(0, Math.min(nx - 1, nxp | 0));
        const cj = Math.max(0, Math.min(ny - 1, nyp | 0));
        const dead = nxp < 0 || nxp > nx - 1 || nyp < 0 || nyp > ny - 1 || f.solid[cj * nx + ci];
        if (!dead) { this.x[k] = nxp; this.y[k] = nyp; }
        let c = this.cnt[k];
        if (c === this.trail) {
          this.buf.copyWithin(k * this.trail * 3, k * this.trail * 3 + 3, k * this.trail * 3 + (this.trail - 1) * 3);
        } else { this.cnt[k] = ++c; }
        const o = (k * this.trail + c - 1) * 3;
        this.buf[o] = this.x[k]; this.buf[o + 1] = this.y[k]; this.buf[o + 2] = this.spd[k];
        if (dead || (this.age[k] * dt > 60 && this.spd[k] < 0.02 * f.uInf)) this.respawn(k, false);
      }
    }

    path(k) {
      const c = this.cnt[k], base = k * this.trail * 3;
      return { pts: this.buf.subarray(base, base + c * 3), len: c };
    }
  }

  /* 过零法估计脱落数：相邻过零间隔是中位数半周期。短窗口 FFT 只会把窗函数
     最低 bin 当成主峰，给出假的低频解。 */
  function strouhal(series, sampleDt, D, uInf) {
    if (series.length < 12) return 0;
    let med = 0;
    const x = Float64Array.from(series);
    const sorted = Array.from(x).sort((a, b) => a - b);
    med = sorted[sorted.length >> 1];
    const cross = [];
    for (let i = 0; i < x.length - 1; i++) {
      const a = x[i] - med, b = x[i + 1] - med;
      if ((a < 0 && b >= 0) || (b < 0 && a >= 0)) {
        const frac = a !== b ? a / (a - b) : 0;
        cross.push((i + frac) * sampleDt);
      }
    }
    if (cross.length < 5) return 0;
    const gaps = [];
    for (let i = 1; i < cross.length; i++) gaps.push(cross[i] - cross[i - 1]);
    gaps.sort((a, b) => a - b);
    const half = gaps[gaps.length >> 1];
    if (!(half > 0)) return 0;
    return D / (2 * half) / uInf;
  }

  global.FluidSim = { Fluid, Tracers, buildMask, sample, strouhal };
})(typeof window !== 'undefined' ? window : globalThis);
