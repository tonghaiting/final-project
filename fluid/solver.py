"""不可压缩 Navier-Stokes 的投影法求解器，带浸没固体与示踪粒子平流。

网格为均匀 cell-centred 网格，节点 (i, j) 对应连续坐标 (i, j)，间距 h = 1，
速度 u, v 与压力 p 同点存储（collocated），固体由布尔掩膜表示。
"""

import numpy as np


def bilinear(f, x, y):
    """在 cell-centred 网格上双线性采样 f[j, i]，越界按最近值夹紧。"""
    ny, nx = f.shape
    xc = np.clip(x, 0.0, nx - 1.0 - 1e-9)
    yc = np.clip(y, 0.0, ny - 1.0 - 1e-9)
    i0 = xc.astype(np.intp)
    j0 = yc.astype(np.intp)
    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)
    sx = xc - i0
    sy = yc - j0
    return ((1.0 - sy) * ((1.0 - sx) * f[j0, i0] + sx * f[j0, i1])
            + sy * ((1.0 - sx) * f[j1, i0] + sx * f[j1, i1]))


def make_solid(nx, ny, shape, cx, cy, size, ratio=1.0, angle=0.0):
    """生成固体掩膜。size 为特征半径（圆即半径，方为半边的长，椭圆为半长轴）。"""
    x, y = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64))
    dx = x - cx
    dy = y - cy
    if shape == "circle":
        return dx * dx + dy * dy <= size * size
    if shape == "square":
        return (np.abs(dx) <= size) & (np.abs(dy) <= size)
    if shape == "ellipse":
        c, s = np.cos(angle), np.sin(angle)
        a = size
        b = size * ratio
        xi = c * dx + s * dy
        eta = -s * dx + c * dy
        return (xi / a) ** 2 + (eta / b) ** 2 <= 1.0
    raise ValueError("未知形状: %r" % shape)


class Fluid:
    """u_inf 来流速度，dt 时间步长，nu 运动粘度，iters 压力 Jacobi 迭代次数。

    所有量均为无量纲：长度以网格间距 h=1 计，速度以来流 u_inf 计。
    Re = u_inf * D / nu，D 为物体特征尺寸。
    """

    def __init__(self, nx, ny, solid=None, u_inf=1.0, dt=0.5, nu=0.05,
                 iters=30, vorticity_gain=1.0, omega=1.0):
        self.nx = int(nx)
        self.ny = int(ny)
        self.u_inf = float(u_inf)
        self.dt = float(dt)
        self.nu = float(nu)
        self.iters = int(iters)
        self.eps = float(vorticity_gain)
        self.omega = float(omega)
        self.solid = np.zeros((self.ny, self.nx), dtype=bool) if solid is None else solid.astype(bool)
        self.u = np.full((self.ny, self.nx), self.u_inf, dtype=np.float64)
        self.v = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.p = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.div_rms = (0.0, 0.0)
        self.time = 0.0
        self._x, self._y = np.meshgrid(np.arange(self.nx, dtype=np.float64),
                                       np.arange(self.ny, dtype=np.float64))
        self._apply_bc()

    # ------------------------------------------------------------------ 网格工具
    @staticmethod
    def _pad(q):
        """四周复制一层边界值，避免差分时的环回(wraparound)。"""
        qp = np.empty((q.shape[0] + 2, q.shape[1] + 2), dtype=q.dtype)
        qp[1:-1, 1:-1] = q
        qp[0, 1:-1] = q[0]
        qp[-1, 1:-1] = q[-1]
        qp[1:-1, 0] = q[:, 0]
        qp[1:-1, -1] = q[:, -1]
        qp[0, 0] = q[0, 0]
        qp[0, -1] = q[0, -1]
        qp[-1, 0] = q[-1, 0]
        qp[-1, -1] = q[-1, -1]
        return qp

    def _neighbors(self, q):
        """四个邻点取值；固体邻点与域外都替换为 q 本身，即零梯度(Neumann)。"""
        qp = self._pad(q)
        sp = self._pad(self.solid)
        l = np.where(sp[1:-1, :-2], q, qp[1:-1, :-2])
        r = np.where(sp[1:-1, 2:], q, qp[1:-1, 2:])
        u = np.where(sp[:-2, 1:-1], q, qp[:-2, 1:-1])
        d = np.where(sp[2:, 1:-1], q, qp[2:, 1:-1])
        return l, r, u, d

    def _apply_bc(self):
        """左入流、右出流、上下自由滑移，固体内速度归零(无滑移)。"""
        u, v = self.u, self.v
        u[:, 0] = self.u_inf
        v[:, 0] = 0.0
        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]
        u[0, :] = u[1, :]
        v[0, :] = 0.0
        u[-1, :] = u[-2, :]
        v[-1, :] = v[-2, :]
        u[self.solid] = 0.0
        v[self.solid] = 0.0

    # ------------------------------------------------------------------ 各算子
    def _advect(self):
        """半拉格朗日平流：沿特征线向后追踪，对旧场双线性插值。"""
        u0, v0 = self.u.copy(), self.v.copy()
        u0[self.solid] = 0.0
        v0[self.solid] = 0.0
        xs = self._x - self.dt * u0
        ys = self._y - self.dt * v0
        self.u = bilinear(u0, xs, ys)
        self.v = bilinear(v0, xs, ys)

    def _diffuse(self):
        """粘性项解 (I - nu*dt*Lap) w = w* 的 Jacobi 迭代。"""
        if self.nu <= 0.0:
            return
        a = self.nu * self.dt
        n = max(4, self.iters // 4)
        for name in ("u", "v"):
            w = getattr(self, name)
            rhs = w.copy()
            for _ in range(n):
                l, r, u, d = self._neighbors(w)
                w = (rhs + a * (l + r + u + d)) / (1.0 + 4.0 * a)
            w[self.solid] = 0.0
            setattr(self, name, w)

    def _confinement(self):
        """涡度约束力 f ∝ N × omega，补偿半拉格朗日平流抹平小尺度涡。

        这里把它归一化成"每步速度增量不超过 eps*u_inf"，因此 eps 与 dt、网格
        分辨率解耦，取值范围约 [0, 0.3]；原始 Fedkiw 形式的力会随 1/dt 放大，
        在本项目的 dt~1 设置下直接把速度推到发散。
        """
        if self.eps <= 0.0:
            return
        w = self.vorticity()
        aw = self._pad(np.abs(w))
        gx = 0.5 * (aw[1:-1, 2:] - aw[1:-1, :-2])
        gy = 0.5 * (aw[2:, 1:-1] - aw[:-2, 1:-1])
        norm = np.hypot(gx, gy) + 1e-12
        fx = (gy / norm) * w
        fy = -(gx / norm) * w
        peak = float(np.hypot(fx, fy).max())
        if peak <= 1e-12:
            return
        gain = self.eps * self.u_inf / (self.dt * peak)
        self.u += self.dt * gain * fx
        self.v += self.dt * gain * fy
        self.u[self.solid] = 0.0
        self.v[self.solid] = 0.0

    def _project(self):
        """Hodge 分解：解 Lap p = div u*，再减梯度得到无散场。

        边界上入流与上下壁取 Neumann(∂p/∂n=0，靠 _pad 的复制实现)，出流取
        Dirichlet p=0。若全域都是 Neumann，则 ∫div 必须为零才有解；而我们固定了
        入流、尾流又使出流通量失衡，Neumann 问题不相容，Jacobi 会把残差停在
        1e-2 量级——表现为"投影后散度反而变大"。
        """
        up = self._pad(self.u)
        vp = self._pad(self.v)
        # 固体内速度为 0，中心差分因此把界面处的质量通量算作发散源
        div = 0.5 * (up[1:-1, 2:] - up[1:-1, :-2]) + 0.5 * (vp[2:, 1:-1] - vp[:-2, 1:-1])
        div[self.solid] = 0.0
        solid = self.solid
        p = self.p
        w = self.omega
        for _ in range(self.iters):
            l, r, u, d = self._neighbors(p)
            p = np.where(solid, self.p, p + w * (0.25 * (l + r + u + d - div) - p))
            p[:, -1] = 0.0
        self.p = p
        l, r, u, d = self._neighbors(p)
        self.u -= 0.5 * (r - l)
        self.v -= 0.5 * (d - u)
        self.u[solid] = 0.0
        self.v[solid] = 0.0
        m = ~solid
        self.div_rms = (float(np.sqrt(np.mean(div[m] ** 2))), self.divergence_norm())

    def step(self):
        self._advect()
        self._diffuse()
        self._confinement()
        self._apply_bc()
        self._project()
        self.time += self.dt
        if not (np.isfinite(self.u).all() and np.isfinite(self.v).all()):
            raise RuntimeError("速度场在 t=%.1f 出现非有限值：把 omega 调回 1.0，"
                               "或减小 vcg / dt 后重试" % self.time)

    # ------------------------------------------------------------------ 诊断
    def vorticity(self):
        vp = self._pad(self.v)
        up = self._pad(self.u)
        w = 0.5 * (vp[1:-1, 2:] - vp[1:-1, :-2]) - 0.5 * (up[2:, 1:-1] - up[:-2, 1:-1])
        w[self.solid] = 0.0
        return w

    def speed(self):
        return np.hypot(self.u, self.v)

    def divergence_norm(self):
        up = self._pad(self.u)
        vp = self._pad(self.v)
        div = 0.5 * (up[1:-1, 2:] - up[1:-1, :-2]) + 0.5 * (vp[2:, 1:-1] - vp[:-2, 1:-1])
        m = ~self.solid
        return float(np.sqrt(np.mean(div[m] ** 2)))


class TracerSet:
    """示踪粒子：被动平流(不反馈流体)，用二阶 Runge-Kutta 积分 dx/dt = u。

    每条轨迹独立成段；粒子出域、入固体或滞止过久即重置，段缓冲存 (x, y, |u|)。
    """

    def __init__(self, fluid, count=160, max_points=600, fill=True, rng=None):
        self.f = fluid
        self.dt = fluid.dt
        self.count = int(count)
        self.max_points = int(max_points)
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.buf = np.zeros((self.count, self.max_points, 3), dtype=np.float64)
        self.segs = []
        self.n = np.zeros(self.count, dtype=np.intp)
        self.seg_t0 = np.zeros(self.count, dtype=np.int64)
        self.age = np.zeros(self.count, dtype=np.float64)
        self.spd = np.zeros(self.count, dtype=np.float64)
        self.x = np.zeros(self.count, dtype=np.float64)
        self.y = np.zeros(self.count, dtype=np.float64)
        self._respawn(np.arange(self.count))
        if fill:
            self._scatter(np.arange(self.count))

    def _inlet(self, idx):
        """按行分层播种在入流边界，附带抖动以免轨迹形成整齐网格。"""
        k = idx.size
        rows = (np.arange(k) + self.rng.random(k)) / k
        self.x[idx] = 0.5
        self.y[idx] = rows * (self.f.ny - 1)

    def _blocked(self):
        f = self.f
        i = np.clip(np.rint(self.x).astype(np.intp), 0, f.nx - 1)
        j = np.clip(np.rint(self.y).astype(np.intp), 0, f.ny - 1)
        return f.solid[j, i]

    def _scatter(self, idx):
        """撒到流体区域内，让画面一开始就有贯穿全场的轨迹。"""
        for _ in range(50):
            k = idx.size
            self.x[idx] = self.rng.uniform(0.5, self.f.nx - 1.5, k)
            self.y[idx] = self.rng.uniform(0.5, self.f.ny - 1.5, k)
            free = idx[self._blocked()[idx]]
            if free.size == 0:
                return
            idx = free

    def _respawn(self, idx, scatter=False):
        if idx.size == 0:
            return
        (self._scatter if scatter else self._inlet)(idx)
        self.n[idx] = 0
        self.age[idx] = 0.0

    def advance(self, record=False, tick=0):
        f = self.f
        ux = bilinear(f.u, self.x, self.y)
        uy = bilinear(f.v, self.x, self.y)
        mx = self.x + 0.5 * self.dt * ux
        my = self.y + 0.5 * self.dt * uy
        vx = bilinear(f.u, mx, my)
        vy = bilinear(f.v, mx, my)
        self.x += self.dt * vx
        self.y += self.dt * vy
        self.spd = np.hypot(vx, vy)
        self.age += 1.0
        if record:
            rows = np.nonzero(self.n < self.max_points)[0]
            fresh = rows[self.n[rows] == 0]
            self.seg_t0[fresh] = tick  # 记录该段起点时刻，前端据此决定它何时开始出现
            self.buf[rows, self.n[rows], 0] = self.x[rows]
            self.buf[rows, self.n[rows], 1] = self.y[rows]
            self.buf[rows, self.n[rows], 2] = self.spd[rows]
            self.n[rows] += 1
        out = ((self.x < 0.0) | (self.x > f.nx - 1.0)
               | (self.y < 0.0) | (self.y > f.ny - 1.0))
        stale = (self.age * self.dt > 60.0) & (self.spd < 0.02 * f.u_inf)
        done = np.nonzero(out | self._blocked() | stale | (self.n >= self.max_points))[0]
        for i in done:
            if self.n[i] > 2:
                self.segs.append((int(self.seg_t0[i]), self.buf[i, :self.n[i]].copy()))
            self._respawn(np.array([i]))

    def flush(self):
        """收尾：把仍在域内的粒子当前轨迹并入结果，返回 (起始时刻, 点列) 列表。"""
        for i in np.nonzero(self.n > 2)[0]:
            self.segs.append((int(self.seg_t0[i]), self.buf[i, :self.n[i]].copy()))
        self.n[:] = 0
        return self.segs
