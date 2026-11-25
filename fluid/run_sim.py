"""运行绕流模拟，把示踪粒子轨迹与涡度场导出给 web/replay.html 播放。

    py fluid/run_sim.py                          # 默认圆柱 Re=300
    py fluid/run_sim.py --shape square --re 120
    py fluid/run_sim.py --shape ellipse --size 14 --ratio 0.3 --angle 20
    py fluid/run_sim.py --settle-transits 1.5 --periods 2

导出 web/flow_data.js（浏览器 file:// 下可直接 <script> 读取）与 data/flow_data.json。
所有物理量无量纲：长度以网格间距 h=1 计，速度以来流 u_inf 计。
"""

import argparse
import base64
import json
import os
import sys
import time

import numpy as np

# Windows 控制台默认按 GBK 输出，中文提示会变成乱码；统一改成 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import Fluid, TracerSet, make_solid  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = 32767.0
ST_GUESS = 0.2  # 圆柱高 Re 下的经验脱落数，仅用于估算记录时长


def encode(values, lo, hi):
    """把浮点数组量化到 int16 再 base64，避免 JSON 里塞上百万个十进制数。"""
    a = np.asarray(values, dtype=np.float64)
    t = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    q = np.rint(np.clip(t, 0.0, 1.0) * (2.0 * Q) - Q).astype("<i2")
    return base64.b64encode(q.tobytes()).decode("ascii")


def block_mean(a, sub):
    ny, nx = a.shape
    hy, hx = ny // sub, nx // sub
    if hy < 1 or hx < 1:
        return a
    return a[: hy * sub, : hx * sub].reshape(hy, sub, hx, sub).mean(axis=(1, 3))


def shed_frequency(series, sample_dt):
    """尾流中心线横速探针的过零周期估计；相邻两次过零算半个周期。

    过零用线性插值定位，间隔取中位数。不用 FFT：记录窗口往往只有 1~2 个周期，
    FFT 只会把窗函数最低的 bin 当成主峰，给出假的低频解。
    """
    x = np.asarray(series, dtype=np.float64)
    if x.size < 12:
        return 0.0
    x = x - np.median(x)
    cross = []
    for i in range(x.size - 1):
        a, b = x[i], x[i + 1]
        if (a < 0.0 <= b) or (b < 0.0 <= a):
            frac = a / (a - b) if a != b else 0.0
            cross.append((i + frac) * sample_dt)
    if len(cross) < 5:
        return 0.0
    half = float(np.median(np.diff(cross)))
    return 1.0 / (2.0 * half) if half > 0.0 else 0.0


def parse_args():
    p = argparse.ArgumentParser(description="绕流模拟与可视化数据导出")
    g = p.add_argument_group("物体")
    g.add_argument("--shape", choices=["circle", "square", "ellipse"], default="circle")
    g.add_argument("--size", type=float, default=14.0, help="特征半径(格数)，D = 2*size")
    g.add_argument("--ratio", type=float, default=0.3, help="ellipse 短轴/长轴")
    g.add_argument("--angle", type=float, default=20.0, help="ellipse 倾角(度)")
    g.add_argument("--x-pos", type=float, default=0.18, help="物体横坐标占域长比例")
    g.add_argument("--y-pos", type=float, default=0.5, help="物体纵坐标占域高比例")
    w = p.add_argument_group("数值")
    w.add_argument("--nx", type=int, default=224)
    w.add_argument("--ny", type=int, default=132)
    w.add_argument("--re", type=float, default=300.0, help="雷诺数 Re = u_inf*D/nu")
    w.add_argument("--u", type=float, default=1.0, help="来流速度 u_inf")
    w.add_argument("--dt", type=float, default=1.0,
                   help="时间步长；取 CFL=u_inf*dt/h≈1 时半拉格朗日的数值粘性最小")
    w.add_argument("--iters", type=int, default=60, help="压力 Poisson 的迭代数")
    w.add_argument("--omega", type=float, default=1.0,
                   help="逐次超松弛因子；本实现里固体邻点已等效欠松弛，取 >1 会发散")
    w.add_argument("--vcg", type=float, default=0.08,
                   help="涡度约束强度 eps，含义是每步速度增量上限占 u_inf 的比例，建议 0~0.3")
    w.add_argument("--settle-transits", type=float, default=2.5, help="稳态等待几个穿越时间")
    w.add_argument("--periods", type=float, default=4.0, help="记录几个涡脱周期")
    w.add_argument("--record-every", type=int, default=2,
                   help="每隔几步存一个轨迹点；取 2 时播放器能插值到 60 fps，取 4 更省体积")
    w.add_argument("--frame-every", type=int, default=6,
                   help="每隔几步存一张涡度场；默认与 record-every 配合成 3 tick 一帧，播放时叠化")
    w.add_argument("--sub", type=int, default=4, help="涡度场导出降采样倍数")
    w.add_argument("--tracers", type=int, default=170)
    w.add_argument("--seed", type=int, default=7)
    w.add_argument("--steady", action="store_true", help="只跑到近定常，低 Re 时用")
    return p.parse_args()


def main():
    a = parse_args()
    D = 2.0 * a.size
    nu = a.u * D / a.re
    cx = int(round(a.x_pos * a.nx))
    cy = int(round(a.y_pos * a.ny))
    solid = make_solid(a.nx, a.ny, a.shape, cx, cy, a.size,
                       a.ratio if a.shape == "ellipse" else 1.0,
                       np.radians(a.angle) if a.shape == "ellipse" else 0.0)
    if a.dt * a.u > 1.0:
        print("warning: CFL = u_inf*dt/h = %.2f > 1，半拉格朗日仍稳定但更耗散" % (a.dt * a.u))

    f = Fluid(a.nx, a.ny, solid=solid, u_inf=a.u, dt=a.dt, nu=nu,
              iters=a.iters, vorticity_gain=a.vcg, omega=a.omega)
    rng = np.random.default_rng(a.seed)
    t_period = D / (ST_GUESS * a.u)
    transit_steps = a.nx / (a.u * a.dt)
    settle = max(50, int(round(a.settle_transits * transit_steps)))

    print("grid %dx%d  solid_cells=%d  D=%.1f  Re=%.0f  nu=%.5f  dt=%.2f  iters=%d  omega=%.2f"
          % (a.nx, a.ny, int(solid.sum()), D, a.re, nu, a.dt, a.iters, a.omega))
    clock = time.time()
    for k in range(settle):
        f.step()
        if (k + 1) % max(1, settle // 6) == 0:
            print("  settling %5d/%d  t=%7.1f  div_rms %.2e -> %.2e"
                  % (k + 1, settle, f.time, f.div_rms[0], f.div_rms[1]), flush=True)

    ticks = 1 if a.steady else max(8, int(round(a.periods * t_period / (a.dt * a.record_every))))
    tr = TracerSet(f, count=a.tracers, max_points=ticks + 4, fill=True, rng=rng)
    probe_x = int(min(a.nx - 2, cx + 4.0 * D))
    probe_dt = a.record_every * a.dt
    frames, probe = [], []
    total = ticks * a.record_every
    tick = 0
    for k in range(total):
        f.step()
        hit = (k + 1) % a.record_every == 0
        if hit:
            tick += 1
            probe.append(float(f.v[cy, probe_x]))
        tr.advance(record=hit, tick=tick - 1)
        if (k + 1) % a.frame_every == 0:
            frames.append(block_mean(f.vorticity(), a.sub))
    segs = tr.flush()
    starts = np.array([s[0] for s in segs], dtype=np.int64)
    pts = [s[1] for s in segs]
    print("recorded %d steps in %.1fs  paths=%d  frames=%d  probe_x=%d"
          % (total, time.time() - clock, len(pts), len(frames), probe_x))

    freq = shed_frequency(probe, probe_dt)
    st = freq * D / a.u

    lens = np.array([p.shape[0] for p in pts], dtype=np.int64) if pts else np.array([0])
    mlen = int(max(1, lens.max()))
    smax = float(max(2.0 * a.u, np.percentile(np.concatenate([p[:, 2] for p in pts]), 99.0))) \
        if pts else 2.0
    paths = np.zeros((len(pts) or 1, mlen, 3), dtype=np.float64)
    for i, p in enumerate(pts):
        paths[i, : p.shape[0], 0] = p[:, 0] / (a.nx - 1)
        paths[i, : p.shape[0], 1] = p[:, 1] / (a.ny - 1)
        paths[i, : p.shape[0], 2] = p[:, 2] / smax
    frames_arr = np.stack(frames) if frames else np.zeros((1, 1, 1))
    wmax = float(np.percentile(np.abs(frames_arr), 99.5)) or 1.0
    fh, fw = int(frames_arr.shape[1]), int(frames_arr.shape[2])

    data = {
        "meta": {
            "nx": a.nx, "ny": a.ny, "dt": a.dt, "u_inf": a.u, "nu": nu, "re": a.re,
            "iters": a.iters, "omega": a.omega, "vcg": a.vcg, "D": D, "cfl": a.u * a.dt,
            "record_every": a.record_every, "frame_every": a.frame_every,
            "ticks": ticks, "periods": 0.0 if a.steady else a.periods,
            "steady": bool(a.steady), "strouhal": st, "probe_x": probe_x,
            "div_rms": [f.div_rms[0], f.div_rms[1]],
            "wmax": wmax, "smax": smax, "path_max": mlen,
            "path_ranges": [[0.0, float(a.nx - 1)], [0.0, float(a.ny - 1)], [0.0, smax]],
            "obstacle": {"shape": a.shape, "cx": float(cx), "cy": float(cy),
                         "size": float(a.size), "ratio": float(a.ratio),
                         "angle": float(a.angle)},
            "frames": {"w": fw, "h": fh, "sub": a.sub},
        },
        "lens": [int(v) for v in lens],
        "t0": [int(v) for v in starts],
        "paths": encode(paths, 0.0, 1.0),
        "frames": encode(frames_arr, -wmax, wmax),
    }

    web = os.path.join(ROOT, "web")
    dat = os.path.join(ROOT, "data")
    os.makedirs(web, exist_ok=True)
    os.makedirs(dat, exist_ok=True)
    blob = json.dumps(data, separators=(",", ":"))
    js_path = os.path.join(web, "flow_data.js")
    json_path = os.path.join(dat, "flow_data.json")
    with open(js_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("window.FLOW_DATA = " + blob + ";\n")
    with open(json_path, "w", encoding="utf-8", newline="\n") as out:
        out.write(blob)

    print("wrote web/flow_data.js (%.0f KB), data/flow_data.json (%.0f KB)"
          % (os.path.getsize(js_path) / 1024.0, os.path.getsize(json_path) / 1024.0))
    print("St=%.3f  freq=%.5f  probe_std=%.3f  div_rms before/after=%.2e/%.2e  |u|/u_inf=%.2f"
          % (st, freq, float(np.std(probe)) if probe else 0.0,
             f.div_rms[0], f.div_rms[1], float(f.speed().max()) / a.u))


if __name__ == "__main__":
    main()
