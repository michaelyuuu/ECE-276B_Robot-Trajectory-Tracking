"""Monte-Carlo evaluation of CEC and GPI: metrics + report figures."""
import os
from time import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import utils

OBSTACLES = np.array([[2.35, 0.95, 0.5], [-2.35, -0.95, 0.5],
                      [1.0, 0.0, 0.5], [-1.0, 0.0, 0.5]])
R_SAFE = 0.8
N_SEEDS = 20
IMAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def clearance(positions):
    """min over time and obstacles of (||p - c|| - r_safe)."""
    d = np.min([np.linalg.norm(positions - OBSTACLES[i, :2], axis=1) - R_SAFE
                for i in range(len(OBSTACLES))], axis=0)
    return d.min()


def rollout(controller, seed):
    np.random.seed(seed)
    state = np.array([utils.x_init, utils.y_init, utils.theta_init])
    states, refs, steptimes = [], [], []
    it = 0
    while it * utils.time_step < utils.sim_time:
        ref = np.array(utils.lemniscate(it))
        t1 = time()
        u = controller(it, state, ref)
        steptimes.append(time() - t1)
        u = [float(np.clip(u[0], utils.v_min, utils.v_max)),
             float(np.clip(u[1], utils.w_min, utils.w_max))]
        states.append(state)
        refs.append(ref)
        state = utils.car_next_state(utils.time_step, state, u, noise=True)
        it += 1
    return np.array(states), np.array(refs), np.array(steptimes)


def metrics_for(controller, name, offline_time=0.0):
    pos_errs, head_errs, max_pos, clears, colls, online = [], [], [], [], 0, []
    sample = None
    for seed in range(N_SEEDS):
        states, refs, steptimes = rollout(controller, seed)
        perr = np.linalg.norm(states[:, :2] - refs[:, :2], axis=1)
        herr = np.abs(np.arctan2(np.sin(states[:, 2] - refs[:, 2]),
                                 np.cos(states[:, 2] - refs[:, 2])))
        pos_errs.append(np.sqrt(np.mean(perr ** 2)))
        head_errs.append(np.sqrt(np.mean(herr ** 2)))
        max_pos.append(perr.max())
        c = clearance(states[:, :2])
        clears.append(c)
        if c < 0:
            colls += 1
        online.append(np.mean(steptimes) * 1000)
        if seed == 0:
            sample = (states, refs, perr, herr)
    res = {
        "rms_pos": np.mean(pos_errs),
        "rms_head": np.mean(head_errs),
        "max_pos": np.mean(max_pos),
        "min_clear": np.min(clears),
        "collisions": colls,
        "success_rate": 100.0 * (N_SEEDS - colls) / N_SEEDS,
        "offline_s": offline_time,
        "online_ms": np.mean(online),
    }
    _plot_tracking(sample[0], sample[1], name)
    _plot_error(sample[2], sample[3], name)
    return res


def _plot_tracking(states, refs, name):
    os.makedirs(IMAGE_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(refs[:, 0], refs[:, 1], marker="x", s=15, c="tab:blue", label="reference")
    ax.plot(states[:, 0], states[:, 1], "k-", lw=2, label="executed")
    for (cx, cy, r) in OBSTACLES:
        ax.add_patch(plt.Circle((cx, cy), r, color="r", alpha=0.4))
        ax.add_patch(plt.Circle((cx, cy), R_SAFE, color="r", alpha=0.12, ls="--", fill=False))
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_aspect("equal")
    ax.set_title(f"{name.upper()} trajectory"); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(IMAGE_DIR, f"{name}_tracking.png"), dpi=150)
    plt.close(fig)


def _plot_error(perr, herr, name):
    t = np.arange(len(perr)) * utils.time_step
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(t, perr, label="position error [m]")
    ax.plot(t, herr, label="heading error [rad]")
    ax.set_xlabel("time [s]"); ax.set_title(f"{name.upper()} tracking error")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(IMAGE_DIR, f"{name}_error.png"), dpi=150)
    plt.close(fig)


def run_experiments():
    from cec import CEC
    from gpi import GPI, make_default_config
    from feature_gpi import FeatureGPI
    results = {}
    # (key, column heading) in the order they appear in the table.
    order = [("simple", "Simple"), ("cec", "CEC"),
             ("gpi", "GPI"), ("feature", "Feature")]
    print("[eval] simple ...")
    results["simple"] = metrics_for(
        lambda t, s, r: utils.simple_controller(s, r), "simple", offline_time=0.0)
    print("[eval] CEC ...")
    results["cec"] = metrics_for(CEC(utils.lemniscate, OBSTACLES), "cec", offline_time=0.0)
    print("[eval] GPI build ...")
    gpi = GPI(make_default_config())
    t0 = time()
    if not gpi.load():
        gpi.compute_policy(num_iters=300)
    results["gpi"] = metrics_for(gpi, "gpi", offline_time=time() - t0)
    print("[eval] Feature-GPI build ...")
    fgpi = FeatureGPI()
    t0 = time()
    if not fgpi.load():
        fgpi.fit()
    results["feature"] = metrics_for(fgpi, "feature", offline_time=time() - t0)
    _print_table(results, order)
    _write_table_tex(results, order)
    return results


_ROWS = [("RMS position error", "rms_pos", "{:.3f}"),
         ("RMS heading error", "rms_head", "{:.3f}"),
         ("Max position error", "max_pos", "{:.3f}"),
         ("Minimum clearance", "min_clear", "{:.3f}"),
         ("Collision count", "collisions", "{:d}"),
         ("Success rate", "success_rate", "{:.0f}"),
         ("Offline time (s)", "offline_s", "{:.1f}"),
         ("Online time (ms)", "online_ms", "{:.2f}")]


def _print_table(results, order):
    header = f"{'Metric':24s} " + " ".join(f"{disp:>12s}" for _, disp in order)
    print("\n" + header)
    for label, key, fmt in _ROWS:
        cells = " ".join(f"{fmt.format(results[name][key]):>12s}" for name, _ in order)
        print(f"{label:24s} {cells}")


def _write_table_tex(results, order):
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results_table.txt"))
    suffix = {"success_rate": r"\%"}
    with open(out, "w") as f:
        for label, key, fmt in _ROWS:
            sfx = suffix.get(key, "")
            cells = " & ".join(f"{fmt.format(results[name][key])}{sfx}" for name, _ in order)
            f.write(f"{label} & {cells} \\\\\n")
    print(f"[eval] wrote {out}")
