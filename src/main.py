from time import time
import argparse
import numpy as np
import utils

OBSTACLES = np.array([
    [2.35, 0.95, 0.5],
    [-2.35, -0.95, 0.5],
    [1.0, 0.0, 0.5],
    [-1.0, 0.0, 0.5],
])


def make_controller(name):
    if name == "cec":
        from cec import CEC
        return CEC(utils.lemniscate, OBSTACLES)
    if name == "gpi":
        from gpi import GPI, make_default_config
        gpi = GPI(make_default_config())
        if not gpi.load():
            print("[main] No saved GPI policy found; computing the offline policy...")
            gpi.compute_policy(num_iters=300)
        return gpi
    if name == "feature":
        from feature_gpi import FeatureGPI
        fgpi = FeatureGPI()
        if not fgpi.load():
            print("[main] No saved Feature-GPI policy found; fitting offline...")
            fgpi.fit()
        return fgpi
    return lambda t, s, r: utils.simple_controller(s, r)


def run_once(controller_name="cec", seed=None, visualize=True, save=False):
    if seed is not None:
        np.random.seed(seed)
    controller = make_controller(controller_name)
    traj = utils.lemniscate
    ref_traj, car_states, times = [], [], []
    error_trans, error_rot = 0.0, 0.0
    cur_state = np.array([utils.x_init, utils.y_init, utils.theta_init])
    cur_iter = 0
    main_loop = time()
    while cur_iter * utils.time_step < utils.sim_time:
        t1 = time()
        cur_ref = traj(cur_iter)
        ref_traj.append(cur_ref)
        car_states.append(cur_state)
        control = controller(cur_iter, cur_state, np.array(cur_ref))
        control = [float(np.clip(control[0], utils.v_min, utils.v_max)),
                   float(np.clip(control[1], utils.w_min, utils.w_max))]
        next_state = utils.car_next_state(utils.time_step, cur_state, control, noise=True)
        cur_state = next_state
        times.append(time() - t1)
        cur_err = cur_state - np.array(cur_ref)
        cur_err[2] = np.arctan2(np.sin(cur_err[2]), np.cos(cur_err[2]))
        error_trans += np.linalg.norm(cur_err[:2])
        error_rot += np.abs(cur_err[2])
        cur_iter += 1
    total = time() - main_loop
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    print(f"[{controller_name}] total {total:.2f}s  avg {times.mean()*1000:.2f} ms/step  "
          f"err_trans {error_trans:.3f}  err_rot {error_rot:.3f}")
    if visualize:
        if save:
            from evaluate import _plot_tracking
            _plot_tracking(car_states, ref_traj, controller_name)
            print(f"[main] Saved results/{controller_name}_tracking.png")
        else:
            utils.visualize(car_states, ref_traj, OBSTACLES, times, utils.time_step)
    return car_states, ref_traj, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="cec",
                    choices=["cec", "gpi", "feature", "simple"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--experiments", action="store_true",
                    help="Run the multi-seed Monte-Carlo evaluation.")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    if args.experiments:
        from evaluate import run_experiments
        run_experiments()
    else:
        run_once(args.controller, seed=args.seed, save=args.save)


if __name__ == "__main__":
    main()
