import numpy as np
import casadi
import utils


DT = utils.time_step          # 0.5
H = 20                        # planning horizon
GAMMA = 0.95
Q = np.diag([20.0, 20.0])
QTH = 4.0
R = np.diag([0.1, 0.05])
QF = 5.0 * Q
QFTH = 5.0 * QTH
R_SAFE = 0.8
WORLD = 3.0                   # [-3,3]^2
M_COLL = 2000.0
COLL_BUFFER = 0.3


def sinc(x):

    if isinstance(x, (casadi.SX, casadi.MX)):
        return casadi.if_else(casadi.fabs(x) < 1e-4, 1.0 - x * x / 6.0, casadi.sin(x) / x)
    return np.sinc(np.asarray(x, dtype=float) / np.pi)


def error_dynamics(t, e, u):

    ex, ey, eth = e[0], e[1], e[2]
    v, w = u[0], u[1]
    rt = utils.lemniscate(int(t))
    rt1 = utils.lemniscate(int(t) + 1)
    alpha_t = rt[2]
    alpha_t1 = rt1[2]
    half = w * DT / 2.0
    s = sinc(half)
    eta = eth + alpha_t + half
    ex_n = ex + DT * s * casadi.cos(eta) * v + (rt[0] - rt1[0])
    ey_n = ey + DT * s * casadi.sin(eta) * v + (rt[1] - rt1[1])
    dalpha = np.arctan2(np.sin(alpha_t - alpha_t1), np.cos(alpha_t - alpha_t1))
    eth_n = eth + DT * w + dalpha
    return casadi.vertcat(ex_n, ey_n, eth_n)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


class CEC:
    def __init__(self, traj=utils.lemniscate, obstacles=None,
                 horizon=H, gamma=GAMMA):
        self.traj = traj
        if obstacles is None:
            obstacles = np.array([[2.35, 0.95, 0.5], [-2.35, -0.95, 0.5],
                                  [1.0, 0.0, 0.5], [-1.0, 0.0, 0.5]])
        self.obstacles = np.asarray(obstacles, dtype=float)
        self.H = horizon
        self.gamma = gamma
        self.prev_sol = None
        self._build_solver()

    def _build_solver(self):
        H = self.H
        nc = self.obstacles.shape[0]
        U = casadi.SX.sym("U", 2 * H)
        E = casadi.SX.sym("E", 3 * (H + 1))
        P = casadi.SX.sym("P", 3 + (H + 1) * 6 + 1)
        e0 = P[0:3]

        def step_param(k):
            base = 3 + k * 6
            return (P[base + 0], P[base + 1], P[base + 2],
                    P[base + 3], P[base + 4], P[base + 5])

        g = []
        lbg = []
        ubg = []
        cost = 0
        g.append(E[0:3] - e0)
        lbg += [0, 0, 0]
        ubg += [0, 0, 0]

        for k in range(H):
            ek = E[3 * k:3 * k + 3]
            uk = U[2 * k:2 * k + 2]
            ekp = E[3 * (k + 1):3 * (k + 1) + 3]
            alpha_t, drx, dry, dalpha, rx, ry = step_param(k)
            half = uk[1] * DT / 2.0
            s = casadi.if_else(casadi.fabs(half) < 1e-4,
                               1.0 - half * half / 6.0, casadi.sin(half) / half)
            eta = ek[2] + alpha_t + half
            ex_n = ek[0] + DT * s * casadi.cos(eta) * uk[0] + drx
            ey_n = ek[1] + DT * s * casadi.sin(eta) * uk[0] + dry
            eth_n = ek[2] + DT * uk[1] + dalpha
            gnext = casadi.vertcat(ex_n, ey_n, eth_n)
            # dynamics defect
            g.append(ekp - gnext)
            lbg += [0, 0, 0]
            ubg += [0, 0, 0]
            # stage cost
            disc = self.gamma ** k
            pe = ek[0:2]
            cost += disc * (pe.T @ Q @ pe
                            + QTH * (1 - casadi.cos(ek[2])) ** 2
                            + uk.T @ R @ uk)

        # terminal cost
        eH = E[3 * H:3 * H + 3]
        peH = eH[0:2]
        cost += peH.T @ QF @ peH + QFTH * (1 - casadi.cos(eH[2])) ** 2
        for k in range(H + 1):
            ek = E[3 * k:3 * k + 3]
            alpha_t, drx, dry, dalpha, rx, ry = step_param(k)
            wx = ek[0] + rx
            wy = ek[1] + ry
            g.append(wx); lbg.append(-WORLD); ubg.append(WORLD)
            g.append(wy); lbg.append(-WORLD); ubg.append(WORLD)
            disc = self.gamma ** min(k, H - 1)
            for i in range(nc):
                cx, cy, _ = self.obstacles[i]
                d2 = (wx - cx) ** 2 + (wy - cy) ** 2
                g.append(d2); lbg.append(R_SAFE ** 2); ubg.append(casadi.inf)
                # soft barrier: penalize clearance below R_SAFE + COLL_BUFFER
                d = casadi.sqrt(d2 + 1e-9)
                slack = (R_SAFE + COLL_BUFFER) - d
                cost += disc * M_COLL * casadi.fmax(slack, 0.0) ** 2

        gvec = casadi.vertcat(*g)
        z = casadi.vertcat(U, E)
        nlp = {"x": z, "f": cost, "g": gvec, "p": P}
        opts = {"ipopt.print_level": 0, "print_time": 0,
                "ipopt.max_iter": 200, "ipopt.sb": "yes"}
        self.solver = casadi.nlpsol("S", "ipopt", nlp, opts)
        self.nU = 2 * H
        self.nE = 3 * (H + 1)
        self.nS = 0
        self.nc = nc
        self._lbg = lbg
        self._ubg = ubg
        self.lbx = np.concatenate([np.tile([0.1, -1.0], H), -np.inf * np.ones(self.nE)])
        self.ubx = np.concatenate([np.tile([1.0, 1.0], H), np.inf * np.ones(self.nE)])

    def _params(self, t0, e0):
        H = self.H
        P = np.zeros(3 + (H + 1) * 6 + 1)
        P[0:3] = e0
        for k in range(H + 1):
            rt = self.traj(t0 + k)
            rt1 = self.traj(t0 + k + 1)
            dalpha = wrap(rt[2] - rt1[2])
            base = 3 + k * 6
            P[base + 0] = rt[2]
            P[base + 1] = rt[0] - rt1[0]
            P[base + 2] = rt[1] - rt1[1]
            P[base + 3] = dalpha
            P[base + 4] = rt[0]
            P[base + 5] = rt[1]
        P[3 + (H + 1) * 6] = t0
        return P

    def _initial_guess(self, e0):
        H = self.H
        if self.prev_sol is not None:
            U_prev = self.prev_sol[:self.nU].reshape(H, 2)
            E_prev = self.prev_sol[self.nU:self.nU + self.nE].reshape(H + 1, 3)
            U_g = np.vstack([U_prev[1:], U_prev[-1]])
            E_g = np.vstack([E_prev[1:], E_prev[-1]])
            E_g[0] = e0
            return np.concatenate([U_g.flatten(), E_g.flatten(), np.zeros(self.nS)])
        U_g = np.tile([0.5, 0.0], H)
        E_g = np.tile(e0, H + 1)
        return np.concatenate([U_g, E_g, np.zeros(self.nS)])

    def __call__(self, t, cur_state, cur_ref_state):
        e0 = np.asarray(cur_state, float) - np.asarray(cur_ref_state, float)
        e0[2] = wrap(e0[2])
        P = self._params(int(t), e0)
        x0 = self._initial_guess(e0)
        try:
            sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx,
                              lbg=self._lbg, ubg=self._ubg, p=P)
            ok = self.solver.stats().get("success", False)
            z = np.array(sol["x"]).flatten()
            if ok and np.all(np.isfinite(z)):
                self.prev_sol = z
                u0 = z[0:2]
                return [float(np.clip(u0[0], 0.1, 1.0)),
                        float(np.clip(u0[1], -1.0, 1.0))]
        except Exception:
            pass
        self.prev_sol = None
        return utils.simple_controller(cur_state, cur_ref_state)
