"""Unabhängige Orakel für REINFORCE: `train` Schritt für Schritt gegen eine Schleifen-Neuimplementierung (direkte Summe für G_t, eigener Softmax) auf identischem
Zufallsstrom; der exakte Politikgradient ∇J(θ) aus dem Politikgradiententheorem (diskontierte Besuchsverteilung per LGS) gegen finite Differenzen von J(θ)
(exakte Politikauswertung); der REINFORCE-Schätzer (`run_episode`, `returns_from_rewards`, `grad_log_prob`) gemittelt gegen diesen exakten Gradienten
(Erwartungswert, mit Standardfehler) – auch mit Zustandswert-Baseline, die den Erwartungswert nicht ändert; `grad_log_prob` gegen finite Differenz von log π;
Wert-Abstand der Analyse gegen LP/LGS."""

import numpy as np
import pytest

import pg_agent as A
import pg_evaluation as E
import pg_grid as G
import pg_policy as PP
import pg_reference as R
from pg_features import one_hot

linprog = pytest.importorskip("scipy.optimize").linprog

_MOVES = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
_NAME = {0: "N", 1: "S", 2: "E", 3: "W"}
_SIDES = {"N": ("E", "W"), "S": ("E", "W"), "E": ("N", "S"), "W": ("N", "S")}


def _outcomes(rows, cols, slip, s, a):
    r, c = divmod(s, cols)
    goal = (rows - 1, cols - 1)
    if (r, c) == goal:
        return [(1.0, s, 0.0, True)]
    cliff = {(rows - 1, k) for k in range(1, cols - 1)}
    m = _NAME[a]
    out = []
    for p, d in ((1 - slip, m), (slip / 2, _SIDES[m][0]), (slip / 2, _SIDES[m][1])):
        nr, nc = r + _MOVES[d][0], c + _MOVES[d][1]
        if not (0 <= nr < rows and 0 <= nc < cols):
            nr, nc = r, c
        if (nr, nc) == goal:
            out.append((p, nr * cols + nc, 10.0, True))
        elif (nr, nc) in cliff:
            out.append((p, (rows - 1) * cols, -100.0, False))
        else:
            out.append((p, nr * cols + nc, -1.0, False))
    return out


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def _vstar_lp(P, Rw, gamma):
    S, nA = Rw.shape
    rows = []
    for s in range(S):
        for a in range(nA):
            row = gamma * P[s, a].copy()
            row[s] -= 1.0
            rows.append(row)
    return linprog(np.ones(S), A_ub=np.array(rows), b_ub=-Rw.reshape(-1), bounds=[(None, None)] * S, method="highs").x


def _replay(rows, cols, slip, gamma, lr, episodes, seed):
    rng = np.random.default_rng(seed)
    theta = np.random.default_rng(seed).normal(0.0, 0.01, size=(rows * cols, 4))
    rets, lens, falls = [], [], []
    for _ in range(episodes):
        s, traj = (rows - 1) * cols, []
        for _ in range(300):
            a = int(rng.choice(4, p=_softmax(theta[s])))
            u = rng.random()
            k = 0 if (slip == 0 or u < 1 - slip) else (1 if u < 1 - slip / 2 else 2)
            _, s2, r, d = _outcomes(rows, cols, slip, s, a)[k]
            traj.append((s, a, r))
            s = s2
            if d:
                break
        T = len(traj)
        for t, (st, a, _) in enumerate(traj):
            Gt = sum(gamma ** (k - t) * traj[k][2] for k in range(t, T))                              # direkte Summe statt Rückwärts-Akkumulation
            g = -_softmax(theta[st])
            g[a] += 1.0
            theta[st] += lr * gamma ** t * Gt * g
        rets.append(sum(x[2] for x in traj))
        lens.append(T)
        falls.append(sum(x[2] == -100.0 for x in traj))
    return theta, np.array(rets), np.array(lens), np.array(falls)


def _j(theta, P, Rw, gamma, s0):
    S = theta.shape[0]
    pi = np.array([_softmax(theta[s]) for s in range(S)])
    Ppi, Rpi = np.einsum("sa,sap->sp", pi, P), (pi * Rw).sum(1)
    V = np.linalg.solve(np.eye(S) - gamma * Ppi, Rpi)
    return V[s0], pi, Ppi, V


def _exact_grad(theta, P, Rw, gamma, s0):
    """Politikgradiententheorem: ∇J = Σ_s d(s) Σ_a Q(s,a) ∇π(a|s), d = γ-diskontierte Besuchsverteilung ab dem Start."""
    S = theta.shape[0]
    _, pi, Ppi, V = _j(theta, P, Rw, gamma, s0)
    Q = Rw + gamma * np.einsum("sap,p->sa", P, V)
    e0 = np.zeros(S)
    e0[s0] = 1.0
    d = np.linalg.solve(np.eye(S) - gamma * Ppi.T, e0)
    grad = np.zeros_like(theta)
    for s in range(S):
        for b in range(4):
            grad[s, b] = d[s] * sum(Q[s, j] * pi[s, j] * ((j == b) - pi[s, b]) for j in range(4))
    return grad


def test_train_replays_step_by_step_on_the_same_random_stream():
    rng = np.random.default_rng(31)
    for _ in range(12):
        rows, cols = int(rng.integers(3, 5)), int(rng.integers(4, 7))
        slip, gamma, lr = float(rng.choice([0, 0.1, 0.3])), float(rng.choice([0.8, 0.95, 0.99])), float(rng.choice([0.001, 0.005, 0.05]))
        ep, seed = int(rng.integers(1, 15)), int(rng.integers(1000))
        g = G.Grid(rows, cols, slip, gamma)
        pol, ret, ln, fl, snaps = A.train(g, lambda s: one_hot(s, g.n_states), g.n_states, lr, ep, seed, checkpoints=(ep,))
        th, ro, lo, fo = _replay(rows, cols, slip, gamma, lr, ep, seed)
        assert np.allclose(pol.theta, th, atol=1e-9, rtol=1e-9)
        assert np.array_equal(ret, ro) and np.array_equal(ln, lo) and np.array_equal(fl, fo)
        assert np.array_equal(snaps[ep].theta, pol.theta)


def test_exact_policy_gradient_matches_finite_differences_of_the_exact_return():
    rng = np.random.default_rng(7)
    for rows, cols, slip, gamma in [(3, 4, 0.1, 0.95), (3, 5, 0.0, 0.8), (4, 5, 0.3, 0.95)]:
        g = G.Grid(rows, cols, slip, gamma)
        P, Rw = G.build_model(g)
        s0 = g.state_of(g.start)
        theta = rng.normal(0, 0.7, size=(g.n_states, 4))
        ga = _exact_grad(theta, P, Rw, gamma, s0)
        gf = np.zeros_like(theta)
        for s in range(g.n_states):
            for b in range(4):
                tp, tm = theta.copy(), theta.copy()
                tp[s, b] += 1e-6
                tm[s, b] -= 1e-6
                gf[s, b] = (_j(tp, P, Rw, gamma, s0)[0] - _j(tm, P, Rw, gamma, s0)[0]) / 2e-6
        assert np.abs(ga - gf).max() < 1e-4 * max(1.0, np.abs(gf).max())
        pi = np.array([_softmax(theta[s]) for s in range(g.n_states)])
        assert R.policy_evaluation_stochastic(P, Rw, pi, gamma)[s0] == pytest.approx(_j(theta, P, Rw, gamma, s0)[0], abs=1e-4)


def test_grad_log_prob_matches_finite_difference_of_log_pi_for_every_action():
    rng = np.random.default_rng(2)
    pol = PP.SoftmaxPolicy(6, 4, 0)
    pol.theta = rng.normal(0, 1.0, size=(6, 4))
    for s in range(6):
        for a in range(4):
            num = np.zeros_like(pol.theta)
            for b in range(4):
                tp, tm = pol.theta.copy(), pol.theta.copy()
                tp[s, b] += 1e-6
                tm[s, b] -= 1e-6
                num[s, b] = (np.log(_softmax(tp[s])[a]) - np.log(_softmax(tm[s])[a])) / 2e-6
            assert np.abs(pol.grad_log_prob(one_hot(s, 6), a) - num).max() < 1e-6


def _mc_gradient(g, theta, n_ep, seed, baseline=None):
    rng = np.random.default_rng(seed)
    pol = PP.SoftmaxPolicy(g.n_states, 4, 0)
    pol.theta = theta.copy()
    ff = lambda s: one_hot(s, g.n_states)
    acc, acc2 = np.zeros_like(theta), np.zeros_like(theta)
    for _ in range(n_ep):
        st, ac, rw = A.run_episode(g, pol, ff, rng)
        Gt = A.returns_from_rewards(rw, g.gamma)
        gr = np.zeros_like(theta)
        for t in range(len(st)):
            b = 0.0 if baseline is None else baseline[st[t]]
            gr += g.gamma ** t * (Gt[t] - b) * pol.grad_log_prob(ff(st[t]), ac[t])
        acc += gr
        acc2 += gr ** 2
    mean = acc / n_ep
    return mean, np.sqrt(np.maximum(acc2 / n_ep - mean ** 2, 0) / n_ep)


def test_reinforce_estimator_is_in_expectation_the_exact_gradient_with_and_without_baseline():
    g = G.Grid(3, 4, 0.1, 0.95)
    P, Rw = G.build_model(g)
    s0 = g.state_of(g.start)
    pi_star = np.argmax(Rw + g.gamma * P @ _vstar_lp(P, Rw, g.gamma), axis=1)
    theta = 2.0 * np.eye(4)[pi_star] + np.random.default_rng(1).normal(0, 0.3, size=(g.n_states, 4))   # überwiegend kurze Episoden: schnell und rauscharm
    exact = _exact_grad(theta, P, Rw, g.gamma, s0)
    V = _j(theta, P, Rw, g.gamma, s0)[3]
    scale = np.abs(exact).max()
    for baseline in (None, V):                                                                       # Baseline V^π ändert den Erwartungswert nicht
        mean, se = _mc_gradient(g, theta, 1500, 5, baseline)
        assert np.all(np.abs(mean - exact) <= 5.0 * se + 0.03 * scale)


def test_analysis_gap_matches_lp_and_linear_solve_and_replayed_returns():
    for seed in range(3):
        s = E.Settings(rows=3, cols=5, slip=0.1, gamma=0.95, lr=0.01, episodes=30 + seed, seed=seed)
        a = E.analyse(s)
        P, Rw = G.build_model(a.grid)
        s0 = a.grid.state_of(a.grid.start)
        assert a.gap == pytest.approx(_vstar_lp(P, Rw, a.grid.gamma)[s0] - _j(a.theta, P, Rw, a.grid.gamma, s0)[0], abs=1e-4)
        th, ro, lo, _ = _replay(3, 5, 0.1, 0.95, 0.01, s.episodes, seed)
        assert np.allclose(a.theta, th, atol=1e-9) and np.array_equal(a.returns, ro) and a.env_steps == int(lo.sum())
