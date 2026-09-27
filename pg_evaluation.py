"""Analyse und das Kernexperiment dieses Stuecks: wie stark der Trainingserfolg von REINFORCE zwischen Zufalls-Seeds schwankt. Jeder Seed
bekommt ein eigenstaendiges, vollstaendiges Training (gleiche Hyperparameter, nur der Zufall unterscheidet sich) - gemessen wird der Wert-Abstand
der gelernten (stochastischen) Politik zu V*(Start). Das ist teurer als DQNs Ablation (Stück 6, dort halfen Checkpoints EINES Trainings), weil
hier gerade die Streuung ZWISCHEN unabhaengigen Trainingslaeufen der Punkt ist, nicht die Streuung ueber die Trainingsdauer."""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

import pg_agent as A
import pg_constants as C
import pg_grid as G
import pg_reference as R
from pg_features import one_hot


@dataclass(frozen=True)
class Settings:
    rows: int = C.DEFAULT_ROWS
    cols: int = C.DEFAULT_COLS
    slip: float = C.DEFAULT_SLIP
    gamma: float = C.DEFAULT_GAMMA
    lr: float = C.DEFAULT_LR
    episodes: int = C.DEFAULT_EPISODES
    seed: int = 0

    @property
    def grid(self):
        return G.Grid(self.rows, self.cols, self.slip, self.gamma)


@lru_cache(maxsize=64)
def _reference(rows, cols, slip, gamma):
    grid = G.Grid(rows, cols, slip, gamma)
    P, Rw = G.build_model(grid)
    V_star, Q_star, pi_star = R.value_iteration(P, Rw, gamma)
    return grid, P, Rw, V_star, Q_star, pi_star


def _feature_fn(grid):
    return lambda s: one_hot(s, grid.n_states)


@dataclass
class Analysis:
    settings: Settings
    grid: G.Grid
    theta: np.ndarray
    returns: np.ndarray
    lengths: np.ndarray
    falls: np.ndarray
    action_probs: np.ndarray
    V_star: np.ndarray
    pi_star: np.ndarray
    V_pi: np.ndarray
    gap: float
    env_steps: int
    snapshots: dict = field(default_factory=dict)


def analyse(s, checkpoints=()):
    grid, P, Rw, V_star, Q_star, pi_star = _reference(s.rows, s.cols, s.slip, s.gamma)
    feature_fn = _feature_fn(grid)
    policy, returns, lengths, falls, snapshots = A.train(grid, feature_fn, grid.n_states, s.lr, s.episodes, s.seed, checkpoints=checkpoints)
    action_probs = A.action_probs_table(policy, feature_fn, grid.n_states)
    V_pi = R.policy_evaluation_stochastic(P, Rw, action_probs, grid.gamma)
    start_s = grid.state_of(grid.start)
    gap = float(V_star[start_s] - V_pi[start_s])
    return Analysis(s, grid, policy.theta, returns, lengths, falls, action_probs, V_star, pi_star, V_pi, gap, int(lengths.sum()), snapshots)


def _gap_for_seed(rows, cols, slip, gamma, lr, episodes, seed):
    grid, P, Rw, V_star, _, _ = _reference(rows, cols, slip, gamma)
    feature_fn = _feature_fn(grid)
    start_s = grid.state_of(grid.start)
    policy, returns, lengths, falls, _ = A.train(grid, feature_fn, grid.n_states, lr, episodes, seed)
    probs = A.action_probs_table(policy, feature_fn, grid.n_states)
    V_pi = R.policy_evaluation_stochastic(P, Rw, probs, gamma)
    return float(V_star[start_s] - V_pi[start_s]), float(returns[-20:].mean())


def seed_variance_experiment(seeds=None, base=None):
    """Ein eigenstaendiges Training je Seed (gleiche Hyperparameter). Rueckgabe: Wert-Abstand und mittlere Spaetphasen-Rueckgabe je Seed, plus
    zusammenfassende Kennzahlen (Anteil nahe-optimal, Anteil dauerhaft haengengeblieben)."""
    seeds = range(C.EXP_SEEDS) if seeds is None else list(seeds)
    base = Settings() if base is None else base
    gaps, final_returns = [], []
    for seed in seeds:
        gap, fr = _gap_for_seed(base.rows, base.cols, base.slip, base.gamma, base.lr, base.episodes, seed)
        gaps.append(gap)
        final_returns.append(fr)
    gaps = np.array(gaps)
    final_returns = np.array(final_returns)
    return {
        "seeds": list(seeds),
        "gaps": gaps,
        "final_returns": final_returns,
        "frac_near_optimal": float(np.mean(gaps < C.NEAR_OPTIMAL_GAP)),
        "frac_stuck": float(np.mean(gaps > C.STUCK_GAP_THRESHOLD)),
        "median_gap": float(np.median(gaps)),
    }
