"""Die Referenzloesung (Value Iteration, nur zur Gegenprobe): Bellman-Formel von Hand, entarteter Fall mit geschlossener Loesung, und die
stochastische Politik-Auswertung gegen die deterministische als Spezialfall."""

import numpy as np
import pytest

import pg_grid as G
import pg_reference as R


def test_q_values_by_hand():
    P = np.array([[[0.5, 0.5], [1.0, 0.0]]])
    Rw = np.array([[1.0, 2.0]])
    V = np.array([10.0, 20.0])
    Q = R.q_values(P, Rw, V, gamma=0.9)
    assert Q[0, 0] == pytest.approx(1.0 + 0.9 * (0.5 * 10 + 0.5 * 20)) and Q[0, 1] == pytest.approx(2.0 + 0.9 * 10.0)


def test_degenerate_single_row_grid_gives_up_and_bounces_forever():
    g = G.Grid(rows=1, cols=3, slip=0.0, gamma=0.95)
    P, Rw = G.build_model(g)
    V, Q, policy = R.value_iteration(P, Rw, g.gamma)
    s0 = g.state_of(g.start)
    assert V[s0] == pytest.approx(-1.0 / (1.0 - g.gamma), abs=1e-4)
    assert policy[s0] == G.NORTH


def test_policy_evaluation_of_the_optimal_policy_matches_value_iteration():
    g = G.Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    P, Rw = G.build_model(g)
    V_star, _, pi_star = R.value_iteration(P, Rw, g.gamma)
    V_pi = R.policy_evaluation(P, Rw, pi_star, g.gamma)
    assert np.allclose(V_star, V_pi, atol=1e-3)


def test_stochastic_evaluation_with_a_one_hot_policy_matches_the_deterministic_version():
    # Eine Aktionswahrscheinlichkeits-Tabelle, die fuer jeden Zustand genau eine Aktion mit Wahrscheinlichkeit 1 waehlt, ist der Spezialfall
    # einer deterministischen Politik - beide Auswertungen muessen dasselbe Ergebnis liefern.
    g = G.Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    P, Rw = G.build_model(g)
    V_star, _, pi_star = R.value_iteration(P, Rw, g.gamma)
    action_probs = np.zeros((g.n_states, 4))
    action_probs[np.arange(g.n_states), pi_star] = 1.0
    V_stoch = R.policy_evaluation_stochastic(P, Rw, action_probs, g.gamma)
    V_det = R.policy_evaluation(P, Rw, pi_star, g.gamma)
    assert np.allclose(V_stoch, V_det, atol=1e-6)


def test_stochastic_evaluation_of_a_uniform_random_policy_by_hand():
    # Ein-Zeilen-Raster, uniforme Zufallspolitik: von Hand nachrechenbarer Fixpunkt (jede Aktion gleich wahrscheinlich).
    g = G.Grid(rows=1, cols=3, slip=0.0, gamma=0.9)
    P, Rw = G.build_model(g)
    action_probs = np.full((g.n_states, 4), 0.25)
    V = R.policy_evaluation_stochastic(P, Rw, action_probs, g.gamma)
    # Von Hand nachgerechnet mit einer unabhaengigen Fixpunkt-Iteration (nicht derselbe Code-Pfad).
    V_manual = np.zeros(g.n_states)
    for _ in range(5000):
        V_new = g.gamma * np.einsum("sa,sap,p->s", action_probs, P, V_manual) + np.einsum("sa,sa->s", action_probs, Rw)
        if np.max(np.abs(V_new - V_manual)) < 1e-10:
            V_manual = V_new
            break
        V_manual = V_new
    assert np.allclose(V, V_manual, atol=1e-4)
