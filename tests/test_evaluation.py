"""Analyse und das Seed-Streuungs-Experiment: Aufbau und Grundeigenschaften (grosse Seed-Zahlen gehoeren in test_claims.py)."""

import numpy as np
import pytest

import pg_evaluation as E


def test_analyse_wiring():
    a = E.analyse(E.Settings(episodes=50, seed=0))
    assert a.action_probs.shape == (a.grid.n_states, 4)
    assert np.allclose(a.action_probs.sum(axis=1), 1.0)
    assert a.returns.shape == (50,) and a.lengths.shape == (50,)
    assert a.env_steps == int(a.lengths.sum())


def test_analyse_is_deterministic_given_the_same_seed():
    s = E.Settings(episodes=30, seed=5)
    a1, a2 = E.analyse(s), E.analyse(s)
    assert np.array_equal(a1.theta, a2.theta)


def test_reference_is_cached_across_settings_with_the_same_grid():
    assert E._reference(3, 4, 0.10, 0.95) is E._reference(3, 4, 0.10, 0.95)


def test_seed_variance_experiment_shape():
    exp = E.seed_variance_experiment(seeds=range(4), base=E.Settings(episodes=50))
    assert len(exp["seeds"]) == 4
    assert exp["gaps"].shape == (4,) and exp["final_returns"].shape == (4,)
    assert 0.0 <= exp["frac_near_optimal"] <= 1.0 and 0.0 <= exp["frac_stuck"] <= 1.0


def test_different_seeds_give_different_gaps():
    # Der Kernpunkt dieses Stuecks: derselbe Algorithmus, dieselben Hyperparameter, andere Seeds -> andere Ergebnisse.
    exp = E.seed_variance_experiment(seeds=range(6), base=E.Settings(episodes=200))
    assert len(set(np.round(exp["gaps"], 3))) > 1
