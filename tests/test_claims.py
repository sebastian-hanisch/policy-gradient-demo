"""Jede Zahl im README gegen den tatsaechlichen Code, nicht nur behauptet. Das Seed-Streuungs-Experiment ist der Befund dieses Stuecks und wird
deshalb mit der VOLLEN Seed-Zahl (EXP_SEEDS_FULL, wie im README) nachgerechnet, nicht mit einer kleineren, schnelleren Stichprobe - deterministisch
bei festen Seeds, deshalb exakt reproduzierbar trotz des Zufalls in der Politik selbst."""

import numpy as np
import pytest

import pg_constants as C
import pg_evaluation as E


def test_lucky_and_unlucky_seed_examples():
    lucky_gap, _ = E._gap_for_seed(C.DEFAULT_ROWS, C.DEFAULT_COLS, C.DEFAULT_SLIP, C.DEFAULT_GAMMA, C.DEFAULT_LR, C.DEFAULT_EPISODES, C.LUCKY_SEED)
    unlucky_gap, _ = E._gap_for_seed(C.DEFAULT_ROWS, C.DEFAULT_COLS, C.DEFAULT_SLIP, C.DEFAULT_GAMMA, C.DEFAULT_LR, C.DEFAULT_EPISODES, C.UNLUCKY_SEED)
    assert lucky_gap == pytest.approx(0.66, abs=0.2)
    assert unlucky_gap == pytest.approx(114.4, abs=5.0)
    assert lucky_gap < C.NEAR_OPTIMAL_GAP
    assert unlucky_gap > C.STUCK_GAP_THRESHOLD


def test_seed_variance_experiment_matches_the_readme():
    # Volle Seed-Zahl wie im README - deterministisch bei festen Seeds 0..29 auf DERSELBEN Plattform, aber ueber 2000 Trainings-Episoden koennen
    # sich winzige Gleitkomma-Unterschiede zwischen Windows und Linux-CI (z.B. in exp()/log() der Softmax) aufschaukeln und einzelne, ohnehin
    # knapp an der Schwelle liegende Seeds in die andere Kategorie kippen lassen - passend zum Thema dieses Stuecks (hohe Empfindlichkeit
    # gegenueber winzigen Unterschieden). Grosszuegige Bandbreite statt exakter Gleichheit (feedback_ci_platform_robust_tests).
    exp = E.seed_variance_experiment(seeds=range(C.EXP_SEEDS_FULL))
    assert exp["frac_near_optimal"] == pytest.approx(0.1667, abs=0.07)
    assert exp["frac_stuck"] == pytest.approx(0.1667, abs=0.07)
    assert exp["median_gap"] == pytest.approx(5.26, abs=1.0)


def test_random_walk_rarely_terminates_on_the_larger_sibling_grid():
    # Begruendung fuer das kleinere Standardraster (Plan-Korrektur, siehe pg_constants.py): auf dem 4x8-Raster der Geschwister erreicht selbst
    # eine rein zufaellige Politik ueberwiegend NICHT Ziel oder Klippe innerhalb von 300 Schritten.
    from pg_grid import ACTIONS, Grid, step
    g = Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    rng = np.random.default_rng(0)
    hit_cap = 0
    n = 100
    for _ in range(n):
        s = g.state_of(g.start)
        for t in range(300):
            s, r, done = step(g, s, int(rng.choice(ACTIONS)), rng)
            if done:
                break
        if not done:
            hit_cap += 1
    assert hit_cap / n > 0.5


def test_random_walk_terminates_reliably_on_the_chosen_default_grid():
    from pg_grid import ACTIONS, Grid, step
    g = Grid(rows=C.DEFAULT_ROWS, cols=C.DEFAULT_COLS, slip=0.1, gamma=0.95)
    rng = np.random.default_rng(0)
    hit_cap = 0
    n = 100
    for _ in range(n):
        s = g.state_of(g.start)
        for t in range(C.MAX_STEPS_PER_EPISODE):
            s, r, done = step(g, s, int(rng.choice(ACTIONS)), rng)
            if done:
                break
        if not done:
            hit_cap += 1
    assert hit_cap / n < 0.3
