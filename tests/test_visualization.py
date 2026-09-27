"""Rauchtest der Plotly-Abbildungen: explizite Keys, ein Balken je Seed."""

import numpy as np

import pg_evaluation as E
from pg_visualization import build_grid, build_learning_curve, build_seed_variance


def test_seed_variance_chart_has_one_bar_per_seed():
    exp = E.seed_variance_experiment(seeds=range(5), base=E.Settings(episodes=50))
    fig = build_seed_variance(exp)
    assert len(fig.data[0].y) == 5


def test_grid_and_curve_render_without_error():
    a = E.analyse(E.Settings(episodes=20, seed=0))
    fig1 = build_grid(a.grid, a.V_pi, a.action_probs)
    fig2 = build_learning_curve(a.returns)
    assert fig1 is not None and fig2 is not None
