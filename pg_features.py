"""Zustands-Darstellung: One-Hot (jeder Zustand hat seine eigene, isolierte Spalte an Aktionspraeferenzen). Anders als bei DQN (Stück 6) ist
Verallgemeinerung auf unbesuchte Zustaende nicht der Hook dieses Stuecks (siehe pg_constants.py) - One-Hot haelt die Politik einfach und macht die
Korrektheits-/Gradienten-Pruefung direkt nachvollziehbar."""

import numpy as np


def one_hot(state, n_states):
    x = np.zeros(n_states)
    x[state] = 1.0
    return x
