"""Referenzloesung (nur zur Gegenprobe, NIE vom lernenden Agenten benutzt): Value Iteration auf dem bekannten Modell (P, R). Zusaetzlich zur
deterministischen Politik-Auswertung der Geschwister eine STOCHASTISCHE Variante: REINFORCE lernt eine echte Wahrscheinlichkeitsverteilung ueber
Aktionen, keine deterministisch-gierige Politik - die Bewertung soll die tatsaechliche (stochastische) Politik bewerten, nicht ihre Argmax-Version."""

import numpy as np

import pg_constants as C


def q_values(P, R, V, gamma):
    return R + gamma * np.einsum("sap,p->sa", P, V)


def value_iteration(P, R, gamma, tol=C.VI_TOL, max_iter=C.VI_MAX_ITER):
    S, A = R.shape
    V = np.zeros(S)
    for _ in range(max_iter):
        Q = q_values(P, R, V, gamma)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = q_values(P, R, V, gamma)
    policy = Q.argmax(axis=1)
    return V, Q, policy


def policy_evaluation(P, R, policy, gamma, tol=C.VI_TOL, max_iter=C.VI_MAX_ITER):
    """Deterministische Politik (ein Aktionsindex je Zustand) - fuer den Vergleich gegen V*."""
    S = P.shape[0]
    idx = np.arange(S)
    P_pi = P[idx, policy]
    R_pi = R[idx, policy]
    V = np.zeros(S)
    for _ in range(max_iter):
        V_new = R_pi + gamma * (P_pi @ V)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return V


def policy_evaluation_stochastic(P, R, action_probs, gamma, tol=C.VI_TOL, max_iter=C.VI_MAX_ITER):
    """action_probs: (S, A), jede Zeile eine Wahrscheinlichkeitsverteilung ueber Aktionen. Bewertet die ECHTE stochastische Politik, wie sie
    REINFORCE tatsaechlich lernt (kein Argmax)."""
    P_pi = np.einsum("sa,sap->sp", action_probs, P)
    R_pi = np.einsum("sa,sa->s", action_probs, R)
    V = np.zeros(P.shape[0])
    for _ in range(max_iter):
        V_new = R_pi + gamma * (P_pi @ V)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return V
