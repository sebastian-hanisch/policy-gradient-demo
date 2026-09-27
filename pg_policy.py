"""Lineare Softmax-Politik: pi(a|s) = softmax(theta^T x(s))_a. Mit One-Hot-Merkmalen ist theta[s, a] direkt die Aktionspraeferenz von Zustand s
fuer Aktion a - kein Bias noetig, jede Zeile ist unabhaengig von jeder anderen (kein Verallgemeinerungs-Anspruch, siehe pg_constants.py)."""

import numpy as np


class SoftmaxPolicy:
    def __init__(self, n_features, n_actions, seed=0):
        rng = np.random.default_rng(seed)
        self.theta = rng.normal(0.0, 0.01, size=(n_features, n_actions))

    def probs(self, X):
        """X: (batch, n_features). Rueckgabe: (batch, n_actions), numerisch stabil (max-Trick)."""
        scores = X @ self.theta
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def sample(self, x, rng):
        """x: (n_features,) fuer EINEN Zustand. Rueckgabe: gezogene Aktion (int)."""
        p = self.probs(x[None, :])[0]
        return int(rng.choice(len(p), p=p))

    def grad_log_prob(self, x, action):
        """d log(pi(action|x)) / d theta, Standardformel der Softmax-Politik: x_i * (1[j=action] - pi(j|x)) je Eintrag theta[i, j]."""
        p = self.probs(x[None, :])[0]
        n_actions = p.shape[0]
        onehot_a = np.zeros(n_actions)
        onehot_a[action] = 1.0
        return np.outer(x, onehot_a - p)

    def step(self, grad, lr):
        self.theta = self.theta + lr * grad
