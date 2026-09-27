"""REINFORCE (Williams 1992): episodischer Monte-Carlo-Politikgradient. Eine ganze Episode wird mit der AKTUELLEN Politik erzeugt, danach wird
fuer jeden Zeitschritt t die tatsaechliche (Monte-Carlo-)Rueckgabe G_t berechnet und die Politik in Richtung gamma^t * G_t * grad(log pi(a_t|s_t))
verschoben - Schritt fuer Schritt, wie im Sutton & Barto (2018)-Pseudocode (Abschnitt 13.3), nicht als ein einziges Batch-Update."""

import numpy as np

import pg_constants as C
from pg_grid import ACTIONS, step
from pg_policy import SoftmaxPolicy

N_ACTIONS = len(ACTIONS)


def run_episode(grid, policy, feature_fn, rng, max_steps=C.MAX_STEPS_PER_EPISODE):
    s = grid.state_of(grid.start)
    states, actions, rewards = [], [], []
    for _ in range(max_steps):
        x = feature_fn(s)
        a = policy.sample(x, rng)
        s_next, r, done = step(grid, s, a, rng)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = s_next
        if done:
            break
    return states, actions, rewards


def returns_from_rewards(rewards, gamma):
    """G_t = sum_{k=t}^{T-1} gamma^(k-t) * r_{k+1}, rueckwaerts akkumuliert (ein Durchlauf reicht)."""
    T = len(rewards)
    G = np.zeros(T)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


def train(grid, feature_fn, n_features, lr, episodes, seed, max_steps=C.MAX_STEPS_PER_EPISODE, checkpoints=()):
    rng = np.random.default_rng(seed)
    policy = SoftmaxPolicy(n_features, N_ACTIONS, seed)
    returns = np.zeros(episodes)
    lengths = np.zeros(episodes, dtype=int)
    falls = np.zeros(episodes, dtype=int)
    snapshots = {}
    checkpoint_set = set(checkpoints)
    for e in range(episodes):
        states, actions, rewards = run_episode(grid, policy, feature_fn, rng, max_steps)
        G = returns_from_rewards(rewards, grid.gamma)
        for t in range(len(states)):
            x = feature_fn(states[t])
            grad = policy.grad_log_prob(x, actions[t])
            policy.step(grad, lr * (grid.gamma ** t) * G[t])
        returns[e] = float(sum(rewards))
        lengths[e] = len(rewards)
        falls[e] = sum(1 for r in rewards if r == C.CLIFF_PENALTY)
        if (e + 1) in checkpoint_set:
            snap = SoftmaxPolicy(n_features, N_ACTIONS, seed)
            snap.theta = policy.theta.copy()
            snapshots[e + 1] = snap
    return policy, returns, lengths, falls, snapshots


def action_probs_table(policy, feature_fn, n_states):
    X = np.stack([feature_fn(s) for s in range(n_states)])
    return policy.probs(X)
