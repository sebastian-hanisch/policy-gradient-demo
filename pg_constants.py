"""Konstanten der Demo "Policy Gradient / REINFORCE" (Stück 7 der Reinforcement-Learning-Linie). Anders als DQN (Stück 6) braucht dieses Stück
KEIN verstecktes Netz: die Politik ist eine lineare Softmax-Regression ueber One-Hot-Zustandsmerkmalen (jeder Zustand hat seine eigene, isolierte
Spalte an Aktionspraeferenzen - kein Verallgemeinerungs-Anspruch, das ist nicht der Hook dieses Stuecks, siehe PLAN.md Hook 7)."""

EPS = 1e-9
SEED_MAX = 999999

# --- Das Raster --------------------------------------------------------------------------------------------------------------------------------
STEP_COST = -1.0
CLIFF_PENALTY = -100.0
GOAL_REWARD = 10.0

# ACHTUNG (Plan-Korrektur): DEFAULT_ROWS/DEFAULT_COLS sind kleiner als bei den Geschwistern (dort 4x8). Gemessen: auf dem 4x8-Raster erreicht
# selbst eine rein zufaellige Politik in 80% der Faelle NICHT einmal Ziel oder Klippe innerhalb von 300 Schritten (sie laeuft in den Deckel).
# REINFORCE ist episodisch/Monte-Carlo - es lernt NUR aus abgeschlossenen Episoden, anders als die TD-Verfahren der Geschwister, die aus JEDEM
# einzelnen Schritt lernen. Auf dem kleineren 3x4-Raster faellt dieser Anteil auf ~12%, das Vehikel wird damit erst lernbar.
ROWS_MIN, ROWS_MAX, DEFAULT_ROWS = 3, 6, 3
COLS_MIN, COLS_MAX, DEFAULT_COLS = 4, 12, 4
SLIP_MIN, SLIP_MAX, SLIP_STEP, DEFAULT_SLIP = 0.0, 0.30, 0.02, 0.10
GAMMA_MIN, GAMMA_MAX, GAMMA_STEP, DEFAULT_GAMMA = 0.80, 0.99, 0.01, 0.95

# --- Referenzloesung (Value Iteration, nur zur Gegenprobe) -------------------------------------------------------------------------------------
VI_TOL = 1e-8
VI_MAX_ITER = 5000

# --- Die Politik (lineare Softmax-Regression, One-Hot-Eingabe - siehe Docstring oben) ------------------------------------------------------------
# ACHTUNG (Plan-Korrektur): rohes REINFORCE ohne Baseline (Williams 1992) ist bei diesem Vehikel hochgradig instabil - die Klippen-Strafe (-100)
# erzeugt gelegentlich riesige Rueckgaben G_t, ein einzelner solcher Schritt kann |theta| in die Saettigung schiessen (gemessen: |theta|>50 schon
# nach 5 Episoden bei lr=0.2). Eine kleinere Lernrate REDUZIERT das Risiko, beseitigt es aber NICHT: bei JEDER getesteten Lernrate bleibt ein
# Teil der Zufalls-Seeds dauerhaft in einer schlechten, gesaettigten Politik haengen (gemessen: ~17% der Seeds bei lr=0,005) - genau das ist der
# Befund dieses Stuecks (hohe Varianz), keine Kalibrierungsfrage, die sich wegtunen laesst.
LR_MIN, LR_MAX, LR_STEP, DEFAULT_LR = 0.001, 0.05, 0.001, 0.005
MAX_STEPS_PER_EPISODE = 300

EPISODES_MIN, EPISODES_MAX, EPISODES_STEP, DEFAULT_EPISODES = 100, 3000, 100, 2000

NEAR_OPTIMAL_GAP = 1.0
STUCK_GAP_THRESHOLD = 50.0

# --- Experiment: wie stark schwankt der Trainingserfolg zwischen Zufalls-Seeds ------------------------------------------------------------------
# EXP_SEEDS in der App kleiner als in test_claims.py (dort die volle Zahl fuer die README-Zahlen) - siehe README/Tests fuer die Zeit-Abwaegung.
EXP_SEEDS = 20
EXP_SEEDS_FULL = 30

# Konkrete Beispiel-Seeds aus der gemessenen 30-Seed-Verteilung bei den Standardeinstellungen (siehe README) - kein Zufallstreffer fuer die
# Presets "Unglücklicher"/"Glücklicher Seed". Auffaellig: mehrere "haengengebliebene" Seeds landen auf fast demselben Wert-Abstand (~114,4) -
# ein Hinweis auf eine gemeinsame, degenerierte Zielpolitik (nicht weiter untersucht, siehe README Grenzen).
LUCKY_SEED = 14
UNLUCKY_SEED = 5
