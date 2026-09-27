import numpy as np
import streamlit as st

import pg_constants as C
from pg_evaluation import Settings, _reference, analyse, seed_variance_experiment
from pg_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, sync_query_params
from pg_visualization import build_grid, build_learning_curve, build_seed_variance

st.set_page_config(page_title="Policy Gradient / REINFORCE", page_icon="🧭", layout="wide")

init_session_state_defaults()
load_permalink_settings()


def de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


@st.cache_data(show_spinner=False)
def _analyse(settings, checkpoints):
    return analyse(settings, checkpoints=checkpoints)


@st.cache_data(show_spinner=False)
def _seed_variance_exp(base):
    return seed_variance_experiment(base=base)


st.title("🧭 Policy Gradient / REINFORCE")
st.markdown(
    """
Derselbe Lagerroboter wie bei den vorigen Stücken - aber ein grundlegend anderer Ansatz: statt eine Wertfunktion zu lernen und daraus eine
Politik abzuleiten (Q-Learning, SARSA, Dyna-Q, DQN), parametrisiert **REINFORCE** (Williams 1992) die Politik direkt als Wahrscheinlichkeits-
verteilung über Aktionen (eine **Softmax-Regression** über Zustandsmerkmale) und verschiebt sie nach jeder kompletten Episode in Richtung der
tatsächlich erlebten (Monte-Carlo-)Rückgabe. Alle Daten sind erzeugt.
"""
)
st.caption(
    "Siebtes Stück der **Reinforcement-Learning-Linie** der \"Konzepte\"-Reihe. **Bezug:** anders als DQN (Stück 6) - Werte lernen, Politik "
    "daraus ableiten - lernt REINFORCE die Politik selbst, direkt; anders als Q-Learning/SARSA/Dyna-Q lernt es nur aus KOMPLETTEN Episoden, "
    "nicht aus jedem einzelnen Schritt."
)

with st.expander("So funktioniert REINFORCE", expanded=True):
    st.markdown(
        r"""
1. **Die Politik** ist eine Softmax-Regression: $\pi(a|s) = \dfrac{\exp(\theta^\top x(s))_a}{\sum_{a'} \exp(\theta^\top x(s))_{a'}}$ - mit
   One-Hot-Merkmalen $x(s)$ hat jeder Zustand seine eigene, unabhängige Spalte an Aktionspräferenzen $\theta$.
2. **Eine ganze Episode** wird MIT dieser Politik gespielt (Aktionen werden gemäß ihrer Wahrscheinlichkeit gezogen, nicht gierig gewählt).
3. **Die Rückgabe** $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1}$ wird für jeden Zeitschritt der abgeschlossenen Episode rückwärts berechnet.
4. **Der Politikgradienten-Schritt:** $\theta \leftarrow \theta + \alpha\, \gamma^t G_t\, \nabla_\theta \log\pi(a_t|s_t)$ - Schritte mit hoher
   Rückgabe werden wahrscheinlicher gemacht, Schritte mit niedriger Rückgabe unwahrscheinlicher. Kein Modell, keine Wertfunktion nötig.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            if st.button(name, key=f"preset_{name}", help=PRESET_HELP[name], width="stretch"):
                apply_preset(name)
st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.subheader("Das Raster")
    rows = st.slider("Zeilen", *bounds("rows_slider"), key="rows_slider")
    cols = st.slider("Spalten", *bounds("cols_slider"), key="cols_slider")
    slip = st.slider("Rutsch-Wahrscheinlichkeit", *bounds("slip_slider"), key="slip_slider", step=C.SLIP_STEP, format="%.2f")
    gamma = st.slider("Diskontfaktor γ", *bounds("gamma_slider"), key="gamma_slider", step=C.GAMMA_STEP, format="%.2f")
    st.subheader("Das Training")
    lr = st.slider("Lernrate", *bounds("lr_slider"), key="lr_slider", step=C.LR_STEP, format="%.3f")
    episodes = st.slider("Trainingsepisoden", *bounds("episodes_slider"), key="episodes_slider", step=C.EPISODES_STEP)
    seed = st.slider("Seed", *bounds("seed_slider"), key="seed_slider")

sync_query_params({
    "rows_slider": int(rows), "cols_slider": int(cols), "slip_slider": round(float(slip), 3), "gamma_slider": round(float(gamma), 3),
    "lr_slider": round(float(lr), 4), "episodes_slider": int(episodes), "seed_slider": int(seed),
})

settings = Settings(int(rows), int(cols), round(float(slip), 3), round(float(gamma), 3), round(float(lr), 4), int(episodes), int(seed))
frames = tuple(sorted({0} | {int(round(x)) for x in np.linspace(1, settings.episodes, min(settings.episodes, 11))}))
with st.spinner("REINFORCE trainiert ..."):
    a = _analyse(settings, frames)
grid = a.grid
s0 = grid.state_of(grid.start)

# --- Episode für Episode -----------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Episode für Episode zur gelernten Politik")
if "pg_frame_idx" not in st.session_state or st.session_state.get("pg_frame_owner") != settings:
    st.session_state["pg_frame_idx"] = len(frames) - 1
    st.session_state["pg_frame_owner"] = settings
idx = st.slider("Trainingsstand", 0, len(frames) - 1, key="pg_frame_idx", help="0 = noch untrainiert (uniforme Zufallspolitik).")
ep = frames[idx]
from pg_agent import action_probs_table
from pg_features import one_hot
from pg_reference import policy_evaluation_stochastic
feature_fn = lambda s: one_hot(s, grid.n_states)
if ep == 0:
    probs_snap = np.full((grid.n_states, 4), 0.25)
elif ep == settings.episodes:
    probs_snap = a.action_probs
else:
    probs_snap = action_probs_table(a.snapshots[ep], feature_fn, grid.n_states)
_, P_ref, Rw_ref, *_ = _reference(settings.rows, settings.cols, settings.slip, settings.gamma)
V_snap = policy_evaluation_stochastic(P_ref, Rw_ref, probs_snap, settings.gamma)
head = "Vor dem Training (uniforme Zufallspolitik)" if ep == 0 else f"Nach {ep} von {settings.episodes} Episoden"
c1, c2 = st.columns([3, 2])
c1.markdown(f"**{head}**")
c1.plotly_chart(build_grid(grid, V_snap, probs_snap), width="stretch", key=f"pg_grid_{ep}")
c2.markdown("**Ertrag je Episode**")
c2.plotly_chart(build_learning_curve(a.returns[:ep] if ep > 0 else np.array([0.0])), width="stretch", key=f"pg_curve_{ep}")

st.markdown("---")

# --- Kernfrage ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Wie stark hängt der Erfolg von REINFORCE vom Zufall ab?")
mcols = st.columns(3)
mcols[0].metric("Wert-Abstand zu V*(Start)", de(a.gap, 2))
mcols[1].metric("Umgebungsschritte", f"{a.env_steps:,}".replace(",", "."))
mcols[2].metric("Trainingsepisoden", settings.episodes)
if a.gap < C.NEAR_OPTIMAL_GAP:
    st.success(f"✅ Bei Seed {settings.seed} lernt REINFORCE hier eine nahezu optimale Politik (Wert-Abstand {de(a.gap,2)}).")
elif a.gap > C.STUCK_GAP_THRESHOLD:
    st.error(f"🛑 Bei Seed {settings.seed} bleibt REINFORCE hier in einer schlechten, festgefahrenen Politik hängen (Wert-Abstand {de(a.gap,2)}) - gleiche Hyperparameter, nur der Zufall unterscheidet sich. Siehe Experiment unten für die volle Streuung.")
else:
    st.warning(f"⚠️ Bei Seed {settings.seed} lernt REINFORCE eine brauchbare, aber nicht optimale Politik (Wert-Abstand {de(a.gap,2)}).")

st.markdown("---")

# --- Experiment: Streuung zwischen Seeds --------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie stark schwankt der Trainingserfolg zwischen Zufalls-Seeds?")
st.caption(
    f"Standardraster, {C.EXP_SEEDS} unabhängige Trainingsläufe mit denselben Hyperparametern (Zeilen/Spalten/Rutschen/Lernrate/Episoden von den "
    "Reglern oben, nur der Seed unterscheidet sich). Gezeigt: Wert-Abstand zu V*(Start) je Seed, aufsteigend sortiert. Dauer: gut eine Minute."
)
if st.button("Streuung durchrechnen (gut eine Minute)", key="variance_start"):
    st.session_state["variance_on"] = True
if st.session_state.get("variance_on"):
    base = Settings(int(rows), int(cols), round(float(slip), 3), round(float(gamma), 3), round(float(lr), 4), int(episodes))
    with st.spinner(f"Trainiert {C.EXP_SEEDS} unabhängige Politiken ..."):
        exp = _seed_variance_exp(base)
    st.plotly_chart(build_seed_variance(exp), width="stretch", key="variance_chart")
    st.warning(
        f"**Befund:** von {len(exp['seeds'])} unabhängigen Trainingsläufen mit identischen Hyperparametern erreichen "
        f"{de(100*exp['frac_near_optimal'],0)} % eine nahezu optimale Politik, während {de(100*exp['frac_stuck'],0)} % dauerhaft in einer "
        f"schlechten, gesättigten Politik hängen bleiben (Median-Wert-Abstand {de(exp['median_gap'],2)}). **Das ist keine Frage der richtigen "
        "Lernrate** (siehe Grenzen-Tabelle) - es ist die hohe Varianz des rohen Monte-Carlo-Politikgradienten selbst: eine einzelne, durch Zufall "
        "sehr schlechte (oder sehr gute) frühe Episode kann die Politik dauerhaft in eine Richtung festlegen, bevor genug Erfahrung für eine "
        "Korrektur da ist."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eine Lernrate, die zur Rückgabe-Skala passt** | Die Klippen-Strafe (-100) erzeugt gelegentlich riesige Rückgaben; eine zu hohe Lernrate lässt die Politik nach einer einzigen schlechten Episode in eine gesättigte Softmax-Verteilung schießen (gemessen: |θ|>50 schon nach 5 Episoden bei Lernrate 0,20). Eine kleinere Lernrate reduziert das Risiko, beseitigt es aber NICHT vollständig (siehe Experiment oben). | Rückgabe-Normalisierung, eine gelernte Basislinie (Actor-Critic, Stück 8) |
| **Genug abgeschlossene Episoden** | REINFORCE lernt NUR aus kompletten Episoden - auf einem größeren Raster erreicht selbst eine zufällige Politik Ziel oder Klippe kaum je innerhalb des Schritt-Deckels, und es gibt schlicht kein Lernsignal. Deshalb ein kleineres Raster als bei den Geschwistern (gemessen, siehe README). | TD-Verfahren (Q-Learning, SARSA), die aus jedem Schritt lernen |
| **Eine niedrige Varianz des Rückgabe-Schätzers** | Der rohe Monte-Carlo-Schätzer $G_t$ hat hohe Varianz - identische Hyperparameter führen bei verschiedenen Zufalls-Seeds zu sehr unterschiedlichen Ergebnissen (gemessen, Experiment oben). | Eine gelernte Zustandswert-Basislinie senkt die Varianz, ohne den Erwartungswert zu ändern (Actor-Critic, Stück 8) |
| **Diskrete, kleine Aktionsräume** | Die Softmax-Politik hier hat vier Aktionen; bei stetigen Aktionsräumen braucht es eine andere Parametrisierung (z. B. eine Gauß-Politik). | Kontinuierliche Policy-Gradient-Verfahren (PPO, SAC) |
"""
)
st.caption("Die Linie: Bandit → Value Iteration und Policy Iteration → Q-Learning → SARSA → Dyna-Q → DQN / Funktionsapproximation → **Policy Gradient / REINFORCE** (dieses Stück) → Actor-Critic.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Politik:** $\pi_\theta(a|s) = \dfrac{\exp(\theta^\top x(s,a))}{\sum_{a'} \exp(\theta^\top x(s,a'))}$, hier mit einer Spalte $\theta_{:,a}$ je Aktion.

**Rückgabe:** $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1}$ (Monte-Carlo, erst nach Ende der Episode berechenbar).

**Politikgradienten-Theorem (Sutton, McAllester, Singh & Mansour, 2000):** $\nabla_\theta J(\theta) = \mathbb{E}_\pi[\nabla_\theta \log\pi_\theta(a_t|s_t)\, Q^\pi(s_t,a_t)]$; REINFORCE (Williams, 1992) schätzt $Q^\pi(s_t,a_t)$ unverzerrt durch die Stichproben-Rückgabe $G_t$.

**Update:** $\theta \leftarrow \theta + \alpha\, \gamma^t G_t\, \nabla_\theta \log\pi_\theta(a_t|s_t)$, mit $\nabla_\theta \log\pi_\theta(a|s)$ (Softmax-Score-Funktion): $\dfrac{\partial \log\pi(a|s)}{\partial \theta_{i,j}} = x(s)_i \big(\mathbb{1}[j=a] - \pi(j|s)\big)$.
"""
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. "
    "Interesse an einer maßgeschneiderten Lösung für Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
