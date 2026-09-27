"""SETTING_SPECS-Permalink-Muster, Presets und Regler-Grenzen (Standardmuster des Portfolios)."""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import pg_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "rows_slider": SettingSpec("rows", int, C.DEFAULT_ROWS, C.ROWS_MIN, C.ROWS_MAX),
    "cols_slider": SettingSpec("cols", int, C.DEFAULT_COLS, C.COLS_MIN, C.COLS_MAX),
    "slip_slider": SettingSpec("slip", float, C.DEFAULT_SLIP, C.SLIP_MIN, C.SLIP_MAX),
    "gamma_slider": SettingSpec("gamma", float, C.DEFAULT_GAMMA, C.GAMMA_MIN, C.GAMMA_MAX),
    "lr_slider": SettingSpec("lr", float, C.DEFAULT_LR, C.LR_MIN, C.LR_MAX),
    "episodes_slider": SettingSpec("episodes", int, C.DEFAULT_EPISODES, C.EPISODES_MIN, C.EPISODES_MAX),
    "seed_slider": SettingSpec("seed", int, 0, 0, C.SEED_MAX),
}
PRESET_KEYS = {
    "rows": "rows_slider", "cols": "cols_slider", "slip": "slip_slider", "gamma": "gamma_slider",
    "lr": "lr_slider", "episodes": "episodes_slider", "seed": "seed_slider",
}
STEPS = {
    "slip_slider": C.SLIP_STEP, "gamma_slider": C.GAMMA_STEP, "lr_slider": C.LR_STEP, "episodes_slider": C.EPISODES_STEP,
}


def _p(**kw):
    base = {
        "rows": C.DEFAULT_ROWS, "cols": C.DEFAULT_COLS, "slip": C.DEFAULT_SLIP, "gamma": C.DEFAULT_GAMMA,
        "lr": C.DEFAULT_LR, "episodes": C.DEFAULT_EPISODES, "seed": 0,
    }
    base.update(kw)
    return base


# Die genauen Seeds fuer "Ungluecklicher"/"Gluecklicher Seed" sind aus der gemessenen 30-Seed-Verteilung bei den Standardeinstellungen gewaehlt
# (siehe README) - kein Zufallstreffer, sondern ein bewusst herausgegriffenes Beispiel je Kategorie.
PRESETS = {
    "Standardfall": _p(),
    "Unglücklicher Seed": _p(seed=C.UNLUCKY_SEED),
    "Glücklicher Seed": _p(seed=C.LUCKY_SEED),
    "Höhere Lernrate": _p(lr=0.02),
    "Kürzer trainiert": _p(episodes=500),
    "Mit Rutschen": _p(slip=0.20),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = round(float(snapped), 4)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


PRESET_HELP = {
    "Standardfall": "Seed 0 mit den Standard-Hyperparametern - ein Beispiellauf, kein garantiert typisches Ergebnis (siehe Experiment unten).",
    "Unglücklicher Seed": "Ein Seed, der bei genau diesen Hyperparametern in einer schlechten, gesättigten Politik hängen bleibt - keine Ausnahme, sondern einer von mehreren in der gemessenen Verteilung.",
    "Glücklicher Seed": "Ein Seed, der bei denselben Hyperparametern nahezu die optimale Politik lernt - identischer Algorithmus, nur der Zufall unterscheidet sich.",
    "Höhere Lernrate": "Lernrate 0,02 statt 0,005 - schnelleres Lernen, aber mehr Risiko für eine verfrühte, falsche Festlegung (siehe Grenzen-Tabelle).",
    "Kürzer trainiert": "500 statt 2000 Episoden - REINFORCE braucht spürbar länger als die tabellarischen Geschwister, weil es nur aus abgeschlossenen Episoden lernt.",
    "Mit Rutschen": "Rutsch-Wahrscheinlichkeit 0,20 - mehr Zufall in der Umgebung selbst, zusätzlich zur Zufälligkeit der Politik.",
}
