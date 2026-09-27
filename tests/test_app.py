"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Trainingsstand-Slider, Permalink-Grenzen/-Raster, Extremwerte, Experiment auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import pg_constants as C
import pg_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def _fast_default_episodes(monkeypatch):
    # AppTest-Rauchtests brauchen keine realistische Konvergenz, nur fehlerfreie Codepfade - DEFAULT_EPISODES=2000 waere hier nur teuer, ohne
    # zusaetzliche Aussagekraft (die echten README-Zahlen kommen ausschliesslich aus test_claims.py mit den vollen Standardwerten).
    # monkeypatch.setitem stellt die urspruenglichen Werte nach jedem Test automatisch wieder her (kein dauerhafter Seiteneffekt); SettingSpec
    # ist ein frozen dataclass, deshalb der ganze Dict-Eintrag statt eines einzelnen Felds.
    for name, preset in P.PRESETS.items():
        if preset["episodes"] == C.DEFAULT_EPISODES:
            monkeypatch.setitem(preset, "episodes", C.EPISODES_MIN)
    spec = P.SETTING_SPECS["episodes_slider"]
    monkeypatch.setitem(P.SETTING_SPECS, "episodes_slider", P.SettingSpec(spec.url_param, spec.caster, C.EPISODES_MIN, spec.lo, spec.hi))
    yield


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.error) + list(at.info):
        assert "{de(" not in el.value, el.value[:120]


def test_default_run_shows_metrics_charts_and_a_verdict():
    at = _run()
    _ok(at)
    assert len(at.metric) == 3 and len(at.get("plotly_chart")) >= 2
    assert len(at.success) + len(at.warning) + len(at.error) >= 1


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == pytest.approx(p[key]) if isinstance(p[key], float) else at.session_state[state_key] == p[key]


def test_frame_slider_survives_a_smaller_episode_count():
    at = _run(pg_frame_idx=5)
    _ok(at)
    at.slider(key="episodes_slider").set_value(C.EPISODES_MIN).run()
    _ok(at)


def test_permalink_values_are_snapped_and_clamped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["rows"] = "999"
    at.query_params["cols"] = "-5"
    at.query_params["slip"] = "abc"
    at.query_params["episodes"] = "999999"
    at.query_params["seed"] = "-1"
    at.run()
    _ok(at)
    s = at.session_state
    assert s["rows_slider"] == C.ROWS_MAX and s["cols_slider"] == C.COLS_MIN
    assert s["slip_slider"] == C.DEFAULT_SLIP and s["episodes_slider"] == C.EPISODES_MAX
    assert s["seed_slider"] == 0


@pytest.mark.parametrize("kw", [
    dict(rows_slider=C.ROWS_MIN, cols_slider=C.COLS_MIN, episodes_slider=C.EPISODES_MIN),
    dict(rows_slider=C.ROWS_MAX, cols_slider=C.COLS_MAX, episodes_slider=C.EPISODES_MIN),
    dict(lr_slider=C.LR_MIN, seed_slider=C.UNLUCKY_SEED),
    dict(lr_slider=C.LR_MAX, slip_slider=C.SLIP_MAX),
])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_seed_variance_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "EXP_SEEDS", 3)
    at = _run(episodes_slider=100)
    next(b for b in at.button if b.key == "variance_start").click().run()
    _ok(at)
    assert at.session_state["variance_on"] and any("Befund" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
