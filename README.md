# 🧭 Policy Gradient / REINFORCE

Siebtes Stück der **Reinforcement-Learning-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Derselbe Lagerroboter wie bei den vorigen Stücken, aber ein grundlegend anderer Ansatz: **REINFORCE** (Williams 1992) parametrisiert die Politik direkt als Softmax-Regression über Zustandsmerkmalen und verschiebt sie nach jeder abgeschlossenen Episode in Richtung der tatsächlich erlebten Monte-Carlo-Rückgabe – ohne Wertfunktion, ohne Modell.

## Kernfrage

**Wie stark hängt der Erfolg von REINFORCE vom Zufall ab?** Mit identischen Hyperparametern erreichen manche Zufalls-Seeds eine nahezu optimale Politik, andere bleiben dauerhaft in einer schlechten, gesättigten Politik hängen – der rohe Monte-Carlo-Politikgradient hat eine hohe Varianz (Williams 1992), das ist keine Kalibrierungsfrage.

## Modell

- **Vehikel:** derselbe Lagerroboter, aber ein **kleineres Raster** als bei den Geschwistern – siehe Plan-Korrektur unten.
- **Die Politik** (`pg_policy.py`): eine lineare Softmax-Regression $\pi(a|s) = \operatorname{softmax}(\theta^\top x(s))_a$ über One-Hot-Zustandsmerkmalen – kein verstecktes Netz nötig (anders als DQN, Stück 6), da Verallgemeinerung nicht der Hook dieses Stücks ist.
- **REINFORCE** (`pg_agent.py`): eine ganze Episode wird MIT der aktuellen Politik gespielt, danach für jeden Zeitschritt die Rückgabe $G_t = \sum_{k=t}^{T-1}\gamma^{k-t} r_{k+1}$ berechnet und $\theta \leftarrow \theta + \alpha\,\gamma^t G_t\,\nabla_\theta\log\pi(a_t|s_t)$ angewendet – Schritt für Schritt wie im Sutton & Barto (2018)-Pseudocode.
- **Korrektheits-Check:** der analytische Gradient von $\log\pi(a|s)$ stimmt exakt mit einem numerisch differenzierten Gradienten auf einer winzigen Politik überein (`test_grad_log_prob_matches_a_numerical_gradient_check`).

## Zwei Plan-Korrekturen (gemessen, nicht behauptet)

**1. Das 4×8-Raster der Geschwister ist für reines Monte-Carlo-REINFORCE ein schlechtes Vehikel.** REINFORCE lernt NUR aus abgeschlossenen Episoden – anders als die TD-Verfahren der Geschwister, die aus jedem einzelnen Schritt lernen. Gemessen: auf dem 4×8-Raster erreicht selbst eine rein zufällige Politik in 80 % der Fälle NICHT einmal Ziel oder Klippe innerhalb von 300 Schritten (sie läuft in den Schritt-Deckel). Auf dem kleineren **3×4-Standardraster** dieses Stücks sinkt dieser Anteil auf ca. 12 % – erst damit wird das Vehikel für REINFORCE lernbar.

**2. Rohes REINFORCE ist instabil – und das lässt sich nicht wegtunen.** Die Klippen-Strafe (−100) erzeugt gelegentlich riesige Rückgaben; ein einzelner solcher Schritt kann die Softmax-Politik in eine Sättigung schießen (gemessen: |θ| > 50 schon nach 5 Episoden bei Lernrate 0,20). Eine kleinere Lernrate reduziert das Risiko, beseitigt es aber nicht: bei **jeder** getesteten Lernrate bleibt ein Teil der Zufalls-Seeds dauerhaft in einer schlechten Politik hängen. Das ist der eigentliche Befund dieses Stücks, nicht ein Kalibrierungsproblem.

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Korrektheits-Check** | Analytischer Gradient von $\log\pi(a\|s)$ stimmt mit einem numerischen Gradienten-Check exakt überein (Toleranz 1e-5). | `test_grad_log_prob_matches_a_numerical_gradient_check` |
| **Streuung über 30 unabhängige Seeds** (Standardeinstellungen, 2000 Episoden) | 16,7 % der Seeds erreichen eine nahezu optimale Politik (Wert-Abstand < 1), 16,7 % bleiben dauerhaft hängen (Wert-Abstand > 50, Median 5,26). Drei der "hängengebliebenen" Seeds landen auf fast demselben Wert-Abstand (≈114,4) – ein Hinweis auf eine gemeinsame degenerierte Zielpolitik. | `test_seed_variance_experiment_matches_the_readme` |
| **Ein Beispiel je Kategorie** | Seed 14: Wert-Abstand 0,66 (nahezu optimal). Seed 5: Wert-Abstand 114,4 (dauerhaft hängengeblieben) – identischer Algorithmus, identische Hyperparameter. | `test_lucky_and_unlucky_seed_examples` |
| **Vehikel-Wahl** | Auf dem 4×8-Raster der Geschwister erreicht eine zufällige Politik in > 50 % der Fälle keinen Endzustand; auf dem gewählten 3×4-Raster in < 30 % der Fälle. | `test_random_walk_rarely_terminates_on_the_larger_sibling_grid`, `test_random_walk_terminates_reliably_on_the_chosen_default_grid` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eine Lernrate, die zur Rückgabe-Skala passt** | Zu hohe Lernraten sättigen die Politik nach einer einzigen schlechten Episode; zu niedrige verlangsamen das Lernen, ohne das Hängenbleiben zu verhindern (gemessen). | Rückgabe-Normalisierung, eine gelernte Basislinie |
| **Genug abgeschlossene Episoden** | REINFORCE lernt nur aus kompletten Episoden – auf einem größeren Raster gibt es kaum Lernsignal (gemessen, siehe Plan-Korrektur 1). | TD-Verfahren (Q-Learning, SARSA, DQN) |
| **Eine niedrige Varianz des Rückgabe-Schätzers** | Identische Hyperparameter führen bei verschiedenen Seeds zu sehr unterschiedlichen Ergebnissen (gemessen, Kernexperiment). | Eine gelernte Zustandswert-Basislinie senkt die Varianz, ohne den Erwartungswert zu ändern (Actor-Critic, Stück 8) |
| **Diskrete, kleine Aktionsräume** | Die Softmax-Politik hier hat vier Aktionen; stetige Aktionsräume brauchen eine andere Parametrisierung. | Kontinuierliche Policy-Gradient-Verfahren (PPO, SAC) |

## Tests

`tests/` prüft das Vehikel, die Referenzlösung (Value Iteration UND die stochastische Politik-Auswertung, die REINFORCEs tatsächliche Verteilung bewertet, nicht ihre Argmax-Version), die Politik (inkl. des numerischen Gradienten-Checks), den Agenten (Rückgabe-Berechnung von Hand, ein struktureller Schnappschuss-Test), die Auswertung, die Presets/Permalinks, die Plotly-Achsen, die App (AppTest: jedes Preset, Trainingsstand-Slider, Permalink-Grenzen, Extremwerte, Experiment auf Abruf) und jede Zahl dieses READMEs (`test_claims.py`, mit der vollen Seed-Zahl – bei festen Seeds deterministisch, aber über 2000 Trainings-Episoden reichen winzige Gleitkomma-Unterschiede zwischen Windows und Linux-CI, um einen ohnehin knapp an der Schwelle liegenden Seed in die andere Kategorie zu kippen – passend zum Thema dieses Stücks, deshalb großzügige statt exakte Bänder). Gesamtlaufzeit einige Minuten statt der sonst üblichen knappen Minute (das Seed-Streuungs-Experiment braucht 30 vollständig unabhängige Trainings); die CI läuft bei jedem Push und wöchentlich.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche: Episode-für-Episode-Ansicht, Kernfrage, Seed-Streuungs-Experiment auf Abruf, Grenzen, Formeln |
| `pg_constants.py` | Regler-Grenzen, feste Rewards, Lernparameter, Experiment-Konstanten |
| `pg_grid.py` | Das Vehikel: Raster, `step`, `build_model` (nur für die Referenz) |
| `pg_features.py` | One-Hot-Zustandsmerkmale |
| `pg_policy.py` | Softmax-Politik: Wahrscheinlichkeiten, Sampling, analytischer Gradient |
| `pg_agent.py` | REINFORCE: Episode erzeugen, Rückgabe berechnen, Politik aktualisieren |
| `pg_reference.py` | Value Iteration und (deterministische UND stochastische) Politik-Auswertung – nur zur Gegenprobe |
| `pg_evaluation.py` | Analyse, Seed-Streuungs-Experiment |
| `pg_visualization.py` | Plotly-Abbildungen |
| `pg_presets.py` | Presets, Permalink |
| `tests/` | Tests (siehe oben) |

## Bewusst nicht umgesetzt

- **Eine gelernte Basislinie zur Varianzreduktion** – das ist genau der Schritt, den Actor-Critic (Stück 8) macht; hier bleibt REINFORCE bewusst in seiner ursprünglichen, hochvarianten Form (Williams 1992).
- **Rückgabe-Normalisierung/Clipping als Stabilitäts-Trick** – hätte die Sättigung abgeschwächt, aber die Kernaussage (hohe Varianz zwischen Seeds) verwässert, ohne sie zu beseitigen.
- **Kontinuierliche Aktionsräume** – die Softmax-Politik bleibt bei vier diskreten Richtungen wie die restliche Linie.

## Lokal ausführen

```bash
pip install -r requirements-dev.txt
streamlit run app.py
python -m pytest tests/ -q
```

Gebaut mit Streamlit, Plotly und numpy.
