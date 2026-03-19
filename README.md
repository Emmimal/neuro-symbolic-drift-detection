# neuro-symbolic-drift-detection
Label-free concept drift detection for neuro-symbolic fraud models — FIDI Z-Score fires before F1 drops, zero labels required.

# neuro-symbolic-drift-detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: CC-0](https://img.shields.io/badge/Dataset-CC--0-lightgrey.svg)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Label-free concept drift detection for neuro-symbolic fraud models using FIDI Z-Score. Part 3 of the neuro-symbolic fraud detection series.

📄 **Article:** [Catching Concept Drift Before F1 Drops (Label-Free)](https://towardsdatascience.com/)
🔗 **Series:** [Part 1](https://towardsdatascience.com/hybrid-neuro-symbolic-fraud-detection-guiding-neural-networks-with-domain-rules/) · [Part 2](https://towardsdatascience.com/how-a-neural-network-learned-its-own-fraud-rules-a-neuro-symbolic-ai-experiment/) · **Part 3**

---

## Setup

```bash
git clone https://github.com/Emmimal/neuro-symbolic-drift-detection.git
cd neuro-symbolic-drift-detection
pip install -r requirements.txt
```

Download [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (CC-0) and place `creditcard.csv` in `data/`.

---

## Run

```bash
# Full pipeline: train → experiment → figures → alert demo
python app.py --step all

# Individual steps
python app.py --step train
python app.py --step experiment
python app.py --step figures
python app.py --step alert

# Options
python app.py --step all --seeds 42 7 --drift_types concept covariate
python app.py --step train --verbose
```

---

## Repo structure

```
neuro-symbolic-drift-detection/
├── app.py                          entry point
├── neuro_symbolic_drift/
│   ├── model.py                    HybridRuleLearner, LearnableDiscretizer, RuleLearner
│   ├── data_loader.py              dataset loading, 70/15/15 stratified split
│   ├── train.py                    training loop, three-part loss, temperature annealing
│   ├── drift_injector.py           covariate / prior / concept drift across 8 windows
│   ├── drift_metrics.py            RWSS, FIDI, RFR, RWSS Velocity, FIDI Z-Score, PSI_rules
│   ├── experiment.py               multi-seed experiment loop, detection lag computation
│   ├── alert_system.py             DriftAlertSystem — build once, call .check() per batch
│   ├── figures.py                  generates all 7 article figures
│   └── __init__.py
├── data/                           place creditcard.csv here
├── checkpoints/                    saved model weights (gitignored)
├── results/
│   ├── summary_table.csv           drift detection results across all seeds
│   └── figures/                    generated PNGs (gitignored)
├── notebooks/
│   └── explore_results.ipynb       interactive result exploration
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Alert system usage

```python
from neuro_symbolic_drift.alert_system import DriftAlertSystem
import torch

# Build once after training
alert_system = DriftAlertSystem.from_trained_model(model, torch.FloatTensor(X_val), feature_names)
alert_system.save("results/drift_alert_baseline_seed42.pkl")

# Every scoring run
alert_system = DriftAlertSystem.load("results/drift_alert_baseline_seed42.pkl")
alert = alert_system.check(model, X_this_week)

if alert.fired:
    print(alert.report())
    # alert.severity          → "warning" | "critical"
    # alert.earliest_layer    → "velocity" | "fidi_z" | "psi_rules" | "rwss"
    # alert.fidi_z_top_feats  → [(feature_name, z_score), ...]
```

---

## Results

| Drift type | F1 fired | FIDIZ fired | FIDIZ mean lag |
|-----------|---------|------------|---------------|
| Covariate | 4/5 | 0/5 | — |
| Prior | 2/5 | 5/5 | −2.00w (late) |
| **Concept** | **5/5** | **5/5** | **+0.40w (early)** |

| Seed | Val PR-AUC | Best epoch | α |
|------|-----------|-----------|---|
| 42 | 0.7717 | 7 | 0.891 |
| 0 | 0.6915 | 6 | 0.889 |
| 7 | 0.6799 | 10 | 0.886 |
| 123 | 0.7899 | 5 | 0.883 |
| 2024 | 0.7951 | 3 | 0.879 |

---

## Known limitations

- Covariate drift: blind spot — FIDIZ fires 0/5 seeds. Use PSI on raw features separately.
- Prior drift: structurally late — FIDIZ needs 3 windows of history before it can fire.
- PSI_rules: silent throughout (PSI_rules = 0.0049) — soft activations at early-stopped checkpoints (τ ≈ 3.5–4.0) produce near-uniform distributions.
- 5 seeds: consistent pattern, not a production guarantee.

---

## Citation

```bibtex
@misc{alexander2026drift,
  author = {Alexander, Emmimal P.},
  title  = {Catching Concept Drift Before F1 Drops: Label-Free Symbolic Monitoring for Fraud Models},
  year   = {2026},
  url    = {https://github.com/Emmimal/neuro-symbolic-drift-detection}
}
```

---

MIT License · Dataset: CC-0 Public Domain
