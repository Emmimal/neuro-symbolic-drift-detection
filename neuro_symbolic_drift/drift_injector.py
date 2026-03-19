"""
drift_injector.py
-----------------
Synthetic drift injection across N time windows.

    Type 1 — covariate : shift V14, V4, V12 distribution by up to +3.0σ
    Type 2 — prior     : increase fraud rate from 0.17% → 2.0%
    Type 3 — concept   : flip sign of V14 for fraud cases (rule becomes wrong)

N_WINDOWS=8 gives gradual enough progression for RWSS to lead F1 by 2-3 windows.
"""

import numpy as np
from dataclasses import dataclass
from typing      import List, Tuple

N_WINDOWS      = 8     # 8 windows — each step is ~14% of full drift severity
DRIFT_FEATURES = ["V14", "V4", "V12"]


@dataclass
class DriftConfig:
    drift_type:             str
    n_windows:              int   = N_WINDOWS
    covariate_shift_max:    float = 3.0    # stronger: pushes activations hard enough
    prior_fraud_rate_end:   float = 0.020  # 2% fraud rate at final window
    concept_flip_strength:  float = 1.0


class DriftInjector:
    """
    Usage:
        injector = DriftInjector(X_test, y_test, config, feature_names)
        windows  = injector.get_windows()   # [(X_w, y_w), ...]
    Window 0 is always the clean baseline (no drift).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 config: DriftConfig, feature_names: List[str],
                 rng: np.random.Generator = None):
        self.X             = X.copy()
        self.y             = y.copy()
        self.config        = config
        self.feature_names = feature_names
        self.rng           = rng or np.random.default_rng(42)
        self.stds          = X.std(axis=0)
        self.means         = X.mean(axis=0)

    def get_windows(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return {
            "covariate": self._covariate,
            "prior":     self._prior,
            "concept":   self._concept,
        }[self.config.drift_type]()

    # ── Type 1 — covariate ──────────────────────────────────────────────── #
    def _covariate(self):
        windows = []
        for w in range(self.config.n_windows):
            severity = w / (self.config.n_windows - 1)
            X_w      = self.X.copy()
            for feat in DRIFT_FEATURES:
                if feat not in self.feature_names:
                    continue
                idx         = self.feature_names.index(feat)
                X_w[:, idx] += severity * self.config.covariate_shift_max * self.stds[idx]
            windows.append((X_w, self.y.copy()))
        return windows

    # ── Type 2 — prior ──────────────────────────────────────────────────── #
    def _prior(self):
        windows   = []
        n         = len(self.y)
        fraud_idx = np.where(self.y == 1)[0]
        legit_idx = np.where(self.y == 0)[0]
        base_rate = self.y.mean()

        for w in range(self.config.n_windows):
            severity    = w / (self.config.n_windows - 1)
            target_rate = base_rate + severity * (
                self.config.prior_fraud_rate_end - base_rate)
            n_fraud = int(n * target_rate)
            n_legit = n - n_fraud
            chosen  = np.concatenate([
                self.rng.choice(fraud_idx, n_fraud, replace=True),
                self.rng.choice(legit_idx, n_legit, replace=False),
            ])
            self.rng.shuffle(chosen)
            windows.append((self.X[chosen], self.y[chosen]))
        return windows

    # ── Type 3 — concept ────────────────────────────────────────────────── #
    def _concept(self):
        windows    = []
        v14_idx    = (self.feature_names.index("V14")
                      if "V14" in self.feature_names else None)
        fraud_mask = self.y == 1

        for w in range(self.config.n_windows):
            severity = w / (self.config.n_windows - 1)
            X_w      = self.X.copy()
            if v14_idx is not None:
                orig     = X_w[fraud_mask, v14_idx]
                flipped  = -orig * self.config.concept_flip_strength
                X_w[fraud_mask, v14_idx] = (
                    (1 - severity) * orig + severity * flipped
                )
            windows.append((X_w, self.y.copy()))
        return windows
