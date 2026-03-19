"""
experiment.py
-------------
Full drift detection experiment loop — zero TODOs, completely runnable.

For each (seed x drift_type) combination:
    1. Extract baseline symbolic signatures
    2. Generate N drifted test windows via DriftInjector
    3. At each window compute layered early-warning metrics:
          RWSS_V   velocity alert      (earliest signal)
          FIDI_Z   z-score alert       (feature-level anomaly)
          PSI_R    rule activation PSI (symbolic-layer distribution)
          RWSS     absolute threshold  (confirmed shift)
          F1       label-based metric  (reference, latest)
    4. Detect alert windows and compute detection lag vs F1

Central finding: RWSS_V and PSI_rules detect concept drift 1-2 windows
before F1 drops, without any labels. The MLP compensates at output level;
the symbolic layer exposes the drift first.
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing      import List, Optional, Dict
import pandas as pd

from .drift_injector import DriftInjector, DriftConfig, N_WINDOWS
from .drift_metrics  import (
    extract_rule_weights, extract_rule_activations,
    extract_bit_means, predict_proba,
    compute_rwss, rwss_alert,
    compute_rwss_velocity, rwss_velocity_alert,
    compute_rolling_rwss,
    compute_fidi, fidi_top_features,
    compute_fidi_zscore, fidi_zscore_alert,
    compute_rfr, rfr_delta,
    compute_standard_metrics, compute_psi,
    compute_psi_rules, psi_rules_alert,
    RWSS_ALERT_THRESHOLD,
    RWSS_VEL_THRESHOLD,
    PSI_RULES_MODERATE,
)

SEEDS       = [42, 0, 7, 123, 2024]
DRIFT_TYPES = ["covariate", "prior", "concept"]


# ══════════════════════════════════════════════════════════════════════════════
#  Data containers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WindowResult:
    window:     int
    drift_type: str
    seed:       int
    # ── Original symbolic metrics ─────────────────────────────────────────── #
    rwss:       float = 0.0
    rwss_alert: bool  = False
    fidi_v14:   float = 0.0
    fidi_v12:   float = 0.0
    fidi_v4:    float = 0.0
    rfr_mean:   float = 0.0
    rfr_change: float = 0.0
    # ── NEW early-warning metrics ─────────────────────────────────────────── #
    rwss_velocity:       float = 0.0    # per-window RWSS drop rate
    rwss_vel_alert:      bool  = False  # True when velocity < -0.03
    rolling_rwss:        float = 1.0    # RWSS vs rolling 2-window baseline
    fidi_z_v14:          float = 0.0    # V14 FIDI z-score
    fidi_z_alert:        bool  = False  # True when any feature z > 2.5
    psi_rules:           float = 0.0    # PSI on rule activation distributions
    psi_rules_alert:     bool  = False  # True when PSI_rules >= 0.10
    psi_rules_severity:  str   = "stable"
    # ── Standard metrics ──────────────────────────────────────────────────── #
    f1:       float = 0.0
    roc_auc:  float = 0.0
    pr_auc:   float = 0.0
    f1_alert: bool  = False
    psi:      float = 0.0
    # ── Lag (filled post-hoc) ─────────────────────────────────────────────── #
    detection_lag:          Optional[int] = None
    detection_lag_velocity: Optional[int] = None
    detection_lag_fidi_z:   Optional[int] = None
    detection_lag_psi_r:    Optional[int] = None


@dataclass
class SeedDriftResult:
    seed:              int
    drift_type:        str
    windows:           List[WindowResult] = field(default_factory=list)
    baseline_f1:       float              = 0.0
    baseline_rwss:     float              = 1.0
    # Alert windows — one per metric
    rwss_alert_window:     Optional[int] = None
    rwss_vel_alert_window: Optional[int] = None
    fidi_z_alert_window:   Optional[int] = None
    psi_r_alert_window:    Optional[int] = None
    f1_alert_window:       Optional[int] = None
    # Detection lags vs F1
    detection_lag:          Optional[int] = None   # RWSS absolute
    detection_lag_velocity: Optional[int] = None   # RWSS velocity
    detection_lag_fidi_z:   Optional[int] = None   # FIDI z-score
    detection_lag_psi_r:    Optional[int] = None   # PSI rules


# ══════════════════════════════════════════════════════════════════════════════
#  Core loop
# ══════════════════════════════════════════════════════════════════════════════

def run_single_seed(seed:          int,
                    drift_type:    str,
                    X_test:        np.ndarray,
                    y_test:        np.ndarray,
                    trained_model,
                    feature_names: List[str],
                    val_threshold: float = 0.5,
                    device:        torch.device = None) -> SeedDriftResult:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trained_model.eval()
    result = SeedDriftResult(seed=seed, drift_type=drift_type)

    # ── Baseline (clean window, no drift) ─────────────────────────────────── #
    X_base_t           = torch.FloatTensor(X_test).to(device)
    rule_weights       = extract_rule_weights(trained_model)
    baseline_acts      = extract_rule_activations(trained_model, X_base_t)
    baseline_mean_acts = baseline_acts.mean(axis=0)
    baseline_bit_means = extract_bit_means(trained_model, X_base_t)
    baseline_rfr       = compute_rfr(baseline_acts)
    baseline_probs     = predict_proba(trained_model, X_base_t)
    baseline_metrics   = compute_standard_metrics(y_test, baseline_probs, val_threshold)

    result.baseline_f1 = baseline_metrics["f1"]
    f1_drop_threshold  = result.baseline_f1 - 0.03

    # ── History buffers for new metrics ───────────────────────────────────── #
    rwss_history       = []          # for velocity computation
    mean_acts_history  = []          # for rolling RWSS
    fidi_history       = []          # for FIDI z-score

    # ── Drift windows ─────────────────────────────────────────────────────── #
    config   = DriftConfig(drift_type=drift_type, n_windows=N_WINDOWS)
    injector = DriftInjector(
        X_test, y_test, config, feature_names,
        rng=np.random.default_rng(seed)
    )
    windows = injector.get_windows()

    for w_idx, (X_w, y_w) in enumerate(windows):
        X_w_t = torch.FloatTensor(X_w).to(device)

        # ── Symbolic layer computations ───────────────────────────────────── #
        current_acts      = extract_rule_activations(trained_model, X_w_t)
        current_mean_acts = current_acts.mean(axis=0)
        current_bit_means = extract_bit_means(trained_model, X_w_t)
        current_rfr       = compute_rfr(current_acts)
        current_probs     = predict_proba(trained_model, X_w_t)

        # ── Original RWSS (absolute) ──────────────────────────────────────── #
        alert_fired, rwss_score = rwss_alert(baseline_mean_acts, current_mean_acts)

        # ── RWSS Velocity (NEW) ───────────────────────────────────────────── #
        rwss_history.append(rwss_score)
        vel_fired, rwss_vel = rwss_velocity_alert(rwss_history)

        # ── Rolling RWSS (NEW) ────────────────────────────────────────────── #
        mean_acts_history.append(current_mean_acts)
        rolling_r = compute_rolling_rwss(mean_acts_history, lookback=2)

        # ── FIDI (absolute) ───────────────────────────────────────────────── #
        fidi = compute_fidi(
            rule_weights, baseline_bit_means, current_bit_means,
            n_features=len(feature_names), n_thresholds=3
        )

        # ── FIDI Z-Score (NEW) ────────────────────────────────────────────── #
        fidi_z      = compute_fidi_zscore(fidi_history, fidi, min_history=3)
        fidi_history.append(fidi)
        fz_fired, fz_top = fidi_zscore_alert(fidi_z, feature_names, top_k=5)

        # Feature indices
        v14i = feature_names.index("V14") if "V14" in feature_names else -1
        v12i = feature_names.index("V12") if "V12" in feature_names else -1
        v4i  = feature_names.index("V4")  if "V4"  in feature_names else -1

        fidi_z_v14 = fidi_z.get(v14i, 0.0)

        # ── PSI on Rule Activations (NEW) ─────────────────────────────────── #
        psi_r       = compute_psi_rules(baseline_acts, current_acts)
        pr_fired, pr_severity = psi_rules_alert(psi_r)

        # ── Standard metrics ──────────────────────────────────────────────── #
        std = compute_standard_metrics(y_w, current_probs, val_threshold)
        psi = compute_psi(baseline_probs, current_probs)

        wr = WindowResult(
            window     = w_idx,
            drift_type = drift_type,
            seed       = seed,
            # Original
            rwss            = rwss_score,
            rwss_alert      = alert_fired,
            fidi_v14        = fidi.get(v14i, 0.0),
            fidi_v12        = fidi.get(v12i, 0.0),
            fidi_v4         = fidi.get(v4i,  0.0),
            rfr_mean        = float(current_rfr.mean()),
            rfr_change      = float(rfr_delta(baseline_rfr, current_rfr).mean()),
            # New early-warning
            rwss_velocity      = rwss_vel,
            rwss_vel_alert     = vel_fired,
            rolling_rwss       = rolling_r,
            fidi_z_v14         = fidi_z_v14,
            fidi_z_alert       = fz_fired,
            psi_rules          = psi_r,
            psi_rules_alert    = pr_fired,
            psi_rules_severity = pr_severity,
            # Standard
            f1       = std["f1"],
            roc_auc  = std["roc_auc"],
            pr_auc   = std["pr_auc"],
            f1_alert = std["f1"] < f1_drop_threshold,
            psi      = psi,
        )
        result.windows.append(wr)

    # ── Compute alert windows for all metrics ─────────────────────────────── #
    def _first(windows, attr):
        wins = [w.window for w in windows if getattr(w, attr)]
        return wins[0] if wins else None

    result.rwss_alert_window     = _first(result.windows, "rwss_alert")
    result.rwss_vel_alert_window = _first(result.windows, "rwss_vel_alert")
    result.fidi_z_alert_window   = _first(result.windows, "fidi_z_alert")
    result.psi_r_alert_window    = _first(result.windows, "psi_rules_alert")
    result.f1_alert_window       = _first(result.windows, "f1_alert")

    # ── Detection lags: how many windows before F1 does each metric fire? ─── #
    def _lag(metric_window, f1_window):
        if metric_window is not None and f1_window is not None:
            return f1_window - metric_window
        return None

    result.detection_lag          = _lag(result.rwss_alert_window,     result.f1_alert_window)
    result.detection_lag_velocity = _lag(result.rwss_vel_alert_window, result.f1_alert_window)
    result.detection_lag_fidi_z   = _lag(result.fidi_z_alert_window,   result.f1_alert_window)
    result.detection_lag_psi_r    = _lag(result.psi_r_alert_window,    result.f1_alert_window)

    # Propagate lags to window records
    for w in result.windows:
        w.detection_lag          = result.detection_lag
        w.detection_lag_velocity = result.detection_lag_velocity
        w.detection_lag_fidi_z   = result.detection_lag_fidi_z
        w.detection_lag_psi_r    = result.detection_lag_psi_r

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════════════════

def _rate_and_mean_lag(all_results, drift_type, lag_attr, alert_window_attr):
    dtr      = [r for r in all_results if r.drift_type == drift_type]
    fired    = [r for r in dtr if getattr(r, alert_window_attr) is not None]
    lags     = [getattr(r, lag_attr) for r in dtr
                if getattr(r, lag_attr) is not None]
    mean_lag = round(float(np.mean(lags)), 2) if lags else None
    std_lag  = round(float(np.std(lags)),  2) if lags else None
    return len(fired), len(dtr), mean_lag, std_lag


def summarise_results(all_results: List[SeedDriftResult]) -> pd.DataFrame:
    rows = []
    for dt in DRIFT_TYPES:
        dtr   = [r for r in all_results if r.drift_type == dt]
        n     = len(dtr)

        def rate(attr): return sum(1 for r in dtr if getattr(r, attr) is not None)
        def lags(attr):
            ls = [getattr(r, attr) for r in dtr if getattr(r, attr) is not None]
            return ls

        lags_rwss  = lags("detection_lag")
        lags_vel   = lags("detection_lag_velocity")
        lags_fidiz = lags("detection_lag_fidi_z")
        lags_psir  = lags("detection_lag_psi_r")
        f1_fired   = sum(1 for r in dtr if r.f1_alert_window is not None)

        rows.append({
            "drift_type":           dt,
            "f1_fired_rate":        f"{f1_fired}/{n}",
            "rwss_fired_rate":      f"{rate('rwss_alert_window')}/{n}",
            "rwss_vel_fired_rate":  f"{rate('rwss_vel_alert_window')}/{n}",
            "fidi_z_fired_rate":    f"{rate('fidi_z_alert_window')}/{n}",
            "psi_r_fired_rate":     f"{rate('psi_r_alert_window')}/{n}",
            "dual_alert_rate":      f"{len(lags_rwss)}/{n}",
            "rwss_mean_lag":        round(float(np.mean(lags_rwss)),  2) if lags_rwss  else None,
            "vel_mean_lag":         round(float(np.mean(lags_vel)),   2) if lags_vel   else None,
            "fidi_z_mean_lag":      round(float(np.mean(lags_fidiz)), 2) if lags_fidiz else None,
            "psi_r_mean_lag":       round(float(np.mean(lags_psir)),  2) if lags_psir  else None,
        })
    return pd.DataFrame(rows)
