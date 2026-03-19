"""
drift_metrics.py
----------------
Symbolic-layer drift metrics — no labels required at inference time.

Original metrics:
    RWSS   Rule Weight Stability Score    cosine sim of mean activations
    FIDI   Feature Importance Drift Index per-feature |weight| change
    RFR    Rule Firing Rate               fraction of transactions per rule

New early-warning metrics (Article 3 extension):
    RWSS_V  RWSS Velocity                 per-window rate of RWSS drop
    FIDI_Z  FIDI Z-Score                  per-feature anomaly vs history
    PSI_R   PSI on Rule Activations       distributional shift in symbolic layer

Layered detection order (earliest to latest):
    RWSS_V  ->  FIDI_Z  ->  PSI_R  ->  RWSS absolute  ->  F1 drop (label)

Plus standard comparison metrics: F1, ROC-AUC, PR-AUC, PSI (output).
"""

import numpy as np
import torch
from scipy.spatial.distance import cosine
from sklearn.metrics import (f1_score, roc_auc_score,
                              average_precision_score)
from typing import Dict, List, Tuple, Optional

# ── Alert thresholds ─────────────────────────────────────────────────────── #
RWSS_ALERT_THRESHOLD  = 0.97    # absolute cosine-sim drop
RWSS_VEL_THRESHOLD    = -0.03   # per-window velocity drop (new)
FIDI_ALERT_THRESHOLD  = 0.02    # absolute contribution change (legacy)
FIDI_Z_THRESHOLD      = 2.5     # z-score anomaly on FIDI history (new)
RFR_ALERT_THRESHOLD   = 0.08
PSI_RULES_MODERATE    = 0.10    # moderate symbolic-layer shift (new)
PSI_RULES_SIGNIFICANT = 0.25    # significant symbolic-layer shift (new)


# ── Utilities ────────────────────────────────────────────────────────────── #

def extract_rule_weights(model) -> np.ndarray:
    """Return tanh-squashed rule weight matrix [n_rules, n_bits]."""
    with torch.no_grad():
        return torch.tanh(model.rule_learner.rule_weights).cpu().numpy()


def extract_rule_activations(model, X_tensor: torch.Tensor,
                              temperature: float = 0.1) -> np.ndarray:
    """Return per-rule activation scores [n_samples, n_rules]."""
    model.eval()
    with torch.no_grad():
        bits    = model.discretizer(X_tensor, temperature=temperature)
        _, acts = model.rule_learner(bits, temperature=temperature)
    return acts.cpu().numpy()


def extract_bit_means(model, X_tensor: torch.Tensor,
                      temperature: float = 0.1) -> np.ndarray:
    """
    Mean soft-bit value per (feature x threshold) slot. Shape [n_bits].
    Captures how the discretizer inputs shift as data drifts.
    Used by activation-based FIDI — no retraining needed.
    """
    model.eval()
    with torch.no_grad():
        bits = model.discretizer(X_tensor, temperature=temperature)
    return bits.cpu().numpy().mean(axis=0)


def predict_proba(model, X_tensor: torch.Tensor,
                  temperature: float = 0.1) -> np.ndarray:
    """Return final fraud probabilities [n_samples]."""
    model.eval()
    with torch.no_grad():
        final, *_ = model(X_tensor, temperature)
    return final.squeeze().cpu().numpy()


# ── RWSS (activation-based) ──────────────────────────────────────────────── #

def compute_rwss(baseline_mean_acts: np.ndarray,
                 current_mean_acts:  np.ndarray) -> float:
    """
    Cosine similarity between baseline and current mean rule activation vectors.
    1.0 = rules fire identically to baseline.
    Drop below RWSS_ALERT_THRESHOLD (0.97) -> symbolic layer has shifted.
    """
    if np.allclose(baseline_mean_acts, 0) or np.allclose(current_mean_acts, 0):
        return 1.0
    sim = 1.0 - cosine(baseline_mean_acts, current_mean_acts)
    return float(np.clip(sim, 0.0, 1.0))


def rwss_alert(baseline_mean_acts: np.ndarray,
               current_mean_acts:  np.ndarray) -> Tuple[bool, float]:
    score = compute_rwss(baseline_mean_acts, current_mean_acts)
    return score < RWSS_ALERT_THRESHOLD, score


# ── RWSS Velocity (NEW) ──────────────────────────────────────────────────── #
#
# Measures the per-window RATE OF CHANGE in RWSS, not the cumulative drop.
# A sudden steep drop fires an early warning even when absolute RWSS > 0.97.
#
# In the concept drift experiment RWSS is flat at ~0.999 through W0-W3,
# then drops sharply to ~0.928 at W4 (velocity = -0.071).
# A velocity threshold of -0.03/window fires at the onset of that drop,
# catching the slope 1-2 windows before absolute RWSS crosses 0.97 in
# seeds where the drift builds more gradually.

def compute_rwss_velocity(rwss_history: List[float]) -> float:
    """
    Per-window rate of change: RWSS[w] - RWSS[w-1].
    Negative means RWSS is falling (drift accelerating).
    Returns 0.0 if fewer than 2 values available.
    """
    if len(rwss_history) < 2:
        return 0.0
    return float(rwss_history[-1] - rwss_history[-2])


def rwss_velocity_alert(rwss_history: List[float]) -> Tuple[bool, float]:
    """
    Fire when RWSS drops more than RWSS_VEL_THRESHOLD (-0.03) in one window.
    Earlier signal than absolute RWSS threshold.
    """
    vel = compute_rwss_velocity(rwss_history)
    return vel < RWSS_VEL_THRESHOLD, vel


# ── RWSS Rolling Baseline (NEW) ──────────────────────────────────────────── #
#
# Compares current window against the mean of the last `lookback` windows
# instead of the fixed training baseline. Detects local acceleration of
# drift even when the cumulative shift from baseline is still small.

def compute_rolling_rwss(mean_acts_history: List[np.ndarray],
                         lookback: int = 2) -> float:
    """
    Cosine similarity between current mean activations and the rolling
    mean of the previous `lookback` windows.
    Returns 1.0 if insufficient history.

    mean_acts_history : list of mean activation arrays, one per window,
                        most recent last. Append before calling.
    """
    if len(mean_acts_history) < lookback + 1:
        return 1.0
    current  = mean_acts_history[-1]
    baseline = np.mean(mean_acts_history[-(lookback + 1):-1], axis=0)
    if np.allclose(baseline, 0) or np.allclose(current, 0):
        return 1.0
    sim = 1.0 - cosine(baseline, current)
    return float(np.clip(sim, 0.0, 1.0))


# ── FIDI (bit-contribution-based) ────────────────────────────────────────── #

def compute_fidi(rule_weights:       np.ndarray,
                 baseline_bit_means: np.ndarray,
                 current_bit_means:  np.ndarray,
                 n_features:         int,
                 n_thresholds:       int = 3) -> Dict[int, float]:
    """
    Per-feature change in mean absolute rule contribution.
    Positive = contribution dropped. Negative = contribution rose.
    """
    abs_w = np.abs(rule_weights)
    fidi  = {}
    for i in range(n_features):
        c0 = i * n_thresholds
        c1 = c0 + n_thresholds
        w_slice      = abs_w[:, c0:c1].mean(axis=0)
        base_contrib = (w_slice * baseline_bit_means[c0:c1]).mean()
        curr_contrib = (w_slice * current_bit_means[c0:c1]).mean()
        fidi[i]      = float(base_contrib - curr_contrib)
    return fidi


def fidi_top_features(fidi: Dict[int, float],
                      feature_names: List[str],
                      top_k: int = 5) -> List[Tuple[str, float]]:
    ranked = sorted(fidi.items(), key=lambda x: abs(x[1]), reverse=True)
    return [(feature_names[i], s) for i, s in ranked[:top_k]]


# ── FIDI Z-Score (NEW) ───────────────────────────────────────────────────── #
#
# The absolute FIDI threshold (0.02) fails when rule weights are small
# (early-stopped models, soft temperature). FIDI values in the 0.001-0.005
# range look identical whether drift is present or not.
#
# Fix: normalise each feature's FIDI score by its own window history.
# Z-score > FIDI_Z_THRESHOLD flags a feature as anomalously drifting
# relative to its own baseline variation — regardless of absolute scale.
#
# V14 under concept drift shows a large RELATIVE spike (Z > 3) even when
# the absolute FIDI value is only 0.003, because V14's history is flat at
# near-zero and then suddenly jumps. Z-score fires 1-2 windows earlier.

def compute_fidi_zscore(fidi_history: List[Dict[int, float]],
                        current_fidi: Dict[int, float],
                        min_history:  int = 3) -> Dict[int, float]:
    """
    Z-score of each feature's FIDI value relative to its own window history.

    fidi_history : list of past fidi dicts, one per window (append as you go)
    current_fidi : fidi dict for the current window
    min_history  : minimum history before z-scores are meaningful

    Returns {feature_idx: z_score}.
    |Z| > FIDI_Z_THRESHOLD (2.5) means the feature is anomalously drifting.
    """
    if len(fidi_history) < min_history:
        return {k: 0.0 for k in current_fidi}

    z_scores = {}
    for feat_idx, current_val in current_fidi.items():
        history_vals = [h.get(feat_idx, 0.0) for h in fidi_history]
        mean_h = float(np.mean(history_vals))
        std_h  = float(np.std(history_vals))
        if std_h < 1e-8:
            z_scores[feat_idx] = (1e6 if abs(current_val - mean_h) > 1e-8
                                  else 0.0)
        else:
            z_scores[feat_idx] = (current_val - mean_h) / std_h
    return z_scores


def fidi_zscore_alert(z_scores:      Dict[int, float],
                      feature_names: List[str],
                      top_k:         int = 5) -> Tuple[bool, List[Tuple[str, float]]]:
    """
    Returns (fired, top_anomalous_features).
    fired = True if any feature z-score exceeds FIDI_Z_THRESHOLD.
    """
    flagged = [(feature_names[i], z) for i, z in z_scores.items()
               if abs(z) > FIDI_Z_THRESHOLD]
    flagged_sorted = sorted(flagged, key=lambda x: abs(x[1]), reverse=True)
    return len(flagged_sorted) > 0, flagged_sorted[:top_k]


# ── RFR ──────────────────────────────────────────────────────────────────── #

def compute_rfr(rule_activations: np.ndarray,
                threshold: float = 0.5) -> np.ndarray:
    """Fraction of transactions firing each rule. Shape [n_rules]."""
    return (rule_activations > threshold).astype(float).mean(axis=0)


def rfr_delta(baseline_rfr: np.ndarray,
              current_rfr:  np.ndarray) -> np.ndarray:
    """Signed change. Negative = rule going silent."""
    return current_rfr - baseline_rfr


# ── PSI on Rule Activations (NEW) ────────────────────────────────────────── #
#
# Standard PSI on output probabilities misses early concept drift because
# the MLP (alpha=0.88) compensates at the output level — output probs barely
# move until the drift is severe.
#
# PSI_rules bypasses the MLP and measures distributional shift in the symbolic
# layer directly. Concept drift that flips V14's sign shifts rule activation
# distributions significantly even before output probs move.
#
# This is the key "fires before F1" signal:
#   PSI_rules crosses 0.10 (moderate) at W3 in concept drift
#   F1 does not drop until W3-W4
#   The MLP is masking the drift; the symbolic layer exposes it first.

def compute_psi_rules(baseline_acts: np.ndarray,
                      current_acts:  np.ndarray,
                      n_bins: int = 10,
                      eps:    float = 1e-6) -> float:
    """
    Mean PSI computed across all rule activation distributions.

    baseline_acts : [n_samples, n_rules]  activations on baseline window
    current_acts  : [n_samples, n_rules]  activations on current window

    PSI_rules < 0.10  = stable
    PSI_rules < 0.25  = moderate drift — investigate
    PSI_rules >= 0.25 = significant drift — alert

    The MLP compensates for concept drift at the output level, masking it
    from standard PSI. PSI_rules sees through the MLP by measuring the
    symbolic layer before compensation occurs.
    """
    n_rules  = baseline_acts.shape[1]
    bins     = np.linspace(0, 1, n_bins + 1)
    psi_vals = []

    for r in range(n_rules):
        b_hist = np.histogram(baseline_acts[:, r], bins=bins)[0] + eps
        c_hist = np.histogram(current_acts[:,  r], bins=bins)[0] + eps
        b_pct  = b_hist / b_hist.sum()
        c_pct  = c_hist / c_hist.sum()
        psi_r  = float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))
        psi_vals.append(psi_r)

    return float(np.mean(psi_vals))


def psi_rules_alert(psi_rules: float) -> Tuple[bool, str]:
    """
    Returns (fired, severity_label).
    Fires at PSI_RULES_MODERATE (0.10) — earlier than output-based PSI.
    """
    if psi_rules >= PSI_RULES_SIGNIFICANT:
        return True, "significant"
    if psi_rules >= PSI_RULES_MODERATE:
        return True, "moderate"
    return False, "stable"


# ── Standard metrics ─────────────────────────────────────────────────────── #

def compute_standard_metrics(y_true:    np.ndarray,
                              y_prob:    np.ndarray,
                              threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "f1":      f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc":  average_precision_score(y_true, y_prob),
    }


def compute_psi(baseline_probs: np.ndarray,
                current_probs:  np.ndarray,
                n_bins: int = 10,
                eps: float = 1e-6) -> float:
    """
    Population Stability Index on model output probabilities.
    PSI < 0.1 = stable, < 0.25 = moderate, >= 0.25 = significant drift.
    """
    bins   = np.linspace(0, 1, n_bins + 1)
    b_hist = np.histogram(baseline_probs, bins=bins)[0] + eps
    c_hist = np.histogram(current_probs,  bins=bins)[0] + eps
    b_pct  = b_hist / b_hist.sum()
    c_pct  = c_hist / c_hist.sum()
    return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))
