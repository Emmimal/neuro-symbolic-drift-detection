"""
alert_system.py
---------------
Production-ready DriftAlertSystem with layered early-warning metrics.

Build once after training. Call .check() after every inference batch.
No labels required — reads the symbolic layer only.

Detection layers (earliest to latest):
    Layer 1  RWSS velocity    rate-of-change alert
    Layer 2  FIDI z-score     per-feature anomaly vs history
    Layer 3  PSI_rules        symbolic-layer distributional shift
    Layer 4  RWSS absolute    confirmed cosine-sim drop
    Ref      F1 drop          label-based (not available here)
"""

import pickle
import numpy as np
import torch
from dataclasses import dataclass, field
from typing      import List, Dict, Optional, Tuple
from pathlib     import Path

from .drift_metrics import (
    extract_rule_weights, extract_rule_activations,
    extract_bit_means,
    compute_rwss, rwss_alert,
    compute_rwss_velocity, rwss_velocity_alert,
    compute_rolling_rwss,
    compute_fidi, fidi_top_features,
    compute_fidi_zscore, fidi_zscore_alert,
    compute_rfr, rfr_delta,
    compute_psi_rules, psi_rules_alert,
    RWSS_ALERT_THRESHOLD,
    RWSS_VEL_THRESHOLD,
    FIDI_Z_THRESHOLD,
    PSI_RULES_MODERATE,
    PSI_RULES_SIGNIFICANT,
    RFR_ALERT_THRESHOLD,
)


@dataclass
class DriftAlert:
    fired:              bool
    severity:           str           # "none" | "warning" | "critical"
    # Original metrics
    rwss_score:         float
    rwss_fired:         bool
    top_drifted_feats:  List[Tuple]
    rfr_fired:          bool
    rfr_delta_mean:     float
    n_rules_silent:     int
    # New early-warning metrics
    rwss_velocity:      float
    rwss_vel_fired:     bool
    fidi_z_fired:       bool
    fidi_z_top_feats:   List[Tuple]   # (feature_name, z_score)
    psi_rules:          float
    psi_rules_fired:    bool
    psi_rules_severity: str
    # Earliest firing layer
    earliest_layer:     str           # "velocity"|"fidi_z"|"psi_rules"|"rwss"|"none"
    metadata:           Dict = field(default_factory=dict)

    def report(self) -> str:
        lines = [
            "═══════════════════════════════════════════════════════",
            f"  DRIFT ALERT  |  severity: {self.severity.upper()}",
            f"  Earliest signal: {self.earliest_layer.upper()}",
            "═══════════════════════════════════════════════════════",
            "",
            "  ── Early-Warning Layer ─────────────────────────────",
            f"  RWSS Velocity : {self.rwss_velocity:+.4f}  "
            f"[threshold {RWSS_VEL_THRESHOLD}]  "
            f"{'⚠ FIRED' if self.rwss_vel_fired else 'OK'}",
            "",
            f"  FIDI Z-Score  : {'⚠ FIRED' if self.fidi_z_fired else 'OK'}",
        ]
        if self.fidi_z_top_feats:
            for feat, z in self.fidi_z_top_feats[:3]:
                lines.append(f"    {feat:>6}  Z = {z:+.2f}")
        lines += [
            "",
            f"  PSI (rules)   : {self.psi_rules:.4f}  "
            f"[moderate≥{PSI_RULES_MODERATE}, "
            f"significant≥{PSI_RULES_SIGNIFICANT}]  "
            f"{'⚠ ' + self.psi_rules_severity.upper() if self.psi_rules_fired else 'stable'}",
            "",
            "  ── Confirmed Layer ─────────────────────────────────",
            f"  RWSS absolute : {self.rwss_score:.4f}  "
            f"[threshold {RWSS_ALERT_THRESHOLD}]  "
            f"{'⚠ FIRED' if self.rwss_fired else 'OK'}",
            "",
            "  Top drifting features (FIDI absolute):",
        ]
        for feat, score in self.top_drifted_feats:
            direction = "collapsed ↓" if score > 0 else "amplified ↑"
            lines.append(f"    {feat:>6}  Δ = {score:+.3f}  ({direction})")
        lines += [
            "",
            f"  Rules gone silent: {self.n_rules_silent}  "
            f"{'⚠ FIRED' if self.rfr_fired else 'OK'}",
            f"  Mean RFR change  : {self.rfr_delta_mean:+.3f}",
            "",
            "  Recommended action:",
            ("    → Retrain immediately. Do not deploy."
             if self.severity == "critical"
             else "    → Schedule retrain within 48 h."
             if self.severity == "warning"
             else "    → No action required."),
            "═══════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "fired":              self.fired,
            "severity":           self.severity,
            "earliest_layer":     self.earliest_layer,
            "rwss_score":         round(self.rwss_score, 4),
            "rwss_fired":         self.rwss_fired,
            "rwss_velocity":      round(self.rwss_velocity, 4),
            "rwss_vel_fired":     self.rwss_vel_fired,
            "fidi_z_fired":       self.fidi_z_fired,
            "fidi_z_top_feats":   self.fidi_z_top_feats,
            "psi_rules":          round(self.psi_rules, 4),
            "psi_rules_fired":    self.psi_rules_fired,
            "psi_rules_severity": self.psi_rules_severity,
            "top_drifted_feats":  self.top_drifted_feats,
            "rfr_fired":          self.rfr_fired,
            "rfr_delta_mean":     round(self.rfr_delta_mean, 4),
            "n_rules_silent":     self.n_rules_silent,
            **self.metadata,
        }


class DriftAlertSystem:
    """
    Build once after training. Save. Load. Call .check() every retrain cycle.

        system = DriftAlertSystem.from_trained_model(model, X_val_t, feature_names)
        system.save("results/drift_baseline.pkl")

        # weekly
        system = DriftAlertSystem.load("results/drift_baseline.pkl")
        alert  = system.check(model, X_this_week)
        if alert.fired:
            print(alert.report())

    Layered detection — earliest to latest:
        velocity  ->  fidi_z  ->  psi_rules  ->  rwss  ->  (F1 label)
    """

    def __init__(self,
                 baseline_mean_acts: np.ndarray,
                 baseline_acts:      np.ndarray,
                 baseline_bit_means: np.ndarray,
                 baseline_rfr:       np.ndarray,
                 rule_weights:       np.ndarray,
                 feature_names:      List[str],
                 n_thresholds:       int = 3):
        self.baseline_mean_acts = baseline_mean_acts
        self.baseline_acts      = baseline_acts
        self.baseline_bit_means = baseline_bit_means
        self.baseline_rfr       = baseline_rfr
        self.rule_weights       = rule_weights
        self.feature_names      = feature_names
        self.n_thresholds       = n_thresholds
        self.n_features         = len(feature_names)
        # History buffers for stateful metrics
        self._rwss_history      = []
        self._mean_acts_history = []
        self._fidi_history      = []
        self.alert_history: List[DriftAlert] = []

    @classmethod
    def from_trained_model(cls, model, X_baseline: torch.Tensor,
                           feature_names: List[str],
                           temperature: float = 0.1) -> "DriftAlertSystem":
        acts      = extract_rule_activations(model, X_baseline, temperature)
        bit_means = extract_bit_means(model, X_baseline, temperature)
        rfr       = compute_rfr(acts)
        weights   = extract_rule_weights(model)
        return cls(acts.mean(axis=0), acts, bit_means, rfr, weights, feature_names)

    @classmethod
    def load(cls, path) -> "DriftAlertSystem":
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def check(self, model, X_current: torch.Tensor,
              temperature: float = 0.1,
              metadata: dict = None) -> DriftAlert:

        current_acts      = extract_rule_activations(model, X_current, temperature)
        current_mean_acts = current_acts.mean(axis=0)
        current_bit_means = extract_bit_means(model, X_current, temperature)
        current_rfr       = compute_rfr(current_acts)

        # ── Layer 4: RWSS absolute ────────────────────────────────────────── #
        rwss_score = compute_rwss(self.baseline_mean_acts, current_mean_acts)
        rwss_fired = rwss_score < RWSS_ALERT_THRESHOLD

        # ── Layer 1: RWSS velocity ────────────────────────────────────────── #
        self._rwss_history.append(rwss_score)
        vel_fired, rwss_vel = rwss_velocity_alert(self._rwss_history)

        # Rolling RWSS (diagnostic)
        self._mean_acts_history.append(current_mean_acts)
        rolling_r = compute_rolling_rwss(self._mean_acts_history, lookback=2)

        # ── FIDI absolute (for report) ────────────────────────────────────── #
        fidi = compute_fidi(self.rule_weights,
                            self.baseline_bit_means, current_bit_means,
                            self.n_features, self.n_thresholds)
        from .drift_metrics import FIDI_ALERT_THRESHOLD
        top_feats = [(f, s) for f, s in
                     fidi_top_features(fidi, self.feature_names, top_k=5)
                     if abs(s) > FIDI_ALERT_THRESHOLD]

        # ── Layer 2: FIDI z-score ─────────────────────────────────────────── #
        fidi_z   = compute_fidi_zscore(self._fidi_history, fidi, min_history=3)
        self._fidi_history.append(fidi)
        fz_fired, fz_top = fidi_zscore_alert(fidi_z, self.feature_names, top_k=5)

        # ── Layer 3: PSI on rule activations ──────────────────────────────── #
        psi_r        = compute_psi_rules(self.baseline_acts, current_acts)
        pr_fired, pr_severity = psi_rules_alert(psi_r)

        # ── RFR ───────────────────────────────────────────────────────────── #
        rfr_change     = rfr_delta(self.baseline_rfr, current_rfr)
        n_silent       = int((rfr_change < -RFR_ALERT_THRESHOLD).sum())
        rfr_fired      = n_silent > 0
        rfr_delta_mean = float(rfr_change.mean())

        # ── Severity and earliest layer ───────────────────────────────────── #
        n_fired = sum([rwss_fired, rfr_fired, len(top_feats) > 0,
                       vel_fired, fz_fired, pr_fired])

        severity = ("critical" if rwss_score < 0.70 or n_fired >= 3
                    else "warning" if n_fired >= 1
                    else "none")

        if vel_fired:
            earliest = "velocity"
        elif fz_fired:
            earliest = "fidi_z"
        elif pr_fired:
            earliest = "psi_rules"
        elif rwss_fired:
            earliest = "rwss"
        else:
            earliest = "none"

        alert = DriftAlert(
            fired              = severity != "none",
            severity           = severity,
            rwss_score         = rwss_score,
            rwss_fired         = rwss_fired,
            top_drifted_feats  = top_feats,
            rfr_fired          = rfr_fired,
            rfr_delta_mean     = rfr_delta_mean,
            n_rules_silent     = n_silent,
            rwss_velocity      = rwss_vel,
            rwss_vel_fired     = vel_fired,
            fidi_z_fired       = fz_fired,
            fidi_z_top_feats   = fz_top,
            psi_rules          = psi_r,
            psi_rules_fired    = pr_fired,
            psi_rules_severity = pr_severity,
            earliest_layer     = earliest,
            metadata           = metadata or {},
        )
        self.alert_history.append(alert)
        return alert

    def timeline(self) -> List[dict]:
        return [a.to_dict() for a in self.alert_history]
