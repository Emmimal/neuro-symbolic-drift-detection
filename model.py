"""
model.py
--------
HybridRuleLearner — complete neuro-symbolic architecture.
No external dependencies on previous articles.

Architecture:
    Input (30 features)
        ├── MLP path  → mlp_prob
        └── Rule path → rule_prob
                        (LearnableDiscretizer → RuleLearner)
    final_prob = α · mlp_prob + (1-α) · rule_prob

α is a *learnable* scalar — not a hyperparameter.
After training converges to ~0.88 (MLP dominant, rules explain).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableDiscretizer(nn.Module):
    """
    Converts continuous features into soft binary features.

    For each feature f and learnable threshold θ_{f,t}:
        b_{f,t} = σ( (x_f − θ_{f,t}) / τ )

    At high τ (early training): sigmoid is near-flat — gradients flow freely.
    At low τ  (late training):  sigmoid is near-step — rules become crisp.

    Output: [batch, n_features * n_thresholds]
    """

    def __init__(self, n_features: int, n_thresholds: int = 3):
        super().__init__()
        self.n_features   = n_features
        self.n_thresholds = n_thresholds
        self.thresholds   = nn.Parameter(
            torch.randn(n_features, n_thresholds) * 0.5
        )

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        x_exp     = x.unsqueeze(-1)              # [B, F, 1]
        t_exp     = self.thresholds.unsqueeze(0) # [1, F, T]
        soft_bits = torch.sigmoid((x_exp - t_exp) / temperature)
        return soft_bits.view(x.size(0), -1)     # [B, F*T]


class RuleLearner(nn.Module):
    """
    Learns soft IF-THEN rules as weighted combinations of binarised features.

    Each rule r:
        rule_r(x) = σ( Σ_i w_{r,i} · b_i / τ )

    After tanh squashing, weight interpretation:
        w > +0.5  →  feature must be HIGH  (IF V14 > threshold)
        w < −0.5  →  feature must be LOW   (IF V14 < threshold)
        |w| < 0.5 →  feature irrelevant to this rule

    Outputs:
        fraud_prob : [B, 1]  weighted combination of rule activations
        rule_acts  : [B, R]  per-rule activation (used by FIDI and RFR)
    """

    def __init__(self, n_bits: int, n_rules: int = 4):
        super().__init__()
        self.n_rules         = n_rules
        self.rule_weights    = nn.Parameter(torch.randn(n_rules, n_bits) * 0.1)
        self.rule_confidence = nn.Parameter(torch.ones(n_rules))

    def forward(self, bits: torch.Tensor, temperature: float = 1.0):
        w          = torch.tanh(self.rule_weights)
        logits     = bits @ w.T                              # [B, R]
        rule_acts  = torch.sigmoid(logits / temperature)    # [B, R]
        conf       = torch.softmax(self.rule_confidence, dim=0)
        fraud_prob = (rule_acts * conf.unsqueeze(0)).sum(dim=1, keepdim=True)
        return fraud_prob, rule_acts


class MLP(nn.Module):
    """Three-layer MLP with batch normalisation and dropout."""

    def __init__(self, n_features: int,
                 hidden_dims=(64, 32, 16),
                 dropout: float = 0.3):
        super().__init__()
        dims   = [n_features] + list(hidden_dims)
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [
                nn.Linear(in_d, out_d),
                nn.BatchNorm1d(out_d),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class HybridRuleLearner(nn.Module):
    """
    Full hybrid model combining MLP detection with differentiable rule learning.

    Args:
        n_features   : input feature count (30 for Kaggle CC fraud)
        n_thresholds : learnable threshold cuts per feature (default 3)
        n_rules      : number of IF-THEN rules to learn (default 4)
        hidden_dims  : MLP hidden layer sizes

    Forward returns:
        final_prob  [B, 1] : α·mlp + (1-α)·rule  — use this for predictions
        mlp_prob    [B, 1] : MLP path output
        rule_prob   [B, 1] : Rule path output
        rule_acts   [B, R] : Per-rule activation scores
    """

    def __init__(self, n_features: int = 30,
                 n_thresholds: int = 3,
                 n_rules: int = 4,
                 hidden_dims=(64, 32, 16)):
        super().__init__()
        n_bits            = n_features * n_thresholds
        self.discretizer  = LearnableDiscretizer(n_features, n_thresholds)
        self.rule_learner = RuleLearner(n_bits, n_rules)
        self.mlp          = MLP(n_features, hidden_dims)
        self.alpha        = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, temperature: float = 1.0):
        mlp_prob               = self.mlp(x)
        bits                   = self.discretizer(x, temperature)
        rule_prob, rule_acts   = self.rule_learner(bits, temperature)
        alpha_s                = torch.sigmoid(self.alpha)
        final_prob             = alpha_s * mlp_prob + (1.0 - alpha_s) * rule_prob
        return final_prob, mlp_prob, rule_prob, rule_acts

    def get_alpha(self) -> float:
        return float(torch.sigmoid(self.alpha).item())

    def extract_rules(self, feature_names, weight_threshold=0.50,
                      n_thresholds=3):
        """
        Extract human-readable IF-THEN rules from trained weights.
        Returns list of dicts, one per active rule.
        """
        with torch.no_grad():
            w    = torch.tanh(self.rule_weights).cpu().numpy()
            conf = torch.softmax(self.rule_learner.rule_confidence,
                                 dim=0).cpu().numpy()
            thresholds = self.discretizer.thresholds.cpu().numpy()

        rules = []
        for r_idx in range(self.rule_learner.n_rules):
            conditions = []
            for feat_idx, feat_name in enumerate(feature_names):
                for t_idx in range(n_thresholds):
                    bit_idx = feat_idx * n_thresholds + t_idx
                    w_val   = w[r_idx, bit_idx]
                    thresh  = thresholds[feat_idx, t_idx]
                    if abs(w_val) > weight_threshold:
                        direction = ">" if w_val > 0 else "<"
                        conditions.append({
                            "feature":    feat_name,
                            "direction":  direction,
                            "threshold":  round(float(thresh), 3),
                            "weight":     round(float(w_val), 3),
                        })
            if conditions:
                rules.append({
                    "rule_id":    r_idx,
                    "confidence": round(float(conf[r_idx]), 3),
                    "conditions": conditions,
                })
        return rules
