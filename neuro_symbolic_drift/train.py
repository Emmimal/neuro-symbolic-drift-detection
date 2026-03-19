"""
train.py
--------
Training loop for HybridRuleLearner.

Three-part loss (Article 2 exact values):
    L = L_BCE + 0.30·L_consistency + 0.25·L_sparsity + 0.01·L_confidence

Temperature annealing:
    τ_start=5.0 → τ_end=0.1 over 80 epochs (exponential)
    Early stopping on val PR-AUC, patience=20, min_epochs=20
    MIN_EPOCHS guard prevents stopping before τ drops below ~1.5,
    giving the rule layer enough time to begin crystallising.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing  import Dict, Optional
from sklearn.metrics import f1_score, average_precision_score

LAMBDA_C     = 0.30
LAMBDA_S     = 0.25
LAMBDA_CONF  = 0.01
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 80
# BUG 6 FIX — PATIENCE raised from 10 to 20.
# All 5 seeds previously stopped at epochs 3–10, when τ was still 3.5–4.5
# (rules barely crystallised). A minimum epoch guard (MIN_EPOCHS) ensures
# the rule layer has reached τ < 1.0 before early stopping is allowed.
# At epoch 20, τ = 5.0 * (0.1/5.0)^(20/79) ≈ 1.52 — rules are tightening.
PATIENCE     = 20
MIN_EPOCHS   = 20   # early stopping not allowed before this epoch
TAU_START    = 5.0
TAU_END      = 0.1


def get_temperature(epoch: int, total: int = MAX_EPOCHS) -> float:
    progress = epoch / max(total - 1, 1)
    return TAU_START * (TAU_END / TAU_START) ** progress


def _compute_loss(final_prob, mlp_prob, rule_prob, y_batch,
                  rule_weights_raw, rule_confidence, pos_w):
    bce = F.binary_cross_entropy(
        final_prob, y_batch,
        weight=torch.where(y_batch == 1, pos_w, torch.ones_like(y_batch))
    )
    with torch.no_grad():
        mask = (mlp_prob > 0.7) | (mlp_prob < 0.3)
        mask = mask.squeeze()
    consistency = (F.mse_loss(rule_prob.squeeze()[mask],
                              mlp_prob.squeeze()[mask].detach())
                   if mask.sum() > 0
                   else torch.tensor(0.0, device=final_prob.device))
    sparsity    = rule_weights_raw.abs().mean()
    conf_pen    = rule_confidence.abs().mean()
    total = (bce
             + LAMBDA_C    * consistency
             + LAMBDA_S    * sparsity
             + LAMBDA_CONF * conf_pen)
    return total, bce, consistency


def find_best_threshold(y_true, y_prob) -> float:
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 100):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def train_model(model, train_loader, val_loader,
                pos_weight: float,
                device: torch.device = None,
                checkpoint_path: Optional[Path] = None,
                verbose: bool = True) -> Dict:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model      = model.to(device)
    optimizer  = torch.optim.Adam(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    pos_w      = torch.tensor(pos_weight, device=device)

    best_pr    = 0.0
    best_epoch = 0
    patience_c = 0
    val_thresh = 0.5
    val_f1     = 0.0
    history    = {"train_loss": [], "val_pr_auc": [], "val_f1": []}

    for epoch in range(MAX_EPOCHS):
        tau = get_temperature(epoch)
        model.train()
        epoch_loss = 0.0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            final, mlp, rule, _ = model(X_b, tau)
            loss, *_ = _compute_loss(
                final, mlp, rule, y_b,
                model.rule_learner.rule_weights,
                model.rule_learner.rule_confidence,
                pos_w,
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                final, *_ = model(X_b.to(device), tau)
                all_probs.extend(final.squeeze().cpu().numpy())
                all_labels.extend(y_b.squeeze().numpy())

        y_arr  = np.array(all_labels)
        p_arr  = np.array(all_probs)
        pr_auc = average_precision_score(y_arr, p_arr)
        thresh = find_best_threshold(y_arr, p_arr)
        f1     = f1_score(y_arr, (p_arr >= thresh).astype(int), zero_division=0)

        history["train_loss"].append(epoch_loss / len(train_loader))
        history["val_pr_auc"].append(pr_auc)
        history["val_f1"].append(f1)

        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  τ={tau:.2f}  "
                  f"loss={epoch_loss/len(train_loader):.4f}  "
                  f"PR-AUC={pr_auc:.4f}  F1={f1:.4f}  α={model.get_alpha():.3f}")

        if pr_auc > best_pr:
            best_pr, best_epoch, patience_c = pr_auc, epoch, 0
            val_thresh, val_f1 = thresh, f1
            if checkpoint_path:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch":         epoch,
                    "model_state":   model.state_dict(),
                    "val_pr_auc":    pr_auc,
                    "val_threshold": thresh,
                    "val_f1":        f1,
                    "alpha":         model.get_alpha(),
                }, checkpoint_path)
        else:
            patience_c += 1
            # BUG 6 FIX — guard: do not allow early stopping before MIN_EPOCHS.
            # Previously all seeds stopped at epochs 3–10 (τ still 3.5–4.5),
            # before the rule layer had meaningful time to crystallise.
            if patience_c >= PATIENCE and epoch >= MIN_EPOCHS:
                if verbose:
                    print(f"  Early stop @ epoch {epoch}  "
                          f"(best PR-AUC={best_pr:.4f} @ epoch {best_epoch})")
                break

    return {
        "best_val_pr_auc": best_pr,
        "best_epoch":      best_epoch,
        "val_threshold":   val_thresh,
        "val_f1":          val_f1,
        "alpha":           model.get_alpha(),
        "history":         history,
    }


def load_checkpoint(path: Path, model, device=None) -> Dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return ckpt
