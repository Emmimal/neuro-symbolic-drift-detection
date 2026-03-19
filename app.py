"""
app.py
------
Single entry point for the Article 3 experiment.

Run this file for everything:

    python app.py --step all          # full pipeline (train → experiment → figures)
    python app.py --step train        # train model across all seeds
    python app.py --step experiment   # run drift simulation
    python app.py --step figures      # generate all article figures
    python app.py --step alert        # demo the DriftAlertSystem

    python app.py --step all --seeds 42 7 --drift_types concept covariate

Dataset required:
    Download creditcard.csv from:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  (CC-0)
    Place it in:  data/creditcard.csv
"""

import argparse
import sys
import json
import numpy as np
import torch
from pathlib import Path
from dataclasses import asdict

# ── Package imports ───────────────────────────────────────────────────────── #
from neuro_symbolic_drift.model        import HybridRuleLearner
from neuro_symbolic_drift.data_loader  import load_data, get_dataloaders, print_dataset_stats
from neuro_symbolic_drift.train        import train_model, load_checkpoint
from neuro_symbolic_drift.experiment   import run_single_seed, summarise_results
from neuro_symbolic_drift.alert_system import DriftAlertSystem
from neuro_symbolic_drift.figures      import generate_all_figures, print_summary_table

SEEDS           = [42, 0, 7, 123, 2024]
DRIFT_TYPES     = ["covariate", "prior", "concept"]
CHECKPOINTS_DIR = Path("checkpoints")
RESULTS_DIR     = Path("results")
RESULTS_FILE    = RESULTS_DIR / "drift_results_all_seeds.json"


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 — Train
# ══════════════════════════════════════════════════════════════════════════════

def step_train(seeds, verbose=False):
    print("\n" + "="*60)
    print("  STEP 1 — Train HybridRuleLearner across all seeds")
    print("="*60)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = []

    for seed in seeds:
        print(f"\n── Seed {seed} ──────────────────────────────────────")
        torch.manual_seed(seed)
        np.random.seed(seed)

        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         scaler, feature_names, pos_weight) = load_data(seed=seed)

        if verbose:
            print_dataset_stats(y_train, y_val, y_test)

        train_loader, val_loader = get_dataloaders(X_train, y_train, X_val, y_val)

        model     = HybridRuleLearner(n_features=X_train.shape[1])
        ckpt_path = CHECKPOINTS_DIR / f"seed_{seed}.pt"

        result = train_model(
            model, train_loader, val_loader,
            pos_weight      = pos_weight,
            device          = device,
            checkpoint_path = ckpt_path,
            verbose         = verbose,
        )

        print(f"  Best val PR-AUC : {result['best_val_pr_auc']:.4f} "
              f"(epoch {result['best_epoch']})")
        print(f"  Val F1          : {result['val_f1']:.4f}  "
              f"threshold={result['val_threshold']:.3f}")
        print(f"  Alpha (MLP wt)  : {result['alpha']:.3f}")
        print(f"  Checkpoint      : {ckpt_path}")
        summary.append({"seed": seed, **result})

    print(f"\n{'Seed':>6}  {'Val PR-AUC':>12}  {'Val F1':>8}  {'Epoch':>6}")
    print("-" * 40)
    for r in summary:
        print(f"{r['seed']:>6}  {r['best_val_pr_auc']:>12.4f}  "
              f"{r['val_f1']:>8.4f}  {r['best_epoch']:>6}")


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2 — Drift experiment
# ══════════════════════════════════════════════════════════════════════════════

def _format_lag(detection_lag):
    # BUG 2 FIX — was f"+{result.detection_lag}w early" with hardcoded "+"
    # which printed "[+-1w early]" for negative lags (RWSS fired after F1).
    # Now correctly labels early, simultaneous, and late cases.
    if detection_lag is None:
        return "no alert fired"
    if detection_lag > 0:
        return f"+{detection_lag}w early"
    if detection_lag < 0:
        return f"{abs(detection_lag)}w LATE"
    return "simultaneous"


def step_experiment(seeds, drift_types):
    print("\n" + "="*60)
    print("  STEP 2 — Drift detection experiment")
    print("="*60)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = []

    for seed in seeds:
        ckpt_path = CHECKPOINTS_DIR / f"seed_{seed}.pt"
        if not ckpt_path.exists():
            print(f"\n  [SKIP] seed={seed} — no checkpoint. Run --step train first.")
            continue

        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         scaler, feature_names, pos_weight) = load_data(seed=seed)

        model = HybridRuleLearner(n_features=X_train.shape[1])
        ckpt  = load_checkpoint(ckpt_path, model, device)
        val_threshold = ckpt.get("val_threshold", 0.5)

        print(f"\n── Seed {seed}  (val_threshold={val_threshold:.3f}) ──────────────")

        for drift_type in drift_types:
            result = run_single_seed(
                seed          = seed,
                drift_type    = drift_type,
                X_test        = X_test,
                y_test        = y_test,
                trained_model = model,
                feature_names = feature_names,
                val_threshold = val_threshold,
                device        = device,
            )
            all_results.append(result)

            print(f"  {drift_type:12s}  "
                  f"F1@W{result.f1_alert_window}  "
                  f"RWSS@W{result.rwss_alert_window}[{_format_lag(result.detection_lag)}]  "
                  f"VEL@W{result.rwss_vel_alert_window}[{_format_lag(result.detection_lag_velocity)}]  "
                  f"FIDIZ@W{result.fidi_z_alert_window}[{_format_lag(result.detection_lag_fidi_z)}]  "
                  f"PSIR@W{result.psi_r_alert_window}[{_format_lag(result.detection_lag_psi_r)}]")

    if not all_results:
        print("No results produced. Run --step train first.")
        return

    # ── Save ──────────────────────────────────────────────────────────────── #
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)
    print(f"\nResults saved → {RESULTS_FILE}")

    # ── Summary table ─────────────────────────────────────────────────────── #
    summary = summarise_results(all_results)
    summary.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    print_summary_table(summary)


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 — Figures
# ══════════════════════════════════════════════════════════════════════════════

def step_figures():
    print("\n" + "="*60)
    print("  STEP 3 — Generate article figures")
    print("="*60)

    if not RESULTS_FILE.exists():
        print(f"  Results not found at {RESULTS_FILE}")
        print("  Run --step experiment first.")
        return

    from neuro_symbolic_drift.experiment import SeedDriftResult, WindowResult

    with open(RESULTS_FILE) as f:
        raw = json.load(f)

    results = []
    for r_dict in raw:
        windows = [WindowResult(**w) for w in r_dict.pop("windows")]
        result  = SeedDriftResult(**r_dict)
        result.windows = windows
        results.append(result)

    print(f"  Loaded {len(results)} results. Generating figures...")
    generate_all_figures(results)
    print(f"  All figures saved → results/figures/")


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Alert demo
# ══════════════════════════════════════════════════════════════════════════════

def step_alert():
    print("\n" + "="*60)
    print("  STEP 4 — DriftAlertSystem demo (seed 42)")
    print("="*60)

    seed      = 42
    ckpt_path = CHECKPOINTS_DIR / f"seed_{seed}.pt"
    if not ckpt_path.exists():
        print(f"  Checkpoint not found. Run --step train first.")
        return

    device = torch.device("cpu")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler, feature_names, pos_weight) = load_data(seed=seed)

    model = HybridRuleLearner(n_features=X_train.shape[1])
    load_checkpoint(ckpt_path, model, device)

    # Build alert system from validation set as baseline
    X_val_t      = torch.FloatTensor(X_val)
    alert_system = DriftAlertSystem.from_trained_model(model, X_val_t, feature_names)
    alert_system.save(RESULTS_DIR / "drift_alert_baseline_seed42.pkl")
    print("  Baseline alert system built and saved.")

    # BUG 5 FIX — was DriftConfig(drift_type="concept", n_windows=5)
    # which ran only 5 windows while the experiment uses N_WINDOWS=8.
    # Now uses the global N_WINDOWS default so both are directly comparable.
    from neuro_symbolic_drift.drift_injector import DriftInjector, DriftConfig
    config   = DriftConfig(drift_type="concept")
    injector = DriftInjector(X_test, y_test, config, feature_names)
    windows  = injector.get_windows()

    print(f"\n  Checking each drift window ({len(windows)} windows):\n")
    for w_idx, (X_w, _) in enumerate(windows):
        X_w_t = torch.FloatTensor(X_w)
        alert = alert_system.check(
            model, X_w_t,
            metadata={"window": w_idx, "drift_type": "concept"}
        )
        print(f"  Window {w_idx}: severity={alert.severity:8s}  "
              f"RWSS={alert.rwss_score:.3f}  "
              f"fired={alert.fired}")

    # Print the first critical alert in full; fall back to first warning
    critical = next((a for a in alert_system.alert_history
                     if a.severity == "critical"), None)
    warning  = next((a for a in alert_system.alert_history
                     if a.severity == "warning"), None)
    first_alert = critical or warning
    if first_alert:
        print("\n" + first_alert.report())


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Neuro-Symbolic Drift Detection — Article 3"
    )
    p.add_argument(
        "--step",
        choices=["train", "experiment", "figures", "alert", "all"],
        default="all",
        help="Which step to run (default: all)"
    )
    p.add_argument("--seeds",       nargs="+", type=int, default=SEEDS)
    p.add_argument("--drift_types", nargs="+", type=str, default=DRIFT_TYPES)
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("\n  Neuro-Symbolic Drift Detection  |  Article 3")
    print(f"  Seeds: {args.seeds}")
    print(f"  Drift types: {args.drift_types}")

    if args.step in ("train", "all"):
        step_train(args.seeds, verbose=args.verbose)

    if args.step in ("experiment", "all"):
        step_experiment(args.seeds, args.drift_types)

    if args.step in ("figures", "all"):
        step_figures()

    if args.step in ("alert", "all"):
        step_alert()

    print("\n  Done.\n")
