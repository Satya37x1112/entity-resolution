"""
Step 4 & Phase 3: Train LightGBM model, perform 5-fold CV threshold sweep,
precision-floor constrained optimization, pass-agreement gate evaluation,
and per-country precision/recall/F0.5 validation.
"""
import json
import sys
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "business_entity_resolution" / "src"))
from config import SEED, cache_path
from features import FEATURES
from metrics import macro_f05
from train import (
    TRAIN_QUERY_FRACTION, MODEL_PATH, PARAMS, N_ROUNDS,
    id_tables, truth_pairs, val_fold, label_table, query_folds,
    load_training_rows, fit, score_split, best_per_query
)

PRECISION_FLOOR = 0.85  # Configurable named constant for Phase 3 Item 2

def evaluate_with_country_breakdown(best: pl.DataFrame, s1: pl.DataFrame, q: pl.DataFrame, tp: pl.DataFrame, thresholds, gate_agreement: bool = False):
    """Evaluates macro F0.5 with full per-country precision, recall, F0.5 breakdown and optional agreement gate."""
    val_s1 = s1.filter(pl.col("is_val"))
    val_s1_indices = set(val_s1["s1_idx"].to_list())
    ids = best.filter(pl.col("s1_idx").is_in(val_s1_indices)) \
              .join(val_s1.select("s1_idx", pl.col("entity_id").alias("s1_id"), "country"), on="s1_idx") \
              .join(q.select("q_idx", pl.col("entity_id").alias("m_id")), on="q_idx")
    tp_val = tp.join(val_s1.select(pl.col("entity_id").alias("s1_id")), on="s1_id", how="inner")
    
    res = []
    countries = val_s1["country"].unique().sort().to_list()
    
    for t in thresholds:
        # Base prediction by threshold
        pred_cand = ids.filter(pl.col("p1") >= t)
        
        # Phase 3 Item 3: Secondary gate on noisy/ambiguous S1 entities
        if gate_agreement and "pass_agreement_count" in pred_cand.columns:
            s1_counts = pred_cand.group_by("s1_id").agg(pl.len().alias("n_cands"))
            pred_cand = pred_cand.join(s1_counts, on="s1_id", how="left")
            pred_cand = pred_cand.filter((pl.col("n_cands") <= 5) | (pl.col("pass_agreement_count") >= 2))
            
        pred = pred_cand.select("s1_id", "m_id")
        m = macro_f05(pred, tp_val, val_s1["entity_id"])
        
        row = {
            "threshold": t,
            "macro_f05": m["macro_f05"],
            "precision": m["macro_precision"],
            "recall": m["macro_recall"],
            "singleton_acc": m["singleton_acc"],
        }
        
        # Per-country precision, recall, and F0.5 breakdown (Phase 2 Item 3)
        for c in countries:
            mc = macro_f05(pred, tp_val, val_s1.filter(pl.col("country") == c)["entity_id"])
            row[f"f05_{c}"] = mc["macro_f05"]
            row[f"prec_{c}"] = mc["macro_precision"]
            row[f"rec_{c}"] = mc["macro_recall"]
            
        res.append(row)
        
        # Format printing
        print(f"t={t:.2f} | Pooled F0.5: {row['macro_f05']:.4f} (P: {row['precision']:.4f}, R: {row['recall']:.4f}) | " +
              " | ".join([f"{c}: F0.5={row[f'f05_{c}']:.4f}, P={row[f'prec_{c}']:.4f}, R={row[f'rec_{c}']:.4f}" for c in countries]),
              flush=True)
              
    return pl.DataFrame(res)

def cv_threshold_selection(best: pl.DataFrame, s1: pl.DataFrame, q: pl.DataFrame, tp: pl.DataFrame, n_folds: int = 5):
    """
    Phase 3 Item 1: 5-Fold cross-validation on the training pool (fold A)
    to select optimal threshold per country, reporting mean and std.
    """
    print("\n" + "=" * 70)
    print("  PHASE 3 ITEM 1: 5-FOLD CV THRESHOLD SELECTION (TRAINING POOL)")
    print("=" * 70)
    
    train_s1 = s1.filter(~pl.col("is_val"))
    train_ids = train_s1["entity_id"].to_numpy()
    
    # 5 folds on training S1 entities
    rng = np.random.default_rng(SEED)
    fold_assignments = rng.integers(0, n_folds, len(train_ids))
    train_s1_folds = pl.DataFrame({"entity_id": train_ids, "cv_fold": fold_assignments})
    
    ids = best.join(s1.select("s1_idx", pl.col("entity_id").alias("s1_id"), "country"), on="s1_idx") \
              .join(q.select("q_idx", pl.col("entity_id").alias("m_id")), on="q_idx") \
              .join(train_s1_folds, left_on="s1_id", right_on="entity_id", how="inner")
              
    # Pre-filter ground truth for training entities to speed up CV
    tp_train = tp.join(train_s1.select(pl.col("entity_id").alias("s1_id")), on="s1_id", how="inner")
    
    countries = train_s1["country"].unique().sort().to_list()
    threshold_grid = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    
    fold_best_thresholds = {c: [] for c in countries}
    fold_best_thresholds["Pooled"] = []
    
    for f in range(n_folds):
        fold_s1 = train_s1_folds.filter(pl.col("cv_fold") == f)
        fold_preds = ids.filter(pl.col("cv_fold") == f)
        
        # Find best threshold for pooled
        best_t_pooled = None
        best_f05_pooled = -1.0
        for t in threshold_grid:
            p = fold_preds.filter(pl.col("p1") >= t).select("s1_id", "m_id")
            m = macro_f05(p, tp_train, fold_s1["entity_id"])
            if m["macro_f05"] > best_f05_pooled:
                best_f05_pooled = m["macro_f05"]
                best_t_pooled = t
        fold_best_thresholds["Pooled"].append(best_t_pooled)
        
        # Find best threshold per country
        for c in countries:
            c_s1 = fold_s1.join(s1.filter(pl.col("country") == c).select("entity_id"), on="entity_id")
            best_t_c = None
            best_f05_c = -1.0
            for t in threshold_grid:
                p = fold_preds.filter((pl.col("country") == c) & (pl.col("p1") >= t)).select("s1_id", "m_id")
                mc = macro_f05(p, tp_train, c_s1["entity_id"])
                if mc["macro_f05"] > best_f05_c:
                    best_f05_c = mc["macro_f05"]
                    best_t_c = t
            fold_best_thresholds[c].append(best_t_c)
            
    print(f"{'Group / Country':<18} | {'Fold Thresholds':<30} | {'Mean':<8} | {'Std':<8} | {'Status'}")
    print("-" * 75)
    for grp, vals in fold_best_thresholds.items():
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        flag = "[FLAG: High Variance std > 0.05]" if std_val > 0.05 else "[STABLE]"
        print(f"{grp:<18} | {str(vals):<30} | {mean_val:<8.4f} | {std_val:<8.4f} | {flag}")
    print("-" * 75)
    
    return fold_best_thresholds

def run_train_and_validate():
    t0 = time.time()
    s1, q, lab, tp = label_table()
    qf = query_folds(lab, s1)
    
    print(f"Labels ready ({time.time()-t0:.0f}s): {lab.height:,} true links, "
          f"{qf.filter(pl.col('q_is_val')).height:,} val-fold queries", flush=True)
          
    best_path = cache_path("train_best.parquet")
    if best_path.exists():
        best = pl.read_parquet(best_path)
        print(f"Loaded existing scored candidates from {best_path.name}")
    else:
        if MODEL_PATH.exists():
            print(f"Loading existing trained model from {MODEL_PATH.name}")
            model = lgb.Booster(model_file=str(MODEL_PATH))
        else:
            Xtr, ytr, Xes, yes = load_training_rows(qf)
            del qf
            print(f"Train rows {len(ytr):,} (pos {ytr.sum():,}); early-stop rows {len(yes):,} ({time.time()-t0:.0f}s)", flush=True)
            
            # Fit LightGBM model
            print("\n--- Training LightGBM Model ---")
            t_train_start = time.time()
            model = fit(Xtr, ytr, Xes, yes)
            train_time = time.time() - t_train_start
            print(f"Training completed in {train_time:.1f}s ({train_time/60:.2f} min)")
            model.save_model(str(MODEL_PATH))
            
        imp = sorted(zip(FEATURES, model.feature_importance("gain")), key=lambda x: -x[1])
        print("\nTop 15 features by gain:")
        for rank, (f, g) in enumerate(imp[:15], 1):
            print(f"  {rank:2d}. {f:<25}: {round(g):,}")
            
        # Check pass_agreement_count importance
        agree_imp = next((g for f, g in imp if f == "pass_agreement_count"), 0)
        print(f"\nFeature 'pass_agreement_count' gain: {round(agree_imp):,}")
        
        # Score candidate pairs
        print(f"Scoring train candidate pairs...", flush=True)
        scored = score_split(model, "train")
        best = best_per_query(scored)
        del scored
        best.write_parquet(best_path)
        print(f"Saved scored candidates to {best_path.name}")
    
    # -------------------------------------------------------------
    # 5-Fold Cross-Validation on Training Pool
    # -------------------------------------------------------------
    cv_threshold_selection(best, s1, q, tp, n_folds=5)
    
    # -------------------------------------------------------------
    # Validation Evaluation on Held-out Fold B
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 2 & 3: HELD-OUT VALIDATION SWEEP (FOLD B)")
    print("=" * 70)
    thresholds = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    
    print("\n--- Run 1: Standard Inference (without pass-agreement gate) ---")
    res_nogate = evaluate_with_country_breakdown(best, s1, q, tp, thresholds, gate_agreement=False)
    
    print("\n--- Run 2: With Pass-Agreement Gate (agreement >= 2 for S1 with >5 candidates) ---")
    res_gated = evaluate_with_country_breakdown(best, s1, q, tp, thresholds, gate_agreement=True)
    
    # -------------------------------------------------------------
    # Precision Floor Constrained Selection (Phase 3 Item 2)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  PHASE 3 ITEM 2: THRESHOLD SELECTION (PRECISION FLOOR >= {PRECISION_FLOOR:.2f})")
    print("=" * 70)
    
    # Unconstrained best
    best_uncons = res_nogate.sort("macro_f05", descending=True).row(0, named=True)
    
    # Constrained best (precision >= PRECISION_FLOOR)
    eligible = res_nogate.filter(pl.col("precision") >= PRECISION_FLOOR)
    if eligible.height > 0:
        best_cons = eligible.sort("macro_f05", descending=True).row(0, named=True)
    else:
        best_cons = best_uncons
        
    print(f"{'Selection Mode':<28} | {'Threshold':<10} | {'Macro F0.5':<12} | {'Precision':<12} | {'Recall':<12} | {'Recall Cost'}")
    print("-" * 88)
    recall_cost = best_uncons["recall"] - best_cons["recall"]
    print(f"{'Unconstrained Best F0.5':<28} | {best_uncons['threshold']:<10.2f} | {best_uncons['macro_f05']:<12.4f} | {best_uncons['precision']:<12.4f} | {best_uncons['recall']:<12.4f} | Baseline")
    print(f"{f'Precision Floor >= {PRECISION_FLOOR:.2f}':<28} | {best_cons['threshold']:<10.2f} | {best_cons['macro_f05']:<12.4f} | {best_cons['precision']:<12.4f} | {best_cons['recall']:<12.4f} | -{recall_cost:.4%}")
    print("-" * 88)
    
    # Save best threshold config
    chosen = best_cons
    val_best_path = cache_path("val_best.json")
    json.dump(chosen, open(val_best_path, "w"), indent=2)
    print(f"\nSaved optimal threshold configuration to {val_best_path}")
    print(f"Chosen threshold: {chosen['threshold']}, Macro F0.5: {chosen['macro_f05']:.4f}")
    
    return chosen

if __name__ == "__main__":
    run_train_and_validate()
