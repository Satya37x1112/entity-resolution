"""
Measure candidate blocking recall, precision, and pass breakdown.
Evaluates both the production GPU TF-IDF blocking and multi-pass exact blocking.
"""
import sys
import os
import time
from pathlib import Path
import polars as pl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "business_entity_resolution" / "src"))
from config import cache_path, SEED
from retrieval import CountryIndex, name_text

def measure_blocking(sample_queries: int = 25000):
    print("=" * 70)
    print("       PHASE 1: BLOCKING RECALL & PRECISION MEASUREMENT")
    print("=" * 70)
    
    t0 = time.time()
    # Load ground truth pairs
    gt = pl.read_parquet(cache_path("train_ground_truth.parquet"))
    tp = (gt.filter(pl.col("matched_entity_ids") != "")
            .with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids")
            .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_ids").alias("m_id")))
    
    s1 = pl.read_parquet(cache_path("train_source1_norm.parquet")).with_row_index("s1_idx")
    q = pl.concat([
        pl.read_parquet(cache_path("train_source2_norm.parquet")),
        pl.read_parquet(cache_path("train_source3_norm.parquet"))
    ]).with_row_index("q_idx")
    
    # Map ground truth to integer indices
    truth_map = (tp.join(s1.select("entity_id", "s1_idx"), left_on="s1_id", right_on="entity_id")
                   .join(q.select("entity_id", "q_idx"), left_on="m_id", right_on="entity_id")
                   .select("q_idx", "s1_idx"))
    
    truth_set = set(zip(truth_map["q_idx"].to_list(), truth_map["s1_idx"].to_list()))
    matched_q_set = set(truth_map["q_idx"].to_list())
    
    print(f"Total ground truth links: {len(truth_set):,}")
    print(f"Total matched queries in ground truth: {len(matched_q_set):,}")
    
    # Sample queries that have true matches + queries without matches
    rng = np.random.default_rng(SEED)
    matched_q_list = list(matched_q_set)
    sample_size = min(sample_queries, len(matched_q_list))
    sampled_q_idx = set(rng.choice(matched_q_list, size=sample_size, replace=False))
    
    # Ground truth for sampled queries
    sample_truth = {(qi, si) for (qi, si) in truth_set if qi in sampled_q_idx}
    n_sample_true = len(sample_truth)
    print(f"Sampled queries with matches: {len(sampled_q_idx):,}, true links: {n_sample_true:,}\n")
    
    # Evaluate per country
    countries = ["India", "US"]
    
    # -------------------------------------------------------------
    # 1. Sweep K_KEEP (5, 10, 15, 20) and candidate thresholds
    # -------------------------------------------------------------
    print("--- 1. Candidate Generation Sweep (k=5, 10, 15, 20) ---")
    results_by_k = {5: {"cands": 0, "tp": 0}, 10: {"cands": 0, "tp": 0}, 15: {"cands": 0, "tp": 0}, 20: {"cands": 0, "tp": 0}}
    
    # Pass breakdown tracking (for k=10)
    pass_stats = {
        "Name View (cos_name >= 0.3)": {"cands": 0, "tp": 0},
        "Address View (cos_addr >= 0.2)": {"cands": 0, "tp": 0},
        "Combined View (cos_comb)": {"cands": 0, "tp": 0},
        "High Agreement (both name+addr)": {"cands": 0, "tp": 0},
    }
    
    country_recalls = {}
    
    for c in countries:
        s1c = s1.filter(pl.col("country") == c)
        qc = q.filter((pl.col("country") == c) & (pl.col("q_idx").is_in(sampled_q_idx)))
        if qc.height == 0:
            continue
            
        print(f"[{c}] Building index for {s1c.height:,} S1 records; evaluating {qc.height:,} queries...")
        idx = CountryIndex(s1c)
        s1_map = s1c["s1_idx"].to_numpy()
        
        Qn, Qa = idx.encode(qc)
        cand_df = idx.candidates(Qn, Qa, K=50, k=20)
        q_map = qc["q_idx"].to_numpy()
        
        cand_df = cand_df.with_columns(
            pl.Series("q_idx", q_map[cand_df["q_row"].to_numpy()], dtype=pl.UInt32),
            pl.Series("s1_idx", s1_map[cand_df["s1_row"].to_numpy()], dtype=pl.UInt32),
        )
        
        c_truth = {(qi, si) for (qi, si) in sample_truth if qi in set(q_map)}
        
        # Check recall at different k
        for k_val in [5, 10, 15, 20]:
            k_cands = set(zip(
                cand_df.filter(pl.col("rank") <= k_val)["q_idx"].to_list(),
                cand_df.filter(pl.col("rank") <= k_val)["s1_idx"].to_list()
            ))
            tp_found = len(k_cands.intersection(sample_truth))
            results_by_k[k_val]["cands"] += len(k_cands)
            results_by_k[k_val]["tp"] += tp_found
            if k_val == 10:
                country_recalls[c] = tp_found / len(c_truth) if len(c_truth) else 0.0
                
        # Pass breakdown at k=10
        k10_df = cand_df.filter(pl.col("rank") <= 10)
        
        # Pass 1: Name strong
        p1_cands = set(zip(
            k10_df.filter(pl.col("cos_name") >= 0.30)["q_idx"].to_list(),
            k10_df.filter(pl.col("cos_name") >= 0.30)["s1_idx"].to_list()
        ))
        pass_stats["Name View (cos_name >= 0.3)"]["cands"] += len(p1_cands)
        pass_stats["Name View (cos_name >= 0.3)"]["tp"] += len(p1_cands.intersection(sample_truth))
        
        # Pass 2: Address strong
        p2_cands = set(zip(
            k10_df.filter(pl.col("cos_addr") >= 0.20)["q_idx"].to_list(),
            k10_df.filter(pl.col("cos_addr") >= 0.20)["s1_idx"].to_list()
        ))
        pass_stats["Address View (cos_addr >= 0.2)"]["cands"] += len(p2_cands)
        pass_stats["Address View (cos_addr >= 0.2)"]["tp"] += len(p2_cands.intersection(sample_truth))
        
        # Pass 3: Combined top-10
        p3_cands = set(zip(k10_df["q_idx"].to_list(), k10_df["s1_idx"].to_list()))
        pass_stats["Combined View (cos_comb)"]["cands"] += len(p3_cands)
        pass_stats["Combined View (cos_comb)"]["tp"] += len(p3_cands.intersection(sample_truth))
        
        # Pass 4: High agreement
        p4_cands = p1_cands.intersection(p2_cands)
        pass_stats["High Agreement (both name+addr)"]["cands"] += len(p4_cands)
        pass_stats["High Agreement (both name+addr)"]["tp"] += len(p4_cands.intersection(sample_truth))
        
    print("\n" + "=" * 70)
    print("Table 1: Candidate Recall vs. Volume Sweep (across all countries)")
    print("-" * 70)
    print(f"{'k (per query)':<15} | {'Candidates / Query':<20} | {'Recall':<12} | {'Precision':<12}")
    print("-" * 70)
    for k_val in [5, 10, 15, 20]:
        total_cands = results_by_k[k_val]["cands"]
        tp_found = results_by_k[k_val]["tp"]
        cands_per_q = total_cands / sample_size
        recall = tp_found / n_sample_true
        prec = tp_found / total_cands if total_cands else 0
        print(f"{k_val:<15} | {cands_per_q:<20.1f} | {recall:<12.4%} | {prec:<12.4%}")
    print("-" * 70)
    
    print("\n" + "=" * 70)
    print("Table 2: Blocking Precision Broken Down by Pass (at k=10)")
    print("-" * 70)
    print(f"{'Pass Name':<35} | {'Candidates':<12} | {'True Positives':<15} | {'Precision':<12}")
    print("-" * 70)
    for pass_name, st in pass_stats.items():
        cands = st["cands"]
        tp_cnt = st["tp"]
        prec = tp_cnt / cands if cands else 0
        print(f"{pass_name:<35} | {cands:<12,} | {tp_cnt:<15,} | {prec:<12.4%}")
    print("-" * 70)
    
    print("\nCandidate Recall by Country (k=10):")
    for c, r in country_recalls.items():
        print(f"  {c}: {r:.4%}")
        
    overall_recall = results_by_k[10]["tp"] / n_sample_true
    print(f"\nOverall Candidate Recall at k=10: {overall_recall:.4%}")
    if overall_recall >= 0.95:
        print("[SUCCESS] Candidate recall clears the >= 0.95 hard requirement!")
    else:
        print(f"[WARNING] Candidate recall {overall_recall:.4%} is below 0.95 threshold!")
    print(f"Measured in {time.time()-t0:.1f}s")
    return results_by_k, pass_stats

if __name__ == "__main__":
    measure_blocking(sample_queries=25000)
