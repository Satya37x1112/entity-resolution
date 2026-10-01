"""
Build training set, partition into train/val splits by S1 ID (leakage-free),
and report row counts, positive rate, and average candidates per S1 entity.
"""
import sys
import time
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "business_entity_resolution" / "src"))
from config import cache_path

def build_training_set_summary():
    print("=" * 70)
    print("      PHASE 2: TRAINING & VALIDATION SET SPLIT ANALYSIS")
    print("=" * 70)
    
    t0 = time.time()
    cand_dir = cache_path("train_cands")
    cand_files = sorted(cand_dir.glob("*.parquet"))
    
    if not cand_files:
        print("[ERROR] No candidate files found in cache/train_cands!")
        return None
        
    print(f"Reading {len(cand_files)} candidate chunk files...")
    cands = pl.scan_parquet(str(cand_dir / "*.parquet"))
    
    # Ground truth links
    gt = pl.read_parquet(cache_path("train_ground_truth.parquet"))
    tp = (gt.filter(pl.col("matched_entity_ids") != "")
            .with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids")
            .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_ids").alias("m_id")))
    
    s1 = pl.read_parquet(cache_path("train_source1.parquet"), columns=["entity_id", "country"]).with_row_index("s1_idx")
    q = pl.concat([
        pl.read_parquet(cache_path("train_source2.parquet"), columns=["entity_id"]),
        pl.read_parquet(cache_path("train_source3.parquet"), columns=["entity_id"])
    ]).with_row_index("q_idx")
    
    # Map ground truth to integer indices
    truth_map = (tp.join(s1.select("entity_id", "s1_idx"), left_on="s1_id", right_on="entity_id")
                   .join(q.select("entity_id", "q_idx"), left_on="m_id", right_on="entity_id")
                   .select("q_idx", pl.col("s1_idx").alias("true_s1")))
    
    # S1 validation fold flag (80/20 numeric modulo)
    s1 = s1.with_columns(
        (pl.col("entity_id").str.slice(3).cast(pl.Int64) % 5 == 0).alias("is_val")
    )
    
    # Query fold flag (assigned to fold of its true S1 entity, or rank-1 retrieved S1 if unmatched)
    top1 = cands.filter(pl.col("rank") == 1).select("q_idx", pl.col("s1_idx").alias("top1")).collect()
    qf = top1.join(truth_map, on="q_idx", how="full", coalesce=True).with_columns(
        pl.coalesce("true_s1", "top1").alias("group_s1")
    )
    qf = qf.join(s1.select(pl.col("s1_idx").alias("group_s1"), pl.col("is_val").alias("q_is_val")),
                 on="group_s1", how="left").select("q_idx", "true_s1", "q_is_val")
    
    # Materialize candidate stats
    print("Aggregating candidate pairs across train and validation splits...")
    cand_df = cands.select("q_idx", "s1_idx").collect()
    total_pairs = cand_df.height
    
    joined = cand_df.join(qf, on="q_idx", how="left").with_columns(
        (pl.col("s1_idx") == pl.col("true_s1")).fill_null(False).alias("is_positive")
    )
    
    train_pairs = joined.filter(~pl.col("q_is_val"))
    val_pairs = joined.filter(pl.col("q_is_val"))
    
    train_count = train_pairs.height
    train_pos = train_pairs["is_positive"].sum()
    train_pos_rate = train_pos / train_count if train_count else 0
    
    val_count = val_pairs.height
    val_pos = val_pairs["is_positive"].sum()
    val_pos_rate = val_pos / val_count if val_count else 0
    
    # S1 candidate coverage
    s1_cand_counts = cand_df.group_by("s1_idx").agg(pl.len().alias("n_cands"))
    total_s1 = s1.height
    covered_s1 = s1_cand_counts.height
    avg_cands_per_s1 = total_pairs / total_s1
    
    print("\n" + "=" * 70)
    print("Table: Training vs. Validation Split Metrics (Leakage-Free 80/20)")
    print("-" * 70)
    print(f"Total candidate pairs:       {total_pairs:,}")
    print(f"Train pairs count:           {train_count:,} ({train_count/total_pairs:.2%})")
    print(f"  Train true positives:      {train_pos:,}")
    print(f"  Train positive rate:       {train_pos_rate:.4%}")
    print(f"Val pairs count:             {val_count:,} ({val_count/total_pairs:.2%})")
    print(f"  Val true positives:        {val_pos:,}")
    print(f"  Val positive rate:         {val_pos_rate:.4%}")
    print("-" * 70)
    print(f"Total S1 entities:           {total_s1:,}")
    print(f"S1 entities with candidates: {covered_s1:,} ({covered_s1/total_s1:.2%})")
    print(f"Average candidates per S1:   {avg_cands_per_s1:.2f}")
    print("-" * 70)
    print(f"Analysis completed in {time.time()-t0:.1f}s")
    
    return {
        "total_pairs": total_pairs,
        "train_count": train_count,
        "train_pos_rate": train_pos_rate,
        "val_count": val_count,
        "val_pos_rate": val_pos_rate,
        "avg_cands_per_s1": avg_cands_per_s1,
    }

if __name__ == "__main__":
    build_training_set_summary()
