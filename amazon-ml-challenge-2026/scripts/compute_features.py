"""
Materialize 65 pairwise features including pass_agreement_count
for all candidate chunks and verify schema and feature shapes.
"""
import sys
import time
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "business_entity_resolution" / "src"))
from config import cache_path
from features import FEATURES
import build_features

def run_compute_features(split: str = "train"):
    print("=" * 70)
    print(f"       PHASE 4 STEP 3: COMPUTE FEATURES ({split.upper()})")
    print("=" * 70)
    print(f"Feature count in specification: {len(FEATURES)}")
    print(f"Contains 'pass_agreement_count': {'pass_agreement_count' in FEATURES}")
    
    t0 = time.time()
    build_features.build(split)
    
    # Verify materialized feature files
    feat_dir = cache_path(f"{split}_feats")
    files = sorted(feat_dir.glob("*.parquet"))
    print(f"\nMaterialized {len(files)} feature chunk files under {feat_dir}")
    
    if files:
        sample_df = pl.read_parquet(files[0])
        print(f"Sample chunk: {files[0].name}")
        print(f"  Shape: {sample_df.shape}")
        print(f"  Columns ({len(sample_df.columns)}): q_idx, s1_idx + {len(sample_df.columns)-2} features")
        assert len(sample_df.columns) == len(FEATURES) + 2, f"Expected {len(FEATURES)+2} columns, got {len(sample_df.columns)}"
        assert "pass_agreement_count" in sample_df.columns, "pass_agreement_count missing from parquet!"
        print("  [SUCCESS] All feature columns verified including pass_agreement_count!")
        
        # Display sample distribution of pass_agreement_count
        dist = sample_df["pass_agreement_count"].value_counts().sort("pass_agreement_count")
        print("\nPass Agreement Count distribution in sample:")
        for row in dist.iter_rows(named=True):
            print(f"  Agreement = {int(row['pass_agreement_count'])}: {row['count']:,} pairs ({row['count']/sample_df.height:.2%})")

    print(f"\nCompleted in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    split_arg = sys.argv[1] if len(sys.argv) > 1 else "train"
    run_compute_features(split_arg)
