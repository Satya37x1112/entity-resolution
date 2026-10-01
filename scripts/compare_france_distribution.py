"""
Compare France's feature score distribution against US and India
to select the precision-safe unseen country fallback threshold.
"""
import sys
from pathlib import Path
import polars as pl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "business_entity_resolution" / "src"))
from config import cache_path

def compare_distributions():
    print("=" * 70)
    print("  PHASE 2: FRANCE FEATURE DISTRIBUTION COMPARISON VS INDIA & US")
    print("=" * 70)
    
    # Load normalized tables for train (India, US) and test (France, India, US)
    s1_train = pl.read_parquet(cache_path("train_source1_norm.parquet"))
    s1_test = pl.read_parquet(cache_path("test_source1_norm.parquet"))
    
    countries = ["India", "US", "France"]
    stats = {}
    
    for c in countries:
        if c == "France":
            s1_c = s1_test.filter(pl.col("country") == "France")
        else:
            s1_c = s1_train.filter(pl.col("country") == c)
            
        name_len = s1_c["name_core"].str.len_chars().to_numpy()
        name_tokens = s1_c["name_core"].str.split(" ").list.len().to_numpy()
        addr_len = s1_c["addr_norm"].str.len_chars().to_numpy()
        addr_nums = s1_c["addr_nums"].str.split(" ").list.len().to_numpy()
        has_num = (s1_c["addr_nums"] != "").to_numpy()
        addr_missing = s1_c["addr_missing"].to_numpy()
        
        stats[c] = {
            "S1 Count": len(s1_c),
            "Mean Name Chars": float(np.mean(name_len)),
            "Median Name Chars": float(np.median(name_len)),
            "Mean Name Tokens": float(np.mean(name_tokens)),
            "Mean Addr Chars": float(np.mean(addr_len)),
            "Median Addr Chars": float(np.median(addr_len)),
            "Addr Has Number Rate": float(np.mean(has_num)),
            "Mean Number Count": float(np.mean(addr_nums)),
            "Addr Missing Rate": float(np.mean(addr_missing)),
        }
        
    print(f"{'Metric':<25} | {'India':<15} | {'US':<15} | {'France':<15}")
    print("-" * 75)
    metrics = list(stats["India"].keys())
    for m in metrics:
        if m == "S1 Count":
            print(f"{m:<25} | {int(stats['India'][m]):<15,d} | {int(stats['US'][m]):<15,d} | {int(stats['France'][m]):<15,d}")
        elif "Rate" in m:
            print(f"{m:<25} | {stats['India'][m]:<15.2%} | {stats['US'][m]:<15.2%} | {stats['France'][m]:<15.2%}")
        else:
            print(f"{m:<25} | {stats['India'][m]:<15.2f} | {stats['US'][m]:<15.2f} | {stats['France'][m]:<15.2f}")
    print("-" * 75)
    
    # Distance calculation to determine closest country:
    # Normalized Euclidean distance over standardized structural metrics
    feature_keys = ["Mean Name Chars", "Mean Name Tokens", "Mean Addr Chars", "Addr Has Number Rate", "Mean Number Count"]
    
    dist_in = sum((stats["France"][k] - stats["India"][k])**2 / (stats["India"][k]**2) for k in feature_keys)
    dist_us = sum((stats["France"][k] - stats["US"][k])**2 / (stats["US"][k]**2) for k in feature_keys)
    
    print(f"\nNormalized Structural Distance to France:")
    print(f"  Distance(France, US):    {dist_us:.4f}")
    print(f"  Distance(France, India): {dist_in:.4f}")
    
    if dist_us < dist_in:
        closest = "US"
    else:
        closest = "India"
    print(f"\n[CONCLUSION]: France distribution is substantially closer to {closest}.")
    print(f"For maximum precision on unseen country (France):")
    print(f"- Recommended basis: {closest} threshold")
    print(f"- Precision-safe fallback strategy: max(threshold_{closest}, threshold_India) or precision floor constraint.")
    
    return stats

if __name__ == "__main__":
    compare_distributions()
