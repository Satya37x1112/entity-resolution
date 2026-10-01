# Amazon ML Challenge 2026: End-to-End Pipeline Execution Guide

This repository contains the complete, production-grade business entity resolution pipeline for the Amazon ML Challenge 2026. The solution achieves **Macro $F_{0.5} = 0.9784$** (**Precision = 0.9886**, **Recall = 0.9552**) on held-out validation data and strictly passes the official submission validator.

---

## 1. System Requirements & Environment Setup

### Prerequisites
- **Operating System**: Windows or Linux
- **Python**: Python 3.10 or 3.11
- **GPU**: NVIDIA GPU with CUDA support (e.g. RTX 3050, 4050, or better, >= 6GB VRAM)
- **RAM**: >= 16 GB RAM
- **Disk Space**: ~30 GB free for raw datasets, parquet caches, and candidate features

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SIBAM890/Business-Entity-Resolution-Amazon-Ml-Challenge-2026.git
   cd Business-Entity-Resolution-Amazon-Ml-Challenge-2026
   git checkout solution-pipeline
   ```

2. **Create and activate a virtual environment**:
   ```powershell
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *(On Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate`)*

3. **Install PyTorch with CUDA**:
   ```powershell
   python -m pip install --upgrade pip
   pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
   ```

4. **Install remaining dependencies**:
   ```powershell
   pip install -r business_entity_resolution/requirements.txt
   ```
   *Core libraries: `polars==1.44.2`, `lightgbm==4.7.0`, `rapidfuzz==3.14.5`, `pandas==2.3.3`, `scipy==1.15.3`, `pyarrow==25.0.1`.*

---

## 2. Dataset Setup

The challenge data must be placed under the `dataset/` directory:
```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

*(If your data is stored in `entity_resolution_solution/student_resource/dataset/`, create an NTFS directory junction on Windows):*
```powershell
New-Item -ItemType Junction -Path dataset -Target entity_resolution_solution\student_resource\dataset
```

---

## 3. Running the Entire Pipeline (One-Click)

To run the complete pipeline from raw TSVs all the way to final submission files:

```powershell
cd business_entity_resolution\src
python run_all.py
```

`run_all.py` executes each stage sequentially. All intermediate tables and models are saved in `cache/`. If interrupted, the pipeline safely resumes from the last completed stage.

---

## 4. Running Phase-by-Phase (Step-by-Step)

Alternatively, each step can be executed individually from the project root:

### Step 0: Data Ingestion (TSV to Parquet)
Converts raw TSVs into fast, columnar Parquet files (~4 seconds for 24M rows):
```powershell
python business_entity_resolution\src\prepare_data.py
```

### Step 1: Text Normalization & Indic Transliteration
Normalizes business names, legal suffixes, addresses, numbers, and learns 526 Indic token transliteration mappings:
```powershell
python business_entity_resolution\src\build_normalized.py
python business_entity_resolution\src\indic_dictionary.py
python business_entity_resolution\src\apply_indic_dict.py train
python business_entity_resolution\src\apply_indic_dict.py test
```

### Step 2: GPU Candidate Blocking & Pair Retrieval
Builds TF-IDF character 3-gram + address token representations and retrieves top-10 candidate pairs per query via GPU CountSketch:
```powershell
# Generate training candidate pairs (103.2M pairs across 52 chunks)
python business_entity_resolution\src\candidates.py train

# Generate test candidate pairs (99.7M pairs across 52 chunks)
python business_entity_resolution\src\candidates.py test
```

### Step 3: Pairwise Feature Extraction (65 Features)
Extracts 65 engineered signals including RapidFuzz ratios, address/number intersections, differences, margins to runner-up candidates, and multi-pass consensus (`pass_agreement_count`):
```powershell
python business_entity_resolution\src\build_features.py train
python business_entity_resolution\src\build_features.py test
```

### Step 4: Model Training & 5-Fold Cross-Validation
Trains the LightGBM classifier with early stopping and optimizes the classification threshold under the precision floor constraint ($\ge 0.85$):
```powershell
python business_entity_resolution\src\train.py
```
*Model artifact saved to `cache/lgbm_model.txt`; optimal threshold config saved to `cache/val_best.json`.*

### Step 5: Test Inference & Submission Generation
Scores all 99.7M test candidate pairs, enforces 1-to-1 query assignments, applies calibrated threshold $t = 0.70$, and writes TSVs in memory-bounded buckets:
```powershell
python business_entity_resolution\src\predict.py 0.70
```
*Outputs generated in `output/`:*
- `output/candidate_pairs.tsv` (~1.30 GB)
- `output/matching_results.tsv` (~98.0 MB)

---

## 5. Official Verification & Packaging

### Format Validation
Run the official challenge validator to verify format, entity completeness, and column schemas:
```powershell
python utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test
```
*Expected output: `PASS — no blocking issues found. Safe to submit.`*

### Creating the Final Submission ZIP
To package the final submission archive with documentation:
```powershell
python make_submission.py <YOUR_TEAM_NAME>
```

---

## 6. Pipeline Benchmark Summary

| Stage | Metric / Count | Time Taken |
|---|---|---|
| Data Ingestion | 24,204,402 records converted | 4.1s |
| Normalization | 24.2M records normalized + 526 Indic tokens learned | 149s |
| Candidate Generation | 103.2M train pairs / 99.7M test pairs | ~35 min (GPU) |
| Feature Extraction | 65 features materialized over 104 chunks | ~25 min |
| Model Training | 1,043 trees, 5-fold CV threshold $t=0.70$ | ~18 min |
| Test Inference | 99,695,890 test pairs scored | ~23 min |
| **Validation Score** | **Macro $F_{0.5} = 0.9784$** (Precision = 98.86%, Recall = 95.52%) | — |
| **Submission Validator** | **PASS (Exit code 0)** | < 1 min |
