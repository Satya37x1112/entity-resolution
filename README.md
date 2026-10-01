# Business Entity Resolution Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-brightgreen.svg)](https://lightgbm.readthedocs.io/)

A high-performance, production-grade business entity resolution pipeline developed for the **Amazon ML Challenge 2026**.

The system matches noisy, multi-lingual, and multi-source business entity records (from Sources 2 and 3) against a canonical reference catalog (Source 1) across different countries (US, India, France). It achieves a **Macro $F_{0.5} = 0.9784$** (**Precision = 0.9886**, **Recall = 0.9552**) on held-out validation data and strictly adheres to the official evaluation criteria.

---

## Table of Contents

- [Overview & Architecture](#overview--architecture)
- [Repository Structure](#repository-structure)
- [System Requirements](#system-requirements)
- [Installation & Environment Setup](#installation--environment-setup)
- [Configuration & Environment Variables](#configuration--environment-variables)
- [Dataset Organization](#dataset-organization)
- [Execution Workflows](#execution-workflows)
  - [1. One-Click Pipeline Execution](#1-one-click-pipeline-execution)
  - [2. Step-by-Step Phase Execution](#2-step-by-step-phase-execution)
- [Validation & Verification](#validation--verification)
- [Benchmark Results](#benchmark-results)
- [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Overview & Architecture

Entity resolution in business domains requires resolving discrepancies in company names, legal abbreviations, addresses, and multi-script representations (such as Devanagari/Indic script transliterations).

```
                      +----------------------------------+
                      | Raw TSVs (Source 1, 2, 3)        |
                      +----------------------------------+
                                        |
                                        v
                      +----------------------------------+
                      | Step 0: Ingestion & Parquet Cast |
                      +----------------------------------+
                                        |
                                        v
                      +----------------------------------+
                      | Step 1: Normalization & Translit |
                      |   - Legal suffix stripping       |
                      |   - Indic transliteration mapping|
                      +----------------------------------+
                                        |
                                        v
                      +----------------------------------+
                      | Step 2: GPU Candidate Blocking   |
                      |   - TF-IDF char-3-grams + address|
                      |   - GPU CountSketch top-10 retrieval|
                      +----------------------------------+
                                        |
                                        v
                      +----------------------------------+
                      | Step 3: Feature Engineering      |
                      |   - 65 pairwise similarity feats |
                      |   - RapidFuzz, house numbers, gap|
                      +----------------------------------+
                                        |
                                        v
                      +----------------------------------+
                      | Step 4: LightGBM Classifier      |
                      |   - 5-Fold Stratified Split      |
                      |   - Threshold tuning for F0.5    |
                      +----------------------------------+
                                        |
                                        v
                      +----------------------------------+
                      | Step 5: Test Inference & Predict |
                      |   - 1-to-1 canonical assignment  |
                      |   - TSV chunked writer           |
                      +----------------------------------+
                                        |
                                        v
     +----------------------------------+----------------------------------+
     |                                                                     |
     v                                                                     v
`output/matching_results.tsv`                                `output/candidate_pairs.tsv`
```

---

## Repository Structure

```text
entity-resolution/
├── README.md                           # Main documentation & setup guide
├── requirements.txt                    # Project-level Python dependencies
├── .gitignore                          # Standard git ignore rules
│
├── business_entity_resolution/         # Core production pipeline implementation
│   ├── README.md                       # Submodule documentation
│   ├── requirements.txt                # Submodule dependencies
│   └── src/
│       ├── config.py                   # Paths, directories, and random seed
│       ├── prepare_data.py             # TSV to parquet ingestion
│       ├── normalization.py            # Text & address normalization routines
│       ├── build_normalized.py         # Multi-process normalization runner
│       ├── indic_dictionary.py         # Indic-to-Latin token dictionary builder
│       ├── apply_indic_dict.py         # Indic transliteration application
│       ├── retrieval.py                # GPU-accelerated candidate retrieval
│       ├── candidates.py               # Blocking & candidate pair generation
│       ├── features.py                 # Feature definitions (65 features)
│       ├── build_features.py           # Feature matrix builder
│       ├── metrics.py                  # Macro F0.5, Precision, Recall metrics
│       ├── train.py                    # LightGBM training & CV threshold tuning
│       ├── predict.py                  # Inference & output generation
│       └── run_all.py                  # End-to-end pipeline driver
│
├── amazon-ml-challenge-2026/           # Amazon ML Challenge bundled workspace
│   ├── HOW_TO_RUN.md                   # Execution runbook
│   ├── PROBLEM_STATEMENT.md            # Challenge problem definition
│   ├── make_submission.py              # Packaging script for submission ZIP
│   ├── scripts/                        # Standalone utility & validation scripts
│   └── utils/                          # Submission validation utility
│
├── scripts/                            # Analysis & diagnostic scripts
│   ├── build_training_set.py           # Training sample builder
│   ├── compare_france_distribution.py  # Geographic distribution analysis
│   ├── compute_features.py             # Feature computation script
│   ├── measure_blocking_recall.py      # Candidate blocking recall analysis
│   └── train_and_validate.py           # Validation script
│
└── utils/
    └── validate_submission.py          # Official format and schema validator
```

---

## System Requirements

### Hardware
- **Processor**: Modern multi-core CPU (8+ cores recommended)
- **RAM**: 16 GB minimum (Pipeline processes data in streaming chunks to stay within ~5 GB active memory)
- **GPU (Recommended)**: NVIDIA GPU with CUDA support (e.g., RTX 3050 / 4050 or higher with $\ge$ 6 GB VRAM) for accelerated CountSketch candidate generation.
  > *Note: A pure CPU fallback is supported, though candidate generation will take substantially longer.*
- **Storage**: $\ge$ 30 GB free space for raw TSVs, Parquet caches, and feature tables.

### Software
- **Operating System**: Windows 10/11 or Linux (Ubuntu 20.04+)
- **Python**: 3.10 or 3.11 (64-bit)
- **CUDA Toolkit**: 12.4 (matches PyTorch 2.6.0)

---

## Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Satya37x1112/entity-resolution.git
cd entity-resolution
```

### 2. Set Up Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If script execution is disabled on PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

#### On Linux / macOS (bash):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch with CUDA 12.4 Support
```bash
python -m pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
*(For CPU-only machines, simply run: `pip install torch==2.6.0`)*

### 4. Install Remaining Dependencies
```bash
pip install -r requirements.txt
```

---

## Configuration & Environment Variables

All path resolutions and execution settings are defined in [`business_entity_resolution/src/config.py`](file:///c:/Users/ANIL-14/Desktop/Github_Upload/entity-resolution/business_entity_resolution/src/config.py).

By default:
- **`DATA_DIR`**: `<repo_root>/dataset`
- **`CACHE_DIR`**: `<repo_root>/cache`
- **`OUTPUT_DIR`**: `<repo_root>/output`
- **`SEED`**: `42`

### Custom Path Overrides
You can customize where datasets, cache files, and outputs are stored by setting environment variables:

#### Windows PowerShell:
```powershell
$env:ER_DATA_DIR = "D:\Datasets\AmazonML"
$env:ER_CACHE_DIR = "D:\Cache\EntityResolution"
$env:ER_OUTPUT_DIR = "D:\Output\Submissions"
```

#### Linux / macOS:
```bash
export ER_DATA_DIR="/mnt/data/AmazonML"
export ER_CACHE_DIR="/mnt/cache/EntityResolution"
export ER_OUTPUT_DIR="/mnt/output/Submissions"
```

---

## Dataset Organization

Place the competition TSV dataset files in the `dataset/` directory structured as follows:

```text
dataset/
├── train/
│   ├── train_source1.tsv           # Canonical catalog (Ground Truth targets)
│   ├── train_source2.tsv           # Source 2 noisy records
│   ├── train_source3.tsv           # Source 3 noisy records
│   └── train_ground_truth.tsv      # True matching links
└── test/
    ├── test_source1.tsv            # Test canonical catalog
    ├── test_source2.tsv            # Test source 2 queries
    └── test_source3.tsv            # Test source 3 queries
```

> [!TIP]
> If your dataset already exists elsewhere on your system, avoid copying large files by creating an NTFS Directory Junction on Windows:
> ```powershell
> New-Item -ItemType Junction -Path dataset -Target "C:\Path\To\Your\Existing\dataset"
> ```
> Or a symbolic link on Linux:
> ```bash
> ln -s /path/to/existing/dataset ./dataset
> ```

---

## Execution Workflows

### 1. One-Click Pipeline Execution

To run the entire pipeline end-to-end automatically:

```bash
cd business_entity_resolution/src
python run_all.py
```

The pipeline automatically handles caching. If any stage is interrupted, simply re-running `run_all.py` will resume from the last completed stage.

---

### 2. Step-by-Step Phase Execution

If you prefer to inspect and run each pipeline stage individually:

#### Step 0: Ingest Raw Data (TSV to Parquet)
Converts large, unquoted TSV files into compressed, columnar Parquet files for fast multi-threaded reads (~4 seconds for 24M records):
```bash
python business_entity_resolution/src/prepare_data.py
```

#### Step 1: Text Normalization & Transliteration
Normalizes entity names, standardizes legal forms (LLC, Inc, Pvt Ltd), normalizes addresses, and creates an Indic transliteration dictionary from the training set:
```bash
python business_entity_resolution/src/build_normalized.py train test
python business_entity_resolution/src/indic_dictionary.py
python business_entity_resolution/src/apply_indic_dict.py train
python business_entity_resolution/src/apply_indic_dict.py test
```

#### Step 2: Candidate Blocking & Pair Retrieval
Uses GPU-accelerated CountSketch hash projection over character 3-grams and address tokens to select the top-10 candidate matches in Source 1 for every query in Source 2 and 3:
```bash
python business_entity_resolution/src/candidates.py train
python business_entity_resolution/src/candidates.py test
```

#### Step 3: Feature Extraction
Calculates 65 pairwise features per candidate (string similarity ratios via RapidFuzz, token intersections, house number matches, margin to next candidate, multi-pass agreement):
```bash
python business_entity_resolution/src/build_features.py train
python business_entity_resolution/src/build_features.py test
```

#### Step 4: LightGBM Model Training
Trains the LightGBM classifier using 5-fold cross-validation and performs an exhaustive grid search to find the optimal decision threshold that maximizes Macro $F_{0.5}$ while maintaining Precision $\ge 0.85$:
```bash
python business_entity_resolution/src/train.py
```
*Outputs: Saved model at `cache/lgbm_model.txt` and best threshold parameters at `cache/val_best.json`.*

#### Step 5: Test Inference & Submission Generation
Scores all candidate test pairs, enforces the 1-to-1 assignment constraint per query, applies the calibrated threshold (default: `0.70`), and writes the final TSV outputs:
```bash
python business_entity_resolution/src/predict.py 0.70
```
*Generated files in `output/`:*
- `output/candidate_pairs.tsv` (~1.3 GB)
- `output/matching_results.tsv` (~98 MB)

---

## Validation & Verification

Always validate the generated output files before submitting to ensure compliance with format, non-empty criteria, and ID schemas:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Expected output:**
```text
PASS — no blocking issues found. Safe to submit.
```

To automatically package the files into a submission-ready archive:
```bash
python amazon-ml-challenge-2026/make_submission.py <YOUR_TEAM_NAME>
```

---

## Benchmark Results

Evaluated on held-out validation data using the official competition evaluation metric:

| Metric | Result |
|---|---|
| **Macro $F_{0.5}$ Score** | **0.9784** |
| **Precision** | **98.86%** |
| **Recall** | **95.52%** |
| **Execution Safety** | Fully idempotent, chunk-based caching |
| **Active Memory Footprint** | Stays strictly under $\le 5$ GB RAM |

### Pipeline Stage Execution Times (RTX 4050 6GB, 16GB RAM)
- **Data Ingestion**: ~4 seconds
- **Normalization & Indic Dictionary**: ~2.5 minutes
- **Candidate Retrieval**: ~35 minutes (GPU)
- **Feature Extraction**: ~25 minutes
- **LightGBM Training**: ~18 minutes
- **Inference & Scoring**: ~23 minutes

---

## Troubleshooting & FAQ

### 1. `CUDA out of memory` during Candidate Retrieval
- In `business_entity_resolution/src/candidates.py`, lower the query batch size parameter (`batch_size=50000` to `batch_size=20000`).
- Ensure no other background applications are occupying GPU VRAM.

### 2. "File Not Found: dataset/train/train_source1.tsv"
- Ensure you have linked or extracted the challenge dataset into the root `dataset/` directory, or set the `ER_DATA_DIR` environment variable to point to your dataset location.

### 3. PowerShell Script Execution Disabled
- Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in your PowerShell terminal to enable virtual environment script execution.

### 4. Running Without a GPU
- PyTorch will fall back to CPU if `torch.cuda.is_available()` returns `False`. The pipeline will execute with complete mathematical equivalence, though candidate generation will require additional time.

---

## License

This project is licensed under the MIT License.
