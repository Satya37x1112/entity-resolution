# Business Entity Resolution — reproducible pipeline

Produces `output/matching_results.tsv` (leaderboard file) and `output/candidate_pairs.tsv`
(the exact candidate set scored by the model) from the challenge TSVs.

## Environment

* Python 3.10, Windows or Linux, 16 GB RAM (the pipeline is written to stay under ~5 GB).
* An NVIDIA GPU is strongly recommended for candidate generation (tested on an RTX 4050 Laptop, 6 GB).
  Without CUDA the same code runs on CPU, but much slower.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python -m pip install -r requirements.txt
```

## Data layout

```
<repo>/dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
<repo>/dataset/test/test_source{1,2,3}.tsv
```

Locations can be overridden with `ER_DATA_DIR`, `ER_CACHE_DIR` and `ER_OUTPUT_DIR`.

## Run

Everything, end to end (each stage caches to `cache/` and resumes if interrupted):

```bash
cd src
python run_all.py
```

Or stage by stage:

| step | command | output |
|---|---|---|
| 0 | `python prepare_data.py` | raw TSV → parquet |
| 1 | `python build_normalized.py train test` | normalized names / addresses |
| 1b | `python indic_dictionary.py` then `python apply_indic_dict.py train test` | Indic-script token dictionary (learned on the training fold) applied to S2/S3 names |
| 2 | `python candidates.py train test` | top-10 S1 candidates for every S2/S3 record |
| 3 | `python build_features.py train test` | 64 pair features per candidate |
| 4 | `python train.py` | LightGBM model + validation threshold sweep |
| 5 | `python predict.py [threshold]` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |

Validate the outputs:

```bash
python ../../utils/validate_submission.py --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv --test-dir ../../dataset/test
```

## Source files

| file | role |
|---|---|
| `config.py` | paths / seed |
| `prepare_data.py` | TSV → parquet (no quoting, all columns as strings) |
| `normalization.py` | name & address normalization, Indic-script transliteration, legal-suffix removal |
| `build_normalized.py` | multiprocess normalization of every record |
| `indic_dictionary.py` | learns transliterated-token → Latin-token mappings from training pairs (fold A only) |
| `apply_indic_dict.py` | applies the dictionary to Indic-script names (idempotent) |
| `retrieval.py` | per-country TF-IDF (name char-3-grams + address tokens), GPU CountSketch top-K, exact re-scoring |
| `candidates.py` | candidate generation for a split (query = S2/S3 record, index = S1); `patch_queries` re-generates a subset |
| `features.py` | pair features (rapidfuzz similarities, token / house-number overlap, query context) |
| `build_features.py` | features for every candidate chunk |
| `metrics.py` | macro F0.5 exactly as defined by the challenge |
| `train.py` | fold split, LightGBM training, end-to-end validation, threshold selection |
| `predict.py` | test scoring, one-S1-per-record assignment, output writing |
| `run_all.py` | orchestrates all stages |
