# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary

We treat the task as **record-to-entity assignment**: in the training data every Source 2/3 record belongs to *at most one* Source 1 entity, so each S2/S3 record retrieves its most similar S1 entities (GPU sparse-TF-IDF retrieval, 98.7% candidate recall at 10 candidates per record), a LightGBM classifier scores the candidates with 64 country-agnostic similarity and "what differs" features, and the record is attached to its single best S1 entity only if the probability clears a threshold tuned for macro F0.5. Two ideas carry most of the gain: a **transliteration + learned dictionary** for the nine Indic scripts (Indic-query recall 91% → 99%), and **hard-negative features** that look only at the tokens / house numbers that differ between two otherwise identical records. Validation macro F0.5 is **0.978** (precision 0.989, recall 0.955), measured end-to-end on held-out Source 1 entities against the full 10.3M-record S2+S3 corpus.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA on the training data:

* **Structure.** 2.21M S1 entities, 10.3M S2+S3 records, 7.64M true links. Every matched S2/S3 record links to exactly one S1 entity (never two), matches never cross countries, 26% of S2/S3 records match nothing (distractors), 5.6% of S1 entities are singletons, and a matched S1 has 3.7 links on average (up to 11).
* **Name noise.** Case changes, accents/diacritics (`Stáffing`), digit-for-letter typos (`Preparat0ry`, `5taffing`), character typos, word-order shuffles (`Private RJ Estate Limited`), legal-suffix changes (`Inc`/`Corp`/`LLC`/`Private Limited`/`Pvt`), prefixes (`M/s`, `Dr`, `Shri`), junk prefixes (`--`, `<<`), `X doing business as Y`, parenthesised country tags (`(India)`), website / hashtag forms (`darkbuildwellclinic.com`, `#technologiesunison`), fully unrelated names where only the address links the records, and **names written in 9 Indic scripts** (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam — about 7% of S2/S3 names) that are word-by-word renderings of the English name (`लक्ष्मी केर प्राइवेट लिमिटेड` = `Lakshmi Care Private Limited`).
* **Address noise.** Missing components (no house number, no city, no state; 3–4% of S2/S3 addresses empty), reordered components, abbreviations (`Rd`/`Road`, `Dr`/`Drive`), state names vs. codes (`Arizona`/`AZ`, `Madhya Pradesh`/`MP`, or the state in native script `মধ্য প্রদেশ`), zero-padded numbers (`004013`), `<NULL>` / `N/A` placeholders, and typos.
* **Hard negatives.** Unmatched S2/S3 records are frequently near-copies of a real S1 entity that differ in *one* word (`Medha Agri` vs `Medha Infra`) or *one* number (`A-777` vs `A-772`, `2413 Fifteenth Ave` vs `2400 15th Ave`). These are the main source of false merges.
* **Test set.** Adds France (15% of test S1), unseen in training: French legal forms (SARL, SAS, SASU, EURL, SCI), `R.`/`BD`/`AV.` street abbreviations, elisions (`l'`, `d'`) and accents. Everything in the pipeline is therefore country-agnostic similarity; the country only partitions the search space and is treated as an open label set.

### 2.2 Solution Strategy

**Approach Type:** Retrieval (blocking) + gradient-boosted pair classifier + constrained assignment  
**Core Innovation:** (1) reversing the direction of matching — each S2/S3 record is assigned to at most one S1 entity, which turns a many-to-many problem into a per-record arg-max with a calibrated abstain option; (2) a rule-based Indic→Latin transliterator plus a word dictionary learned from training pairs; (3) GPU CountSketch retrieval over sparse TF-IDF so that all 10M records can be searched against every S1 of their country; (4) "difference" and "margin versus runner-up" features aimed at the hard negatives.

Pipeline:

1. **Normalize** every name and address (§3.1).
2. **Retrieve** the 10 most similar S1 entities of the same country for every S2/S3 record (§3).
3. **Featurize** each (record, candidate) pair with 64 features (§4).
4. **Classify** with LightGBM → probability that the pair is a true link.
5. **Assign** each record to its highest-probability candidate if p ≥ 0.7 (tuned on validation); otherwise leave it unmatched. Group assignments by S1 entity to produce `matching_results.tsv`.

---

## 3. Candidate Generation (Blocking)

### 3.1 Normalization

* **Indic transliteration.** All nine Indic Unicode blocks share the ISCII-derived layout, so one offset→Latin table (consonants, independent vowels, vowel signs, virama, anusvara/candrabindu/visarga, nukta forms, native digits, Malayalam chillu letters) transliterates every script, with inherent-vowel handling and word-final schwa deletion.
* **Learned Indic dictionary.** Transliteration alone gives `praivet`, `injiniyaring`, `phaundeshan`. For training pairs whose S2/S3 name was in an Indic script and has the same number of tokens as the S1 name, tokens are aligned by position and a mapping is kept when it occurs ≥ 3 times and accounts for ≥ 60% of the token's alignments. This yields 526 mappings (`praivet→private`, `injiniyaring→engineering`, `teknolojij→technologies`, `phyuchar→future`, `kanastrakashan→construction`, …). The dictionary is learned **only from fold-A training entities** (see §5), so it does not leak into validation, and is applied identically to test.
* **Names.** ASCII folding (accents, ß, œ, curly quotes), lower-casing, `doing business as` / `d/b/a` / `aka` handling (keep the trade name), junk-prefix stripping, website / hashtag detection (`.com`, `.in`, `.fr`, `www.`, `#`), `M/s` removal, `&`/`+` → `and`, dotted abbreviations (`S.A.S.U.` → `sasu`), French elisions, digit-for-letter repair inside alphabetic tokens (`preparat0ry` → `preparatory`). Two views are kept: `name_norm` (all tokens) and `name_core` (legal/stop words removed: Inc, LLC, Corp, Ltd, Pvt, Private, LLP, PC, SARL, SAS, EURL, SCI, GmbH, the/of/and/de/du/la …).
* **Addresses.** Same folding + transliteration, `<NULL>`/`N/A` removal, number/suffix splitting with leading-zero stripping (`004013` → `4013`, `94B` → `94 b`), canonical street-type abbreviations (English and French), US and Indian state names (incl. transliterated native-script spellings) → codes, removal of filler tokens (`unit`, `apt`, `no`, `door`, `#`).

### 3.2 Retrieval

* **Direction.** Query = each S2/S3 record; index = the S1 entities of the same country. Country is an open label set — every value present is processed independently (US, India, France).
* **Representation.** Two TF-IDF views fitted on S1 of the country: *name* = character 3-grams of `name_core` **with spaces removed** (so `dark buildwell clinic` ≡ `darkbuildwellclinic.com`, robust to typos and re-ordering; sublinear TF, max_df 5%), and *address* = word tokens (max_df 10%). Score = 0.5·cos_name + 0.5·cos_addr. Combining the views matters: on India, name-only recall@10 was 61%, address-only 86%, combined 97%.
* **GPU search.** The L2-normalised concatenated sparse vector is compressed with a signed feature hash (CountSketch, 1024 dims) to fp16 and a brute-force top-40 is run on the GPU (RTX 4050, 6 GB); the 40 approximate neighbours are re-scored **exactly** on the sparse vectors and the best 10 are kept. This processes 10M queries in about an hour per split, versus many hours for CPU sparse top-k.
* **Blocking keys used:** country partition + TF-IDF(name char-3-grams) + TF-IDF(address tokens).
* **Candidate pairs generated:** 10 per S2/S3 record — 103.2M for train, 99.7M for test (≈ 57 candidates per test S1 entity on average). `candidate_pairs.tsv` is exactly this set, regrouped by S1 entity.
* **How true matches were not lost:** candidate recall on the full training set is **98.7%** (India 98.3%, US 98.9%; rank-1 alone 97.1%). The Indic dictionary raised Indic-script recall@10 from 91.3% to 99.4%. The remaining misses are mostly records with an empty address and a generic name shared by several S1 entities.

---

## 4. Matching Model

**Features used (64, all country-agnostic):**

- *Retrieval / query context:* cos_name, cos_addr, combined score, rank, number of candidates, best score of the query, gap to the best **other** candidate, gap to the next candidate, name and address gaps to the query's best.
- *Name features:* rapidfuzz ratio, token-set, token-sort and partial ratio on the core name; ratio, partial ratio and Jaro-Winkler on the space-free name key; ratio on the full name; exact-match flags; token intersection, Jaccard and coverage on both sides; first-token equality; lengths and length ratio.
- *Address features:* ratio, token-set, token-sort, partial ratio, token Jaccard and query coverage; house-number counts, shared numbers, query-number coverage, first-number equality, number conflict (both have numbers, none shared); address-missing flag; lengths.
- *What differs (hard negatives):* number of name tokens only on the query side / only on the S1 side, and the similarity of those leftover tokens (a typo scores high, a substituted word scores low); query-only address tokens and whether they appear (fuzzily) in the S1 address; numbers only on one side, relative difference between the first house numbers, equality of the number sets.
- *Margins versus runner-up:* for cos_name, cos_addr, name token-set, full-name ratio, address token-set and number coverage: value minus the best value among the query's *other* candidates; number of candidates with near-tied names.
- *Other:* S3-vs-S2 flag, website-name flag, Indic-script flag, number of S1 entities in the country with the same core name (ambiguity), number of records that retrieved this S1 at rank 1.

**Model type:** LightGBM binary classifier (num_leaves 255, learning rate 0.05, min_data_in_leaf 200, feature/bagging fraction 0.8, L2 1.0), 892 trees chosen by early stopping on held-out fold-A queries. Trained on 8.2M candidate pairs (10% of fold-A records, all 10 candidates each). Most important features by gain: gap to the best other candidate, rank, combined score, count of query-only numbers, address token-set margin, full-name ratio.

**Assignment / threshold selection:** each S2/S3 record is linked to its single highest-probability candidate if p ≥ t. t was chosen by sweeping {0.3 … 0.8} and maximizing validation macro F0.5 (best t = 0.7; 0.6 and 0.7 are within 0.0001). We also tried an expected-F0.5-optimal subset selection per S1 entity; it gained under 0.001, so the simpler threshold is kept.

---

## 5. Results & Error Analysis

**Validation protocol.** S1 entities are split by id into fold A (80%) and fold B (20%). Every S2/S3 record goes to the fold of its true S1 entity (or, if unmatched, of its rank-1 candidate). The model and the Indic dictionary are fit on fold A only. All 10.3M records are then scored and assigned, and macro F0.5 is computed over the fold-B S1 entities with the official definition (per-entity F0.5, singletons count 1 when predicted empty). This matches the test setting: full corpus, every distractor present.

| model | macro F0.5 | precision | recall | singleton acc. | India | US |
|---|---|---|---|---|---|---|
| v1: retrieval + 48 features | 0.9707 | 0.9835 | 0.9436 | 0.970 | 0.9608 | 0.9774 |
| **v2: + Indic dictionary + difference/margin features** | **0.9783** | **0.9886** | **0.9549** | **0.981** | **0.9754** | **0.9802** |

(For reference, the earlier prototype in this repository reported 0.850 on an easier sampled corpus with 72% candidate recall.)

- **F_0.5 Score (macro):** 0.9783 on the validation fold.
- **Robustness to the test distribution:** test has more S2/S3 records per S1 entity than train (5.8 vs 4.7), which implies about 40% distractor records instead of 26%. Re-scoring validation with every distractor assignment duplicated (2× distractors) gives 0.9759 at t = 0.7 and at t = 0.75, so the chosen threshold stays near-optimal. On test, 58–63% of records are matched per country (India 57.6%, US 58.5%, France 63.3%), consistent with that estimate.
- **Common false positives (wrong merges):** unmatched near-copies of a real entity that differ by one word (`Gowthami Fuels` → `Gowthami Travels`) or one house/unit number (`N1355 Migl Rd` → `N1348 Mill Road`); records with no address whose generic name exists several times in S1 (`First Talent`, `Raj Advisors`).
- **Common false negatives (missed matches):** records with an empty address (about 4% of records; only about half are recoverable because several S1 entities share the name); heavily corrupted names combined with partial addresses; unrelated trade names (`Nylapyrakor`) whose address is also truncated.

---

## 6. Conclusion

Reframing the problem as "attach each S2/S3 record to at most one S1 entity" gave an exact, cheap decision rule. A high-recall GPU retrieval stage (98.7% recall at 10 candidates) with a LightGBM matcher reaches 0.978 validation macro F0.5 end-to-end. The largest single improvement came from handling the Indic scripts (transliteration plus a dictionary learned from training pairs), and precision came from features that look at *what differs* between two otherwise similar records. The pipeline uses only the provided data and open-source libraries; no external lookup, API, geocoder or pretrained language model is used.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (all source in `src/`, with `README.md` and `requirements.txt`). Entry point `python src/run_all.py` runs the stages in order, caching each one:

| step | script | output |
|---|---|---|
| 0 | `prepare_data.py` | TSV → parquet |
| 1 | `build_normalized.py` | normalized names/addresses (multiprocess) |
| 1b | `indic_dictionary.py`, `apply_indic_dict.py` | learned Indic token dictionary applied to S2/S3 names |
| 2 | `candidates.py` | top-10 S1 candidates per S2/S3 record (GPU) |
| 3 | `build_features.py` (`features.py`) | 64 pair features |
| 4 | `train.py` | LightGBM model, validation sweep, threshold |
| 5 | `predict.py` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |

Hardware used: laptop with 16 GB RAM, 16 threads, NVIDIA RTX 4050 (6 GB). The whole pipeline runs in about 4–5 hours, dominated by candidate generation and feature computation.

### B. Additional Results

Candidate recall (train, full corpus, 10 candidates per record):

| segment | recall@1 | recall@10 |
|---|---|---|
| US | 0.975 | 0.989 |
| India, Latin-script names | 0.960 | 0.981 |
| India, Indic-script names (before dictionary) | — | 0.913 |
| India, Indic-script names (with dictionary) | 0.980 | 0.994 |

Threshold sweep (v2, validation fold):

| t | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|
| macro F0.5 | 0.9725 | 0.9755 | 0.9773 | 0.9782 | 0.9783 | 0.9774 |
| precision | 0.9776 | 0.9820 | 0.9851 | 0.9874 | 0.9886 | 0.9895 |
| recall | 0.9660 | 0.9640 | 0.9615 | 0.9584 | 0.9549 | 0.9495 |

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
