# Dependencies and Data Manifest

This file provides a short, human-readable manifest of the project's runtime and test dependencies and the data files the main pipeline uses.

## Python packages (from `requirements.txt`)
- pandas >= 1.3.0 — CSV reading/filtering (used by `chatbot_pipeline.py`'s fast path and tests)
- numpy >= 1.21.0 — numerical utilities used by data processing
- sentence-transformers >= 2.2.0 — heavy ML scorer in `integrated/facility_scorer.py` (optional for fast/demo runs)
- tqdm >= 4.62.0 — progress bars (used by scoring scripts)
- scikit-learn >= 0.24.0 — ML utilities (used by scorer/scikit pipelines)
- pathlib — (stdlib backport; safe to keep)
- typing — (stdlib; safe to keep)
- pytest >= 7.0.0 — test runner for `tests/`

Notes:
- The code supports a "fast" interactive path that only requires `pandas` and `numpy`. The heavy `sentence-transformers` model is only needed if you run `integrated/facility_scorer.py` to re-score facilities.
- If you create a fresh venv, install using:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset / data files used by the pipeline
Files located in `Group3_dataset/` (used by the fast search / scoring pipeline):

- `all_facilities_scored.csv` — pre-scored facilities CSV. The pipeline `chatbot_pipeline.py` prefers this file for the fast search path (no ML model required).
- `facilities_final.csv` — final facility records (source data used for scoring/exports).
- `locations_affordable_normalized.csv` — normalized location mapping used in affordability processing.
- `locations_affordable_plpfile.csv` — another locations file used in preprocessing (PLP format).
- `nsumhss_affordability_min(Janet).csv` — affordability-related CSV from an external source.
- `Placesaffordablehealth_cleaned.csv` — cleaned place-level data.
- `SAMHSA_cleaned.csv` — cleaned SAMHSA dataset (renamed to remove stray space in filename).

Files in `result_of_second_group/` (utilities / examples):

- `patient_case.txt` — sample patient case text
- `generate_patient_case.py` — small script to produce example patient case(s)
- `test.py`, `test.txt` — miscellaneous test/example files

## Which scripts reference these files
- `chatbot_pipeline.py` (entrypoint) — uses `Group3_dataset/all_facilities_scored.csv` for the fast search path (`fast_search_scored_csv`).
- `integrated/facility_scorer.py` — loads raw facility CSV(s) and (optionally) the sentence-transformers model to compute embeddings and scores. Use this if you want to re-score facilities (heavy step).
- Tests under `tests/` use a small sample CSV (packaged in the test) to verify `fast_search_scored_csv` behavior.

## Quick recommendations
- For interactive/demo runs: ensure `pandas` and `numpy` are installed; `all_facilities_scored.csv` should exist in `Group3_dataset/`.
- For production or re-scoring: install `sentence-transformers` and `scikit-learn` and ensure you have enough memory and a network connection (model downloads).
- Fixed the filename `SAMHSA_cleaned.csv` (removed accidental space) to avoid issues when programmatically opening/globbing files.

## Clarification about the earlier second bullet point you asked about
The second bullet in the todo was "List dependencies and data files" — it meant: identify (a) Python package dependencies needed to run the code, and (b) dataset files the pipeline expects (CSV files under `Group3_dataset/` and helper files under `result_of_second_group/`). This manifest consolidates both into one place.

---
Generated: November 3, 2025
