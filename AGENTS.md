# AGENTS.md

## Repository profile
This repository is a research-oriented codebase focused on adversarial perturbations, image protection, watermark-related robustness, and attack / defense evaluation for image or video generation systems.

Typical components may include:
- training / optimization scripts
- evaluation scripts
- dataset loaders and preprocessing
- checkpoint loading / exporting
- perturbation generation pipelines
- visualization / analysis utilities
- experiment configs and shell launchers

Treat this repository as an experimental codebase where reproducibility, minimal diffs, and safe validation matter more than broad refactors.

---

## Working principles

### 1. Preserve experiment logic unless explicitly asked
Do not change the mathematical objective, loss weighting, perturbation budget, preprocessing convention, normalization rule, or evaluation metric unless the task explicitly requires it.

### 2. Prefer minimal, local changes
Prefer the smallest safe diff that solves the requested problem.
Avoid broad refactors, cross-file renames, or API redesign unless requested.

### 3. Do not launch expensive jobs by default
Never start full training, long-running optimization, large-batch evaluation, or dataset-wide inference unless the user explicitly asks for it.

Prefer:
- static inspection
- config inspection
- import checks
- CLI `--help`
- batch-size 1 smoke tests
- one-step / few-step sanity runs
- subset or toy input verification

### 4. Protect data and artifacts
Do not modify or delete:
- raw datasets
- cached datasets
- pretrained checkpoints
- experiment outputs
- result folders
- logs
unless the user explicitly asks.

When generating temporary files, keep them isolated and easy to remove.

### 5. Avoid hidden environment changes
Do not install packages, upgrade dependencies, download model weights, or alter system state unless explicitly requested.

If an environment issue blocks progress, identify the missing dependency or mismatch clearly before proposing changes.

---

## Repository discovery order

When starting a task, inspect the repository in this order if needed:

1. `README*`
2. `pyproject.toml`, `requirements*.txt`, `environment*.yml`, `setup.py`
3. `configs/`, `scripts/`, `sh/`, `bash/`
4. main entry scripts such as:
   - `train*.py`
   - `eval*.py`
   - `test*.py`
   - `infer*.py`
   - `attack*.py`
   - `generate*.py`
5. dataset and model modules
6. existing tests

Use the repository's actual commands and conventions rather than inventing new ones.

---

## Research-specific safety checks

For any task involving perturbations, image preprocessing, or model I/O, explicitly check for:

- value range mismatch: `[0,1]` vs `[0,255]`
- normalization mismatch
- channel order mismatch: RGB vs BGR
- image tensor layout mismatch
- resize / crop inconsistency between train and eval
- dtype / device mismatch
- perturbation projection or clipping bugs
- epsilon scale confusion
- checkpoint key mismatch
- accidental no-grad / detached graph behavior
- wrong target label or prompt conditioning path

If image quality is relevant, pay special attention to:
- visible artifacts caused by incorrect clipping
- over-large perturbation norms
- repeated resizing / interpolation damage
- mismatch between optimization-space tensor and saved image-space tensor

---

## Code editing policy

When editing code:

- preserve existing CLI arguments unless required
- preserve existing config field names unless required
- add comments only where they improve clarity materially
- do not rewrite unrelated code for style alone
- do not add dependencies unless necessary
- keep new logic near the existing relevant code path

If the task is ambiguous, prefer a conservative implementation.

---

## Validation ladder

After making changes, validate from cheap to expensive:

1. syntax / parse check
2. import smoke check
3. targeted unit-level or function-level check
4. CLI help or argument parsing check
5. smallest realistic runtime smoke test
6. broader validation only if explicitly justified

If no formal tests exist, still perform the strongest safe verification available.

Do not claim a fix is verified unless you actually ran a relevant check.

---

## Response format

When reporting work, include:

1. what changed
2. why it changed
3. what was validated
4. what was not validated
5. remaining risks or assumptions

Be concrete and concise.