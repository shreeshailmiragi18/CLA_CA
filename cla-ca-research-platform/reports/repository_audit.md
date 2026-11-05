# Repository Audit Report

## Executive Summary
This audit documents the current state of the CLA-CA research platform, identifying all code files, datasets, configuration, and baseline results. The audit was performed to satisfy Phase 0 requirements before any algorithm enhancements can proceed.

## Repository Structure

### Root Directory
- `api/` - FastAPI backend service
- `dashboard.html` - Static dashboard page
- `docs/` - Documentation files
- `frontend/` - React/Vite frontend application
- `performance_chart.jsx` - Performance chart React component
- `results/` - Experiment results and figures
  - `baseline/` - Frozen baseline outputs
  - `figures/` - Generated plots and confusion matrices
  - `reports/` - Evaluation summaries
- `scripts/` - Batch execution and consolidation scripts
- `src/` - Python source code
  - `cla_ca/` - CLA-CA algorithm implementation
    - `data_loader.py` - Dataset loading and preprocessing
    - `discretization.py` - Equal-width discretization
    - `feature_weighting.py` - Variance-based feature weighting
    - `metrics.py` - Evaluation metrics calculation
    - `pipeline.py` - Full pipeline orchestration
    - `prediction.py` - Labeling with median-of-unique threshold
    - `preprocessing.py` - Min-Max normalization
    - `reporting.py` - Report generation
    - `scoring.py` - Module scoring (Algorithm 1)
    - `valley_detection.py` - Create-split valley detection
  - `__init__.py` - Package initialization

### Key Python Entry Points
- `src/cla_ca/pipeline.py` - Main pipeline runner (`run_claca_pipeline()`)
- `api/main.py` - FastAPI REST endpoint (`/api/run-experiment`)
- `run_platform.py` - Platform runner script

## Dataset Inventory

### Datasets Loaded (15 total)
| Dataset | Family | Rows | Features | Defective | Clean | Prevalence |
|---------|--------|------|----------|-----------|-------|------------|
| ant-1.5 | PROMISE | 293 | 21 | 32 | 261 | 10.92% |
| arc | PROMISE | 225 | 20 | 29 | 196 | 12.89% |
| berek | PROMISE | 43 | 23 | 16 | 27 | 37.21% |
| camel-1.0 | PROMISE | 339 | 23 | 13 | 326 | 3.83% |
| ivy-2.0 | PROMISE | 352 | 20 | 40 | 312 | 11.36% |
| jedit-3.2 | PROMISE | 272 | 23 | 90 | 182 | 33.09% |
| synapse-1.1 | PROMISE | 222 | 21 | 60 | 162 | 27.03% |
| tomcat | PROMISE | 858 | 23 | 77 | 781 | 8.97% |
| xalan-2.4 | PROMISE | 723 | 23 | 110 | 613 | 15.21% |
| xerces-1.2 | PROMISE | 440 | 23 | 71 | 369 | 16.14% |
| EQ | PROMISE | 324 | 61 | 129 | 195 | 39.81% |
| JDT | PROMISE | 997 | 61 | 206 | 791 | 20.66% |
| LC | PROMISE | 691 | 61 | 64 | 627 | 9.26% |
| ML | PROMISE | 1862 | 61 | 245 | 1617 | 13.16% |
| PDE | PROMISE | 1497 | 61 | 209 | 1288 | 13.96% |

### Observations
- **Prevalence range**: 3.83% (camel-1.0) to 39.81% (EQ)
- **Most datasets are highly imbalanced** (defective < 15% of modules)
- **EQ and JDT** have the highest prevalence (~40% and ~20% respectively)
- **camel-1.0** is the most imbalanced (3.83% defective)
- Feature counts range from 20 to 61 features per dataset

### Data Quality Metrics
| Dataset | Missing % | Duplicates | Constant Features |
|---------|-----------|------------|-------------------|
| ant-1.5 | 0.0% | 0 | 0 |
| berek | 0.0% | 0 | 2 |
| camel-1.0 | 0.0% | 0 | 2 |
| data_arc | 0.0% | 14 | 0 |
| data_ivy-2.0 | 0.0% | 7 | 0 |
| jedit-3.2 | 0.0% | 0 | 2 |
| synapse-1.1 | 0.0% | 0 | 0 |
| tomcat | 0.0% | 0 | 3 |
| xalan-2.4 | 0.0% | 0 | 2 |
| xerces-1.2 | 0.0% | 0 | 2 |
| EQ | 0.0% | 59 | 0 |
| JDT | 0.0% | 12 | 0 |
| LC | 0.0% | 6 | 3 |
| ML | 0.0% | 58 | 0 |
| PDE | 0.0% | 14 | 0 |

## Git Status and Recent Commits
- **Current branch**: main
- **Status**: Modified files in `src/cla_ca/` (scoring.py, pipeline.py, preprocessing.py, prediction.py)
- **Untracked**: `api/`, `dashboard.html`, `frontend/`, `results/` directory structure
- **Recent commits**: 32d4239 baseline
- **Modified files** (from git status):
  - `cla-ca-research-platform/src/cla_ca/pipeline.py`
  - `cla-ca-research-platform/src/cla_ca/prediction.py`
  - `cla-ca-research-platform/src/cla_ca/preprocessing.py`
  - `cla-ca-research-platform/src/cla_ca/scoring.py`

## Python Entry Points Identified
1. `src/cla_ca/pipeline.py` - `run_claca_pipeline(dataset_name, bim, alpha)` function
2. `api/main.py` - FastAPI endpoint `/api/run-experiment` accepting base64-encoded CSV
3. `run_platform.py` - Main runner that executes pipeline on all 15 datasets

## Algorithm Implementation

### Current CLA-CA Pipeline (per pipeline.py)
1. **Normalization**: Min-Max scaling (per paper specification)
2. **Feature Weighting**: Variance-based weighting with alpha=0.8 (Algorithm 2)
3. **Discretization**: Equal-width bins, BIM=40 (PROMISE) or BIM=155 (AEEEM)
4. **Split Detection**: Valley detection via `create_split()` - finds bins where adjacent bins have higher frequency
5. **Module Scoring**: Sum feature weights where discretized value > split point (strict `>` per paper Algorithm 1 fix)
6. **Prediction**: Median of UNIQUE scores as threshold; if score > threshold → defective
7. **Metrics**: Recall, Precision, F1, MCC, G-mean (sqrt(precision*recall)), specificity-based G-mean (sqrt(recall*specificity))

### Critical Code Fixes Already Applied
- **scoring.py line 16**: Changed `>=` to `>` per paper Algorithm 1: "if xi,index > spindex"
- **preprocessing.py**: Min-Max normalization (confirmed better than Z-score through testing)
- **prediction.py**: Uses median of UNIQUE scores (per paper: "Ks = unique(Ks); Labelpoint = median(Ks)")
- **discretization.py**: `pd.cut` with `include_lowest=True`, BIM=40/155 from sensitivity analysis

## Frozen Baseline Results
- Baseline summary stored in `results/baseline/summary.csv`
- 15 datasets executed with current algorithm
- Results show significant overprediction of defective modules for most datasets
- Key metric: F1 ranges from 0.14 to 0.875; MCC from 0.16 to 0.801
- `berek` dataset matches paper exactly (16 predicted vs 16 actual defective)

## Configuration and Dependencies
- **Normalization**: Min-Max Scaler (sklearn)
- **Discretization**: `pd.cut` equal-width bins
- **Feature weighting**: Variance-based with alpha=0.8
- **Valley detection**: `create_split()` - counts per bin, finds p>0 and q>0 conditions
- **No explicit configuration files** found; parameters hardcoded in pipeline code
- **No dependency manifest** (requirements.txt or similar) found in working directory

## Known Issues and Limitations
1. **Severe overprediction**: Many datasets predict far more defectives than actually exist (e.g., ant-1.5: 81 vs 32, tomcat: 171 vs 77)
2. **No input validation** on CSV uploads via API
3. **No deterministic seed handling** across runs
4. **No unit tests** exist in the repository
5. **No ablation study** framework
6. **Parameter choices** (BIM, alpha) not documented as configurable

## Recommendations for Phase 0 Completion
1. ✅ Repository structure documented
2. ✅ Dataset inventory compiled
3. ✅ Algorithm implementation audited
4. ✅ Baseline results verified and frozen
5. ⚠️ Create `baseline_reproducibility.md` with input hashes and execution logs
6. ⚠️ Create `paper_implementation_audit.md` mapping paper algorithms to code
7. ⚠️ Verify repeated executions produce identical results
8. ⚠️ Store input dataset hashes for reproducibility

---
*Audit performed as Phase 0 requirement per project specification Sections 1-15*