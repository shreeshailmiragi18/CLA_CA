---
name: cla-ca-research-spec
description: Research specification and paper analysis for CLA-CA (A Novel Unsupervised Method for Software Defect Prediction Based on Clustering Assumption)
metadata:
  type: project
---

# CLA-CA Research Specification & Analysis

Based on the research paper **"A Novel Unsupervised Method for Software Defect Prediction Based on Clustering Assumption"** (Shiqi Tang et al., IEEE Access 2026):

## 1. Core Algorithm Summary
CLA-CA is an unsupervised software defect prediction (UDP) method with two main stages:
1. **Score Calculation Stage**:
   - **Step 1: Set Weight**: Normalize features, compute variance of each feature (`Vars[i]`), find `maxVar`. Calculate weight `W[i]`:
     - If `maxVar / Vars[i] > 1.5`, `W[i] = e` (Euler's number $\approx 2.71828$).
     - If `maxVar / Vars[i] <= 1.5`, `W[i] = e + alpha * (Vars[i] / maxVar)`, where `alpha = 0.8`.
   - **Step 2: Discretization**: Equal-width discretization into `bim` regions.
     - PROMISE datasets: `bim = 35`.
     - AEEEM datasets: `bim = 150`.
   - **Step 3: Feature Split (Createsplit / Valley Detection)**:
     - For each feature, count samples in each region (`num[i]` for $i \in \{1, \dots, \text{bim}\}$).
     - Find candidate split points for $i \in \{2, \dots, \text{bim}-1\}$ where $p = \text{num}[i+1] - \text{num}[i] > 0$ and $q = \text{num}[i-1] - \text{num}[i] > 0$ (a valley where adjacent bins have higher frequencies than the valley bin).
     - Val = $p + q$.
     - Select `maxindex` maximizing Val. The split point index `sp` is this index.
     - If no candidate valley exists (`Splist.size == 0`), return `-1` (feature is considered redundant and excluded from scoring).
   - **Step 4: Module Scoring**:
     - For each module $X_i$, score $k_i = \sum_{j \in \text{Splist}} (\text{if } x_{i, \text{index}} > \text{spindex} \text{ then } W[\text{index}] \text{ else } 0)$ — wait, let's verify exact scoring condition in Algorithm 1.
2. **Labeling Stage**:
   - Extract unique scores `Ks = unique(Ks)`.
   - Threshold `Labelpoint = median(Ks)`.
   - If score $k > \text{Labelpoint}$, label as Defective (`defective`), else Non-Defective (`non-defective`).

## 2. Evaluation Metrics
- Accuracy, Precision, Recall, F1-measure, MCC, G-mean ($\sqrt{\text{precision} \times \text{recall}}$).
- Protocol: 10 runs (though deterministic for CLA-CA, averaged).
