---
name: cla-ca-assumptions
description: Documented implementation assumptions for CLA-CA
metadata:
  type: project
---

# Implementation Assumptions

1. **Normalization method**: Paper mentions normalization in SetWeight (Z-score or Min-Max). We implement Z-score normalization as standard for software defect prediction datasets.
2. **Missing values**: Handled by filling NaN with column median or mean.
3. **Target column identification**: Automatically identifying target column (commonly named `bug`, `defects`, `isDefective`, `class`, or last column).
