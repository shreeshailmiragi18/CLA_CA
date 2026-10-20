---
name: cla-ca-protocol
description: Experimental protocol and assumptions for reproducing CLA-CA
metadata:
  type: project
---

# Experimental Protocol and Assumptions

1. **Datasets**: 
   - PROMISE (10 projects: `ant-1.5`, `arc`, `berek`, `camel-1.0`, `ivy-2.0`, `jedit-3.2`, `synapse-1.1`, `tomcat`, `xalan-2.4`, `xerces-1.2`) with `bim = 35`. Note: filenames in our dataset folder correspond to these projects (e.g., `data_arc.csv`, `data_ivy-2.0.csv`, etc.).
   - AEEEM (5 projects: `EQ`, `JDT`, `LC` (Lucene), `ML` (Mylyn), `PDE`) with `bim = 150`.
2. **Parameters**: Alpha = 0.8.
3. **Normalization**: Min-Max or Z-score (paper specifies Z-score normalization filter / normalization in SetWeight).
