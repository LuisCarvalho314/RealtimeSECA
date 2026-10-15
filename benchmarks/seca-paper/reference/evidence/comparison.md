# Shared reconstructed evaluator: C# versus Rust

Scores are percentages; differences are percentage points. Published values are
12-cycle means, including in per-cycle rows. — denotes an unavailable metric.

| Theme / metric | Scoreable paired cycles | C# paired | Rust paired | Abs. Δ | Published | C# paired − published | Rust paired − published | Rust full |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| G / nmi | 1,2 | 86.48 | 86.25 | 0.22 | 88.30 | -1.82 | -2.05 | 87.17 |
| G / bcubed_f1 | 1,2 | 59.55 | 59.33 | 0.23 | 63.03 | -3.48 | -3.70 | 60.89 |
| G / purity | 1,2 | 76.28 | 75.88 | 0.41 | 80.36 | -4.08 | -4.48 | 79.77 |
| H / nmi | 1,2,3,5 | 83.43 | 83.05 | 0.37 | 87.52 | -4.09 | -4.47 | 85.25 |
| H / bcubed_f1 | 1,2,3,5 | 66.90 | 66.82 | 0.08 | 71.14 | -4.24 | -4.32 | 71.05 |
| H / purity | 1,2,3,5 | 90.80 | 90.34 | 0.46 | 89.18 | 1.62 | 1.16 | 93.05 |

Partial paired means are not estimates of missing cycles or replacements for the published average.

| Theme / cycle | C# NMI | Rust NMI | C# F1 | Rust F1 | C# purity | Rust purity | C# status |
|---|---:|---:|---:|---:|---:|---:|---|
| G / 1 | 85.53 | 85.07 | 57.44 | 56.36 | 72.30 | 71.49 | scored |
| G / 2 | 87.42 | 87.44 | 61.66 | 62.29 | 80.27 | 80.27 | scored |
| G / 3 | — | 86.61 | — | 59.00 | — | 78.24 | clustering_input_failure |
| G / 4 | — | 86.91 | — | 60.35 | — | 78.92 | clustering_input_failure |
| G / 5 | — | 87.05 | — | 59.63 | — | 81.22 | clustering_input_failure |
| G / 6 | — | 86.69 | — | 59.64 | — | 77.84 | clustering_input_failure |
| G / 7 | — | 86.99 | — | 60.83 | — | 79.86 | clustering_input_failure |
| G / 8 | — | 87.71 | — | 62.39 | — | 82.16 | clustering_input_failure |
| G / 9 | — | 87.05 | — | 60.34 | — | 80.95 | clustering_input_failure |
| G / 10 | — | 87.87 | — | 62.50 | — | 82.03 | clustering_input_failure |
| G / 11 | — | 87.81 | — | 62.30 | — | 82.03 | clustering_input_failure |
| G / 12 | — | 88.79 | — | 65.02 | — | 82.30 | clustering_input_failure |
| H / 1 | 84.09 | 84.30 | 69.27 | 70.71 | 90.10 | 89.95 | scored |
| H / 2 | 83.88 | 83.21 | 69.77 | 69.36 | 89.05 | 88.35 | scored |
| H / 3 | 83.77 | 83.44 | 66.88 | 66.50 | 92.30 | 92.10 | scored |
| H / 4 | — | 80.34 | — | 60.54 | — | 87.55 | clustering_input_failure |
| H / 5 | 81.97 | 81.27 | 61.69 | 60.71 | 91.75 | 90.95 | scored |
| H / 6 | — | 84.42 | — | 68.82 | — | 95.60 | clustering_input_failure |
| H / 7 | — | 86.49 | — | 70.78 | — | 93.55 | clustering_input_failure |
| H / 8 | — | 88.16 | — | 75.96 | — | 94.80 | clustering_input_failure |
| H / 9 | — | 88.28 | — | 77.00 | — | 94.25 | clustering_input_failure |
| H / 10 | — | 88.65 | — | 77.83 | — | 96.40 | clustering_input_failure |
| H / 11 | — | 86.13 | — | 73.39 | — | 96.90 | clustering_input_failure |
| H / 12 | — | 88.30 | — | 81.03 | — | 96.20 | clustering_input_failure |
