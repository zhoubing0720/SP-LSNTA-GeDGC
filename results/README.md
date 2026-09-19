# Results

`paper_results.csv` contains the GeDGC and SP-LSNTA-GeDGC values reported in
the manuscript for the seven study datasets. The values are provided for direct
comparison with the manuscript; they are not a substitute for independently
rerunning the code. All ACC, NMI, ARI, and Purity values in this file are
**percentages** on a 0--100 scale.

New execution outputs are written to `results/<dataset>/seed_<seed>/` and are
ignored by Git. Each completed run produces per-seed metrics and dataset-level
summary files. Runtime clustering metrics are proportions on a 0--1 scale, while
`Time` is measured in seconds. Multiply runtime clustering metrics by 100 before
comparing them with `paper_results.csv`.

For repeated runs, `runs.csv` contains one row per seed and `summary.csv`
contains the population mean and population standard deviation across those
seeds. Each seed directory also contains `run_config.json`, which records the
resolved dataset specification, fixed experiment settings, seed, and checkpoint
reuse status.
