# Metric units for locally generated analyses

ACC, NMI, ARI, Purity and auxiliary FMI are computed from predicted clusters and reference labels. The analysis CSV uses the 0--1 scale; displayed percentages and differences use the 0--100 scale. Sample SD uses ddof=1. These are reporting conventions, not experiment findings.

The unit of replication is an independent full training run on fixed data, not an individual cell or donor. Runtime is in seconds and PyTorch allocated/reserved memory is in MiB, with the scopes stated in the analysis plan.
