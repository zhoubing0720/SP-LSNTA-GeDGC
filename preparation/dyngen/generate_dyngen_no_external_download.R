user_lib <- Sys.getenv("R_LIBS_USER")
if (!nzchar(user_lib)) {
  user_lib <- file.path(
    Sys.getenv("LOCALAPPDATA"),
    "R",
    "win-library",
    paste(
      R.version$major,
      strsplit(R.version$minor, "\\.")[[1]][1],
      sep = "."
    )
  )
}

dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))

if (!requireNamespace("dyngen", quietly = TRUE)) {
  stop(
    "dyngen is not installed in the current user's R library. ",
    "Run the previous dyngen installation tool first."
  )
}
if (!requireNamespace("Matrix", quietly = TRUE)) {
  stop("The Matrix package is required but is not installed.")
}

suppressPackageStartupMessages(library(dyngen, lib.loc = user_lib))

set.seed(1)

output_dir <- file.path("downloaded_data", "dyngen_branching")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("============================================================\n")
cat("DYNGEN BRANCHING COMPATIBLE GENERATOR - NO EXTERNAL CACHE\n")
cat("============================================================\n")
cat("dyngen version:", as.character(packageVersion("dyngen")), "\n")
cat("User library:", user_lib, "\n\n")

# -------------------------------------------------------------------------
# 1. Build a local weighted sparse GRN.
#
# dyngen officially accepts a custom weighted sparse matrix as `realnet`.
# A directed ring guarantees connectivity; random positive edges add
# heterogeneous local structure. This avoids downloading the large
# data_files archive and avoids corrupted realnet cache files.
# -------------------------------------------------------------------------
num_realnet_nodes <- 1200L
node_names <- sprintf("G%04d", seq_len(num_realnet_nodes))

ring_i <- seq_len(num_realnet_nodes)
ring_j <- c(seq.int(2L, num_realnet_nodes), 1L)
ring_x <- rep(1.0, num_realnet_nodes)

random_edges_per_node <- 8L
random_i <- rep(seq_len(num_realnet_nodes), each = random_edges_per_node)
random_j <- sample.int(
  num_realnet_nodes,
  length(random_i),
  replace = TRUE
)
random_x <- runif(length(random_i), min = 0.1, max = 1.0)

realnet <- Matrix::sparseMatrix(
  i = c(ring_i, random_i),
  j = c(ring_j, random_j),
  x = c(ring_x, random_x),
  dims = c(num_realnet_nodes, num_realnet_nodes),
  dimnames = list(node_names, node_names),
  giveCsparse = TRUE
)
realnet <- Matrix::drop0(realnet)

cat(
  "Local sparse GRN:",
  nrow(realnet), "x", ncol(realnet),
  "nonzero edges =", length(realnet@x), "\n"
)

# -------------------------------------------------------------------------
# 2. Build a local sparse reference count matrix.
#
# generate_experiment() requires a realcount matrix even when CPM/library
# size mapping is disabled. This local matrix prevents a second network
# download. It is used only as a valid reference object; the mapping flags
# below are both FALSE.
# -------------------------------------------------------------------------
num_reference_cells <- 2000L
num_features <- 600L
nonzero_per_cell <- 24L

rc_i <- rep(seq_len(num_reference_cells), each = nonzero_per_cell)
rc_j <- unlist(
  replicate(
    num_reference_cells,
    sample.int(num_features, nonzero_per_cell, replace = FALSE),
    simplify = FALSE
  ),
  use.names = FALSE
)
rc_x <- stats::rpois(length(rc_i), lambda = 5) + 1L

realcount <- Matrix::sparseMatrix(
  i = rc_i,
  j = rc_j,
  x = rc_x,
  dims = c(num_reference_cells, num_features),
  dimnames = list(
    sprintf("reference_cell_%04d", seq_len(num_reference_cells)),
    sprintf("feature_%04d", seq_len(num_features))
  ),
  giveCsparse = TRUE
)

cat(
  "Local sparse reference counts:",
  nrow(realcount), "x", ncol(realcount),
  "nonzero entries =", length(realcount@x), "\n\n"
)

# -------------------------------------------------------------------------
# 3. Configure and run dyngen.
# -------------------------------------------------------------------------
backbone <- backbone_branching()
num_tfs <- nrow(backbone$module_info)
num_targets <- floor((num_features - num_tfs) / 2)
num_hks <- num_features - num_tfs - num_targets

cat(
  "Requested simulation:",
  "cells=1000",
  paste0("features=", num_features),
  paste0("TFs=", num_tfs),
  paste0("targets=", num_targets),
  paste0("HKs=", num_hks),
  "\n\n"
)

config <- initialise_model(
  backbone = backbone,
  num_cells = 1000L,
  num_tfs = num_tfs,
  num_targets = num_targets,
  num_hks = num_hks,
  feature_network_params = feature_network_default(
    realnet = realnet,
    damping = 0.01,
    target_resampling = 100L,
    max_in_degree = 5L
  ),
  simulation_params = simulation_default(
    total_time = 1000,
    census_interval = 2,
    ssa_algorithm = ssa_etl(tau = 300 / 3600),
    experiment_params = simulation_type_wild_type(
      num_simulations = 100
    ),
    compute_rna_velocity = TRUE
  ),
  experiment_params = experiment_snapshot(
    realcount = realcount,
    map_reference_cpm = FALSE,
    map_reference_ls = FALSE,
    weight_bw = 0.1
  ),
  verbose = TRUE,
  download_cache_dir = NULL
)

out <- generate_dataset(
  config,
  format = "list",
  make_plots = FALSE,
  store_dimred = TRUE,
  store_cellwise_grn = FALSE,
  store_rna_velocity = TRUE
)

# -------------------------------------------------------------------------
# 4. Strict dimensional validation.
# -------------------------------------------------------------------------
premrna <- out$model$experiment$counts_premrna
mrna <- out$model$experiment$counts_mrna
protein <- out$model$experiment$counts_protein

expected_dim <- c(1000L, 600L)
actual_dims <- list(
  premrna = dim(premrna),
  mrna = dim(mrna),
  protein = dim(protein)
)

for (name in names(actual_dims)) {
  if (!identical(as.integer(actual_dims[[name]]), expected_dim)) {
    stop(
      "Unexpected ", name, " dimensions: ",
      paste(actual_dims[[name]], collapse = " x "),
      "; expected 1000 x 600."
    )
  }
}

output_path <- file.path(
  output_dir,
  "dyngen_branching_compatible_seed1.rds"
)
saveRDS(out, output_path, compress = "xz")

info <- list(
  status = "compatible_replica_not_exact_paper_file",
  generator = "dyngen_custom_local_sparse_realnet",
  seed = 1L,
  num_cells = 1000L,
  num_features = 600L,
  num_tfs = num_tfs,
  num_targets = num_targets,
  num_hks = num_hks,
  backbone = "backbone_branching",
  dyngen_version = as.character(packageVersion("dyngen")),
  modalities_available = c(
    "counts_premrna",
    "counts_mrna",
    "counts_protein",
    "rna_velocity"
  ),
  external_cache_used = FALSE,
  note = paste(
    "The exact GeDGC dyngen simulation file/configuration is not published.",
    "This fixed-seed dataset uses an officially supported custom weighted",
    "sparse GRN and is shape-compatible, not bit-identical to the paper data."
  )
)

dput(
  info,
  file = file.path(
    output_dir,
    "dyngen_branching_compatible_seed1_info.R"
  )
)

cat("\n============================================================\n")
cat("DYNGEN COMPATIBLE DATASET GENERATED SUCCESSFULLY\n")
cat("Output:", output_path, "\n")
cat("pre-mRNA:", paste(dim(premrna), collapse = " x "), "\n")
cat("mRNA:", paste(dim(mrna), collapse = " x "), "\n")
cat("protein:", paste(dim(protein), collapse = " x "), "\n")
cat("External cache used: FALSE\n")
cat("============================================================\n")
