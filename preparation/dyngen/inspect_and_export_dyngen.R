#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(Matrix))

input_path <- file.path(
  "downloaded_data",
  "dyngen_branching",
  "dyngen_branching_compatible_seed1.rds"
)
output_dir <- file.path(
  "downloaded_data",
  "dyngen_branching",
  "exported"
)

if (!file.exists(input_path)) {
  stop("Missing RDS: ", input_path)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

obj <- readRDS(input_path)

premrna <- obj$model$experiment$counts_premrna
mrna <- obj$model$experiment$counts_mrna
protein <- obj$model$experiment$counts_protein
rna <- premrna + mrna

matrices <- list(
  rna = rna,
  premrna = premrna,
  mrna = mrna,
  adt = protein
)

for (name in names(matrices)) {
  matrix <- as(matrices[[name]], "dgCMatrix")
  if (!identical(as.integer(dim(matrix)), c(1000L, 600L))) {
    stop(
      name,
      " dimensions are ",
      paste(dim(matrix), collapse = " x "),
      ", expected 1000 x 600."
    )
  }
  Matrix::writeMM(
    matrix,
    file.path(output_dir, paste0(name, ".mtx"))
  )
}

barcodes <- rownames(premrna)
if (is.null(barcodes)) {
  barcodes <- sprintf("dyngen_cell_%04d", seq_len(nrow(premrna)))
}
features <- colnames(premrna)
if (is.null(features)) {
  features <- sprintf("dyngen_feature_%04d", seq_len(ncol(premrna)))
}

writeLines(barcodes, file.path(output_dir, "barcodes.txt"))
writeLines(features, file.path(output_dir, "features.txt"))

candidate_cell_info <- list(
  top_level_cell_info = obj$cell_info,
  model_cell_info = obj$model$cell_info,
  simulation_cell_info = obj$model$simulation$cell_info,
  experiment_cell_info = obj$model$experiment$cell_info
)

cell_info <- NULL
source_name <- NULL
for (name in names(candidate_cell_info)) {
  candidate <- candidate_cell_info[[name]]
  if (!is.null(candidate) && is.data.frame(candidate)) {
    if (nrow(candidate) == nrow(premrna)) {
      cell_info <- candidate
      source_name <- name
      break
    }
  }
}

if (is.null(cell_info)) {
  cell_info <- data.frame(
    cell_barcode = barcodes,
    stringsAsFactors = FALSE
  )
  source_name <- "generated_barcode_only"
} else {
  cell_info <- as.data.frame(cell_info)
  if (!"cell_barcode" %in% colnames(cell_info)) {
    cell_info$cell_barcode <- barcodes
  }
}

write.csv(
  cell_info,
  file.path(output_dir, "cell_metadata.csv"),
  row.names = FALSE
)

metadata <- list(
  source_rds = input_path,
  export_directory = output_dir,
  cell_info_source = source_name,
  cell_metadata_columns = colnames(cell_info),
  modalities = names(matrices),
  dimensions = lapply(matrices, dim),
  scientific_boundary = paste(
    "The exact GeDGC dyngen file/configuration was not public.",
    "This is a fixed-seed compatible simulation."
  )
)
dput(metadata, file.path(output_dir, "export_info.R"))

cat("\nDYNGEN EXPORT COMPLETE\n")
cat("Output:", output_dir, "\n")
cat("Cell metadata source:", source_name, "\n")
cat("Cell metadata columns:\n")
print(colnames(cell_info))
cat(
  "\nChoose a biologically meaningful label column that yields 3 classes, ",
  "then pass it to build_dyngen_h5mu.py.\n",
  sep = ""
)
