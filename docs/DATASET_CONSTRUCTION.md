# Dataset construction and input identity

This page gives the data explanations associated with the controlled benchmark. The manuscript keeps the basic input table, corrected references and availability pointer. Detailed construction, pairing, annotation use and molecular-view definitions are documented here and in [data/README.md](../data/README.md).

## Current access status

Four smaller exact inputs remain in the source tree. Pbmc10k, opmultiome and TEA-seq, annotation-corrected Kidney and the compatible dyngen file are prepared as versioned assets. The release contains processed data assets and checksums; it does not contain experimental result archives. [Release status](../data/revision_release_status.json) and [manifest](../data/dataset_manifest.csv) distinguish input identities from publication state. Asset publication state is verified through the live release API rather than inferred from reserved URLs.

## Kidney pairing and use of labels

The actual source is paired sci-CAR adult mouse kidney from Cao et al., Science 2018, DOI [10.1126/science.aau0730](https://doi.org/10.1126/science.aau0730), GEO GSE117089, RNA GSM3271044 and ATAC GSM3271045. The old attribution to Muto 2021 human kidney was incorrect. It should not be used to infer that separately assayed human libraries were converted to paired nuclei.

The recovered constructor joins RNA/ATAC by shared raw sample barcode. RNA cell_name annotation removes missing labels and guides proportional stratified subsampling of 8,837 paired labelled cells to 4,753 with NumPy default_rng(0). Annotation does not computationally match the two modalities. Features are selected by raw-count variance in selected cells, giving 5,000 RNA and 10,000 ATAC columns. This is the author's constructed benchmark and not proof of identity with unpublished original GeDGC preparation.

An initial CSV-header error shifted feature IDs by one source row. The original experiment input is retained with SHA256 c095bef25a489e7a9d7462827225a91a027004c9ab276e3c309f375dafc837c9. The separately versioned corrected file has SHA256 8d85673284c34cd0d1bb77a21acb1474a03d7967870b5750a7db8a19c9eea14d. Matrix CSR structure/values, numeric column order, cells and labels are identical; corrected gene/peak IDs and raw-source mappings were independently checked, including reconstruction from the six original files. See [constructor](../preparation/kidney/create_kidney_h5mu_header_fixed.py) and [verification records](../data/provenance/).

## Exact real-data inputs

The seven frozen inputs are described by dimensions and SHA256 in [the manifest](../data/dataset_manifest.csv) and [the portable protocol](../reproduction/protocol_template.json). Chen-2019 is the SNARE-seq mouse-brain benchmark; Pbmc10k derives from GLUE-distributed paired 10x Multiome PBMC material; opmultiome uses GSE194122 site1_donor1_multiome; multiome uses the documented GSE194122 batches. Direct release of the exact final processed inputs is the reproducibility route where historical raw-to-final preparation was not independently rebuilt. Matching reported dimensions is not evidence of original-paper byte identity.

The original-preparation alternative and direct-processed-input alternative are distinguished. This update supplies actual experiment files, recorded source identities and recovered preparation programs, without pretending every missing historical intermediate was recovered.

## TEA-seq reconstruction

Public processed RNA/ATAC/ADT material corresponds to GSE158013/GSM5123951 and the official eLife 63632 Figure 4 source-data 2 cell-type table. Recovered preparation scripts use label suffix -3 and the three-view/label barcode intersection, initially 6,314 cells with 13 broad labels. RNA/ATAC features detected in at least 1% of retained cells are kept. This study's final reconstruction excludes DC.Plasmacytoid (3 cells) and T.DoubleNegative (1 cell), then selects 14,732 ATAC features by sparse TF-IDF variance while retaining their original counts. Final identity is 6,310 cells, 11 categories and 9,855/14,732/46 features.

These exclusions and peak-selection rules belong to this study's reconstruction. They are not asserted to be the unpublished original GeDGC rules. Exact final processed input SHA256 is 393a4f21785a89c50a6ad4004bd83aee929cd9713c90c3066fc8e6582054a0ac. [Preparation scripts](../preparation/teaseq/) document the recovered steps. Historical intermediate data/reports were not independently recovered during this audit; the author's consistency confirmation is distinct from independent numerical checks. Controlled raw access under dbGaP phs002316 remains subject to its source conditions and is not bypassed.

## Compatible dyngen and the adt key

| Stored key | Generator quantity | Meaning |
|---|---|---|
| premrna | counts_premrna | Simulated pre-mRNA |
| mrna | counts_mrna | Simulated mature mRNA |
| rna | premrna + mrna | Derived total RNA |
| adt | counts_protein | Simulated protein |

The adt key is not an actual antibody-derived-tag experiment. Total RNA is deterministically derived from two other views, so the four views are not four independent measured omics.

The recovered generator uses dyngen 1.1.1, simulation seed 1, backbone_branching, a local 1,200-node weighted synthetic network and a local 2,000-by-600 reference count matrix, with no external empirical-network cache. The RNG sequence includes network/reference construction before simulation. The actual 1,000-cell input has 600 features per view (67 TF / 266 target / 267 housekeeping) and 13 from-to transition classes. No classes are merged to force 11 or 3. Simulation seed 1 differs from training seeds 0–4.

[Generator/export/final conversion](../preparation/dyngen/) and verification identify exact feature IDs/order, shared cell/label order, nonnegative integer counts and entry-wise RNA addition. The historical RDS/exports were not independently recovered for a fresh per-value protein comparison; the author's consistency confirmation is retained separately. The compatible simulation is not represented as the exact unpublished original GeDGC dyngen file. Its final SHA256 is c02241529fe3683027fb56c3dc824ee66f24527e13483f4ded3bd3cded112d89.

## Training and analysis distinction

Input labels provide the known cluster count and evaluation references. Classifier pretraining uses GMM pseudo-labels. The execution configurations keep data preparation, training seed, simulation seed and local output identity separate. See [reproduction documentation](../reproduction/README.md).
