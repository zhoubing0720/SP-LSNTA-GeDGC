# Processed data: exact experiment files and preparation

**Versioned processed inputs:** the files and SHA256 checksums have been verified. Check the linked Release and live API status for public access; four smaller exact inputs remain hosted in the source tree.

Target versioned assets: https://github.com/zhoubing0720/SP-LSNTA-GeDGC/releases/tag/v1.0.0. Input identities are listed in dataset_manifest.csv and reproduction/protocol_template.json. Use scripts/download_processed_revision.py and scripts/verify_data.py --scope all.

## Kidney

The actual benchmark is sci-CAR adult **mouse** kidney from Cao et al., Science (2018), DOI 10.1126/science.aau0730; GEO GSE117089, RNA GSM3271044 and ATAC GSM3271045. Earlier attribution to Muto's separately assayed human kidney data was incorrect.

The recovered constructor matches RNA/ATAC by shared raw sample barcode. RNA cell_name annotation is used to remove missing labels and for proportional stratified subsampling of 8,837 paired labelled cells to 4,753 (NumPy default_rng(0)); labels are not used to computationally match modalities. Feature selection uses raw-count variance in selected cells, 5,000 RNA and 10,000 ATAC features. This is an author-prepared benchmark with a size based on GeDGC, not evidence of identity with unpublished original GeDGC processing.

The recovered reader initially treated CSV headers as feature IDs, shifting all selected IDs by one source row. `data/processed/Kidney.h5mu` retains the exact frozen input checksum c095bef25a489e7a9d7462827225a91a027004c9ab276e3c309f375dafc837c9. The corrected release asset `Kidney_4753_annotations_v1.h5mu` has checksum 8d85673284c34cd0d1bb77a21acb1474a03d7967870b5750a7db8a19c9eea14d, corrected feature annotations and source-column mappings. CSR contents, cell order, numeric feature-column order and labels are identical; raw-source reconstruction also passed. Reproduction of original byte hashes uses the legacy file; annotation-dependent downstream work should use the corrected version.

Constructor: preparation/kidney/create_kidney_h5mu_header_fixed.py. Run with --raw-dir pointing to the six unpacked GEO files and --output pointing to a new file; no overwrite is permitted. The returned matrices and feature IDs were independently checked against the corrected input.

## Other real datasets

- Chen-2019: paired SNARE-seq mouse brain, GSE126074. The exact file remains hosted in data/processed.
- Pbmc10k: annotated paired RNA/ATAC files distributed by GLUE for 10x Multiome PBMC10k. Exact 9,631-cell, 19-class input is prepared as a release asset (versioned asset).
- opmultiome: GSE194122, site1_donor1_multiome; exact 6,137-cell, 21-class input is prepared as a release asset (versioned asset).
- multiome: GSE194122 processed benchmark, 16,847 cells/21 classes; exact input remains hosted.
- TEA-seq: public processed GSE158013/GSM5123951 RNA/ATAC/ADT and Figure4_SourceData2_TypeLabelsUMAP.csv from eLife 63632 Figure 4 source data 2. The recovered pipeline uses official label suffix -3, takes the three-modality plus label barcode intersection (6,314 cells/13 broad classes), and filters RNA/ATAC features detected in at least 1% of retained cells. The final author reconstruction removes DC.Plasmacytoid (3 cells) and T.DoubleNegative (1 cell), then selects 14,732 ATAC features by sparse TF-IDF variance while retaining their original counts. Final dimensions are 6,310 cells, 11 classes and 9,855/14,732/46 features. These are this study's reconstruction rules; they are not proof that the original GeDGC authors used the same exclusions or peaks. The exact final processed file is supplied directly. Controlled raw data under dbGaP phs002316.v1.p1 remain subject to their original access route.

Preparation code supports provenance and future construction. Missing historical intermediate TEA/dyngen objects were not independently recovered in this audit; direct release of the exact final inputs enables use of the same files without asserting an independent numerical re-creation of those historical intermediate objects.

## Compatible dyngen: four views

| Key | Definition | Interpretation |
|---|---|---|
| premrna | model$experiment$counts_premrna | Simulated pre-mRNA counts |
| mrna | model$experiment$counts_mrna | Simulated mature mRNA counts |
| rna | premrna + mrna | Derived total RNA counts |
| adt | model$experiment$counts_protein | Simulated protein counts; ADT is a storage key |

This synthetic ADT view is not an actual antibody-derived-tag measurement. The four views are correlated and total RNA is deterministically derived from two components. The generator uses dyngen 1.1.1, simulation seed=1, backbone_branching and a locally generated weighted 1,200-node synthetic GRN, without downloading an empirical-network cache. It produces 1,000 cells with 600 features per view (67 TF/266 targets/267 housekeeping). Snapshot reference CPM/library-size mapping is disabled. Actual labels are 13 from->to trajectory transition classes; no merging is used to force 11 or 3. Simulation seed=1 is distinct from training seeds 0--4.

Recovered generator, exporter and final converter: preparation/dyngen. Run them sequentially in a NEW empty working directory. The old exporter's final prompt about 3 classes is superseded by build_dyngen_compatible_actual_classes.py. Independent checks confirm feature IDs/order, view cell/label ordering, nonnegative integer counts and exact total-RNA addition. The original RDS-to-current-protein numerical identity was not independently rechecked; this limitation is preserved rather than described as a new completed audit. This dataset is an author compatible simulation, not an exact original GeDGC file.

## What is shared

Four existing frozen files stay hosted in the source tree. Release assets provide Pbmc10k.h5mu, opmultiome.h5mu, TEA_seq.h5mu, the exact compatible dyngen file, and the corrected Kidney annotations. Original source identity/conditions, construction scripts, explicit input hashes accompany them. All binary uploads have passed server SHA256 verification; check the live release API to establish current public access. No DOI is claimed. Data licences are not newly assigned by this code release.
