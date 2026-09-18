# Checkpoints

Generated files are stored at `checkpoints/<dataset>/seed_<seed>/` and ignored
by Git. A complete reusable pretraining set contains:

- `pretrained_vae.pkl`
- `pretrained_GMM.pkl`
- `pretrained_classifier.pkl`

Place historical files either in that canonical seed directory, directly in
`checkpoints/<dataset>/`, or in a historical directory named by
`legacy_checkpoint_name` in `configs/datasets.json`. Unlabelled historical
directories are treated as seed 0 only; seed-specific repeats must use their own
canonical directories or pretrain afresh. `run.py` recognizes a complete set and
copies it into the canonical seed directory. An incomplete set is not mixed
with a new run: the entry point performs fresh pretraining.

PyTorch full-object pickle files depend on the retained `module.*` import paths
and are unsafe when obtained from untrusted sources. Only load checkpoints you
created or received from a trusted archive.
