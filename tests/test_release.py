import json
import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_PROCESSED_DATA = {
    Path("data/processed/Chen-2019.h5mu"),
    Path("data/processed/Kidney.h5mu"),
    Path("data/processed/multiome.h5mu"),
    Path("data/processed/dyngen_branching.h5mu"),
}


class ReleaseTests(unittest.TestCase):
    def test_public_metadata_matches_current_manuscript(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        title = (
            "SP-LSNTA-GeDGC: Task-Oriented Shared-Private Routing and "
            "Multilevel Structural Alignment for Single-Cell Multi-Omics Clustering"
        )
        self.assertIn(title, citation)
        self.assertIn("submission to *Cells*", readme)
        ordered_names = ["Shizheng", "Bing", "Yanbu", "Min", "Wanwei", "Dongyuan"]
        positions = [citation.index("given-names: " + name) for name in ordered_names]
        self.assertEqual(positions, sorted(positions))

    def test_dataset_registry_has_exactly_seven_entries(self):
        registry = json.loads((ROOT / "configs" / "datasets.json").read_text(encoding="utf-8"))
        self.assertEqual(
            list(registry),
            ["Chen-2019", "Kidney", "Pbmc10k", "opmultiome", "multiome", "TEA-seq", "dyngen_branching"],
        )

    def test_confirmed_nmodal_shapes_are_reflected_in_config(self):
        registry = json.loads((ROOT / "configs" / "datasets.json").read_text(encoding="utf-8"))
        self.assertEqual([x["name"] for x in registry["TEA-seq"]["modalities"]], ["rna", "atac", "adt"])
        self.assertEqual([x["name"] for x in registry["dyngen_branching"]["modalities"]], ["rna", "premrna", "mrna", "adt"])

    def test_cli_lists_datasets_without_scientific_dependencies(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "run.py"), "--list-datasets"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Chen-2019", completed.stdout)
        self.assertIn("dyngen_branching", completed.stdout)

    def test_no_unapproved_data_or_model_artifacts_are_in_tree(self):
        forbidden = {".h5ad", ".pkl", ".pt", ".pth", ".rar", ".zip"}
        artifacts = []
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if path.suffix.lower() == ".h5mu" and relative not in ALLOWED_PROCESSED_DATA:
                artifacts.append(relative)
            elif path.suffix.lower() in forbidden:
                artifacts.append(relative)
        self.assertEqual(artifacts, [])

    def test_gitignore_whitelists_only_selected_processed_datasets(self):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for path in ALLOWED_PROCESSED_DATA:
            self.assertIn("!" + path.as_posix(), text)
        for filename in ("Pbmc10k.h5mu", "opmultiome.h5mu", "TEA-seq.h5mu"):
            self.assertNotIn("!data/processed/" + filename, text)

    def test_five_run_csv_summary(self):
        spec = importlib.util.spec_from_file_location("release_run", ROOT / "run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = []
        for seed in range(5):
            row = {"Dataset": "Chen-2019", "Seed": seed}
            row.update({field: float(seed) for field in module.METRIC_FIELDS})
            rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            module.write_summary("Chen-2019", rows, Path(directory))
            with (Path(directory) / "Chen-2019" / "summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                summary = next(csv.DictReader(handle))
            self.assertEqual(summary["Runs"], "5")
            self.assertEqual(float(summary["ACC_mean"]), 2.0)

    def test_complete_historical_checkpoint_set_is_reused(self):
        spec = importlib.util.spec_from_file_location("release_run", ROOT / "run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            historical = base / "historical"
            canonical = base / "canonical"
            historical.mkdir()
            for filename in module.CHECKPOINT_FILENAMES:
                (historical / filename).write_bytes(filename.encode("ascii"))
            reusable, source = module.prepare_checkpoints(
                canonical, [historical], force_pretrain=False
            )
            self.assertTrue(reusable)
            self.assertEqual(source, historical)
            self.assertTrue(module.checkpoint_complete(canonical))


if __name__ == "__main__":
    unittest.main()
