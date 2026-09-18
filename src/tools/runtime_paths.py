"""Runtime output locations configured by the public entry point."""

from pathlib import Path

_checkpoint_dir = None
_result_dir = None


def configure(checkpoint_dir, result_dir):
    global _checkpoint_dir, _result_dir
    _checkpoint_dir = Path(checkpoint_dir).resolve()
    _result_dir = Path(result_dir).resolve()
    _checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _result_dir.mkdir(parents=True, exist_ok=True)


def checkpoint_file(filename):
    if _checkpoint_dir is None:
        raise RuntimeError("Runtime checkpoint directory is not configured.")
    return _checkpoint_dir / filename


def result_file(filename):
    if _result_dir is None:
        raise RuntimeError("Runtime result directory is not configured.")
    return _result_dir / filename
