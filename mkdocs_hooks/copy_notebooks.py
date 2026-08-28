"""Make the example notebooks part of the documentation site.

The notebooks live in ``examples/`` at the repository root, where users of a clone
expect them, but MkDocs only ever looks inside ``docs_dir``. This hook copies them
into ``docs/examples/`` before the file tree is collected, so ``mkdocs-jupyter`` can
render (and execute) them as ordinary pages. The copies are build artefacts and are
git-ignored; ``examples/`` stays the single source of truth.

The hook itself lives outside ``docs/`` on purpose: mkdocs-jupyter converts the .py
files it finds in the docs directory, and would try to execute this one as a notebook.
"""
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SOURCE_DIR = REPO_ROOT / "examples"
TARGET_DIR = REPO_ROOT / "docs" / "examples"


def on_pre_build(config, **kwargs):
	"""Refresh docs/examples/ with the notebooks from examples/."""
	TARGET_DIR.mkdir(exist_ok=True)
	for notebook in sorted(SOURCE_DIR.glob("*.ipynb")):
		shutil.copy2(notebook, TARGET_DIR / notebook.name)
