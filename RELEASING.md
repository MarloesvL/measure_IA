# Releasing MeasureIA

Maintainer notes. The version number lives in four places that must agree, and they have drifted before
(`pyproject.toml` sat at 0.1.0 while the tags and the Zenodo release were at v0.3.0), so work through this
list rather than from memory.

## Where the version lives

| Place | What it is | Updated by |
|---|---|---|
| `pyproject.toml` `version` | the source of truth | hand |
| `uv.lock` | mirrors it | `uv lock` (**not** `uv sync` — see below) |
| `measureia.__version__` | read from installed metadata | automatic |
| `CITATION.cff` `version` + `date-released` | what people cite | hand |
| `CITATION.cff` `identifiers` | the version DOI of the *previous* release | hand, after Zenodo mints the new one |
| git tag `vX.Y.Z` | what Zenodo archives | hand |

## Before releasing

1. **Everything green.** `uv run pytest` — no failures, no warnings (the suite runs with
   `filterwarnings = ["error", ...]`, so a new warning fails CI).
2. **Docs build clean.** `uv run --group docs mkdocs build --strict`.
3. **Examples run.** From `examples/`, run `example_measure_IA_box.py`, `example_read_and_plot.py` and
   `example_measure_IA_lightcone.py`. They use `measureia.mocks`, so they need no data.
4. **A clean install works.** Build and install the wheel into a fresh environment and import it — this is
   what catches a dependency that is not actually installable:

   ```bash
   uv build
   uv venv /tmp/relcheck --python 3.11 --seed
   /tmp/relcheck/bin/pip install dist/measureia-X.Y.Z-py3-none-any.whl
   /tmp/relcheck/bin/python -c "import measureia; print(measureia.__version__)"
   ```

5. **CHANGELOG.** Move the `Unreleased` entries under a new `## [X.Y.Z] - YYYY-MM-DD` heading, and update
   the link definitions at the bottom.
6. **Breaking changes are flagged.** Anything that changes results or removes an argument belongs in the
   changelog with a migration note, and requires a major version bump under semver.

## Releasing

1. Bump `version` in `pyproject.toml`, then `uv lock` so `uv.lock` follows.

   Use `uv lock`, not `uv sync`. A bare `uv sync` installs only the default dependency set and
   **removes everything else from the environment** — the `validation` extra (halotools, treecorr),
   the `docs` group (mkdocs, mkdocstrings), and anything installed by hand such as `pytest-cov`. It
   will also replace a locally built package with the published wheel, which matters if you have
   built treecorr from source to get OpenMP. `uv lock` updates the lockfile, which is all this step
   needs, and leaves the environment alone. If you do want to sync, `uv sync --all-extras
   --all-groups` keeps the extras, but still reinstalls anything you built yourself.
2. Update `CITATION.cff`: `version`, `date-released`, and the description of the versioned DOI identifier.
   Leave the concept DOI (`10.5281/zenodo.17252215`) alone — it always resolves to the latest release.
3. Commit, open the PR from `dev` into `main`, and merge once CI passes on all five Python versions.
4. Tag `main` and push the tag:

   ```bash
   git checkout main && git pull
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

5. Create the GitHub release from the tag, with the changelog section as its body. This is what triggers
   the Zenodo deposit.

## After releasing

1. **Zenodo.** Confirm the new deposit appeared and note the new *version* DOI. Add it to `CITATION.cff`
   under `identifiers`, replacing the previous version's entry (or keeping both, if you want the history).
   Verify that the concept DOI still resolves to the newest version.
2. **PyPI.** Publish the built artefacts:

   ```bash
   uv build
   uv publish            # needs a PyPI token
   ```

   A version can never be re-uploaded to PyPI. If a release is wrong, yank it and publish a new patch
   version — do not try to replace it.
3. **Docs.** The `docs` workflow deploys on any push to `main`, so the site updates with the merge. Check
   the site picked it up.
4. **Verify the published package.** `pip install measureia==X.Y.Z` into a clean environment and import it.

## Known state to be careful about

- **PyPI 0.3.0 does not match the repository's 0.3.0.** The published 0.3.0 declares `numpy~=1.26.2`,
  `astropy~=6.1.0`, `scipy~=1.11.4` and `pathos`; it predates the NumPy 2 migration. Those pins do match
  the `v0.3.0` git tag, so the upload was made from that tree — but that tree's `pyproject.toml` still said
  `version = "0.1.0"`, which is where the drift this checklist guards against first appeared. PyPI does not
  allow re-uploading a version, so this is corrected simply by the next release carrying a new number, not
  by trying to fix 0.3.0.
- **Every published version of MeasureIA before 1.0.0 fails to `pip install`**, because `kmeans-radec` was
  declared as a dependency but does not exist on PyPI. Nothing can be done about the published artefacts;
  1.0.0 is the first release that installs in one command.
