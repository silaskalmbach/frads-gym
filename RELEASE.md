# Release checklist

Publishing is manual on purpose: every step that leaves the machine is listed separately.

A GitHub release has side effects:

- **Zenodo** archives it as a new version under the concept DOI
  [10.5281/zenodo.18504304](https://doi.org/10.5281/zenodo.18504304) (repository webhook) and reads the
  metadata from `CITATION.cff` (a `.zenodo.json`, if ever added, would take precedence).
- **GitHub Pages**: the docs workflow publishes the documentation. Before the first release, set
  Settings → Pages → Source to "GitHub Actions".

PyPI is not touched by any workflow; uploads use `twine`.

## 1. Prepare (local)

- [ ] `main` is up to date and the CI and Docs workflows are green on it.
- [ ] Version in `pyproject.toml` and `CITATION.cff` (`version`), and add `date-released: YYYY-MM-DD` to
      `CITATION.cff`.
- [ ] `CHANGELOG.md`: rename the section to `## [X.Y.Z] - YYYY-MM-DD`, keep an empty `## [Unreleased]`.
- [ ] Commit: `Release X.Y.Z`.

## 2. Build and check (local)

```bash
rm -rf dist
python -m build
twine check --strict dist/*
```

Smoke test from the wheel in a fresh environment:

```bash
python -m venv /tmp/fg && /tmp/fg/bin/pip install dist/frads_gym-X.Y.Z-py3-none-any.whl
cp -r examples /tmp/fg-examples && cd /tmp/fg-examples && /tmp/fg/bin/python quickstart.py
```

## 3. TestPyPI (outward)

```bash
twine upload --repository testpypi dist/*
python -m venv /tmp/fg-test && /tmp/fg-test/bin/pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ frads-gym==X.Y.Z
```

## 4. Tag and GitHub release (outward: Zenodo and docs follow)

```bash
git tag -a vX.Y.Z -m "frads-gym X.Y.Z"
git push origin main vX.Y.Z
```

Create the release from the tag on GitHub, with the changelog section as notes. Mark it as a full release
(the earlier `v0.1.1` is a pre-release).

## 5. PyPI (outward)

```bash
twine upload dist/*
```

## 6. Afterwards

- [ ] The Zenodo record of the new version has the right title, authors, version and license.
- [ ] https://silaskalmbach.github.io/frads-gym/ shows the new version.
- [ ] `pip install frads-gym==X.Y.Z` works in a fresh environment.
