# Releasing pyTracer to PyPI

pyTracer is a monorepo of 12 packages published together at one version. This
guide covers the one-time setup and the steps to cut a release. Publishing uses
PyPI Trusted Publishing (OIDC), so there are no long-lived tokens to store.

## The packages

All are published at the same version. `pip install pytracer` is not available
(that name is held by an unrelated project), so the "install everything" bundle
is `pytracer-all`, and the common install is `pytracer-sdk`.

```
pytracer-sdk        pytracer-stores     pytracer-gateway    pytracer-replay
pytracer-runner     pytracer-registry   pytracer-cli        pytracer-all
pytracer-langgraph  pytracer-openai-agents  pytracer-crewai  pytracer-autogen
```

## One-time setup

1. Create accounts on <https://pypi.org> and <https://test.pypi.org>.
2. For **each** of the 12 names above, add a **Trusted Publisher** (PyPI ->
   the project, or "pending publisher" if the project does not exist yet ->
   Publishing -> Add a new pending publisher) with:
   - **PyPI Project Name:** the package name (e.g. `pytracer-sdk`)
   - **Owner:** `vi6120`
   - **Repository name:** `pyTracer`
   - **Workflow name:** `publish.yml`
   - **Environment:** leave blank
   Do the same on TestPyPI if you want dry runs.

Pending publishers let the first upload create the project, so you configure all
12 before the very first release.

## Cutting a release

1. **Bump the version** in every package (they must match):
   - `version = "..."` in each `packages/*/pyproject.toml`
   - the inter-package pins `==...` in those pyproject files
   - `__version__ = "..."` in each `packages/*/src/*/__init__.py`

2. **Verify locally** that everything builds and validates:
   ```bash
   pip install build twine
   rm -rf dist && mkdir dist
   for p in sdk stores gateway replay runner registry cli pytracer \
            langgraph openai-agents crewai autogen; do
     python -m build --outdir dist "packages/$p"
   done
   twine check dist/*
   ```

3. **Dry run to TestPyPI** (Actions -> Publish to PyPI -> Run workflow ->
   target `testpypi`), then confirm an install works:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ pytracer-sdk
   ```

4. **Release to PyPI:** create a GitHub Release with a tag like `v0.1.0`. The
   `publish.yml` workflow builds all packages and publishes them to PyPI.

5. **Confirm:**
   ```bash
   pip install pytracer-sdk==0.1.0
   pip install pytracer-all==0.1.0
   ```

## Manual fallback

If you prefer to publish by hand (with a PyPI API token instead of Trusted
Publishing), after building and checking as above:

```bash
twine upload dist/*
```

## Notes

- Because packages depend on each other with exact pins (`==<version>`), publish
  all 12 at the same version; a partial release leaves the meta-package
  uninstallable until the rest land.
- To reclaim the bare `pytracer` name later, the abandoned project (last released
  2019) could be requested through PyPI's PEP 541 process; it is slow and not
  guaranteed, so `pytracer-all` is the name to use now.
