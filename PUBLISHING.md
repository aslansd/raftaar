# Publishing Raftaar

## Routine release

```bash
cd raftaar
source .venv/bin/activate

# 1. Bump the version in TWO places — they must match
#    src/raftaar/__init__.py   __version__ = "0.2.0"
#    pyproject.toml            version = "0.2.0"

# 2. Write the CHANGELOG entry first, not last

# 3. Verify
pytest -q                                  # 53 passed
python -m raftaar.cli synth clean --episodes 120
raftaar scan datasets/clean --out reports/clean

# 4. Build and check
rm -rf dist build src/*.egg-info
python -m build
python -m twine check dist/*

# 5. Test the artifact, not the source tree
python -m venv /tmp/check
/tmp/check/bin/pip install dist/raftaar-*.whl
/tmp/check/bin/raftaar --help
/tmp/check/bin/python -c "import raftaar; print(raftaar.__version__)"

# 6. Upload
python -m twine upload dist/*
git tag v0.2.0 && git push --tags

# 7. Confirm — pip caches the index page and PyPI's CDN lags a minute or two
pip install --no-cache-dir -U raftaar
```

Step 5 is the one people skip. It catches packaging mistakes the test suite
cannot see: a module missing from the wheel, a data file absent from
`MANIFEST.in`, an entry point that does not resolve.

---

## First-time setup

**An API token, not a password.** PyPI → Account settings → API tokens. Scope
it to "entire account" for the first upload, then re-scope to the project.
Paste it including the `pypi-` prefix; Twine 7 asks for the token directly, so
you do not type `__token__` as a username.

To avoid re-pasting, put it in `~/.pypirc` (mode 600):

```ini
[distutils]
index-servers = pypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...
```

---

## Things that have already cost time

### A name can be free and still refused

`raftar` returned 404 from the PyPI JSON API — unregistered — and the upload was
still rejected:

```
400 The name 'raftar' isn't allowed.
```

PyPI blocks names too similar to existing ones as typosquatting protection, and
`rafter` already existed. **Unregistered is not the same as registrable.**

To test a name without building the real package, upload a `0.0.0` stub to
TestPyPI, which applies the same rules:

```bash
mkdir /tmp/nametest && cd /tmp/nametest
printf '[build-system]\nrequires=["setuptools>=77"]\nbuild-backend="setuptools.build_meta"\n[project]\nname="yourname"\nversion="0.0.0"\ndescription="name check"\n' > pyproject.toml
mkdir -p src/yourname && touch src/yourname/__init__.py
python -m build -q && python -m twine upload --repository testpypi dist/*
```

TestPyPI needs its **own account and its own token**. A PyPI token there returns
a bare `403`, which looks like a name problem and is not.

### A version number can never be reused

Not even after deleting the release. If 0.2.0 is wrong, ship 0.2.1. Do not
delete and retry.

### `pip install -U` will show you the old version

For a minute or two after upload. Use `--no-cache-dir`, and check what is really
published, bypassing pip entirely:

```bash
curl -s https://pypi.org/pypi/raftaar/json | python -c \
  "import json,sys; print(json.load(sys.stdin)['info']['version'])"
```

### `rm -rf src/*.egg-info` fails on a clean tree in zsh

```
zsh: no matches found: src/*.egg-info
```

Harmless — the glob matched nothing. Use `rm -rf src/*.egg-info 2>/dev/null` or
`setopt null_glob` if it bothers you.

---

## Do not ship these

The repository should not contain, and the sdist should not carry:

| | Why |
|---|---|
| `datasets/`, `reports/` | generated output; `.gitignore` covers them, but they are easy to include by accident when zipping |
| `dist/`, `build/`, `*.egg-info` | build artifacts |
| `.DS_Store` | macOS noise; `find . -name .DS_Store -delete` |
| `rename.py` | existed to keep the name cheap to change before release. Delete it once published — a script that rewrites every occurrence of the package name does not belong inside the published package |

---

## Before announcing a release

- [ ] `pytest -q` green
- [ ] the LeRobot adapter run against at least one real hub dataset
- [ ] wheel installed into a clean venv and the CLI invoked
- [ ] README renders correctly on the live PyPI page
- [ ] `git tag && git push --tags`

---

## On the licence

`LICENSE` contains the full Apache-2.0 text and `pyproject.toml` declares
`license = "Apache-2.0"` as an SPDX expression. This is not bookkeeping: the
strategy rests on publishing freely available source code, and a public
repository with a permissive licence and a licence file present is a different
legal object from a private or licence-gated artifact. Keep it that way.
