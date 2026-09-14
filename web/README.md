# Raftaar browser demo

The Flask app behind the hosted demo. **Not part of the installable package** —
it is a demo surface, and shipping Flask and gunicorn as dependencies of a CPU
dataset audit would be wrong.

```bash
pip install "raftaar[web]"
python web/app.py
```

Container build for Cloud Run:

```bash
docker build -f web/Dockerfile -t raftaar-demo .
```

The app imports `raftaar` as a normal dependency, so install the package
first (`pip install -e .` from the repository root).
