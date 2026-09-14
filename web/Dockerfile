# Build context is the REPOSITORY ROOT, not web/:
#     docker build -f web/Dockerfile -t raftaar-demo .
#
# The app is a thin layer over the library, so the image installs the package
# from source rather than duplicating it. That way the demo can never drift from
# the code it is demonstrating.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Dependencies first, so edits to app.py do not invalidate the install layer.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[web,plot]"

COPY web/app.py ./
COPY web/templates/ ./templates/

# Cloud Run injects PORT. One worker with a few threads: a scan is CPU-bound and
# short, and each request builds its dataset in /tmp.
ENV PORT=8080
EXPOSE 8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 120 app:app
