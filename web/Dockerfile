FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY raftaar/ ./raftaar/
COPY templates/ ./templates/
COPY app.py .

# Cloud Run injects PORT. One worker, a few threads: the scan is CPU-bound and
# short, and each request builds its dataset in /tmp.
ENV PORT=8080
EXPOSE 8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 120 app:app
