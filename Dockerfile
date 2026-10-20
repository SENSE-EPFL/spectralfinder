# Public web app (app.py). On start the container fetches the line lists it lacks into
# data/ (prefetch.py), then serves; mount a volume there so the fetch happens once.
#
#   docker compose up        -> http://localhost:8501 (see compose.yaml)

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# dependencies first: this layer is rebuilt only when the lock changes
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev

RUN useradd --create-home --uid 1000 app
COPY --chown=app atm_standard.csv app.py prefetch.py ./
COPY --chown=app assets/ assets/
COPY --chown=app hitran_tool/ hitran_tool/
RUN mkdir data && chown app data

USER app
ENV PATH="/app/.venv/bin:$PATH" \
    HITRAN_OFFLINE=1 \
    MAX_PARALLEL_SEARCHES=2 \
    PYTHONUNBUFFERED=1
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/health')"
CMD ["sh", "-c", "HITRAN_OFFLINE=0 python prefetch.py && exec gunicorn --workers 1 --threads 16 \
     --bind 0.0.0.0:8501 app:server"]
