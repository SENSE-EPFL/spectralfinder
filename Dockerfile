# Public web app (app.py). Build from a checkout whose data/ is filled -- `uv run prefetch.py`
# first: the container never downloads from HITRAN (HITRAN_OFFLINE=1), it only uses the
# line lists copied in below.
#
#   docker build -t hitran-spectro .
#   docker run -p 8501:8501 hitran-spectro        -> http://localhost:8501

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# dependencies first: this layer is rebuilt only when the lock changes
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev

RUN useradd --create-home app
# line lists (~400 MB): before the code, so a code change doesn't recopy them
COPY --chown=app data/*.lines.npz data/
COPY --chown=app atm_standard.csv app.py ./
COPY --chown=app .streamlit/ .streamlit/
COPY --chown=app hitran_tool/ hitran_tool/
RUN chown app /app/data

USER app
ENV PATH="/app/.venv/bin:$PATH" \
    HITRAN_OFFLINE=1 \
    MAX_PARALLEL_SEARCHES=2 \
    PYTHONUNBUFFERED=1
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
