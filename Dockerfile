FROM python:3.12-slim

WORKDIR /app

# CPU-only torch. The default wheel drags in ~2 GB of CUDA libraries that are
# dead weight for a 128x128 MLP policy on a free CPU host.
#
# The `||` fallback is deliberate: download.pytorch.org intermittently serves an
# empty index, which fails the build outright with "No matching distribution
# found for torch" on a Dockerfile that built fine an hour earlier.
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 120 torch \
        --index-url https://download.pytorch.org/whl/cpu \
 || pip install --no-cache-dir --retries 5 --timeout 120 torch
RUN pip install --no-cache-dir --retries 5 --timeout 120 -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY artifacts/ ./artifacts/
COPY reports/ ./reports/
COPY NOTES.md README.md ./

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/showcase.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
