# JyutVoice TTS API, CPU only. Build context is the repository root (see docker-compose.yml):
#     docker build -f docker/Dockerfile -t jyutvoice-api:cpu .
# The model weights are not in the image: mount onnx_models/ (scripts/export_onnx.py) and the pretrained decoder files (docker-compose.yml does).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/opt/nltk_data \
    JYUTVOICE_TN_CACHE=/opt/tn_cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch from the CPU wheel index (the default PyPI wheel brings about 3 GB of CUDA libraries)
RUN pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

COPY api/requirements.txt api/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r api/requirements.txt \
    && pip install https://github.com/indiejoseph/WeTextProcessing/archive/ae36e7c2781ff79373610d0158f195e25690c4b0.zip

# English g2p reads these at import; fetch them now so that the container starts offline. The downloader writes the zips as 0600: the app user (below) could not read
# them and g2p_en would download them again on every start.
RUN python -m nltk.downloader -d "$NLTK_DATA" cmudict averaged_perceptron_tagger_eng averaged_perceptron_tagger \
    && chmod -R a+rX "$NLTK_DATA"

# The code, then the compiled text-normalisation graphs (about a minute for the English one) so that the first request does not build them.
# WORKDIR matters: the English g2p loads ./bert/deberta-v3-large relative to it.
COPY jyutvoice jyutvoice
COPY bert bert
RUN python -c "from jyutvoice.text.text_processor import normalize_text; normalize_text('我有3個', 'yue'); normalize_text('It costs 3 dollars', 'en')" \
    && python -c "from g2p_en import G2p; G2p()"

COPY configs/onnx_infer.yaml configs/onnx_infer.yaml
COPY infer_onnx.py infer_onnx.py
COPY api api

RUN useradd --create-home --uid 1000 app && chown -R app:app /opt/tn_cache
USER app

ENV API_HOST=0.0.0.0 \
    API_PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('API_PORT', '8000'), timeout=8)" || exit 1

# One worker: the models are loaded once and synthesis is serialised by a lock anyway
CMD ["sh", "-c", "exec uvicorn api.main:app --host \"$API_HOST\" --port \"$API_PORT\""]
