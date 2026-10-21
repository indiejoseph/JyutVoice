# JyutVoice playground: the page and its proxy (standard library only, no pip install).
FROM python:3.13-slim

WORKDIR /app
COPY server.py index.html /app/
COPY voices /app/voices

ENV HOST=0.0.0.0 PORT=8898
USER nobody
EXPOSE 8898

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8898/api/voices', timeout=4)"]

CMD ["python3", "server.py"]
