FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["python", "/app/lynse/api/http_api/http_api.py", "run", "--host", "0.0.0.0", "--port", "7637"]
