FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch

ENV PYTHONPATH "${PYTHONPATH}:/app"

EXPOSE 80

CMD ["python", "/app/min_vec/api/http_api.py", "run", "--host", "0.0.0.0", "--port", "7637"]
