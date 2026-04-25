FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn pydantic requests

COPY environment/ ./environment/
COPY app.py .
COPY client.py .
COPY openenv.yaml .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
