FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install uv && uv sync --no-dev

EXPOSE 8080
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
