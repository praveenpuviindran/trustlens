# Build frontend
FROM node:20-bullseye AS frontend
WORKDIR /app
COPY web/package.json web/package-lock.json* ./web/
WORKDIR /app/web
RUN npm install --no-fund --no-audit
COPY web/ /app/web/
RUN npm run build

# Build backend
FROM python:3.11-slim AS backend
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .
COPY src/ /app/src/
COPY --from=frontend /app/src/trustlens/api/static /app/src/trustlens/api/static
EXPOSE 8000
CMD ["uvicorn", "trustlens.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
