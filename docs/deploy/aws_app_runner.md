# TrustLens Deployment: AWS App Runner + RDS Postgres

## Overview
Single-service deployment: FastAPI serves both API and the built React UI.

## RDS Postgres Setup
1. Create an RDS Postgres instance (db.t3.micro ok for demo).
2. Create a database + user (e.g., `trustlens` / `trustlens_user`).
3. Ensure security group allows inbound from App Runner VPC connector.
4. Note the connection string:
   ```
   postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME
   ```

## App Runner Setup
1. Create new App Runner service from GitHub repo.
2. Configure build from `Dockerfile` at repo root.
3. Environment variables:
   - `DATABASE_URL` = Postgres URL
   - Optional: `TRUSTLENS_GDELT_DOC_BASE_URL`, `TRUSTLENS_GDELT_TIMEOUT_S`
4. Deploy.

## Verify
- API health:
  ```
  curl https://<app-runner-url>/api/health
  ```
- UI:
  Open `https://<app-runner-url>/` in browser.

## Local Build (Docker)
```bash
docker build -t trustlens .
docker run -p 8000:8000 -e DATABASE_URL=duckdb:///data/trustlens.duckdb trustlens
```
