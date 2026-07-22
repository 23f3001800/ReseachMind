# Convenience image for running the whole stack locally in one container.
#
# For real deploys use the two service images instead — backend/Dockerfile and
# frontend/Dockerfile — one process per container, which is what Azure
# Container Apps (and every other managed platform) expects. See deploy/azure.sh.

FROM node:22-alpine AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build


FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential nginx gettext-base \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY --from=ui /ui/dist /usr/share/nginx/html
COPY frontend/nginx.conf.template /etc/nginx/templates/default.conf.template
COPY frontend/docker-entrypoint.sh /usr/local/bin/render-nginx.sh
RUN chmod +x /usr/local/bin/render-nginx.sh \
 && rm -f /etc/nginx/sites-enabled/default

EXPOSE 8080

# Config comes from --env-file or -e at run time; no .env is baked in, so a
# missing GROQ_API_KEY fails fast at startup rather than at the first LLM call.
# nginx serves the SPA on 8080 and proxies /api/ to uvicorn on 8000.
#
# bash, not sh: this image's /bin/sh is dash, which has no `wait -n`.
# `wait -n` plus `kill 0` means if either process dies the container exits,
# rather than sitting "healthy" with half the app gone.
ENV PORT=8080 BACKEND_URL=http://127.0.0.1:8000
CMD ["bash", "-c", "cd backend && uvicorn main:app --host 127.0.0.1 --port 8000 & \
     /usr/local/bin/render-nginx.sh & \
     wait -n; kill 0"]
