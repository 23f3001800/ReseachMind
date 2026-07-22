#!/bin/sh
# Render nginx.conf from the environment at container start.
#
# The bundle is a static build, so the backend URL and API key cannot be baked
# in at build time — they belong to the deployment, not the artifact. Only the
# named variables are substituted so nginx's own $variables survive.
set -eu

: "${PORT:=8080}"
: "${BACKEND_URL:=http://127.0.0.1:8000}"
: "${BACKEND_API_KEY:=}"

# proxy_pass needs a bare host[:port] for the Host header.
BACKEND_HOST="$(printf '%s' "$BACKEND_URL" | sed -e 's#^[a-z]*://##' -e 's#/.*$##')"
export PORT BACKEND_URL BACKEND_API_KEY BACKEND_HOST

envsubst '${PORT} ${BACKEND_URL} ${BACKEND_API_KEY} ${BACKEND_HOST}' \
  < /etc/nginx/templates/default.conf.template \
  > /etc/nginx/conf.d/default.conf

if [ -z "$BACKEND_API_KEY" ]; then
  echo "WARN: BACKEND_API_KEY is empty — requests will be proxied unauthenticated." >&2
fi
echo "Serving on :${PORT}, proxying /api/ -> ${BACKEND_URL}"

exec nginx -g 'daemon off;'
