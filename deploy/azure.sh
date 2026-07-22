#!/usr/bin/env bash
# Deploy researchMind to Azure Container Apps.
#
# Two container apps on the Container Apps Consumption plan:
#   API — FastAPI/uvicorn
#   UI  — React SPA on nginx, which also reverse-proxies /api/ to the API and
#         attaches the API key server side, so the browser never holds a
#         credential and there is no cross-origin request to configure.
#
# Images are built LOCALLY and pushed: `az acr build` (ACR Tasks) is rejected on
# Azure for Students subscriptions with TasksOperationsNotAllowed, so a local
# Docker daemon is required.
#
# Registry auth uses the app's system-assigned managed identity, so no registry
# password is ever stored in the app config or passed on a command line.
#
# Cost shape:
#   Container Apps  — free grant covers 180k vCPU-s / 360k GiB-s / 2M requests
#                     per subscription per month. Both apps scale to zero, so
#                     idle time costs nothing.
#   ACR Basic       — ~$5/month. The only standing charge. Swap for ghcr.io if
#                     you want a strict $0 footprint.
#   Log Analytics   — pay-as-you-go ingestion, cents at this volume.
#
# Both apps are pinned to max-replicas=1 on purpose: the vector index, the
# LangGraph checkpointer and the SQLite history all live in process memory
# (PROJECT_GUIDE.md G-6/G-7). A second replica would serve requests that can't
# see the first replica's uploads. Raise this only after moving that state out.
#
# Usage:  ./deploy/azure.sh [--groq-key KEY]

set -euo pipefail

RG="${RG:-research-agent-rg}"
LOCATION="${LOCATION:-centralindia}"
ENV_NAME="${ENV_NAME:-research-agent-env}"
API_APP="${API_APP:-research-agent-api}"
UI_APP="${UI_APP:-research-agent-ui}"
TAG="${TAG:-v1}"

GROQ_KEY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --groq-key) GROQ_KEY="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_ID="$(az account show --query id -o tsv)"
# ACR names are globally unique and alphanumeric-only.
ACR_NAME="${ACR_NAME:-racr$(echo "$SUB_ID" | tr -d '-' | cut -c1-12)}"

echo "==> Subscription : $SUB_ID"
echo "==> Resource grp : $RG ($LOCATION)"
echo "==> Registry     : $ACR_NAME"

# ── Providers ────────────────────────────────────────────
for p in Microsoft.App Microsoft.OperationalInsights Microsoft.ContainerRegistry; do
  az provider register -n "$p" --wait
done

# ── Resource group + registry ────────────────────────────
az group create -n "$RG" -l "$LOCATION" -o none

az acr create -g "$RG" -n "$ACR_NAME" --sku Basic -o none

ACR_SERVER="$(az acr show -g "$RG" -n "$ACR_NAME" --query loginServer -o tsv)"

# ── Build and push locally ───────────────────────────────
# `az acr login` mints a short-lived token from the current az session, so no
# admin credential needs to be enabled or handled.
az acr login -n "$ACR_NAME"

docker build -t "$ACR_SERVER/research-agent/backend:$TAG" \
  -f "$REPO_ROOT/backend/Dockerfile" "$REPO_ROOT/backend"
docker push "$ACR_SERVER/research-agent/backend:$TAG"

docker build -t "$ACR_SERVER/research-agent/frontend:$TAG" \
  -f "$REPO_ROOT/frontend/Dockerfile" "$REPO_ROOT/frontend"
docker push "$ACR_SERVER/research-agent/frontend:$TAG"

# ── Container Apps environment ───────────────────────────
az containerapp env create -g "$RG" -n "$ENV_NAME" -l "$LOCATION" -o none

# A generated key so the API is never exposed unauthenticated.
API_KEY="${API_KEY:-$(openssl rand -hex 24)}"

# ── Persistent storage ───────────────────────────────────
# Without this the vector index, document catalog and conversation history sit
# on the container's own filesystem and are destroyed on every scale-to-zero:
# uploads appear to work, then silently vanish. An Azure Files share mounted at
# DATA_DIR is the cheapest fix that genuinely survives a restart.
STORAGE_ACCOUNT="${STORAGE_ACCOUNT:-rasa$(openssl rand -hex 6)}"
SHARE_NAME="researchmind-data"
STORAGE_REF="rmdata"

az storage account create -g "$RG" -n "$STORAGE_ACCOUNT" -l "$LOCATION" \
  --sku Standard_LRS --kind StorageV2 -o none

STORAGE_KEY="$(az storage account keys list -g "$RG" -n "$STORAGE_ACCOUNT" --query "[0].value" -o tsv)"

az storage share create --account-name "$STORAGE_ACCOUNT" --account-key "$STORAGE_KEY" \
  -n "$SHARE_NAME" --quota 5 -o none

az containerapp env storage set -g "$RG" -n "$ENV_NAME" --storage-name "$STORAGE_REF" \
  --azure-file-account-name "$STORAGE_ACCOUNT" \
  --azure-file-account-key "$STORAGE_KEY" \
  --azure-file-share-name "$SHARE_NAME" \
  --access-mode ReadWrite -o none

# ── Backend ──────────────────────────────────────────────
az containerapp create -g "$RG" -n "$API_APP" \
  --environment "$ENV_NAME" \
  --image "$ACR_SERVER/research-agent/backend:$TAG" \
  --target-port 8000 \
  --ingress external \
  --cpu 1.0 --memory 2.0Gi \
  --min-replicas 0 --max-replicas 1 \
  --secrets "groq-key=$GROQ_KEY" "api-key=$API_KEY" "tavily-key=${TAVILY_KEY:-}" \
  --env-vars \
      "PORT=8000" \
      "GROQ_API_KEY=secretref:groq-key" \
      "API_KEY=secretref:api-key" \
      "TAVILY_API_KEY=secretref:tavily-key" \
      "REQUEST_TIMEOUT=180" \
      "RATE_LIMIT_PER_MINUTE=20" \
      "DATA_DIR=/app/data" \
      "LOG_FORMAT=json" \
  -o none

# The volume can only be attached via YAML — `az containerapp create` has no
# flag for it — so the app is created first and then patched in place.
TMP_YAML="$(mktemp)"
az containerapp show -g "$RG" -n "$API_APP" -o yaml > "$TMP_YAML"
python3 - "$TMP_YAML" <<'PY'
import sys

path = sys.argv[1]
out = []
for line in open(path).read().splitlines():
    if line.strip() == "volumes: null":
        out += ["    volumes:", "    - name: rmdata",
                "      storageName: rmdata", "      storageType: AzureFile"]
        continue
    out.append(line)
    if line.strip() == "name: research-agent-api" and line.startswith("      "):
        out += ["      volumeMounts:", "      - mountPath: /app/data",
                "        volumeName: rmdata"]
open(path, "w").write("\n".join(out) + "\n")
PY
az containerapp update -g "$RG" -n "$API_APP" --yaml "$TMP_YAML" -o none
rm -f "$TMP_YAML"

API_FQDN="$(az containerapp show -g "$RG" -n "$API_APP" --query properties.configuration.ingress.fqdn -o tsv)"

# ── Frontend ─────────────────────────────────────────────
az containerapp create -g "$RG" -n "$UI_APP" \
  --environment "$ENV_NAME" \
  --image "$ACR_SERVER/research-agent/frontend:$TAG" \
  --target-port 8080 \
  --ingress external \
  --cpu 0.5 --memory 1.0Gi \
  --min-replicas 0 --max-replicas 1 \
  --secrets "api-key=$API_KEY" \
  --env-vars \
      "PORT=8080" \
      "BACKEND_URL=https://$API_FQDN" \
      "BACKEND_API_KEY=secretref:api-key" \
  -o none

# No sticky sessions needed: the SPA holds its own state client side and nginx
# is stateless, so any replica can serve any request.

# ── Managed-identity registry access ─────────────────────
# Grant each app AcrPull on the registry and switch it off password auth, so
# no registry credential is stored anywhere in the app configuration.
ACR_ID="$(az acr show -g "$RG" -n "$ACR_NAME" --query id -o tsv)"
for APP in "$API_APP" "$UI_APP"; do
  az containerapp identity assign -g "$RG" -n "$APP" --system -o none
  PRINCIPAL="$(az containerapp show -g "$RG" -n "$APP" --query identity.principalId -o tsv)"
  az role assignment create --assignee-object-id "$PRINCIPAL" \
    --assignee-principal-type ServicePrincipal \
    --role AcrPull --scope "$ACR_ID" -o none
  az containerapp registry set -g "$RG" -n "$APP" --server "$ACR_SERVER" --identity system -o none
done

# Lock the API's CORS to the UI origin. Browser traffic goes through the UI's
# nginx proxy so it is same-origin and never triggers CORS, but this keeps the
# API closed to any browser that finds it directly.
#
# NOTE: ALLOWED_ORIGINS is parsed as a comma-separated STRING by config.py.
# It must not be typed as List[str] there — pydantic-settings JSON-decodes
# complex types straight from the environment, before any validator runs, so a
# bare URL raises SettingsError and the container exits 1 on startup.
UI_FQDN="$(az containerapp show -g "$RG" -n "$UI_APP" --query properties.configuration.ingress.fqdn -o tsv)"
az containerapp update -g "$RG" -n "$API_APP" \
  --set-env-vars "ALLOWED_ORIGINS=https://$UI_FQDN" -o none

echo
echo "API : https://$API_FQDN"
echo "UI  : https://$UI_FQDN"
echo "API key: $API_KEY"
echo
echo "Set a working Groq key with:"
echo "  az containerapp secret set -g $RG -n $API_APP --secrets groq-key=<KEY>"
echo "  az containerapp revision restart -g $RG -n $API_APP \\"
echo "    --revision \$(az containerapp show -g $RG -n $API_APP --query properties.latestRevisionName -o tsv)"
