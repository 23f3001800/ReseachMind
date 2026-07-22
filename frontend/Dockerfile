# ── Build ────────────────────────────────────────────────
FROM node:22-alpine AS build

WORKDIR /app

# Copy manifests first so the dependency layer caches independently of source.
COPY package.json package-lock.json* ./
RUN npm install --no-audit --no-fund

COPY . .
RUN npm run build


# ── Serve ────────────────────────────────────────────────
FROM nginx:1.27-alpine

# envsubst
RUN apk add --no-cache gettext

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY docker-entrypoint.sh /docker-entrypoint.d/99-render-config.sh

# Replace the stock config; ours is rendered at start into conf.d.
RUN rm -f /etc/nginx/conf.d/default.conf \
 && chmod +x /docker-entrypoint.d/99-render-config.sh \
 && mv /docker-entrypoint.d/99-render-config.sh /entrypoint.sh

ENV PORT=8080
EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
