# Deployment Guide

## 1. Single-node Docker deployment (recommended for v0.x)

### Prerequisites

- Docker 24+
- Docker Compose v2+
- 2 GB+ RAM (for MockBackend); 16 GB+ RAM for HF backend with small models; GPU for larger models

### Steps

1. **Build the image:**
   ```bash
   docker build -f docker/Dockerfile -t chen:latest .
   ```

2. **Run with docker-compose:**
   ```bash
   cd docker
   docker compose up -d
   ```

3. **Verify:**
   ```bash
   curl http://localhost:8000/v1/health
   # {"status":"ok","version":"0.1.0"}
   ```

4. **Make an inference request:**
   ```bash
   curl -X POST http://localhost:8000/v1/infer \
     -H "Content-Type: application/json" \
     -d '{"prompt":"Explain recursion.","phase":1,"backend":"mock"}'
   ```

### With monitoring (Prometheus)

```bash
cd docker
docker compose --profile monitoring up -d
# Prometheus UI at http://localhost:9090
```

## 2. Kubernetes deployment

A Helm chart is on the roadmap. For now, use a simple Deployment:

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chen
  labels:
    app: chen
spec:
  replicas: 1
  selector:
    matchLabels:
      app: chen
  template:
    metadata:
      labels:
        app: chen
    spec:
      containers:
        - name: chen
          image: ghcr.io/your-org/chen:0.1.0
          ports:
            - containerPort: 8000
          env:
            - name: CHEN_LOG_LEVEL
              value: "INFO"
            - name: CHEN_LOG_JSON
              value: "1"
            - name: CHEN_DEFAULT_BACKEND
              value: "mock"
            # For HF backend:
            # - name: HUGGING_FACE_HUB_TOKEN
            #   valueFrom:
            #     secretKeyRef:
            #       name: chen-secrets
            #       key: hf-token
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
          readinessProbe:
            httpGet:
              path: /v1/health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /v1/health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: chen
spec:
  type: ClusterIP
  ports:
    - port: 80
      targetPort: 8000
  selector:
    app: chen
```

```bash
kubectl apply -f k8s-deployment.yaml
```

## 3. Bare-metal deployment

### Systemd unit

```ini
# /etc/systemd/system/chen.service
[Unit]
Description=CHEN API server
After=network.target

[Service]
Type=simple
User=chen
WorkingDirectory=/opt/chen
Environment="CHEN_LOG_LEVEL=INFO"
Environment="CHEN_LOG_JSON=1"
Environment="CHEN_RUN_STORE_PATH=/var/lib/chen/runs.sqlite3"
ExecStart=/opt/chen/.venv/bin/chen serve --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now chen
sudo systemctl status chen
```

### Nginx reverse proxy (TLS termination)

```nginx
server {
    listen 443 ssl http2;
    server_name chen.example.com;

    ssl_certificate /etc/letsencrypt/live/chen.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/chen.example.com/privkey.pem;

    client_max_body_size 10M;  # cap prompt size

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 4. Production hardening

Before exposing CHEN to the internet:

1. **TLS**: terminate TLS at nginx/Caddy/Cloudflare — never expose the API directly over HTTP.
2. **Auth**: add an authenticating reverse proxy (OAuth2 Proxy, Caddy with Basic Auth, etc.).
3. **Rate limiting**: configure nginx `limit_req` to prevent abuse.
4. **Body size limits**: set `client_max_body_size` to ~1 MB to prevent prompt-based DoS.
5. **CORS**: edit `src/chen/server/app.py` to restrict `allow_origins` to your frontend.
6. **Secrets**: store `HUGGING_FACE_HUB_TOKEN` in a secret manager (Vault, AWS Secrets Manager), not in `.env`.
7. **Backups**: back up `chen_data/runs.sqlite3` daily.
8. **Monitoring**: scrape `/v1/metrics` with Prometheus; alert on error rate and KV transfer failures.

## 5. Upgrading

```bash
git pull
pip install -e ".[all]"
# Run migrations (when added)
# Restart the server
sudo systemctl restart chen
```

Always check [`https://github.com/your-org/chen/blob/main/CHANGELOG.md`](https://github.com/your-org/chen/blob/main/CHANGELOG.md) for breaking changes
before upgrading.
