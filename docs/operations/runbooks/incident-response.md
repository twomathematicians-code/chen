# Incident Response Runbook

## Severity levels

| Severity | Definition | Response time |
|----------|------------|---------------|
| **SEV-1** | Service down, data loss, security breach | < 15 min |
| **SEV-2** | Major feature broken, performance degraded | < 1 hour |
| **SEV-3** | Minor feature broken, workaround exists | < 1 business day |
| **SEV-4** | Cosmetic, no user impact | Next release |

## Triage steps

1. **Acknowledge** the alert in your monitoring tool (PagerDuty, Slack, etc.).
2. **Assess severity** using the table above.
3. **Open a bridge** in Slack/Zoom for SEV-1/SEV-2.
4. **Page additional on-call** if you need help (database expert, security expert).

## Common incidents

### Incident A: API returning 500 errors

**Symptom**: `chen_requests_total{status="500"}` is increasing.

**Investigation**:
1. Check container logs: `docker logs chen --tail 100`
2. Look for `run.failed` log events with stack traces.
3. Check `/v1/health` — if it returns 500, the app itself is broken.

**Common causes & fixes**:
- **OOM**: container hit memory limit → increase `resources.limits.memory` in k8s or `--memory` in Docker.
- **SQLite locked**: too many concurrent writes → set `CHEN_RUN_STORE_PATH` per-process, or migrate to Postgres.
- **Backend crash**: HuggingFace model failed to load → check `HUGGING_FACE_HUB_TOKEN` and disk space for model cache.

**Recovery**:
```bash
docker restart chen
# Verify
curl http://localhost:8000/v1/health
```

### Incident B: KV-cache transfer failures spiking

**Symptom**: `rate(chen_kv_cache_transfers_total{result="failure"}[5m]) > 0.1`.

**Investigation**:
1. Check logs for `IncompatibleCacheError` messages.
2. Identify which (source, target) pair is failing.
3. Check if a backend was upgraded (changed model architecture).

**Common causes & fixes**:
- **Cross-family transfer**: expected behavior. Pipeline falls back to text automatically. No action needed unless accuracy regresses.
- **Same-family transfer failing**: model architecture changed in an upgrade. Pin the model version in your config.

### Incident C: Latency spike

**Symptom**: `histogram_quantile(0.99, chen_request_latency_seconds_bucket[5m]) > 5`.

**Investigation**:
1. Check `chen_active_pipelines` — if high, you're overloaded. Scale horizontally.
2. Check `chen_expert_invocations_total` — has the router started waking more experts? (routing regression?)
3. Check GPU utilization if using HF backend: `nvidia-smi`.

**Recovery**:
- Add more replicas.
- Lower `max_tokens_per_expert` in the request.
- Switch to a faster backend (MockBackend for dev, vLLM for prod).

### Incident D: Run store corruption

**Symptom**: `sqlite3.OperationalError: database disk image is malformed`.

**Recovery**:
```bash
# Stop the server
docker stop chen

# Backup the corrupt file
cp chen_data/runs.sqlite3 chen_data/runs.sqlite3.corrupt.$(date +%s)

# Try to recover
sqlite3 chen_data/runs.sqlite3 ".recover" > recovered.sql
sqlite3 chen_data/runs_new.sqlite3 < recovered.sql
mv chen_data/runs_new.sqlite3 chen_data/runs.sqlite3

# Restart
docker start chen
```

If recovery fails, delete the file — the schema will be recreated
empty on next run. Past runs are lost (acceptable for v0.x; backups
are the mitigation).

### Incident E: Security breach

**Symptom**: Suspicious activity in logs, unauthorized requests, leaked token.

**Steps**:
1. **Rotate the HuggingFace token** immediately at https://huggingface.co/settings/tokens.
2. **Revoke any API keys** if you've added auth.
3. **Take the server offline**: `docker stop chen`.
4. **Preserve logs** for forensics: `docker logs chen > /tmp/chen-forensic-$(date +%s).log`.
5. **Notify** users if their prompts may have been exposed.
6. **File a security report** — see [https://github.com/your-org/chen/blob/main/SECURITY.md](https://github.com/your-org/chen/blob/main/SECURITY.md).

## Post-incident

Within 48 hours of a SEV-1/SEV-2 incident:

1. Write a postmortem (blameless) in `docs/operations/postmortems/YYYY-MM-DD-title.md`.
2. Identify root cause and contributing factors.
3. Create action items (with owners and due dates) to prevent recurrence.
4. Review at the next maintainer sync.
