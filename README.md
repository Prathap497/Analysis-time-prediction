# Astrée ETA + Progress + Notifications (Phase-1 MVP)

This repo contains a production-ready, Phase-1 Python system for predicting Astrée runtime ETA with uncertainty bands (P10/P50/P90), computing progress and remaining time, and sending idempotent email notifications from Splunk logs.

## Architecture

```
Splunk (logs) -> extract-history (parquet)
                     |
                     v
                 train (model.joblib)
                     |
                     v
Splunk (running logs) -> predict-live (parquet) -> notify (email)
```

## Environment Variables

Required:

- `SPLUNK_BASE_URL`
- `SPLUNK_TOKEN`
- `SPLUNK_VERIFY_TLS` (true/false)
- `SPLUNK_APP`
- `SPLUNK_INDEX`
- `SPLUNK_SOURCETYPE`
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USER`
- `SMTP_PASS`
- `EMAIL_FROM`
- `EMAIL_TO` (comma-separated)

Optional:

- `SPLUNK_HOST_FILTER` (required to send notifications; use the Linux AAaaS Astrée server host value)

## CLI Commands

```
python -m astree_eta extract-history --months 6 --out data/history.parquet
python -m astree_eta train --in data/history.parquet --out models/
python -m astree_eta predict-live --interval-min 15 --out data/live_predictions.parquet
python -m astree_eta notify --in data/live_predictions.parquet
python -m astree_eta evaluate --in data/history.parquet --out reports/metrics.json
```

## Splunk SPL Templates

Use placeholders for `index`, `sourcetype`, `app`, and `host` (when required).

### Historical completed runs (training)

```
search index=$SPLUNK_INDEX$ sourcetype=$SPLUNK_SOURCETYPE$ app=$SPLUNK_APP$ host=$SPLUNK_HOST_FILTER$ status=COMPLETED
| stats latest(project_id) as project_id
    latest(astree_version) as astree_version
    latest(config_profile) as config_profile
    latest(host) as host
    latest(loc) as loc
    latest(kloc) as kloc
    latest(num_files) as num_files
    latest(num_tu) as num_tu
    latest(cpu_avg) as cpu_avg
    latest(server_load_avg) as server_load_avg
    min(start_time) as start_time
    max(end_time) as end_time
    latest(status) as status
    by run_id
```

### Currently running runs (live)

```
search index=$SPLUNK_INDEX$ sourcetype=$SPLUNK_SOURCETYPE$ app=$SPLUNK_APP$ host=$SPLUNK_HOST_FILTER$ status=RUNNING
| stats latest(project_id) as project_id
    latest(astree_version) as astree_version
    latest(config_profile) as config_profile
    latest(host) as host
    latest(loc) as loc
    latest(kloc) as kloc
    latest(num_files) as num_files
    latest(num_tu) as num_tu
    latest(cpu_avg) as cpu_avg
    latest(server_load_avg) as server_load_avg
    min(start_time) as start_time
    latest(_time) as last_log_time
    latest(status) as status
    by run_id
```

### Completion detection / status transitions

```
search index=$SPLUNK_INDEX$ sourcetype=$SPLUNK_SOURCETYPE$ app=$SPLUNK_APP$ host=$SPLUNK_HOST_FILTER$ (status=COMPLETED OR status=FAILED OR status=ABORTED)
| stats latest(project_id) as project_id
    latest(astree_version) as astree_version
    latest(config_profile) as config_profile
    latest(host) as host
    latest(status) as status
    max(end_time) as end_time
    by run_id
```

## Cron Example

See `cron/astree_eta.cron` for a minimal cron setup that extracts predictions and sends notifications every 15 minutes.

## Runbook

1. **Extract history**: Run `extract-history` to capture completed runs into parquet.
2. **Train**: Run `train` to produce `models/model.joblib`.
3. **Predict**: Run `predict-live` on a 15-minute schedule.
4. **Notify**: Run `notify` on the same schedule. Notifications are idempotent per run.
5. **Evaluate**: Run `evaluate` weekly to track MAE, MdAPE, and P10-P90 coverage.

## Notes

- Training uses completed runs only and drops outliers outside the 1st/99th percentile.
- Target is `log1p(runtime_sec)`; predictions are inverted via `expm1`.
- Missing LOC/TU features are imputed to zero; categorical features are one-hot encoded.
- Email notifications are throttled to 1 per run per 15 minutes.
