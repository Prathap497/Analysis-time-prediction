# Astrée Execution Time Prediction & Notification System

## Architecture Overview

```
Astrée Linux Server
        ↓
   Splunk (logs + metrics)
        ↓
Python Data Pipeline
        ↓
Prediction Engine (ML + stats)
        ↓
Email Notification System
```

## Data Source (Splunk)

### What Already Exists

From Splunk you can extract:

- Analysis start timestamp
- Current elapsed time
- Project identifier
- Astrée version
- Status (running / completed / failed)

### Required Additions

| Feature                      | Why it matters                 |
| ---------------------------- | ------------------------------ |
| Lines of code (LOC)          | Strongest runtime driver       |
| Number of translation units  | Impacts path explosion         |
| Astrée configuration profile | Huge time variance             |
| Server load / CPU usage      | Runtime distortion             |
| Failure flag                 | Must be excluded from training |

## Python Data Pipeline

### Extraction

- Use the Splunk REST API.
- Pull completed runs only for training.
- Pull running runs for live inference.

Fields to extract:

```
project_id
astree_version
config_profile
loc
modules
start_time
end_time
total_runtime_sec
```

### Preprocessing

- Drop:
  - Aborted runs
  - Incomplete runs
  - Outliers (>99th percentile unless explained)
- Normalize:
  - LOC (log scale)
  - Runtime (log scale)

## Modeling Strategy

### Phase 1 – Baseline

Multiple Linear Regression (log-log):

```
log(runtime) =
  a * log(LOC) +
  b * modules +
  c * astree_version +
  d * config_profile +
  e
```

### Phase 2 – Improved Accuracy (Optional)

Pick one:

- Gradient Boosted Regression (XGBoost / LightGBM)
- Quantile Regression (confidence bounds)

## Confidence Intervals

Predictions must include uncertainty, for example:

```
Estimated completion time: 18.5 hours
Confidence interval (90%): 15.2 – 22.8 hours
```

Approaches:

- Quantile regression (P10 / P50 / P90)
- Residual-based confidence intervals

## Live Progress Estimation

```
Predicted Total Runtime = model(features)
Remaining Time = Predicted Total – Elapsed Time
Completion % = Elapsed / Predicted Total
```

Update cadence:

- Recalculate every N minutes.
- Shrink confidence intervals as elapsed time increases.

## Anomaly Detection

Flag runs when:

- Elapsed time > P90 prediction.
- Runtime slope deviates sharply from historical patterns.

## Email Notification System

### Triggers

- Analysis started
- 50% / 75% completion reached
- Predicted delay beyond expected CI
- Analysis completed

### Sample Email

```
Subject: Astrée Analysis – ETA Update (Project X)

Current status: Running
Elapsed time: 12.4 hours
Estimated remaining time: 6.1 hours
Confidence interval (90%): 4.5 – 8.9 hours

No anomalies detected.
```

## Implementation Roadmap

### Week 1

- Splunk field validation
- Historical data extraction
- Baseline regression model

### Week 2

- Confidence intervals
- Live prediction logic
- Email notifications

### Week 3

- Anomaly detection
- Dashboard integration
- Accuracy validation
