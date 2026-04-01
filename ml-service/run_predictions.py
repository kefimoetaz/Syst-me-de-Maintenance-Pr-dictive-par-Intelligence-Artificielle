"""
Standalone script: wipe seeded predictions, run RF for all machines, save to DB.
Run from ml-service directory: python run_predictions.py

Fix: uses each machine's actual latest data timestamp as end_date,
     so old seeded data (Feb 2026) is found correctly.
"""
import sys
import json
import psycopg2
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

DB_PARAMS = dict(
    host=os.getenv('DB_HOST', 'localhost'),
    port=os.getenv('DB_PORT', 5432),
    database=os.getenv('DB_NAME', 'predictive_maintenance'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', '123')
)


def get_conn():
    return psycopg2.connect(**DB_PARAMS)


# ── Step 0: check active RF model ─────────────────────────────────────────────
conn = get_conn()
cur = conn.cursor()
cur.execute("SELECT model_id FROM ml_models WHERE is_active = true AND model_type = 'random_forest'")
row = cur.fetchone()
cur.close()
conn.close()

if not row:
    print("[ERROR] No active random_forest model. Train first:")
    print('  python -c "from src.training_pipeline import TrainingPipeline; TrainingPipeline().run()"')
    sys.exit(1)

print("Active RF model:", row[0])

# ── Step 1: wipe all existing predictions ─────────────────────────────────────
conn = get_conn()
cur = conn.cursor()
cur.execute("DELETE FROM predictions")
deleted = cur.rowcount
conn.commit()
cur.close()
conn.close()
print(f"[1/3] Cleared {deleted} old predictions")

# ── Step 2: get all machines + their latest data timestamp ────────────────────
conn = get_conn()
cur = conn.cursor()
cur.execute("""
    SELECT m.id, MAX(sm.created_at) as latest
    FROM machines m
    LEFT JOIN system_metrics sm ON sm.machine_id = m.id
    GROUP BY m.id
    ORDER BY m.id
""")
machines = cur.fetchall()  # [(machine_id, latest_timestamp), ...]
cur.close()
conn.close()

print(f"[2/3] Running RF predictions for {len(machines)} machines...")

# ── Step 3: predict per machine using its own data window ─────────────────────
from src.model_registry import ModelRegistry
from src.feature_extractor import FeatureExtractor

registry = ModelRegistry()
model, meta = registry.get_active_model('random_forest')
registry.close()

if model is None:
    print("[ERROR] Could not load RF model from registry")
    sys.exit(1)

print("      Model:", meta['model_id'])


def calculate_risk_level(prob_pct):
    if prob_pct >= 70:
        return 'CRITICAL'
    elif prob_pct >= 50:
        return 'HIGH'
    elif prob_pct >= 30:
        return 'MEDIUM'
    return 'LOW'


predictions = []
skipped = 0

for machine_id, latest_ts in machines:
    if latest_ts is None:
        skipped += 1
        continue

    # Use the machine's actual latest timestamp as end_date
    end_date = latest_ts
    start_date = end_date - timedelta(days=90)  # 90-day window around the data

    try:
        fe = FeatureExtractor()
        features_df = fe.extract_features(
            machine_id=machine_id,
            start_date=start_date,
            end_date=end_date
        )
        fe.close()

        if features_df.empty:
            skipped += 1
            continue

        drop_cols = ['machine_id', 'timestamp']
        X = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns])
        X = X.fillna(0)

        proba = model.predict_proba(X)[0]
        failure_prob = proba[1] if len(proba) > 1 else 0.0

        fp_30d = float(failure_prob * 100)
        fp_7d  = float(failure_prob * 70)
        fp_14d = float(failure_prob * 85)
        risk   = calculate_risk_level(fp_30d)

        # contributing factors
        factors = []
        if hasattr(model, 'feature_importances_'):
            pairs = sorted(
                zip(X.columns.tolist(), model.feature_importances_),
                key=lambda x: x[1], reverse=True
            )[:5]
            factors = [{'feature': n, 'importance': float(i), 'value': float(X[n].iloc[0])}
                       for n, i in pairs]

        predictions.append({
            'machine_id': machine_id,
            'prediction_date': datetime.now(),
            'failure_probability_7d': fp_7d,
            'failure_probability_14d': fp_14d,
            'failure_probability_30d': fp_30d,
            'risk_level': risk,
            'model_version': meta['model_id'],
            'contributing_factors': factors
        })
        print(f"      Machine {machine_id:3d}: {risk:8s} ({fp_30d:.1f}%)")

    except Exception as e:
        print(f"      Machine {machine_id}: ERROR - {e}")
        skipped += 1
        continue

print(f"\n      Generated: {len(predictions)} | Skipped: {skipped}")

if not predictions:
    print("[ERROR] No predictions generated")
    sys.exit(1)

# ── Step 4: store to DB ────────────────────────────────────────────────────────
print("[3/3] Saving to DB...")
conn = get_conn()
cur = conn.cursor()
stored = 0
for p in predictions:
    try:
        cur.execute("DELETE FROM predictions WHERE machine_id = %s", (p['machine_id'],))
        cur.execute("""
            INSERT INTO predictions (
                machine_id, prediction_date,
                failure_probability_7d, failure_probability_14d, failure_probability_30d,
                risk_level, model_version, contributing_factors
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            p['machine_id'], p['prediction_date'],
            p['failure_probability_7d'], p['failure_probability_14d'],
            p['failure_probability_30d'], p['risk_level'],
            p['model_version'], json.dumps(p['contributing_factors'])
        ))
        stored += 1
    except Exception as e:
        print(f"  Store failed for machine {p['machine_id']}: {e}")
        conn.rollback()
        continue

conn.commit()
cur.close()
conn.close()

print(f"      Saved {stored} predictions\n")
print("=== Risk Distribution ===")
dist = {}
for p in predictions:
    dist[p['risk_level']] = dist.get(p['risk_level'], 0) + 1
for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
    if level in dist:
        print(f"  {level:8s}: {dist[level]} machine(s)")

print("\nDone. Refresh the dashboard to see real RF predictions.")
