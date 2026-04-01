# ⚠️ READ THIS BEFORE ANYTHING ELSE

> This file complements `README.md`. Read both.
> `README.md` covers the general overview and installation.
> This file covers the critical setup details that will save you hours.

---

## 1. Environment Files — Required Before Starting

You need THREE `.env` files. None of them are in the repo (they're gitignored).

### `backend/.env`
```env
PORT=3000
NODE_ENV=development

DB_HOST=localhost
DB_PORT=5432
DB_NAME=maintenance_predictive
DB_USER=postgres
DB_PASSWORD=123

JWT_SECRET=your-super-secret-jwt-key-change-this
API_TOKEN=dev-token-12345

ML_SERVICE_URL=http://localhost:5000
ML_SERVICE_TOKEN=dev-token-12345

SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@gmail.com
SMTP_PASS=your-app-password
ALERT_EMAIL_FROM=your@gmail.com
ALERT_EMAIL_TO=admin@maintenance.com

OLLAMA_URL=http://localhost:11434
```

### `ml-service/.env`
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=maintenance_predictive
DB_USER=postgres
DB_PASSWORD=123
API_HOST=0.0.0.0
API_PORT=5000
API_TOKEN=dev-token-12345
MODEL_DIR=./models
LOG_LEVEL=INFO
```

### `agent/config.json`
```json
{
  "api_url": "http://localhost:3000/api/data",
  "agent_id": "550e8400-e29b-41d4-a716-446655440001",
  "token": "token_support_2024_secure_003",
  "collection_interval_hours": 1,
  "log_level": "INFO",
  "log_file": "agent.log",
  "max_log_size_mb": 10
}
```

> The `agent_id` and `token` must match a row in the `agents` table in the database.
> Run `node backend/src/database/seed.js` first to create them.

---

## 2. Correct Startup Order

Services must start in this exact order:

```
1. PostgreSQL        (must be running first)
2. Backend           (needs DB)
3. ML Service        (needs DB + backend)
4. Frontend          (needs backend)
5. Agent             (needs backend running)
6. Ollama (optional) (for chatbot AI responses)
```

---

## 3. Default Login Credentials

| Email | Password | Role |
|-------|----------|------|
| admin@maintenance.com | admin123 | Admin (full access) |
| technicien@maintenance.com | tech123 | Technician |

> If login fails, run: `node backend/create-users.js`

---

## 4. SMART Data on Windows

The agent reads real disk health via `smartctl`. Without it, all machines show `GOOD / 40°C` (fallback).

**Install smartmontools:**
Download from https://www.smartmontools.org/wiki/Download

After install, verify:
```bash
smartctl --scan
smartctl -H /dev/sda -d nvme   # for NVMe drives
```

The agent auto-detects it at:
`C:\Program Files\smartmontools\bin\smartctl.exe`

---

## 5. ML Models

Two models are used:

| Model | File | Purpose |
|-------|------|---------|
| LSTM | `ml-service/models/lstm_model.pth` | Real-time risk from SMART sequences |
| Random Forest | `ml-service/models/random_forest_vX.joblib` | 30-day failure probability |

Models are **not in the repo** (gitignored — binary files).

To train the LSTM:
```bash
python train_lstm_schema.py
```

To train the Random Forest:
```bash
cd ml-service
python -m src.training_pipeline
```

Without trained models:
- LSTM returns `UNKNOWN` if no `.pth` file exists (uses untrained weights)
- Random Forest returns no predictions until trained

---

## 6. Chatbot (Ollama)

The chatbot works in two modes:
- **With Ollama:** natural language responses for greetings and general questions
- **Without Ollama:** deterministic rule-based responses (still fully functional)

To enable Ollama:
```bash
# Install from https://ollama.com
ollama pull llama3.2:1b
ollama serve
```

---

## 7. Seeding Demo Data

To populate the dashboard with realistic demo machines:

```bash
cd backend
node src/database/seed.js                    # base data
node seed-diverse-machines.js               # 15 machines with varied risk levels
node seed-smart-data-all-machines.js        # SMART history for all machines
node seed-lstm-model.js                     # seed ML model record in DB
```

---

## 8. Ports Summary

| Service | Port | URL |
|---------|------|-----|
| Frontend | 5173 | http://localhost:5173 |
| Backend API | 3000 | http://localhost:3000 |
| ML Service | 5000 | http://localhost:5000 |
| PostgreSQL | 5432 | localhost:5432 |
| Ollama | 11434 | http://localhost:11434 |

---

## 9. Full Technical Documentation

See `PFE_ADVANCED_TECHNICAL_DOCUMENTATION.md` for:
- Complete architecture analysis
- All API endpoints
- Database schema
- ML model details
- Security model
- Defense preparation (jury Q&A)
