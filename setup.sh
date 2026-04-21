#!/usr/bin/env bash
# telco-llm-platform — tek komutluk kurulum ve smoke test
# Kullanim:
#   bash setup.sh                 # kur + API'yi arka planda baslat + test et
#   bash setup.sh --train         # ayrica mini LoRA egitimi de yapar
#   bash setup.sh --stop          # arka plandaki API'yi durdurur

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR=".venv"
LOG_DIR=".logs"
API_PID_FILE="$LOG_DIR/api.pid"
MCP_PID_FILE="$LOG_DIR/mcp.pid"

mkdir -p "$LOG_DIR"

log() { echo -e "\033[1;36m[setup]\033[0m $*"; }
err() { echo -e "\033[1;31m[err]\033[0m $*" >&2; }

# --- stop flag ---
if [[ "${1:-}" == "--stop" ]]; then
    for f in "$API_PID_FILE" "$MCP_PID_FILE"; do
        if [[ -f "$f" ]]; then
            pid="$(cat "$f")"
            if kill "$pid" 2>/dev/null; then
                log "pid $pid durduruldu ($f)"
            fi
            rm -f "$f"
        fi
    done
    exit 0
fi

# --- 0) python kontrol ---
if ! command -v python3 >/dev/null 2>&1; then
    err "python3 bulunamadi. Lutfen Python 3.10+ kurun."
    exit 1
fi

PY_VER="$(python3 -c 'import sys; print("%d.%d"%sys.version_info[:2])')"
log "Python versiyonu: $PY_VER"

# --- 1) venv ---
if [[ ! -d "$VENV_DIR" ]]; then
    log "venv olusturuluyor..."
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

log "pip guncelleniyor..."
pip install --upgrade pip >/dev/null

# --- 2) bagimliliklar ---
REQ_FILE="requirements-cpu.txt"
log "bagimliliklar kuruluyor ($REQ_FILE) — bu birkac dakika surebilir..."
pip install -r "$REQ_FILE"

# --- 3) env ---
export PYTHONPATH="$ROOT_DIR"
export TELCO_MODEL_MODE="${TELCO_MODEL_MODE:-mock}"

# --- 4) (opsiyonel) mini egitim ---
if [[ "${1:-}" == "--train" ]]; then
    log "mini LoRA egitimi baslatiliyor (Qwen2.5-0.5B, 1 epoch)..."
    python finetune/train.py \
        --base_model Qwen/Qwen2.5-0.5B \
        --data_path finetune/data/telco_qa.jsonl \
        --output_dir finetune/outputs/telco-lora \
        --epochs 1 \
        --batch_size 2 2>&1 | tee "$LOG_DIR/train.log"
fi

# --- 5) API'yi arka planda baslat ---
log "API mock modda arka planda baslatiliyor (port 8000)..."
nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 \
    > "$LOG_DIR/api.log" 2>&1 &
echo $! > "$API_PID_FILE"

# --- 6) MCP HTTP dev sunucusunu arka planda baslat ---
log "MCP HTTP sunucusu arka planda baslatiliyor (port 8765)..."
nohup python -m mcp_server.server --http --port 8765 \
    > "$LOG_DIR/mcp.log" 2>&1 &
echo $! > "$MCP_PID_FILE"

# --- 7) saglik kontrol ---
log "API hazir olana kadar bekleniyor..."
for i in {1..30}; do
    if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
        log "API ayakta"
        break
    fi
    sleep 1
    if [[ $i -eq 30 ]]; then
        err "API 30 saniyede ayaga kalkmadi. '$LOG_DIR/api.log' dosyasina bakin."
        exit 1
    fi
done

for i in {1..20}; do
    if curl -fsS http://localhost:8765/health >/dev/null 2>&1; then
        log "MCP HTTP ayakta"
        break
    fi
    sleep 1
done

# --- 8) smoke test ---
log "smoke test calistiriliyor..."
bash scripts/test_smoke.sh || true

cat <<EOF

=========================================================
 telco-llm-platform hazir
 ---------------------------------------------------------
 API:       http://localhost:8000
 API docs:  http://localhost:8000/docs
 MCP HTTP:  http://localhost:8765
 Loglar:    $LOG_DIR/api.log, $LOG_DIR/mcp.log

 Ornek istek:
   curl -s -X POST http://localhost:8000/chat \\
     -H 'content-type: application/json' \\
     -d '{"message":"Internet yavas","phone":"5551234567","region":"Istanbul"}'

 Durdurmak icin:  bash setup.sh --stop
=========================================================
EOF
