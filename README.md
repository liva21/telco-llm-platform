# telco-llm-platform

Telekomünikasyon domain'ine fine-tuned, **MCP destekli**, **OpenShift-ready** LLM platformu.

- **Fine-tuning:** Qwen2.5 / Mistral üzerinde LoRA (PEFT + TRL), MLflow ile deney takibi
- **MCP Server:** `query_knowledge_base`, `get_customer_info`, `check_network_status` araçları
- **API:** FastAPI — `/chat` endpoint, tool orchestration, metrics
- **Deployment:** Docker, docker-compose, OpenShift manifests (Deployment, Route, ImageStream)

---

## 0) Ön koşullar

- Python 3.10+
- Docker 20+ (opsiyonel, konteyner çalıştırmak için)
- `git` (opsiyonel)
- GPU gerekmez — her şey CPU'da çalışır. GPU varsa daha hızlı olur.

---

## 1) Hızlı başlangıç (copy-paste)

```bash
# Projeyi aç
cd telco-llm-platform

# Venv + bagimliliklar
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# PYTHONPATH
export PYTHONPATH="$(pwd)"

# API'yi MOCK modda baslat (model yuklemez, test icin)
export TELCO_MODEL_MODE=mock
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Başka bir terminalde:

```bash
cd telco-llm-platform
bash scripts/test_smoke.sh
```

Beklenen çıktı: `/health`, `/tools` ve `/chat` endpoint'lerinden JSON cevap.

---

## 2) Fine-tuning (LoRA)

CPU'da küçük bir modelle hızlı denemek için:

```bash
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

python finetune/train.py \
    --base_model Qwen/Qwen2.5-0.5B \
    --data_path finetune/data/telco_qa.jsonl \
    --output_dir finetune/outputs/telco-lora \
    --epochs 1 \
    --batch_size 2
```

Google Colab Free (T4) üzerinde 7B model ile:

```bash
python finetune/train.py \
    --base_model mistralai/Mistral-7B-v0.1 \
    --epochs 3 \
    --batch_size 4 \
    --use_4bit
```

MLflow UI:

```bash
mlflow ui --backend-store-uri file:./finetune/mlflow_tracking --port 5000
# http://localhost:5000
```

Değerlendirme:

```bash
python finetune/evaluate.py \
    --base_model Qwen/Qwen2.5-0.5B \
    --adapter_dir finetune/outputs/telco-lora \
    --eval_file finetune/data/telco_qa.jsonl
```

Çıktı: `domain_accuracy`, `bleu`, `rougeL` metrikleri.

---

## 3) MCP Server

Model Context Protocol sunucusu iki modda çalışır.

### HTTP dev modu (test için)

```bash
export PYTHONPATH="$(pwd)"
python -m mcp_server.server --http --port 8765

# Test:
curl -s http://localhost:8765/tools | python -m json.tool
curl -s -X POST http://localhost:8765/call \
  -H "content-type: application/json" \
  -d '{"name":"check_network_status","arguments":{"region":"Istanbul-Kadikoy"}}'
```

### Stdio modu (üretim / Claude Desktop / IDE entegrasyonu)

```bash
python -m mcp_server.server
# Bu mod stdin/stdout üzerinden JSON-RPC bekler; host uygulamadan çağrılır.
```

Claude Desktop `mcp.json` örneği:

```json
{
  "mcpServers": {
    "telco": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "env": { "PYTHONPATH": "/ABSOLUTE/PATH/telco-llm-platform" }
    }
  }
}
```

### Tool listesi

| Tool | Açıklama |
|------|----------|
| `query_knowledge_base(query, top_k=3)` | Telekom bilgi tabanında arama |
| `get_customer_info(phone)` | Telefon numarasına göre müşteri profili |
| `check_network_status(region)` | Bölge bazlı şebeke durumu |

---

## 4) API (FastAPI)

### Lokal çalıştırma

```bash
export PYTHONPATH="$(pwd)"
export TELCO_MODEL_MODE=mock           # model yüklemeden çalışır (CI/test)
# export TELCO_MODEL_MODE=local         # LoRA adapter yüklü gerçek üretim
# export TELCO_BASE_MODEL=Qwen/Qwen2.5-0.5B
# export TELCO_ADAPTER_DIR=finetune/outputs/telco-lora

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Ornek istek

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "content-type: application/json" \
  -d '{
    "message": "Internet baglantim yavas, ne yapmaliyim?",
    "phone": "5551234567",
    "region": "Istanbul-Kadikoy"
  }' | python -m json.tool
```

### Endpoints

| Endpoint | Açıklama |
|----------|----------|
| `GET /health` | Liveness |
| `GET /ready` | Readiness (model yüklü mü) |
| `GET /tools` | MCP tool şeması |
| `POST /chat` | Chat + tool orchestration |
| `GET /metrics` | Prometheus uyumlu sayaçlar |

---

## 5) Docker & docker-compose

```bash
# Build
docker build -t telco-llm-platform:latest .

# Tek container
docker run --rm -p 8000:8000 \
  -e TELCO_MODEL_MODE=mock \
  telco-llm-platform:latest

# Compose (api + mcp HTTP)
docker compose up --build
# http://localhost:8000  -> API
# http://localhost:8765  -> MCP HTTP
```

---

## 6) OpenShift deployment

OpenShift'i lokal kurmana **gerek yok**. Manifestler hazır; gerçek bir cluster'a veya `OpenShift Local` (CRC) kurduğunda şu sırayla uygula:

```bash
# 1) Namespace
oc apply -f openshift/namespace.yaml
oc project telco-llm

# 2) ImageStream + BuildConfig (OpenShift içinde build almak için)
oc apply -f openshift/imagestream.yaml
oc apply -f openshift/buildconfig.yaml

# 3) (Opsiyonel) Build'i tetikle — git URI'yi buildconfig.yaml icinde duzenle
oc start-build telco-llm-platform --follow

# 4) Deployment + Service + PVC
oc apply -f openshift/deployment.yaml

# 5) Route (public URL)
oc apply -f openshift/route.yaml
oc get route telco-llm-platform -o jsonpath='{.spec.host}'
```

Harici bir registry'den çalıştıracaksan:

```bash
# imagestream / deployment icindeki image adresini kendi registry'nle degistir
# ornek: quay.io/<user>/telco-llm-platform:latest
```

### OpenShift Local (CRC) — ücretsiz lokal cluster

```bash
# Mac/Linux
brew install crc        # macOS
# veya: https://developers.redhat.com/products/openshift-local

crc setup
crc start
eval $(crc oc-env)
oc login -u developer https://api.crc.testing:6443
# sonra yukaridaki 'oc apply' komutlari calisir
```

---

## 7) Proje yapısı

```
telco-llm-platform/
├── finetune/
│   ├── train.py                 # LoRA fine-tuning (peft + trl)
│   ├── evaluate.py              # BLEU, ROUGE, domain accuracy
│   ├── data/telco_qa.jsonl      # ornek telekom Q&A seed verisi
│   └── mlflow_tracking/         # MLflow deney kayitlari (runtime)
├── mcp_server/
│   ├── server.py                # MCP stdio + HTTP dev modu
│   └── tools/
│       ├── knowledge_base.py    # query_knowledge_base
│       ├── customer_info.py     # get_customer_info
│       └── network_status.py    # check_network_status
├── api/
│   └── main.py                  # FastAPI servis katmani
├── openshift/
│   ├── namespace.yaml
│   ├── deployment.yaml          # Deployment + Service + PVC
│   ├── route.yaml               # OpenShift Route (TLS edge)
│   ├── imagestream.yaml
│   └── buildconfig.yaml
├── scripts/
│   ├── run_api.sh
│   ├── run_mcp.sh
│   ├── train_local.sh
│   └── test_smoke.sh
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-cpu.txt
├── setup.sh                     # tek komutluk kurulum
└── README.md
```

---

## 8) Tek satirlik kurulum

```bash
bash setup.sh
```

Bu script:
- Python venv oluşturur
- CPU bağımlılıklarını kurar
- PYTHONPATH ayarlar
- API'yi mock modda arka planda başlatır
- Smoke test çalıştırır

---

## 9) Test

```bash
# API ayakta iken:
bash scripts/test_smoke.sh

# MCP dogrudan tool cagrisi:
curl -s -X POST http://localhost:8765/call \
  -H "content-type: application/json" \
  -d '{"name":"query_knowledge_base","arguments":{"query":"fatura itirazi"}}'
```

---

## 10) Özellikler 

- **LoRA PEFT** — tam ağırlık güncellemek yerine adapter eğitimi. 7B model Colab T4'te sığar.
- **TRL SFTTrainer** — instruction-tuning için hazır pipeline.
- **MLflow** — deney parametreleri + artifact versiyonlama.
- **MCP mimarisi** — LLM sadece metin üretmez, tool'lar çağırır (agentic pattern).
- **FastAPI + pydantic** — tip güvenli istek/cevap şemaları, OpenAPI auto-doc.
- **OpenShift-ready** — root-olmayan container, PVC ile adapter persistency, Route + TLS.
- **Mock mode** — CI/CD'de model yüklemeden tüm pipeline test edilebilir.

---

## Lisans

MIT 



<img width="1270" height="1544" alt="image" src="https://github.com/user-attachments/assets/97ae9ce5-e1ad-49df-8342-df7201bbcf21" />
<img width="1358" height="1482" alt="image" src="https://github.com/user-attachments/assets/3251139f-8663-4f68-bd32-8af20b835606" />


