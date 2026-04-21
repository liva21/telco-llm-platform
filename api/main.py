"""
FastAPI servis katmani.
- /chat: kullanicidan gelen soruyu alir, gerekirse MCP araclarini cagirir,
         sonrasinda fine-tuned LLM'e gonderir.
- /health: liveness/readiness probe
- /tools: MCP tool listesini yayinlar
- /metrics: basit prometheus uyumlu sayaclar

Model katmani:
- TELCO_MODEL_MODE=mock  -> gercek model yuklemeden kural tabanli cevap (CI/test icin)
- TELCO_MODEL_MODE=local -> transformers + PEFT ile LoRA adapter yukler
"""
import json
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from mcp_server.server import TOOL_REGISTRY, call_tool, list_tools_spec

MODE = os.getenv("TELCO_MODEL_MODE", "mock").lower()
BASE_MODEL = os.getenv("TELCO_BASE_MODEL", "Qwen/Qwen2.5-0.5B")
ADAPTER_DIR = os.getenv("TELCO_ADAPTER_DIR", "finetune/outputs/telco-lora")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Telco LLM Platform",
    version="1.0.0",
    description="Turk Telekom domain fine-tuned + MCP tool destekli asistan.",
)

# CORS izni
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_COUNTERS = {"requests_total": 0, "tool_calls_total": 0, "errors_total": 0}


class ChatRequest(BaseModel):
    message: str = Field(..., description="Kullanicinin mesaji")
    phone: Optional[str] = Field(None, description="Musteri telefon numarasi (opsiyonel)")
    region: Optional[str] = Field(None, description="Musteri bolgesi (opsiyonel)")
    max_tokens: int = Field(256, ge=16, le=2048)
    is_first_turn: bool = Field(
        False, description="True ise selamlama uretilir, aksi halde dogrudan konu"
    )


class ChatResponse(BaseModel):
    answer: str
    intent: str = "general"
    tool_calls: List[Dict[str, Any]] = []
    mode: str
    latency_ms: int


# ------------------- Model katmani -------------------
_llm = {"tokenizer": None, "model": None}


def load_local_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[api] local model yukleniyor base={BASE_MODEL} adapter={ADAPTER_DIR}", flush=True)
    print(f"[api] tokenizer yukleniyor...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[api] base model yukleniyor...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = base
    if os.path.isdir(ADAPTER_DIR):
        print(f"[api] LoRA adapter yukleniyor...", flush=True)
        try:
            model = PeftModel.from_pretrained(base, ADAPTER_DIR)
            print(f"[api] LoRA adapter yuklendi: {ADAPTER_DIR}", flush=True)
        except Exception as e:
            print(f"[api] adapter yuklenemedi ({e}), base model kullanilacak", flush=True)
    model.eval()
    print(f"[api] model.eval() tamamlandi.", flush=True)
    _llm["tokenizer"] = tok
    _llm["model"] = model


@app.on_event("startup")
def _startup():
    if MODE == "local":
        load_local_model()
    else:
        print(f"[api] MODE={MODE} (mock) - gercek model yuklenmeyecek")


# ------------------- Intent detection -------------------
INTENT_KEYWORDS = {
    "billing": [
        "fatura", "ödeme", "odeme", "borç", "borc", "ödenmemiş", "odenmemis",
        "tutar", "ödenmedi", "odenmedi", "gecikme", "son ödeme", "itiraz",
    ],
    "network": [
        "bağlantı", "baglanti", "yavaş", "yavas", "internet kesintisi", "kesinti",
        "sebeke", "şebeke", "sinyal", "kopuyor", "hat çekmiyor",
    ],
    "tariff": [
        "tarife", "paket", "kota", "kota aşımı", "hız düştü", "hızım düşük",
        "ek paket", "gb", "dakika", "sms", "tarife değiştir", "tarife yükselt",
        "yukselt", "downgrade", "upgrade",
    ],
    "roaming": ["roaming", "yurtdışı", "yurtdisi", "yurt dışı", "yurt disi"],
    "porting": ["numara taşıma", "numara tasima", "taşıma", "tasima", "operator degis"],
    "technical": ["modem", "router", "arıza", "ariza", "statik ip", "port", "wifi", "şifre", "sifre", "kablosuz", "wpa", "192.168"],
    "account": [
        "dondur", "dondurma", "askıya", "askiya", "iptal", "sonlandır", "sonlandir",
        "kapama", "kapat", "hat dondur", "abonelik", "aboneliğim", "hesabım",
        "pasif", "aktif et", "yeniden aç",
    ],
}


def _detect_intent(msg: str) -> str:
    msg = msg.lower()
    scores = {k: sum(1 for kw in kws if kw in msg) for k, kws in INTENT_KEYWORDS.items()}
    top = max(scores.items(), key=lambda x: x[1])
    return top[0] if top[1] > 0 else "general"


def _decide_tools(req: ChatRequest, intent: str) -> List[Dict[str, Any]]:
    """Intent-aware tool selection — her intent icin yalnizca ilgili tool'lar."""
    calls = []

    # musteri bilgisi: telefon varsa her zaman fayda saglar
    if req.phone:
        calls.append({"name": "get_customer_info", "arguments": {"phone": req.phone}})

    # sebeke durumu SADECE network intent'inde
    if intent == "network":
        region = req.region or "Türkiye"
        calls.append({"name": "check_network_status", "arguments": {"region": region}})

    # KB: general ve technical disindaki intent'lerde kisa sorgu, aksi halde tam mesaj
    kb_query = req.message
    if intent in {"billing", "roaming", "porting", "tariff"}:
        kb_query = f"{intent} {req.message}"
    calls.append({"name": "query_knowledge_base", "arguments": {"query": kb_query, "top_k": 3}})
    return calls


# ------------------- Cevap uretim (mock + local) -------------------
INTENT_OPENINGS = {
    "billing": "Fatura bilgilerinizi kontrol ettim.",
    "network": "Şebeke ve bağlantı durumunuza baktım.",
    "tariff": "Tarife ve paket seçeneklerini inceledim.",
    "roaming": "Roaming seçeneklerini taradım.",
    "porting": "Numara taşıma bilgilerini getirdim.",
    "technical": "Teknik tarafta atacağımız adımlara baktım.",
    "account": "Hesap/abonelik işleminizi inceledim.",
    "general": "Sorunuzu inceledim.",
}

INTENT_ACTIONS = {
    "billing": (
        "Ödeme için operatörünüzün mobil uygulaması veya Online İşlemler portali üzerinden "
        "kredi kartı ya da havale ile yapabilirsiniz. Fatura itirazı için "
        "444 0 375 numarasını arayın (15 iş günü içinde dönüş yapılır)."
    ),
    "network": (
        "Önce modeminizi 30 saniye kapatıp açın. Sorun devam ederse "
        "444 0 375 numarasından arıza kaydı açtırabilirsiniz."
    ),
    "tariff": (
        "Tarife değişikliği için mobil uygulama → Tarifem → Tarife Değiştir "
        "menüsünü kullanabilirsiniz. Taahhütlü hatlarda cayma bedeli oluşabilir."
    ),
    "roaming": (
        "Yolculuk öncesi Online İşlemler → Roaming menüsünden ülke bazlı paket aktive edin. "
        "Aktivasyon en az 24 saat önceden yapılması önerilir."
    ),
    "porting": (
        "Numara taşıma için kimliğinizle en yakın bayiye başvurun. "
        "1-3 iş günü içinde tamamlanır."
    ),
    "technical": (
        "Teknik destek için 444 0 375'i arayarak arıza kaydı açabilirsiniz. "
        "Modem arızası doğrulanırsa değişim ücretsiz gerçekleştirilir."
    ),
    "account": (
        "Hat dondurma/askıya alma işlemi için operatörünüzün müşteri hizmetlerini "
        "(444 0 375 veya benzeri hat) arayın ya da Online İşlemler portalından "
        "Hattım → Abonelik İşlemleri bölümüne gidin. "
        "Dondurma süresi genellikle 1-12 ay arasında seçilebilir."
    ),
    "general": "Daha fazla destek için 444 0 375 numarasını arayabilirsiniz.",
}


def _extract_customer_summary(tool_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for tc in tool_results:
        if tc["name"] == "get_customer_info":
            r = tc["result"] or {}
            # Tool artik 'found' donmeyebilir, phone varsa bulundu sayalim
            if "phone" in r:
                return r
    return None


def _extract_network_summary(tool_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for tc in tool_results:
        if tc["name"] == "check_network_status":
            return tc["result"]
    return None


def _extract_kb_docs(tool_results: List[Dict[str, Any]]) -> List[Any]:
    for tc in tool_results:
        if tc["name"] == "query_knowledge_base":
            res = tc["result"] or {}
            return res.get("results", [])
    return []


def _render_billing_facts(cust: Optional[Dict[str, Any]]) -> str:
    if not cust:
        return "Musteri bilgileri saglanmadigi icin fatura ayrintisina erisemedim."
    bal = cust.get("balance_try")
    due = cust.get("invoice_due")
    name = cust.get("name", "Sayin musterimiz")
    if bal is None:
        return f"{name} adina kayitli hesabi inceledim; guncel fatura bilgisi sistemde gozukmuyor."
    if bal > 0:
        return (
            f"{name}, sisteme gore {bal:.2f} TL tutarinda odenmemis bakiyeniz var "
            f"ve son odeme tarihi {due}."
        )
    if bal < 0:
        return f"{name}, hesabiniz {abs(bal):.2f} TL alacakli gorunuyor, odenmemis fatura yok."
    return f"{name}, odenmemis faturaniz bulunmuyor. Son odeme tarihi: {due}."


def _render_network_facts(net: Optional[Dict[str, Any]]) -> str:
    if not net:
        return ""
    status = net.get("status", "bilinmiyor")
    region = net.get("region", "bölgeniz")
    latency = net.get("latency_ms")
    loss = net.get("packet_loss_pct")
    extra = ""
    if latency is not None and loss is not None:
        extra = f" (gecikme {latency} ms, kayip %{loss})"
    if status.upper() == "OK":
        return f"{region} bölgesinde şebeke sağlıklı{extra}. Bölgesel bir arıza görünmüyor."
    if status.upper() == "DEGRADED":
        return f"{region} bölgesinde hafif kalite düşüklüğü var{extra}."
    return f"{region} bölgesinde şebeke arızası tespit ettim (durum: {status})."


def _render_kb_snippet(docs: List[Any], limit: int = 2) -> str:
    if not docs:
        return ""
    lines = ["Bilgi tabanindan ilgili makaleler:"]
    for d in docs[:limit]:
        if isinstance(d, dict):
            lines.append(f"- {d.get('title','?')}: {d.get('content','').strip()}")
        else:
            lines.append(f"- {d}")
    return "\n".join(lines)


def _mock_generate(
    user_msg: str,
    tool_results: List[Dict[str, Any]],
    intent: str,
    is_first_turn: bool,
) -> str:
    """Intent-aware, akiskan, selamlama icermeyen mock cevap uretici."""
    cust = _extract_customer_summary(tool_results)
    net = _extract_network_summary(tool_results)
    docs = _extract_kb_docs(tool_results)

    parts = []

    # --- Ana teshis / durum bölümü ---
    if intent == "network":
        net_txt = _render_network_facts(net)
        if net_txt:
            parts.append("🔍 **Şebeke Durumu**\n" + net_txt)
        if cust:
            parts.append(
                f"📋 **Hat Bilginiz:** {cust.get('plan', '-')} paketiniz aktif görünüyor."
            )
    elif intent == "billing":
        billing_txt = _render_billing_facts(cust) if cust else ""
        if billing_txt:
            parts.append("💳 **Fatura Durumu**\n" + billing_txt)
    # tariff/account/roaming/porting/general: hesap bilgisi istenmiyor

    # --- Bilgi tabanı önerileri ---
    kb_txt = _render_kb_snippet(docs)
    if kb_txt:
        parts.append("📚 **Önerilen Adımlar**\n" + kb_txt)

    # --- Eylem çağrısı ---
    action = INTENT_ACTIONS.get(intent, INTENT_ACTIONS["general"])
    parts.append("💡 " + action)

    return "\n\n".join(p for p in parts if p)


def _local_generate(
    user_msg: str,
    tool_results: List[Dict[str, Any]],
    max_tokens: int,
    intent: str,
    is_first_turn: bool,
) -> str:
    import torch
    tok = _llm["tokenizer"]
    model = _llm["model"]

    ctx_txt = ""
    for tc in tool_results:
        ctx_txt += f"\n[{tc['name']}] {json.dumps(tc['result'], ensure_ascii=False)[:400]}"

    greeting_rule = (
        "Bu konusmanin ilk mesaji, kisa bir selamlama ile baslayabilirsin."
        if is_first_turn
        else "Bu devam eden bir konusma, SELAMLAMA YAPMA ve kendini tekrar tanitma. Dogrudan konuya gir."
    )
    intent_rule = {
        "billing": "Kullanicinin sorusu FATURA/ODEME ile ilgili. Modem/sebeke onerilerinden bahsetme.",
        "network": "Kullanicinin sorusu INTERNET/SEBEKE ile ilgili. Fatura detayina girme.",
        "tariff": "Kullanicinin sorusu TARIFE/PAKET ile ilgili.",
        "roaming": "Kullanicinin sorusu ROAMING ile ilgili.",
        "porting": "Kullanicinin sorusu NUMARA TASIMA ile ilgili.",
        "technical": "Kullanicinin sorusu TEKNIK ARIZA ile ilgili.",
        "general": "Genel bir soru; uygun ve kisa cevap ver.",
    }.get(intent, "Genel bir soru; uygun ve kisa cevap ver.")

    prompt = f"Kullanici: {user_msg}\nAsistan:"
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


# ------------------- Endpoints -------------------
@app.get("/health")
def health():
    return {"status": "ok", "mode": MODE}


@app.get("/ready")
def ready():
    if MODE == "local" and _llm["model"] is None:
        raise HTTPException(status_code=503, detail="model hazir degil")
    return {"ready": True}


@app.get("/tools")
def tools():
    return {"tools": list_tools_spec()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    _COUNTERS["requests_total"] += 1
    t0 = time.time()
    try:
        intent = _detect_intent(req.message)
        planned = _decide_tools(req, intent)
        executed: List[Dict[str, Any]] = []
        for c in planned:
            _COUNTERS["tool_calls_total"] += 1
            executed.append({"name": c["name"], "arguments": c["arguments"],
                             "result": call_tool(c["name"], c["arguments"])})

        if MODE == "local":
            answer = _local_generate(req.message, executed, req.max_tokens,
                                     intent, req.is_first_turn)
        else:
            answer = _mock_generate(req.message, executed, intent, req.is_first_turn)

        return ChatResponse(
            answer=answer,
            intent=intent,
            tool_calls=executed,
            mode=MODE,
            latency_ms=int((time.time() - t0) * 1000),
        )
    except HTTPException:
        raise
    except Exception as e:
        _COUNTERS["errors_total"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = []
    for k, v in _COUNTERS.items():
        lines.append(f"# TYPE telco_{k} counter")
        lines.append(f"telco_{k} {v}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)