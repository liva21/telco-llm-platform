"""
Fine-tuned LoRA adapter icin degerlendirme.
BLEU, ROUGE-L ve basit domain-accuracy hesaplar.

Kullanim:
    python finetune/evaluate.py \
        --base_model Qwen/Qwen2.5-0.5B \
        --adapter_dir finetune/outputs/telco-lora \
        --eval_file finetune/data/telco_qa.jsonl
"""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import evaluate as hf_evaluate
    bleu = hf_evaluate.load("bleu")
    rouge = hf_evaluate.load("rouge")
    HF_METRICS = True
except Exception:
    HF_METRICS = False


TELCO_KEYWORDS = [
    "tarife", "fatura", "roaming", "modem", "bağlantı", "internet", "numara",
    "taşıma", "5g", "4.5g", "ip", "hız", "paket", "mobil", "destek",
]


def domain_accuracy(pred: str, gold: str) -> float:
    """Basit heuristik: altin cevap anahtar kelimelerinin kac tanesi predictte var."""
    gold_tokens = {t for t in gold.lower().split() if t in TELCO_KEYWORDS}
    if not gold_tokens:
        return 1.0 if any(k in pred.lower() for k in TELCO_KEYWORDS) else 0.5
    hits = sum(1 for t in gold_tokens if t in pred.lower())
    return hits / len(gold_tokens)


def generate(model, tokenizer, question: str, max_new_tokens: int = 256) -> str:
    prompt = (
        "<|im_start|>system\nSen Turk Telekom musteri hizmetleri asistanisin.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter_dir", default="finetune/outputs/telco-lora")
    parser.add_argument("--eval_file", default="finetune/data/telco_qa.jsonl")
    parser.add_argument("--max_samples", type=int, default=20)
    args = parser.parse_args()

    print(f"[info] base: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    if Path(args.adapter_dir).exists():
        print(f"[info] LoRA adapter yukleniyor: {args.adapter_dir}")
        model = PeftModel.from_pretrained(base, args.adapter_dir)
    else:
        print("[warn] adapter yok, base model degerlendirilecek")
        model = base

    model.eval()

    preds, refs, domain_scores = [], [], []
    with open(args.eval_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.max_samples:
                break
            row = json.loads(line)
            q = row.get("question") or row.get("instruction")
            a = row.get("answer") or row.get("response")
            pred = generate(model, tokenizer, q)
            preds.append(pred)
            refs.append(a)
            d = domain_accuracy(pred, a)
            domain_scores.append(d)
            print(f"\n[{i+1}] Q: {q}\n    A*: {a}\n    A^: {pred}\n    domain={d:.2f}")

    result = {"domain_accuracy": sum(domain_scores) / max(len(domain_scores), 1)}
    if HF_METRICS:
        try:
            result["bleu"] = bleu.compute(predictions=preds, references=[[r] for r in refs])["bleu"]
            result["rougeL"] = rouge.compute(predictions=preds, references=refs)["rougeL"]
        except Exception as e:
            result["metrics_error"] = str(e)

    print("\n=== Sonuclar ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()