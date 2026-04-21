"""
LoRA Fine-tuning for Telecom Domain LLM
----------------------------------------
Qwen2.5-7B / Mistral-7B / TinyLlama uyumlu LoRA fine-tuning.
CPU/MPS/CUDA uyumlu. Google Colab Free tier (T4) uzerinde calisir.

Kullanim:
    python finetune/train.py --base_model Qwen/Qwen2.5-0.5B --epochs 1
    python finetune/train.py --base_model mistralai/Mistral-7B-v0.1 --epochs 3 --use_4bit
"""
import argparse
import json
import os
from pathlib import Path

import mlflow
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


def build_prompt(example: dict) -> dict:
    """Telekom Q&A kayitlarini chat formatina cevirir."""
    question = example.get("question") or example.get("instruction") or example.get("prompt", "")
    answer = example.get("answer") or example.get("response") or example.get("output", "")
    text = (
        "<|im_start|>system\n"
        "Sen Turk Telekom musteri hizmetleri asistanisin. "
        "Kullanici sorularini net, dogru ve saygili sekilde yanitla.\n<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>"
    )
    return {"text": text}


def load_telco_dataset(data_path: str) -> Dataset:
    """Yerel JSONL veriyi yukler. Yoksa seed ornek veri olusturur."""
    path = Path(data_path)
    if not path.exists():
        print(f"[warn] {data_path} yok, ornek seed veri olusturuluyor...")
        seed = [
            {"question": "Internet bağlantım çok yavaş, ne yapmalıyım?",
             "answer": "Modeminizi 30 saniye kapatıp yeniden başlatın, kablo bağlantılarını kontrol edin ve hız testi yapın. Sorun devam ederse 444 0 375 hattından destek alabilirsiniz."},
            {"question": "Fatura itirazı nasıl yapılır?",
             "answer": "Online İşlemler üzerinden 'Fatura İtirazı' başlığından dilekçenizi iletebilirsiniz. İtiraz başvurusu sonrası 15 iş günü içinde dönüş sağlanır."},
            {"question": "Tarife değişikliği nasıl yapılır?",
             "answer": "turktelekom.com.tr veya Türk Telekom mobil uygulaması üzerinden 'Tarifem' menüsünden mevcut tarifenizi değiştirebilirsiniz. Taahhütlü hatlarda cayma bedeli uygulanabilir."},
            {"question": "5G kapsama alanım nasıl öğrenirim?",
             "answer": "Türk Telekom kapsama haritası üzerinden adresinizi girerek 4.5G/5G hizmetlerinin sağlanıp sağlanmadığını öğrenebilirsiniz."},
            {"question": "Yurt dışında roaming açtırmak istiyorum.",
             "answer": "Online İşlemler > Roaming bölümünden gideceğiniz ülkeye uygun paketi aktive edebilirsiniz. Yolculuk öncesi 24 saat açtırmanız önerilir."},
            {"question": "Modem arızalı, değişim nasıl olur?",
             "answer": "Arızalı modem bildiriminizi 444 0 375 üzerinden açın. Teknik ekip uzaktan test sonrası arızayı doğrularsa değişim ücretsiz yapılır."},
            {"question": "Numara taşıma süreci nasıl işler?",
             "answer": "Türk Telekom bayilerine kimliğinizle başvurduktan sonra 1-3 iş günü içinde numara taşıma tamamlanır. Eski operatördeki borçların kapatılmış olması gerekir."},
            {"question": "Statik IP almak istiyorum.",
             "answer": "Bireysel müşteriler için statik IP hizmeti aylık ek ücretle sunulur. Online İşlemler üzerinden 'Ek Hizmetler' bölümünden aktive edebilirsiniz."},
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in seed:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    ds = load_dataset("json", data_files=str(path), split="train")
    return ds.map(build_prompt, remove_columns=ds.column_names)


def get_device_config(use_4bit: bool):
    """Donanima gore quantization ve dtype secer."""
    if torch.cuda.is_available():
        if use_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ), torch.float16, "cuda"
        return None, torch.float16, "cuda"
    if torch.backends.mps.is_available():
        return None, torch.float32, "mps"
    return None, torch.float32, "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B",
                        help="HF base model id (kucuk modeller CPU'da da calisir)")
    parser.add_argument("--data_path", default="finetune/data/telco_qa.jsonl")
    parser.add_argument("--output_dir", default="finetune/outputs/telco-lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--mlflow_uri", default="file:./finetune/mlflow_tracking")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("telco-llm-finetune")

    bnb_config, dtype, device = get_device_config(args.use_4bit)
    print(f"[info] cihaz={device} dtype={dtype} 4bit={bool(bnb_config)}")

    print(f"[info] tokenizer yukleniyor: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[info] model yukleniyor...")
    model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_telco_dataset(args.data_path)
    print(f"[info] egitim orneği sayisi: {len(dataset)}")

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        max_length=args.max_seq_length,
        packing=False,
        dataset_text_field="text",
        fp16=(device == "cuda" and not args.use_4bit),
        bf16=False,
        optim="adamw_torch",
    )

    with mlflow.start_run():
        mlflow.log_params({
            "base_model": args.base_model,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": device,
        })

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        trainer.train()

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        mlflow.log_artifacts(str(out_dir), artifact_path="lora_adapter")
        print(f"[done] LoRA adapter kaydedildi: {out_dir}")


if __name__ == "__main__":
    main()
    