import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_PATH = os.environ.get("DATA_PATH", "00_finetune_artifact/data/train.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "00_finetune_artifact/outputs/qwen_qlora_adapter")

def build_prompt(example):
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]

    if input_text.strip() != "":
        text = "### Instruction:\n" + instruction + "\n\n### Input:\n" + input_text + "\n\n### Answer:\n" + output
    else:
        text = "### Instruction:\n" + instruction + "\n\n### Answer:\n" + output

    return {"text": text}

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA needs CUDA. Run this on RunPod/A100, not CPU.")

    print("Base model:", BASE_MODEL)
    print("Data path:", DATA_PATH)
    print("Output dir:", OUTPUT_DIR)
    print("CUDA available:", torch.cuda.is_available())

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    dataset = dataset.map(build_prompt)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        bf16=True
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        dataset_text_field="text",
        max_seq_length=512
    )

    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Saved QLoRA adapter to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
