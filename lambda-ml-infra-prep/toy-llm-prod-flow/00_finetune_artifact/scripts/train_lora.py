import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_PATH = os.environ.get("DATA_PATH", "00_finetune_artifact/data/train.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "00_finetune_artifact/outputs/qwen_lora_adapter")

def build_prompt(example):
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]

    if input_text.strip() != "":
        text = (
            "### Instruction:\n"
            + instruction
            + "\n\n### Input:\n"
            + input_text
            + "\n\n### Answer:\n"
            + output
        )
    else:
        text = (
            "### Instruction:\n"
            + instruction
            + "\n\n### Answer:\n"
            + output
        )

    return {"text": text}

def main():
    print("Base model:", BASE_MODEL)
    print("Data path:", DATA_PATH)
    print("Output dir:", OUTPUT_DIR)
    print("CUDA available:", torch.cuda.is_available())

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    dataset = dataset.map(build_prompt)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device_map = "auto"
    else:
        torch_dtype = torch.float32
        device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
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
        fp16=False,
        bf16=torch.cuda.is_available()
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

    print("Saved LoRA adapter to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
