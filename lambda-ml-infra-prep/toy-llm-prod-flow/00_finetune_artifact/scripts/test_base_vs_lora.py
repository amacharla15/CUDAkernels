import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "00_finetune_artifact/outputs/qwen_lora_adapter")

PROMPTS = [
    "What is TTFT?",
    "What is the difference between fine-tuning and serving?",
    "Explain LoRA forward pass.",
    "What is QLoRA?",
    "What is distillation loss?"
]

def load_base_model():
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

    model.eval()
    return tokenizer, model

def generate(tokenizer, model, question):
    prompt = "### Instruction:\n" + question + "\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text

def main():
    print("Base model:", BASE_MODEL)
    print("Adapter dir:", ADAPTER_DIR)
    print("CUDA available:", torch.cuda.is_available())

    tokenizer, base_model = load_base_model()

    print("\n================ BASE MODEL ================\n")
    for prompt in PROMPTS:
        print("Question:", prompt)
        print(generate(tokenizer, base_model, prompt))
        print()

    print("\nLoading LoRA adapter...\n")
    lora_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    lora_model.eval()

    print("\n================ LORA MODEL ================\n")
    for prompt in PROMPTS:
        print("Question:", prompt)
        print(generate(tokenizer, lora_model, prompt))
        print()

if __name__ == "__main__":
    main()
