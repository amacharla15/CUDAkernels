import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "00_finetune_artifact/outputs/qwen_lora_adapter")
MERGED_DIR = os.environ.get("MERGED_DIR", "00_finetune_artifact/outputs/qwen_lora_merged")

def main():
    print("Base model:", BASE_MODEL)
    print("Adapter dir:", ADAPTER_DIR)
    print("Merged dir:", MERGED_DIR)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device_map = "auto"
    else:
        torch_dtype = torch.float32
        device_map = None

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    lora_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    merged_model = lora_model.merge_and_unload()

    merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)

    print("Saved merged model to:", MERGED_DIR)

if __name__ == "__main__":
    main()
