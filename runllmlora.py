from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel
import torch

# Paths
base_model = "./Qwen3-0.6B-Base/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
lora_model = "./Qwen3-0.6B-LoRA-Alpaca"

# Toggle quantization here:
use_quantized = False  # Set True to enable 4-bit quantized loading, False for 16-bit fp16

if use_quantized:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model_kwargs = {
        "quantization_config": bnb_config,
        "trust_remote_code": True,
        "device_map": "auto",
        "local_files_only": True,
    }
else:
    model_kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "device_map": "auto",
        "local_files_only": True,
    }

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, lora_model)
model.eval()

# Qwen-Base style prompt (no <|im_start|>, plain instruction prompt)
prompt = (
    "### Instruction:\nWhat is the capital of France?\n\n### Response:\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generation config
generation_config = GenerationConfig(
    max_new_tokens=2000,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,  # Needed for Qwen to avoid warning
    eos_token_id=tokenizer.eos_token_id,
)

# Generate
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        generation_config=generation_config,
    )

# Decode and print result, skipping special tokens
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
