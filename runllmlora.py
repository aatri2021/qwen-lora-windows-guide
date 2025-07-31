from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import torch

# Paths
base_model = "D:/AI/models/Qwen-0.6B-Base"
lora_model = "D:/AI/qwen_lora_output"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    trust_remote_code=True,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16,
    local_files_only=True,
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, lora_model)
model.eval()

# Prompt
prompt = "### Instruction:\nTell me a joke about programmers.\n\n### Input:\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generation config
generation_config = GenerationConfig(
    max_new_tokens=100,
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

# Decode and print result
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

