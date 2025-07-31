import time
import json
import os
import torch
from torch.optim import AdamW
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_alpaca_data(filepath, start, limit=None):
    with open(filepath, "r", encoding="utf-8") as f:
        alpaca_data = json.load(f)

    total_samples = len(alpaca_data)

    if start >= total_samples:
        print(f"Start index {start} exceeds dataset size {total_samples}. No data loaded.")
        return []

    if limit is not None and start + limit > total_samples:
        limit = total_samples - start
        print(f"Limit adjusted to {limit} to avoid going past dataset end.")

    def format_entry(entry):
        if entry["input"].strip():
            prompt = f"{entry['instruction']}\n{entry['input']}"
        else:
            prompt = entry["instruction"]
        answer = entry["output"]
        return prompt, answer

    sliced_data = alpaca_data[start:start + limit] if limit else alpaca_data[start:]
    data = [format_entry(entry) for entry in sliced_data]
    print(f"Loaded {len(data)} training samples from {filepath} starting at {start}")
    return data


def load_model_and_tokenizer(model_id, device, lora_adapter_path, lora_config):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Make sure padding is valid for Qwen
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True
    )
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()

    if os.path.exists(lora_adapter_path):
        print("Found existing LoRA model, loading for additional training")
        try:
            model = PeftModel.from_pretrained(base_model, lora_adapter_path)
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
            model.print_trainable_parameters()
        except Exception as e:
            print(f"Error loading LoRA adapter from {lora_adapter_path}: {e}")
            exit(1)
    else:
        print("Training a brand new LoRA model")
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()

    model.train()
    return model, tokenizer


def train_loop(model, tokenizer, training_data, device, batch_size, num_epochs, learning_rate, lora_adapter_path, start):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        epoch_start_time = time.time()
        batch_times = []

        for i in range(0, len(training_data), batch_size):
            batch_start_time = time.time()

            batch = training_data[i:i + batch_size]
            optimizer.zero_grad()

            prompts = [f"{p}\nAnswer:" for p, _ in batch]
            answers = [a for _, a in batch]
            full_texts = [f"{p}\n{a}" for p, a in zip(prompts, answers)]

            encodings = tokenizer(
                full_texts,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            )

            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            labels = input_ids.clone()
            for idx, (prompt, _) in enumerate(batch):
                prompt_tokens = tokenizer(f"{prompt}", add_special_tokens=False)["input_ids"]
                labels[idx, :len(prompt_tokens)] = -100  # mask prompt part

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            batch_duration = time.time() - batch_start_time
            batch_times.append(batch_duration)

            avg_batch_time = sum(batch_times) / len(batch_times)
            batches_left = (len(training_data) - (i + batch_size)) / batch_size
            est_total_remaining = avg_batch_time * (batches_left + (num_epochs - epoch - 1) * len(training_data) / batch_size)

            print(
                f"✔️ Batch {i//batch_size + 1}, Loss: {loss.item():.4f}, "
                f"Time: {batch_duration:.2f}s, ETA: {est_total_remaining:.1f}s"
            )

        print(f"📅 Epoch {epoch + 1} completed in {time.time() - epoch_start_time:.2f} seconds")

    print(f"🏁 Training completed in {time.time() - total_start_time:.2f} seconds")

    model.save_pretrained(lora_adapter_path)
    tokenizer.save_pretrained(lora_adapter_path)

    end_sample_index = start + len(training_data)
    save_training_metadata(lora_adapter_path, batch_size, end_sample_index, num_epochs, learning_rate)


def save_training_metadata(path, batch_size, end_sample_index, num_epochs, learning_rate):
    metadata = {
        "batch_size": batch_size,
        "alpaca_end_sample_index": end_sample_index,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
    }
    with open(os.path.join(path, "alpaca_training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"📁 Saved training metadata to {path}/alpaca_training_metadata.json")


def main():
    model_id = "D:/AI/models/Qwen3-0.6B-Base"  # Adjust as needed
    lora_adapter_path = "D:/AI/models/Qwen3-0.6B-LoRA-Alpaca"
    alpaca_json_path = "D:/AI/data/alpaca.json"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    batch_size = 4
    num_epochs = 3
    learning_rate = 3e-5
    data_limit = 100
    start = 0

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_data = load_alpaca_data(alpaca_json_path, start, limit=data_limit)
    model, tokenizer = load_model_and_tokenizer(model_id, device, lora_adapter_path, lora_config)
    train_loop(model, tokenizer, training_data, device, batch_size, num_epochs, learning_rate, lora_adapter_path, start)

if __name__ == "__main__":
    main()
