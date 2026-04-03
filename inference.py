from mlx_lm import generate, load

adapter_path = "adapters"

model_path = "mlx-community/Qwen3.5-0.8B-4bit"
model_lora, tokenizer = load(model_path, adapter_path=adapter_path)

prompt = "The truth is rarely pure and never simple. ->: "
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

response = generate(
    model_lora, 
    tokenizer, 
    prompt=prompt, 
    verbose=True
    )