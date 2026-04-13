from mlx_lm import generate, load

adapter_path = "adapters"

model_path = "mlx-community/Qwen3.5-0.8B-4bit"
model_lora, tokenizer = load(model_path, adapter_path=adapter_path)

prompt = "I'm quite illiterate, but I read a lot. ->: "

response = generate(
    model_lora, 
    tokenizer, 
    prompt=prompt, 
    verbose=True
)