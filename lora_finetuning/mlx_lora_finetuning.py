import json
import os
from typing import Dict, List,Tuple, Union
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.tuner import TrainingArgs, datasets, linear_to_lora_layers, train


from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()
login()

# Load the base model and tokenizer
model_path = "mlx-community/Qwen3.5-0.8B-4bit"
model, tokenizer = load(model_path)

# # Test inference on base model
# prompt = "What is fine-tuning in machine learning?"
# messages = [{"role": "user", "content": prompt}]
# prompt = tokenizer.apply_chat_template(
#     messages, 
#     tokenize=False, 
#     add_generation_prompt=True
# )
# response = generate(
#     model, 
#     tokenizer, 
#     prompt=prompt, 
#     verbose=True,
#     max_tokens=-1
# )

# Paths to save LoRA parameters and trained weights
adapter_path = "adapters"
os.makedirs(adapter_path, exist_ok=True)
adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
adapter_file_path = os.path.join(adapter_path, "adapters.safetensors")

# Define LoRA parameters
lora_config = {
    "num_layers": 8,
    "lora_parameters": {
        "rank": 8,
        "scale": 20.0,
        "dropout": 0.0,
    },
}

with open(adapter_config_path, "w") as f:
    json.dump(lora_config, f, indent=4)

model.freeze()
linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"])
num_train_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
print(f"Number of trainable parameters: {num_train_params}")
model.train()

# Class to follow the training metrics
class Metrics:
    def __init__(self) -> None:
        self.train_losses: List[Tuple[int, float]] = []
        self.val_losses: List[Tuple[int, float]] = []

    def on_train_loss_report(self, info: Dict[str, Union[float, int]]) -> None:
        self.train_losses.append((info["iteration"], info["train_loss"]))

    def on_val_loss_report(self, info: Dict[str, Union[float, int]]) -> None:
        self.val_losses.append((info["iteration"], info["val_loss"]))

metrics = Metrics()

# Load Dataset
train_set, valid_set, test_set = (
    datasets.CacheDataset(ds) for ds in 
        datasets.load_local_dataset(
        data_path=Path('./data'),
        tokenizer=tokenizer,
        config=datasets.TextDataset
    )
)

# print('\nTrain Data Example: \n', train_set[0])
# print('\nValid Data Example: \n', valid_set[0])
# print('\nTest Data Example: \n', test_set[0])

# Define training parameters and Run finetuning
training_args = TrainingArgs(
    adapter_file=adapter_file_path,
    iters=100,
    steps_per_eval=10,
    steps_per_save=50,
    grad_checkpoint=True,
    batch_size=4,
    grad_accumulation_steps=4,
    steps_per_report=1
)

# Cosine schedule
lr_schedule = optim.cosine_decay(init=1e-8, decay_steps=50, end=1e-4)

train(
    model=model,
    args=training_args,
    optimizer=optim.Adam(learning_rate=lr_schedule),
    train_dataset=train_set,
    val_dataset=valid_set,
    training_callback=metrics
)

# Save training loss plot
train_its, train_losses = zip(*metrics.train_losses)
validation_its, validation_losses = zip(*metrics.val_losses)
plt.plot(train_its, train_losses, "-o", label="Train")
plt.plot(validation_its, validation_losses, "-o", label="Validation")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.savefig('outputs/mlx_lora_loss.png')