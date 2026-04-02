import json
import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# import matplotlib.pyplot as plt
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.tuner import TrainingArgs, datasets, linear_to_lora_layers, train
from transformers import PreTrainedTokenizer

from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()
login()

model_path = "mlx-community/Qwen3.5-0.8B-4bit"
model, tokenizer = load(model_path)

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

adapter_path = "adapters"
os.makedirs(adapter_path, exist_ok=True)
adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
adapter_file_path = os.path.join(adapter_path, "adapters.safetensors")

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

training_args = TrainingArgs(
    adapter_file=adapter_file_path,
    iters=200,
    grad_checkpoint = True,
    grad_accumulation_steps = 4,
    steps_per_report=1
)

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

### Load Dataset ###
# def custom_load_hf_dataset(
#     data_id: str,
#     tokenizer: PreTrainedTokenizer,
#     names: Tuple[str, str, str] = ("train", "valid", "test"),
# ):
#     from datasets import exceptions, load_dataset

#     try:
#         dataset = load_dataset(data_id)
#         dataset = transform_dataset(dataset)

#         train, valid, test = [
#             (
#                 datasets.create_dataset(dataset[n], tokenizer, config=datasets.TextDataset)
#                 if n in dataset.keys()
#                 else []
#             )
#             for n in names
#         ]

#     except exceptions.DatasetNotFoundError:
#         raise ValueError(f"Not found Hugging Face dataset: {data_id} .")

#     return train, valid, test


# def transform_dataset(dataset):
#     def merge_columns(example):
#         # example["prompt"] = example["quote"] + " ->: " 
#         # example["completion"] = str(example["tags"])
#         example['text'] = example["quote"] + " ->: " + str(example["tags"])
#         return example
    
#     dataset = dataset.map(merge_columns)
#     dataset = dataset.remove_columns(['quote', 'author', 'tags'])
#     return dataset

# train_set, val_set, test_set = custom_load_hf_dataset(
#     data_id="Abirate/english_quotes",
#     tokenizer=tokenizer,
#     names=("train", "", ""),
# )

train_set, val_set, test_set = (
    datasets.CacheDataset(ds) for ds in 
        datasets.load_local_dataset(
        data_path=Path('./data'),
        tokenizer=tokenizer,
        config=datasets.CompletionsDataset
    )
)

# train_set = datasets.CacheDataset(datasets.CompletionsDataset(
#     train_set, 
#     tokenizer, 
#     prompt_key='prompt', 
#     completion_key='completion', 
#     mask_prompt=True
# ))

# print(train_set[0])

train(
    model=model,
    args=training_args,
    optimizer=optim.Adam(learning_rate=1e-5),
    train_dataset=train_set,
    training_callback=metrics
)