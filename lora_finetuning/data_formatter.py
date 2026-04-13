import json
from datasets import load_dataset

# 1. Load the dataset from Hugging Face
# This dataset typically contains only a 'train' split
ds = load_dataset("Abirate/english_quotes")

# 2. Define the transformation function
def reformat_entry(example):
    curly_quotes = '“”'
    return {
        "text": example["quote"].strip(curly_quotes) + " ->: " + str(example['tags'])
    }
# def reformat_entry(example):
#     return {
#         "prompt": example["quote"],
#         "completion": str(example['tags'])
#     }

# 3. Apply the transformation and remove original columns
formatted_ds = ds["train"].map(reformat_entry, remove_columns=ds["train"].column_names)

# 4.1. First split: Separate 10% for the Test set
# The remaining 90% stays in 'train'
split_ds = formatted_ds.train_test_split(test_size=0.1, seed=42)

# 4.2. Second split: Split that 90% 'train' again to get a Validation set
# To get a 10% Val set of the TOTAL, use test_size=0.11 (which is 1/9th of the 90%)
train_val_split = split_ds["train"].train_test_split(test_size=0.11, seed=42)

# 5. Save the splits as JSONL files
def save_as_jsonl(dataset, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for record in dataset:
            f.write(json.dumps(record) + "\n")

save_as_jsonl(train_val_split["train"], "./data/train.jsonl")
save_as_jsonl(train_val_split["test"], "./data/valid.jsonl")
save_as_jsonl(split_ds["test"], "./data/test.jsonl")

print("Successfully saved 'train.jsonl', 'test.jsonl' and 'val.jsonl'.")
