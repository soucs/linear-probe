import os
import json
import numpy as np

JSON_PATH = "data/deceptive_outputs_labeled.jsonl"
ACTIVATIONS_DIR = "data/activations/"

# Layer Mapping
# 0: Embedding, 1-24: Hidden Layers, 25: Final Norm
LAYER_MAP = {
    'emb': 0,
    **{str(i): i + 1 for i in range(24)},
    'norm': 25
}

JSON_PATH = "data/deceptive_outputs_labeled.jsonl"
ACTIVATIONS_DIR = "data/activations/"

def load_dataset(layer_key):
    layer_idx = LAYER_MAP[str(layer_key)]
    X = []
    y = []
    
    with open(JSON_PATH, 'r') as f:
        for line in f:
            entry = json.loads(line)
            uuid = entry['id']
            label = 1 if entry['is_deceptive'] == 'positive' else 0
            
            file_path = os.path.join(ACTIVATIONS_DIR, f"{uuid}.npy")
            
            if os.path.exists(file_path):
                # mmap_mode='r' prevents loading the whole 26-layer file into RAM at once
                acts = np.load(file_path, mmap_mode='r') 
                
                # Shape: (26, seq_len, 1, 2048) -> select layer and squeeze
                # We take the activation of the LAST token in the sequence 
                # as it usually contains the aggregated information for the response.
                layer_acts = acts[layer_idx, -1, 0, :] 
                
                X.append(layer_acts)
                y.append(label)
                
    return np.array(X), np.array(y)