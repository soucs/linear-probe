import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuration
JSON_PATH = "data/deceptive_outputs_labeled.jsonl"
ACTIVATIONS_DIR = "data/activations/"
TARGET_LAYER = 16  # Options: 'emb' (0), 0-23 (1-24), 'norm' (25)

# Layer Mapping
# 0: Embedding, 1-24: Hidden Layers, 25: Final Norm
LAYER_MAP = {
    'emb': 0,
    **{str(i): i + 1 for i in range(24)},
    'norm': 25
}

def load_dataset(json_path, layer_key):
    layer_idx = LAYER_MAP[str(layer_key)]
    X = []
    y = []
    
    with open(json_path, 'r') as f:
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

# 1. Load data
print(f"Loading activations for layer: {TARGET_LAYER}...")
X, y = load_dataset(JSON_PATH, TARGET_LAYER)

# Save for later
np.save(f"data/layer_dataframes/layer_{TARGET_LAYER}_X.npy", X)
np.save(f"data/layer_dataframes/layer_{TARGET_LAYER}_y.npy", y)

# Load for later
# X = np.load(f"data/layer_dataframes/layer_{TARGET_LAYER}_X.npy")
# y = np.load(f"data/layer_dataframes/layer_{TARGET_LAYER}_y.npy")

# Print shapes
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Scaling (Highly recommended for PCA)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Apply PCA
n_components = 0.99
pca = PCA(n_components) 
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Original features: {X_train.shape}")
print(f"Reduced features ({n_components*100}% variance): {X_train_pca.shape}")

# 3. Fit Logistic Regression
# 'liblinear' is good for smaller datasets; 'saga' is faster for large ones.
clf = LogisticRegression(max_iter=1000, solver='lbfgs')
clf.fit(X_train_pca, y_train)

# 4. Evaluation
y_pred = clf.predict(X_test_pca)
print('Predictions:', y_pred)
print("\n--- Output Metrics ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))