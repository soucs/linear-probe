import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_helper import load_dataset

# Configuration
target_layer = 16  # Options: 'emb' (0), 0-23 (1-24), 'norm' (25)

# Layer Mapping
# 0: Embedding, 1-24: Hidden Layers, 25: Final Norm
LAYER_MAP = {
    'emb': 0,
    **{str(i): i + 1 for i in range(24)},
    'norm': 25
}

# 1. Load data
print(f"Loading activations for layer: {target_layer}...")
X, y = load_dataset(target_layer)

# Save for later
np.save(f"data/layer_dataframes/layer_{target_layer}_X.npy", X)
np.save(f"data/layer_dataframes/layer_{target_layer}_y.npy", y)

# Load for later
# X = np.load(f"data/layer_dataframes/layer_{target_layer}_X.npy")
# y = np.load(f"data/layer_dataframes/layer_{target_layer}_y.npy")

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