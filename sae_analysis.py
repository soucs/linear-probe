from sae_lens import SAE
from data_helper import load_dataset
import torch

target_layer = 16  # Options: 'emb' (0), 0-23 (1-24), 'norm' (25)

X, y = load_dataset(target_layer)
X, y = torch.from_numpy(X), torch.from_numpy(y)

honest_X = X[y == 0]
deceptive_X = X[y == 1]

print(honest_X.shape, deceptive_X.shape)

release = "qwen-scope-3.5-2b-base-w32k-l100"
sae_id = "layer15"
sae = SAE.from_pretrained(release, sae_id)

honest_features = sae.encode(honest_X)
deceptive_features = sae.encode(deceptive_X)

print(honest_features.shape, deceptive_features.shape)

# Compute mean activation difference
honest_mean = honest_features.mean(dim=0)
deceptive_mean = deceptive_features.mean(dim=0)

# Compute variances
honest_var = honest_features.var(dim=0)
deceptive_var = deceptive_features.var(dim=0)

diff = deceptive_mean - honest_mean

# Find strongest features
k = 30

top_pos = torch.topk(diff, k=k)
top_neg = torch.topk(-diff, k=k)

print("Most deceptive features:")
print(top_pos.indices)

print("Most honest features:")
print(top_neg.indices)

# View variances of top features (and maximum pos and neg variances)
print("Honest Variances:")
print(honest_var[top_pos.indices])
print("Max Honest Variance:", honest_var.max().item())

print("Deceptive Variances:")
print(deceptive_var[top_pos.indices])
print("Max Deceptive Variance:", deceptive_var.max().item())

# Top feature differences
import matplotlib.pyplot as plt

k = 30

top_values, top_indices = torch.topk(
    diff.abs(),
    k=k
)

signed_values = diff[top_indices]

plt.figure(figsize=(12,6))
plt.bar(range(k), signed_values.detach().numpy())

plt.axhline(0)
plt.xticks(range(k), top_indices.detach().numpy(), rotation=90)

plt.ylabel("Mean Activation Difference")
plt.xlabel("SAE Feature ID")
plt.title("Top SAE Features Differing Between Honest and Deceptive")
plt.tight_layout()
plt.savefig(f"outputs/layer_{target_layer}_diff.png")