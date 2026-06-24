import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------
# Generate Dataset
# --------------------------------------------------
num_samples = 1000
signal_length = 256

noisy_signals = []
clean_signals = []

for _ in range(num_samples):

    t = np.linspace(0, 1, signal_length)

    # Clean signal
    clean = (
        np.sin(2 * np.pi * 5 * t)
        + 0.5 * np.sin(2 * np.pi * 15 * t)
    )

    # Random Gaussian noise
    noise = np.random.normal(0, 0.5, signal_length)

    noisy = clean + noise

    noisy_signals.append(noisy)
    clean_signals.append(clean)

# Convert to tensors
X = torch.tensor(noisy_signals, dtype=torch.float32)
Y = torch.tensor(clean_signals, dtype=torch.float32)

# Shape: (batch, channels, length)
X = X.unsqueeze(1)
Y = Y.unsqueeze(1)

# Train/Test Split
train_size = int(0.8 * len(X))

X_train = X[:train_size]
Y_train = Y[:train_size]

X_test = X[train_size:]
Y_test = Y[train_size:]

train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# --------------------------------------------------
# 1D CNN Denoiser
# --------------------------------------------------
class DenoiseCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv1d(16, 1, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.net(x)

model = DenoiseCNN()

# --------------------------------------------------
# Loss and Optimizer
# --------------------------------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------------------------------
# Training
# --------------------------------------------------
epochs = 20

for epoch in range(epochs):

    model.train()
    epoch_loss = 0

    for noisy_batch, clean_batch in train_loader:

        optimizer.zero_grad()

        output = model(noisy_batch)

        loss = criterion(output, clean_batch)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {epoch_loss/len(train_loader):.6f}"
    )

# --------------------------------------------------
# Inference
# --------------------------------------------------
model.eval()

with torch.no_grad():

    sample_noisy = X_test[0].unsqueeze(0)

    sample_clean = Y_test[0].squeeze().numpy()

    sample_denoised = (
        model(sample_noisy)
        .squeeze()
        .numpy()
    )

# --------------------------------------------------
# Plot Results
# --------------------------------------------------
plt.figure(figsize=(12, 5))

#plt.plot(
#    sample_noisy.squeeze().numpy(),
#    label="Noisy Signal",
 #   alpha=0.7
#)

plt.plot(
    sample_clean,
    label="Clean Signal",
    linewidth=2
)

plt.plot(
    sample_denoised,
    label="CNN Denoised",
    linewidth=2
)

plt.legend()
plt.grid(True)
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("1D CNN Signal Denoising using PyTorch")
plt.show()