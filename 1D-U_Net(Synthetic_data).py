import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# =====================================================
# Configuration
# =====================================================
NUM_SAMPLES = 1000
SIGNAL_LENGTH = 256
EPOCHS = 20
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# Generate Dataset
# Clean sinusoid + Gaussian noise + random spike
# =====================================================
clean_signals = []
noisy_signals = []

for _ in range(NUM_SAMPLES):

    t = np.linspace(0, 1, SIGNAL_LENGTH)

    # Clean signal
    clean = (
        np.sin(2 * np.pi * 5 * t)
        + 0.5 * np.sin(2 * np.pi * 15 * t)
    )

    # Gaussian noise
    noise = np.random.normal(
        loc=0.0,
        scale=0.3,
        size=SIGNAL_LENGTH
    )

    # Add large spike near middle
    spike_idx = np.random.randint(
        SIGNAL_LENGTH // 2 - 20,
        SIGNAL_LENGTH // 2 + 20
    )

    spike_amp = np.random.uniform(3, 8)

    noise[spike_idx] += (
        np.random.choice([-1, 1]) * spike_amp
    )

    noisy = clean + noise

    clean_signals.append(clean)
    noisy_signals.append(noisy)

# Convert to tensors
X = torch.tensor(
    np.array(noisy_signals),
    dtype=torch.float32
).unsqueeze(1)

Y = torch.tensor(
    np.array(clean_signals),
    dtype=torch.float32
).unsqueeze(1)

# =====================================================
# Train/Test Split
# =====================================================
train_size = int(0.8 * len(X))

X_train = X[:train_size]
Y_train = Y[:train_size]

X_test = X[train_size:]
Y_test = Y[train_size:]

train_loader = DataLoader(
    TensorDataset(X_train, Y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =====================================================
# U-Net Building Blocks
# =====================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# =====================================================
# 1D U-Net
# =====================================================
class UNet1D(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(1, 16)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(32, 64)

        # Decoder
        self.up2 = nn.ConvTranspose1d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(64, 32)

        self.up1 = nn.ConvTranspose1d(
            32,
            16,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(32, 16)

        self.final = nn.Conv1d(
            16,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)

        return out

# =====================================================
# Model
# =====================================================
model = UNet1D().to(DEVICE)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)

# =====================================================
# Training
# =====================================================
print("Training...")

for epoch in range(EPOCHS):

    model.train()
    epoch_loss = 0

    for noisy_batch, clean_batch in train_loader:

        noisy_batch = noisy_batch.to(DEVICE)
        clean_batch = clean_batch.to(DEVICE)

        optimizer.zero_grad()

        output = model(noisy_batch)

        loss = criterion(
            output,
            clean_batch
        )

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss = {epoch_loss/len(train_loader):.6f}"
    )

# =====================================================
# Test Example
# =====================================================
model.eval()

with torch.no_grad():

    sample_noisy = X_test[0:1].to(DEVICE)

    sample_denoised = (
        model(sample_noisy)
        .cpu()
        .squeeze()
        .numpy()
    )

sample_clean = (
    Y_test[0]
    .squeeze()
    .numpy()
)

sample_noisy = (
    X_test[0]
    .squeeze()
    .numpy()
)

# =====================================================
# Plot Results
# =====================================================
plt.figure(figsize=(14, 6))

plt.plot(
    sample_clean,
    label="Clean Signal",
    linewidth=2
)

plt.plot(
    sample_noisy,
    label="Noisy Signal",
    alpha=0.7
)

plt.plot(
    sample_denoised,
    label="U-Net Denoised",
    linewidth=2
)

plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("1D U-Net Denoising (Gaussian Noise + Large Spike)")
plt.legend()
plt.grid(True)

plt.show()
