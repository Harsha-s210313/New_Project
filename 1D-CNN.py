import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# =====================================================
# CONFIGURATION
# =====================================================

# ===== CHANGE THIS =====
NOISY_FOLDER = r"PATH_TO_NOISY_FILES"

# ===== CHANGE THIS =====
CLEAN_FOLDER = r"PATH_TO_CLEAN_FILES"

WINDOW_SIZE = 2048
STRIDE = 512

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# DATASET
# =====================================================

class UnderwaterDataset(Dataset):

    def __init__(
        self,
        noisy_folder,
        clean_folder,
        window_size=2048,
        stride=512
    ):

        self.samples = []

        noisy_files = sorted(
            Path(noisy_folder).glob("*.txt")
        )

        print(f"Found {len(noisy_files)} noisy files")

        for noisy_file in noisy_files:

            clean_file = (
                Path(clean_folder) /
                noisy_file.name
            )

            if not clean_file.exists():
                print(
                    f"Skipping {noisy_file.name}"
                )
                continue

            noisy_signal = np.loadtxt(
                noisy_file
            )

            clean_signal = np.loadtxt(
                clean_file
            )

            if len(noisy_signal) != len(clean_signal):
                print(
                    f"Length mismatch in {noisy_file.name}"
                )
                continue

            # Create overlapping windows
            for start in range(
                0,
                len(noisy_signal) - window_size + 1,
                stride
            ):

                noisy_window = noisy_signal[
                    start:start+window_size
                ]

                clean_window = clean_signal[
                    start:start+window_size
                ]

                # Normalization
                noisy_window = (
                    noisy_window -
                    np.mean(noisy_window)
                ) / (
                    np.std(noisy_window) + 1e-8
                )

                clean_window = (
                    clean_window -
                    np.mean(clean_window)
                ) / (
                    np.std(clean_window) + 1e-8
                )

                self.samples.append(
                    (
                        noisy_window.astype(
                            np.float32
                        ),
                        clean_window.astype(
                            np.float32
                        )
                    )
                )

        print(
            f"Total training windows: "
            f"{len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        noisy, clean = self.samples[idx]

        noisy = torch.tensor(
            noisy
        ).unsqueeze(0)

        clean = torch.tensor(
            clean
        ).unsqueeze(0)

        return noisy, clean


# =====================================================
# U-NET BLOCK
# =====================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(
                out_channels
            ),

            nn.ReLU(inplace=True),

            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(
                out_channels
            ),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        return self.block(x)


# =====================================================
# 1D U-NET
# =====================================================

class UNet1D(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder

        self.enc1 = DoubleConv(
            1,
            32
        )

        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = DoubleConv(
            32,
            64
        )

        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = DoubleConv(
            64,
            128
        )

        self.pool3 = nn.MaxPool1d(2)

        # Bottleneck

        self.bottleneck = DoubleConv(
            128,
            256
        )

        # Decoder

        self.up3 = nn.ConvTranspose1d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose1d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose1d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        self.final = nn.Conv1d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        d3 = self.up3(b)
        d3 = torch.cat(
            [d3, e3],
            dim=1
        )
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat(
            [d2, e2],
            dim=1
        )
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat(
            [d1, e1],
            dim=1
        )
        d1 = self.dec1(d1)

        return self.final(d1)


# =====================================================
# LOAD DATA
# =====================================================

dataset = UnderwaterDataset(
    noisy_folder=NOISY_FOLDER,
    clean_folder=CLEAN_FOLDER,
    window_size=WINDOW_SIZE,
    stride=STRIDE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =====================================================
# MODEL
# =====================================================

model = UNet1D().to(DEVICE)

criterion = nn.SmoothL1Loss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# =====================================================
# TRAINING
# =====================================================

print("Training started...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for noisy, clean in loader:

        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)

        optimizer.zero_grad()

        output = model(noisy)

        loss = criterion(
            output,
            clean
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = (
        running_loss /
        len(loader)
    )

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
        f" | Loss = {avg_loss:.6f}"
    )

# =====================================================
# SAVE MODEL
# =====================================================

torch.save(
    model.state_dict(),
    "underwater_unet.pth"
)

print(
    "\nTraining complete."
)

print(
    "Model saved as underwater_unet.pth"
)
