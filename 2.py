import numpy as np
import pandas as pd
import os
from scipy.signal import firwin, lfilter

# --------------------------------------------------
# Parameters
# --------------------------------------------------
fs = 15000          # Sampling frequency (Hz)
N = 2048            # Samples per signal
num_signals = 100   # Number of signals
f1 = 1000           # Lower cutoff (Hz)
f2 = 4000           # Upper cutoff (Hz)
output_clean = "dataset/clean"
output_noisy = "dataset/noisy"

os.makedirs(output_clean, exist_ok=True)
os.makedirs(output_noisy, exist_ok=True)

# --------------------------------------------------
# Bandpass Filter (FIR)
# --------------------------------------------------
nyq = fs / 2
fir_coeffs = firwin(
    numtaps=101,
    cutoff=[f1/nyq, f2/nyq],
    pass_zero=False
)

# --------------------------------------------------
# Signal Generation
# --------------------------------------------------
for i in range(num_signals):
    # Random signal power between -40dB and 6dB
    power_db = np.random.uniform(-40, 6)
    power_linear = 10 ** (power_db / 10)

    # Generate white noise and filter to get bandpass signal
    white = np.random.randn(N)
    bandpass_signal = lfilter(fir_coeffs, 1.0, white)

    # Normalize to desired power
    current_power = np.mean(bandpass_signal ** 2)
    bandpass_signal = bandpass_signal * np.sqrt(power_linear / (current_power + 1e-8))

    # Clean signal is the bandpass signal
    clean = bandpass_signal

    # Add AWGN noise (fixed noise power at -30dB)
    noise_power = 10 ** (-30 / 10)
    noise = np.random.randn(N) * np.sqrt(noise_power)
    noisy = clean + noise

    # Save to CSV
    pd.DataFrame(clean).to_csv(
        os.path.join(output_clean, f"signal_{i+1:03d}.csv"),
        index=False, header=False
    )
    pd.DataFrame(noisy).to_csv(
        os.path.join(output_noisy, f"signal_{i+1:03d}.csv"),
        index=False, header=False
    )

    print(f"Generated signal {i+1:03d} | Power: {power_db:.2f} dB")

print("\nDataset generation complete!")
print(f"Clean signals: {output_clean}")
print(f"Noisy signals: {output_noisy}")
