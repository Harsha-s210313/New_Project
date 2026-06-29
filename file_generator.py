import numpy as np
from pathlib import Path

# --- Configuration ---
FS             = 1000          # Sampling frequency in Hz (change here later)
N_SAMPLES      = 2048          # Samples per signal
F_START        = 21            # LFM start frequency in Hz (scaled to FS)
F_END          = 23            # LFM end frequency in Hz (scaled to FS)
N_SIGNALS      = 100           # Number of signal pairs to generate
SEED           = 42

# Power range for clean signals (dB)
CLEAN_PWR_MIN  = -40.0
CLEAN_PWR_MAX  =   6.0

# Power range for noise (dB) — change these two lines if needed later
NOISE_PWR_MIN  = -40.0
NOISE_PWR_MAX  =   6.0

CLEAN_DIR      = Path("data/clean")
NOISY_DIR      = Path("data/noisy")

# --- Helpers ---
def db_to_linear(db):
    """Convert dB power to linear scale."""
    return 10 ** (db / 10.0)

def scale_to_power(signal, target_power_db):
    """Scale signal to a target power in dB."""
    current_power = np.mean(signal ** 2)
    target_power  = db_to_linear(target_power_db)
    scale         = np.sqrt(target_power / (current_power + 1e-12))
    return signal * scale

def generate_lfm(n_samples, fs, f_start, f_end):
    """Generate a single LFM (chirp) signal."""
    t      = np.linspace(0, n_samples / fs, n_samples, endpoint=False)
    # Apply Tukey window for gradual rise and fall
    window = np.zeros(n_samples)
    alpha  = 0.2  # taper ratio — 10% rise, 10% fall
    from scipy.signal import windows
    window = windows.tukey(n_samples, alpha=alpha)
    chirp  = np.sin(2 * np.pi * (f_start * t + (f_end - f_start) / (2 * (n_samples / fs)) * t ** 2))
    return chirp * window

def generate_awgn(n_samples):
    """Generate unit-power AWGN."""
    noise = np.random.randn(n_samples)
    return noise

# def generate_colored_noise(n_samples, fs, color='pink'):
#     """Generate colored noise (pink/brown/blue).
#     color: 'pink'  → PSD ∝ 1/f
#            'brown' → PSD ∝ 1/f²
#            'blue'  → PSD ∝ f
#     """
#     freqs = np.fft.rfftfreq(n_samples, d=1/fs)
#     freqs[0] = 1  # avoid divide by zero at DC
#     if color == 'pink':
#         power_spectrum = 1 / freqs
#     elif color == 'brown':
#         power_spectrum = 1 / freqs ** 2
#     elif color == 'blue':
#         power_spectrum = freqs
#     else:
#         raise ValueError(f"Unknown color: {color}")
#     amplitudes  = np.sqrt(power_spectrum)
#     phases      = np.random.uniform(0, 2 * np.pi, len(freqs))
#     spectrum    = amplitudes * np.exp(1j * phases)
#     noise       = np.fft.irfft(spectrum, n=n_samples)
#     noise       = noise / (np.std(noise) + 1e-12)  # normalize to unit power
#     return noise

# --- Main ---
def main():
    np.random.seed(SEED)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    NOISY_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(1, N_SIGNALS + 1):
        # Generate clean LFM
        chirp = generate_lfm(N_SAMPLES, FS, F_START, F_END)

        # Scale clean signal to random power in [CLEAN_PWR_MIN, CLEAN_PWR_MAX]
        clean_pwr_db = np.random.uniform(CLEAN_PWR_MIN, CLEAN_PWR_MAX)
        clean        = scale_to_power(chirp, clean_pwr_db)

        # Generate AWGN and scale to random power in [NOISE_PWR_MIN, NOISE_PWR_MAX]
        noise_pwr_db = np.random.uniform(NOISE_PWR_MIN, NOISE_PWR_MAX)
        noise        = generate_awgn(N_SAMPLES)
        noise        = scale_to_power(noise, noise_pwr_db)

        # Noisy = clean + noise (same clean signal, paired)
        noisy = clean + noise

        # Save as .dat files
        np.savetxt(CLEAN_DIR / f"{i}.dat", clean)
        np.savetxt(NOISY_DIR / f"{i}.dat", noisy)

        print(f"Signal {i:03d} | Clean power: {clean_pwr_db:.2f} dB | "
              f"Noise power: {noise_pwr_db:.2f} dB | "
              f"SNR: {clean_pwr_db - noise_pwr_db:.2f} dB")

    print(f"\nDone. {N_SIGNALS} pairs saved to {CLEAN_DIR} and {NOISY_DIR}")

if __name__ == "__main__":
    main()