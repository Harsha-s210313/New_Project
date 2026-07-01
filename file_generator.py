import numpy as np
from pathlib import Path

# --- Configuration ---
FS              = 1000          # Sampling frequency in Hz (change here later)
N_SAMPLES       = 2048          # Samples per signal
F_START         = 21            # LFM start frequency in Hz (scaled to FS)
F_END           = 23            # LFM end frequency in Hz (scaled to FS)
SEED            = 42

# Power sweep range (dB) — clean and noise both drawn from this set
PWR_MAX         =   6.0
PWR_MIN         = -40.0
PWR_STEP        =   0.5

REPEATS_PER_LVL = 22            # repetitions per power level (93 levels x 22 = 2046 pairs)

CLEAN_DIR      = Path(r"C:\Users\HARSHA\Desktop\dataset\clean")
NOISY_DIR      = Path(r"C:\Users\HARSHA\Desktop\dataset\noisy")

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
    from scipy.signal import windows
    t      = np.linspace(0, n_samples / fs, n_samples, endpoint=False)
    window = windows.tukey(n_samples, alpha=0.2)  # gradual rise and fall
    chirp  = np.sin(2 * np.pi * (f_start * t + (f_end - f_start) / (2 * (n_samples / fs)) * t ** 2))
    return chirp * window

def generate_awgn(n_samples):
    """Generate unit-power AWGN."""
    return np.random.randn(n_samples)

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

    # Deterministic descending power sequence: 6, 5.5, ..., -40
    power_levels = np.arange(PWR_MAX, PWR_MIN - PWR_STEP, -PWR_STEP)
    n_levels     = len(power_levels)

    # Build clean-power sequence: each level repeated REPEATS_PER_LVL times, in descending order
    clean_powers = np.repeat(power_levels, REPEATS_PER_LVL)
    n_signals    = len(clean_powers)

    # Noise power: same multiset of values, independently shuffled
    noise_powers = clean_powers.copy()
    np.random.shuffle(noise_powers)

    print(f"Power levels: {n_levels} | Repeats per level: {REPEATS_PER_LVL} | "
          f"Total pairs: {n_signals}\n")

    for i in range(1, n_signals + 1):
        idx = i - 1

        # Generate clean LFM, scale to its deterministic power
        chirp        = generate_lfm(N_SAMPLES, FS, F_START, F_END)
        clean_pwr_db = clean_powers[idx]
        clean        = scale_to_power(chirp, clean_pwr_db)

        # Generate AWGN, scale to its shuffled power
        noise_pwr_db = noise_powers[idx]
        noise        = generate_awgn(N_SAMPLES)
        noise        = scale_to_power(noise, noise_pwr_db)

        # Noisy = clean + noise
        noisy = clean + noise

        # Save as .dat files
        np.savetxt(CLEAN_DIR / f"{i}.dat", clean)
        np.savetxt(NOISY_DIR / f"{i}.dat", noisy)

        if i % 100 == 0 or i == n_signals:
            print(f"Signal {i:04d}/{n_signals} | Clean power: {clean_pwr_db:6.2f} dB | "
                  f"Noise power: {noise_pwr_db:6.2f} dB | "
                  f"SNR: {clean_pwr_db - noise_pwr_db:6.2f} dB")

    print(f"\nDone. {n_signals} pairs saved to {CLEAN_DIR} and {NOISY_DIR}")

if __name__ == "__main__":
    main()
