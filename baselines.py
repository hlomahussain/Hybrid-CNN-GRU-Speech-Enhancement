"""
Reference methods evaluated on exactly the same test mixtures as the neural models.

spectral_subtraction : classical magnitude spectral subtraction
wiener_filter        : decision-directed Wiener filter
rnnoise              : the official RNNoise library (optional, needs `pip install pyrnnoise`)
"""
import numpy as np
import librosa

import config as C
from audio_utils import stft, istft, EPS


def _noise_psd(power):
    """Per-bin noise power estimate from the 30th percentile over time.
    For exponentially distributed noise power, the 30th percentile = 0.357 x mean."""
    return np.percentile(power, 30, axis=1, keepdims=True) / 0.357


def spectral_subtraction(noisy, alpha=2.0, beta=0.02):
    S = stft(noisy)
    mag, phase = np.abs(S), np.angle(S)
    noise_mag = np.sqrt(_noise_psd(mag ** 2))
    out = np.maximum(mag - alpha * noise_mag, beta * mag)
    return istft(out * np.exp(1j * phase), len(noisy)).astype(np.float32)


def wiener_filter(noisy, dd=0.98, g_min=0.1):
    S = stft(noisy)
    power = np.abs(S) ** 2
    npsd = _noise_psd(power)[:, 0] + EPS
    G = np.zeros_like(power)
    prev = np.zeros(power.shape[0])
    for t in range(power.shape[1]):
        post = power[:, t] / npsd
        prio = dd * prev / npsd + (1 - dd) * np.maximum(post - 1.0, 0.0)
        g = np.maximum(prio / (1.0 + prio), g_min)
        G[:, t] = g
        prev = (g ** 2) * power[:, t]
    return istft(S * G, len(noisy)).astype(np.float32)


# ------------------------------------------------------------------ RNNoise
def rnnoise_available():
    try:
        import pyrnnoise.rnnoise  # noqa: F401
        return True
    except Exception:
        return False


def _align(reference, x, max_lag=2400):
    """Remove the processing delay of x relative to reference (cross-correlation peak)."""
    n = min(len(reference), len(x))
    a, b = reference[:n], x[:n]
    corr = np.correlate(b[: min(n, 96000)], a[: min(n, 96000)], mode="full")
    mid = min(n, 96000) - 1
    window = corr[mid - max_lag: mid + max_lag + 1]
    lag = int(np.argmax(window)) - max_lag
    if lag > 0:
        x = np.concatenate([x[lag:], np.zeros(lag, dtype=x.dtype)])
    elif lag < 0:
        x = np.concatenate([np.zeros(-lag, dtype=x.dtype), x[:lag]])
    return x


def rnnoise(noisy):
    from pyrnnoise.rnnoise import FRAME_SIZE, create, destroy, process_mono_frame

    x = noisy
    if C.SAMPLE_RATE != 48000:
        x = librosa.resample(x, orig_sr=C.SAMPLE_RATE, target_sr=48000)
    pcm = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
    n_frames = int(np.ceil(len(pcm) / FRAME_SIZE))
    pcm = np.pad(pcm, (0, n_frames * FRAME_SIZE - len(pcm)))
    state = create()
    out = np.zeros_like(pcm)
    for i in range(n_frames):
        frame, _ = process_mono_frame(state, pcm[i * FRAME_SIZE:(i + 1) * FRAME_SIZE].copy())
        out[i * FRAME_SIZE:(i + 1) * FRAME_SIZE] = frame
    destroy(state)
    y = out.astype(np.float32) / 32767.0
    if C.SAMPLE_RATE != 48000:
        y = librosa.resample(y, orig_sr=48000, target_sr=C.SAMPLE_RATE)
    y = librosa.util.fix_length(y, size=len(noisy))
    return _align(noisy, y).astype(np.float32)
