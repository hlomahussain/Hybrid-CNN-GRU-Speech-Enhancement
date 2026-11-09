"""
Signal-processing helpers shared by every script:
loading, mixing at a target SNR, STFT / log-Mel features, ideal band gains,
gain application + reconstruction, and objective metrics.
"""
import os
import re
import warnings

import numpy as np
import librosa
import soundfile as sf

import config as C

AUDIO_EXT = (".wav", ".flac", ".ogg")
EPS = 1e-10


# ----------------------------------------------------------------- file I/O
def list_audio(root):
    out = []
    for d, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(AUDIO_EXT):
                out.append(os.path.join(d, f))
    return sorted(out)


def _to_mono(x):
    return x.mean(axis=1) if x.ndim > 1 else x


def load_audio(path, sr=C.SAMPLE_RATE, max_seconds=None):
    """Read a whole file as mono float32 at `sr`."""
    info = sf.info(path)
    frames = -1 if max_seconds is None else int(max_seconds * info.samplerate)
    x, native = sf.read(path, frames=frames, dtype="float32", always_2d=False)
    x = _to_mono(x)
    if native != sr:
        x = librosa.resample(x, orig_sr=native, target_sr=sr)
    return x.astype(np.float32)


def read_segment(path, start_sec, dur_sec, sr=C.SAMPLE_RATE):
    """Read only [start_sec, start_sec+dur_sec) from a file (memory friendly for long noise files)."""
    info = sf.info(path)
    start = int(start_sec * info.samplerate)
    frames = int(np.ceil(dur_sec * info.samplerate)) + 8
    x, native = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=False)
    x = _to_mono(x)
    if native != sr:
        x = librosa.resample(x, orig_sr=native, target_sr=sr)
    return x.astype(np.float32)


def noise_type_of(path, noise_root=C.NOISE_DIR):
    """noise/<type>/file.wav -> type ;  noise/Babble_3.wav -> babble ;  noise/babble.wav -> babble"""
    rel = os.path.relpath(path, noise_root)
    parts = rel.split(os.sep)
    if len(parts) > 1:
        return parts[0].lower()
    stem = os.path.splitext(parts[0])[0]
    return re.sub(r"[_\-\s.]*\d+$", "", stem).lower()


def speaker_of(path, clean_root=C.CLEAN_DIR):
    """Top-level sub-folder of the clean directory is treated as the speaker id."""
    rel = os.path.relpath(path, clean_root)
    parts = rel.split(os.sep)
    return parts[0] if len(parts) > 1 else None


# ------------------------------------------------------------------- mixing
def fit_length(x, n):
    """Loop (tile) or crop a signal to exactly n samples."""
    if len(x) >= n:
        return x[:n]
    reps = int(np.ceil(n / max(len(x), 1)))
    return np.tile(x, reps)[:n]


def mix_at_snr(clean, noise, snr_db):
    """Return (clean, noisy) where noisy = clean + scaled noise at the requested SNR (dB)."""
    noise = fit_length(noise, len(clean))
    p_clean = np.mean(clean ** 2) + EPS
    p_noise = np.mean(noise ** 2) + EPS
    scale = np.sqrt(p_clean / (p_noise * 10 ** (snr_db / 10)))
    noisy = clean + scale * noise
    peak = np.max(np.abs(noisy))
    if peak > 0.99:                       # avoid clipping, keep the SNR
        clean, noisy = clean * 0.99 / peak, noisy * 0.99 / peak
    return clean.astype(np.float32), noisy.astype(np.float32)


# ----------------------------------------------------------------- features
def vorbis_window(n=C.WINDOW_SIZE):
    """Vorbis power-complementary window (Princen-Bradley), as in RNNoise."""
    k = np.arange(n) + 0.5
    return np.sin(0.5 * np.pi * np.sin(np.pi * k / n) ** 2).astype(np.float32)


_WINDOW = vorbis_window()


def stft(x):
    return librosa.stft(x, n_fft=C.WINDOW_SIZE, hop_length=C.HOP_LENGTH, window=_WINDOW, center=True)


def istft(S, length):
    return librosa.istft(S, hop_length=C.HOP_LENGTH, window=_WINDOW, center=True, length=length)


_MEL_FB = None
_BAND_MATRIX = None


def mel_filterbank():
    global _MEL_FB
    if _MEL_FB is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # a few low-frequency filters are empty at 50 Hz resolution
            _MEL_FB = librosa.filters.mel(sr=C.SAMPLE_RATE, n_fft=C.WINDOW_SIZE, n_mels=C.N_MELS).astype(np.float32)
    return _MEL_FB


def band_matrix():
    """(22, n_bins) 0/1 matrix: which FFT bins belong to which critical band."""
    global _BAND_MATRIX
    if _BAND_MATRIX is None:
        freqs = librosa.fft_frequencies(sr=C.SAMPLE_RATE, n_fft=C.WINDOW_SIZE)
        edges = C.BARK_EDGES
        B = np.zeros((C.NUM_BANDS, len(freqs)), dtype=np.float32)
        for b in range(C.NUM_BANDS):
            hi_ok = freqs <= edges[b + 1] if b == C.NUM_BANDS - 1 else freqs < edges[b + 1]
            B[b, (freqs >= edges[b]) & hi_ok] = 1.0
        _BAND_MATRIX = B
    return _BAND_MATRIX


def logmel_features(S):
    """Complex STFT (bins, T) -> log-Mel power in dB, shape (T, N_MELS).
    No per-utterance normalisation: every frame depends only on itself (real-time friendly)."""
    power = np.abs(S) ** 2
    mel = mel_filterbank() @ power
    return (10.0 * np.log10(mel + EPS)).T.astype(np.float32)


def ideal_band_gains(S_clean, S_noisy):
    """Training target, RNNoise eq.:  g_b = sqrt(E_clean(b) / E_noisy(b)), clipped to [0, 1]. Shape (T, 22)."""
    B = band_matrix()
    e_clean = B @ (np.abs(S_clean) ** 2)
    e_noisy = B @ (np.abs(S_noisy) ** 2)
    g = np.sqrt(e_clean / (e_noisy + EPS))
    return np.clip(g, 0.0, 1.0).T.astype(np.float32)


def apply_band_gains(S_noisy, gains, length):
    """gains (T, 22) -> per-bin gain -> enhanced waveform (noisy phase is kept)."""
    G = band_matrix().T @ gains.T            # (bins, T)
    return istft(S_noisy * G, length).astype(np.float32)


# ------------------------------------------------------------------ metrics
def si_sdr(ref, est):
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + EPS)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target ** 2) + EPS) / (np.sum(noise ** 2) + EPS)))


def compute_metrics_many(clean, estimates):
    """PESQ (wide-band, 16 kHz), STOI, ESTOI, SI-SDR for several enhanced versions of one mixture.
    estimates: {method_name: waveform}  ->  {method_name: {metric: value}}"""
    try:
        from pesq import pesq
    except ImportError:                        # pesq needs a C compiler on Windows; other metrics still work
        pesq = None
    from pystoi import stoi

    ref16 = librosa.resample(clean, orig_sr=C.SAMPLE_RATE, target_sr=C.PESQ_SR)
    out = {}
    for name, est in estimates.items():
        est = fit_length(est, len(clean)) if len(est) >= len(clean) else np.pad(est, (0, len(clean) - len(est)))
        deg16 = librosa.resample(est, orig_sr=C.SAMPLE_RATE, target_sr=C.PESQ_SR)
        m = {"si_sdr": si_sdr(clean, est)}
        try:
            m["pesq"] = float(pesq(C.PESQ_SR, ref16, deg16, "wb")) if pesq else np.nan
        except Exception:                      # PESQ can fail on very short / silent signals
            m["pesq"] = np.nan
        m["stoi"] = float(stoi(ref16, deg16, C.PESQ_SR, extended=False))
        m["estoi"] = float(stoi(ref16, deg16, C.PESQ_SR, extended=True))
        out[name] = m
    return out


# --------------------------------------------------------- noise stationarity
def nonstationarity_index(x):
    """Non-stationarity index (dB): median over the critical bands up to 8.5 kHz of the standard
    deviation over time of the band level, after 100 ms moving-average smoothing.
    Low values = stationary noise (fan, air conditioner); high values = babble, typing, door slams."""
    from scipy.ndimage import uniform_filter1d
    power = band_matrix() @ (np.abs(stft(x)) ** 2)
    keep = [b for b in range(C.NUM_BANDS) if C.BARK_EDGES[b + 1] <= 8500]   # also valid for 16 kHz sources
    smoothed = uniform_filter1d(power[keep], size=max(1, int(round(0.1 * C.SAMPLE_RATE / C.HOP_LENGTH))), axis=1)
    level = 10.0 * np.log10(smoothed + EPS)
    active = level.max(axis=0) > level.max() - 60          # ignore digital silence
    return float(np.median(level[:, active].std(axis=1)))


def stationarity_category(nsi_db):
    lo, hi = C.STATIONARITY_THRESHOLDS_DB
    return "stationary" if nsi_db < lo else ("quasi-stationary" if nsi_db < hi else "non-stationary")
