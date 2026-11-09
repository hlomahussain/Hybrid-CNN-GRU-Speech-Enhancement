"""
Dataset bookkeeping: scanning the folders, leak-free train/val/test splits,
noise sampling and deterministic re-creation of the test mixtures.
"""
import collections
import os

import numpy as np
import soundfile as sf

import config as C
from audio_utils import (list_audio, load_audio, read_segment, noise_type_of, speaker_of,
                         fit_length, mix_at_snr)

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------- clean side
def split_clean(rng):
    """Split clean files into train/val/test. Whole speakers are kept together when
    the clean folder has speaker sub-folders, so test speakers are never seen in training."""
    files = list_audio(C.CLEAN_DIR)
    files = [f for f in files if sf.info(f).duration >= C.MIN_UTT_SECONDS]
    if not files:
        raise SystemExit(f"No usable audio found in {C.CLEAN_DIR}")
    speakers = [speaker_of(f) for f in files]
    by_speaker = len(set(speakers) - {None}) >= 5 and None not in speakers
    groups = collections.defaultdict(list)
    for f, s in zip(files, speakers):
        groups[s if by_speaker else f].append(f)
    keys = sorted(groups)
    rng.shuffle(keys)
    n = len(keys)
    n_test = max(1, int(round(n * C.CLEAN_SPLIT[2])))
    n_val = max(1, int(round(n * C.CLEAN_SPLIT[1])))
    if n < 3:
        raise SystemExit("Need at least 3 clean files (and realistically hundreds) to build train/val/test.")
    parts = {"test": keys[:n_test], "val": keys[n_test:n_test + n_val], "train": keys[n_test + n_val:]}
    out = {k: sorted(f for key in v for f in groups[key]) for k, v in parts.items()}
    info = {"split_by": "speaker" if by_speaker else "file",
            "n_groups": {k: len(v) for k, v in parts.items()}}
    return out, info


# --------------------------------------------------------------- noise side
def scan_noise():
    """-> {noise_type: [(path, duration_sec), ...]}"""
    table = collections.defaultdict(list)
    for f in list_audio(C.NOISE_DIR):
        t = noise_type_of(f)
        if C.USE_NOISE_TYPES and t not in [x.lower() for x in C.USE_NOISE_TYPES]:
            continue
        d = sf.info(f).duration
        if d >= 1.0:
            table[t].append((f, d))
    if not table:
        raise SystemExit(f"No usable audio found in {C.NOISE_DIR}")
    return dict(sorted(table.items()))


def split_bounds(duration, split):
    """Time range of one noise file that belongs to a split (no overlap between splits)."""
    a = np.cumsum((0.0,) + tuple(C.NOISE_SPLIT)) * duration
    i = SPLITS.index(split)
    return float(a[i]), float(a[i + 1])


def pick_noise(noise_table, noise_type, split, dur_sec, rng):
    """Choose (file, offset, read_duration) inside the time range reserved for `split`."""
    files = noise_table[noise_type]
    w = np.array([d for _, d in files])
    path, d = files[rng.choice(len(files), p=w / w.sum())]
    lo, hi = split_bounds(d, split)
    if hi - lo > dur_sec:
        return path, float(rng.uniform(lo, hi - dur_sec)), float(dur_sec)
    return path, lo, hi - lo          # short portion: it will be looped


def get_noise(path, offset, read_dur, n_samples):
    return fit_length(read_segment(path, offset, read_dur), n_samples)


# ---------------------------------------------------------------- test set
def build_test_manifest(test_files, noise_table, seen_types, rng):
    import pandas as pd
    files = list(test_files)
    rng.shuffle(files)
    files = files[:C.N_TEST_UTTS]
    rows = []
    for u, cf in enumerate(sorted(files)):
        dur = min(sf.info(cf).duration, C.MAX_UTT_SECONDS)
        for nt in noise_table:
            path, off, rd = pick_noise(noise_table, nt, "test", dur, rng)
            for snr in C.TEST_SNRS:        # same speech + same noise segment at every SNR
                rows.append(dict(utt=u, clean=cf, noise_type=nt, noise_file=path, noise_offset=round(off, 4),
                                 noise_read=round(rd, 4), snr=snr, seen=nt in seen_types))
    return pd.DataFrame(rows)


def make_test_mixture(row):
    """Re-create one test mixture exactly (clean, noisy) from a manifest row."""
    clean = load_audio(row["clean"], max_seconds=C.MAX_UTT_SECONDS)
    noise = get_noise(row["noise_file"], row["noise_offset"], row["noise_read"], len(clean))
    return mix_at_snr(clean, noise, row["snr"])
