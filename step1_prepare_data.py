"""
STEP 1 - build the experiment data.

  * scans datasets/clean and datasets/noise
  * makes leak-free train / val / test splits (speech by speaker or file, noise by time range)
  * synthesises noisy training/validation segments at random SNR and stores
    features (log-Mel) + targets (22 ideal band gains) as .npy
  * writes the fixed test manifest (every test utterance x every noise type x every SNR)

Run:  python step1_prepare_data.py
"""
import _env  # noqa: F401  (Run button: uses the project environment automatically)
import collections
import json
import os

import numpy as np
import soundfile as sf
from tqdm import tqdm

import config as C
from audio_utils import (load_audio, stft, logmel_features, ideal_band_gains, mix_at_snr, list_audio,
                         read_segment, nonstationarity_index, stationarity_category)
from dataset import split_clean, scan_noise, split_bounds, pick_noise, get_noise, build_test_manifest


def measure_stationarity(noise_table, unseen):
    """Non-stationarity index of every noise type, measured on the training time range of up to
    5 files (60 s each). Unseen types are measured on their test range (they have no training use)."""
    import pandas as pd
    rows = []
    for t, files in noise_table.items():
        split = "test" if t in unseen else "train"
        vals = []
        for path, dur in files[:5]:
            lo, hi = split_bounds(dur, split)
            x = read_segment(path, lo, min(60.0, hi - lo))
            if len(x) > C.SAMPLE_RATE and np.mean(x ** 2) > 1e-10:
                vals.append(nonstationarity_index(x))
        nsi = float(np.mean(vals)) if vals else float("nan")
        cat = C.NOISE_CATEGORY_OVERRIDES.get(t, stationarity_category(nsi) if vals else "unknown")
        rows.append(dict(noise_type=t, nsi_db=round(nsi, 2), category=cat, seen=t not in unseen,
                         files=len(files), minutes=round(sum(d for _, d in files) / 60, 1)))
    return pd.DataFrame(rows).sort_values("nsi_db").reset_index(drop=True)


def synth_segments(clean_files, noise_table, noise_types, split, n_segments, rng, desc):
    X = np.lib.format.open_memmap(os.path.join(C.WORK_DIR, f"X_{split}.npy"), mode="w+", dtype=np.float32,
                                  shape=(n_segments, C.TIME_STEPS, C.N_MELS))
    Y = np.lib.format.open_memmap(os.path.join(C.WORK_DIR, f"Y_{split}.npy"), mode="w+", dtype=np.float32,
                                  shape=(n_segments, C.TIME_STEPS, C.NUM_BANDS))
    n, i = 0, 0
    order = rng.permutation(len(clean_files))
    usage = collections.Counter()
    bar = tqdm(total=n_segments, desc=desc)
    while n < n_segments:
        if i >= len(order):
            order, i = rng.permutation(len(clean_files)), 0
        clean = load_audio(clean_files[order[i]], max_seconds=C.MAX_UTT_SECONDS)
        i += 1
        if np.mean(clean ** 2) < 1e-8:
            continue
        nt = noise_types[rng.integers(len(noise_types))]
        path, off, rd = pick_noise(noise_table, nt, split, len(clean) / C.SAMPLE_RATE, rng)
        noise = get_noise(path, off, rd, len(clean))
        if np.mean(noise ** 2) < 1e-10:
            continue
        snr = rng.uniform(*C.TRAIN_SNR_RANGE)
        clean, noisy = mix_at_snr(clean, noise, snr)
        level = 10 ** (rng.uniform(-24, -1) / 20) / (np.max(np.abs(noisy)) + 1e-9)   # level augmentation
        clean, noisy = clean * level, noisy * level

        S_clean, S_noisy = stft(clean), stft(noisy)
        feats, gains = logmel_features(S_noisy), ideal_band_gains(S_clean, S_noisy)
        for k in range(feats.shape[0] // C.TIME_STEPS):
            if n >= n_segments:
                break
            sl = slice(k * C.TIME_STEPS, (k + 1) * C.TIME_STEPS)
            X[n], Y[n] = feats[sl], gains[sl]
            n += 1
            usage[nt] += 1
            bar.update(1)
    bar.close()
    X.flush(); Y.flush()
    return X, usage


def main():
    os.makedirs(C.WORK_DIR, exist_ok=True)
    rng = np.random.default_rng(C.SEED)

    # ------------------------------------------------------------ inventory
    splits, split_info = split_clean(rng)
    noise_table = scan_noise()
    unseen = [t.lower() for t in C.UNSEEN_NOISE_TYPES]
    missing = [t for t in unseen if t not in noise_table]
    if missing:
        print(f"note: UNSEEN_NOISE_TYPES not present in {C.NOISE_DIR} and ignored: {missing}")
        unseen = [t for t in unseen if t in noise_table]
    seen = [t for t in noise_table if t not in unseen]
    if not seen:
        raise SystemExit("All noise types are marked unseen - nothing left for training.")

    rates = collections.Counter(sf.info(f).samplerate for f in list_audio(C.CLEAN_DIR)[:200])
    n_rates = collections.Counter(sf.info(f).samplerate for fl in noise_table.values() for f, _ in fl)

    report = []
    report.append(f"Clean speech : {sum(len(v) for v in splits.values())} files, split by {split_info['split_by']}")
    for k in ("train", "val", "test"):
        hours = sum(min(sf.info(f).duration, C.MAX_UTT_SECONDS) for f in splits[k]) / 3600
        report.append(f"  {k:5s}: {len(splits[k]):5d} files, {split_info['n_groups'][k]:4d} {split_info['split_by']}s, {hours:.2f} h")
    report.append(f"  native sample rates (first 200 files): {dict(rates)}")
    report.append(f"Noise        : {len(noise_table)} types   (train/val/test = {C.NOISE_SPLIT} of every file, by time)")
    for t, fl in noise_table.items():
        tag = "UNSEEN (test only)" if t in unseen else "seen"
        report.append(f"  {t:22s} {len(fl):3d} files, {sum(d for _, d in fl) / 60:6.1f} min   {tag}")
    report.append(f"  native sample rates: {dict(n_rates)}")

    stat = measure_stationarity(noise_table, unseen)
    stat.to_csv(os.path.join(C.WORK_DIR, "noise_stationarity.csv"), index=False)
    lo, hi = C.STATIONARITY_THRESHOLDS_DB
    report.append(f"Noise stationarity (NSI, dB; stationary < {lo} <= quasi-stationary < {hi} <= non-stationary):")
    for _, r in stat.iterrows():
        report.append(f"  {r.noise_type:22s} NSI {r.nsi_db:6.2f} dB   {r.category:17s} {'seen' if r.seen else 'UNSEEN'}")
    counts = stat["category"].value_counts().to_dict()
    report.append(f"  categories: {counts}")

    warn = []
    if len(splits["train"]) < 200:
        warn.append(f"Only {len(splits['train'])} clean training files. A thesis-level experiment needs hundreds of "
                    "utterances from many speakers.")
    if split_info["split_by"] == "file":
        warn.append("Clean files are split by file, not by speaker. Put each speaker in its own sub-folder of "
                    "datasets/clean to guarantee unseen test speakers.")
    if len(noise_table) < 3:
        warn.append("Fewer than 3 noise types. Add more (babble, car, street, cafe, ...) for meaningful results.")
    if max(list(rates) + list(n_rates)) < C.SAMPLE_RATE:
        warn.append(f"No source file reaches {C.SAMPLE_RATE} Hz: the data is up-sampled, so bands above "
                    f"{max(list(rates) + list(n_rates)) // 2} Hz contain nothing. Use 48 kHz recordings to claim "
                    "full-band enhancement.")
    if warn:
        report.append("")
        report += [f"WARNING: {w}" for w in warn]
    text = "\n".join(report)
    print("\n" + text + "\n")
    with open(os.path.join(C.WORK_DIR, "data_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(C.WORK_DIR, "splits.json"), "w", encoding="utf-8") as f:
        json.dump({"clean": splits, "seen_noise": seen, "unseen_noise": unseen}, f, indent=1)

    # ------------------------------------------------------- train / val data
    Xtr, usage = synth_segments(splits["train"], noise_table, seen, "train", C.N_TRAIN_SEGMENTS, rng, "train segments")
    synth_segments(splits["val"], noise_table, seen, "val", C.N_VAL_SEGMENTS, rng, "val segments  ")
    s1 = np.zeros(C.N_MELS)
    s2 = np.zeros(C.N_MELS)
    for i in range(0, len(Xtr), 1000):                    # chunked: avoids a multi-GB float64 copy
        blk = np.asarray(Xtr[i:i + 1000], dtype=np.float64).reshape(-1, C.N_MELS)
        s1 += blk.sum(0)
        s2 += (blk ** 2).sum(0)
    cnt = len(Xtr) * C.TIME_STEPS
    mean = s1 / cnt
    np.savez(os.path.join(C.WORK_DIR, "norm.npz"), mean=mean, var=s2 / cnt - mean ** 2 + 1e-3)
    print("training segments per noise type:", dict(sorted(usage.items())))

    # ------------------------------------------------------------- test set
    manifest = build_test_manifest(splits["test"], noise_table, seen, rng)
    manifest.to_csv(os.path.join(C.WORK_DIR, "test_manifest.csv"), index=False)
    print(f"test manifest: {len(manifest)} mixtures = {manifest['utt'].nunique()} utterances x "
          f"{manifest['noise_type'].nunique()} noise types x {len(C.TEST_SNRS)} SNRs")
    print(f"done -> {C.WORK_DIR}/")


if __name__ == "__main__":
    main()
