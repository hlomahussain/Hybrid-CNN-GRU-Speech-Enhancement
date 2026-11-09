"""
STEP 0 (alternative source) - build the 48 kHz thesis data from the Microsoft DNS Challenge
full-band corpus (ICASSP/Interspeech 2021 release), downloaded file-by-file from GitHub.
Use this when VCTK / DEMAND cannot be downloaded (slow or blocked connection).

What it builds:
  datasets/clean/<reader>/*.wav            read English speech, 48 kHz, gender-balanced readers,
                                           every 31 s clip cut into three ~10 s utterances
  datasets/noise/<class>/<class>.wav       labelled 48 kHz noise classes (fan, typing, door, ...),
                                           clips of one class joined into one long recording,
                                           ordered by source recording (keeps time-based splits clean)
  datasets/noise/babble_<N>talkers/*.wav   synthetic babble from readers that are NOT target speakers
  datasets/noise/{white,pink,speech_shaped}_noise/*.wav   synthetic stationary reference noises
  datasets/speakers.json                   target / babble readers with estimated gender
  datasets/noise_sources.json              which source clips went into each noise class

Requires git (for the file list) and internet access to github.com.

Run:  python step0_get_data_dns.py
"""
import _env  # noqa: F401  (Run button: uses the project environment automatically)
import argparse
import collections
import concurrent.futures as cf
import json
import os
import re
import subprocess
import urllib.request

import numpy as np
import soundfile as sf

import config as C
from step0_get_data import make_babble

REPO = "https://github.com/microsoft/DNS-Challenge"
BRANCH = "interspeech2021/adddata"
MEDIA = "https://media.githubusercontent.com/media/microsoft/DNS-Challenge/" + BRANCH + "/"
WORK = os.path.join("downloads", "dns")
NOISE_CLASSES = {          # class folder name -> file-name prefix in noise_fullband
    "fan": "fan",
    "typing": "typing",
    "door": "door",
    "munching": "munching",
    "copy_machine": "copy-machine",
    "squeaky_chair": "squeak_squeakyChair",
}


def file_list():
    """All file paths of the full-band part of the DNS repository (no audio is downloaded)."""
    cache = os.path.join(WORK, "filelist.txt")
    if not os.path.exists(cache):
        repo = os.path.join(WORK, "repo")
        if not os.path.exists(repo):
            subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", "--depth", "1",
                            "-b", BRANCH, REPO, repo], check=True, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
        out = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only", "HEAD", "datasets/fullband"],
                             check=True, capture_output=True, text=True).stdout
        with open(cache, "w", encoding="utf-8") as f:
            f.write(out)
    with open(cache, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip().endswith(".wav") and not l.startswith('"')]


def fetch(path, tries=4):
    """Download one LFS file (cached, retried). Returns local path or None."""
    import time
    local = os.path.join(WORK, "raw", path.replace("/", "__"))
    if os.path.exists(local):
        return local
    os.makedirs(os.path.dirname(local), exist_ok=True)
    for t in range(tries):
        try:
            req = urllib.request.Request(MEDIA + path, headers={"User-Agent": "thesis-data-script"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if not data.startswith(b"RIFF"):
                return None
            with open(local + ".part", "wb") as f:
                f.write(data)
            os.replace(local + ".part", local)
            return local
        except Exception as e:
            if t == tries - 1:
                print(f"  failed {path}: {e}")
            time.sleep(2 * (t + 1))
    return None


def fetch_many(paths, workers=8):
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fetch, paths))


def read48(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != C.SAMPLE_RATE:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=C.SAMPLE_RATE)
    return x


def median_f0(x):
    import librosa
    y = librosa.resample(x[: 10 * C.SAMPLE_RATE], orig_sr=C.SAMPLE_RATE, target_sr=16000)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=16000)
    f0 = f0[voiced & ~np.isnan(f0)]
    return float(np.median(f0)) if len(f0) else float("nan")


# ------------------------------------------------------------------ speech
def build_speech(files, n_target, n_babble, clips_target, clips_babble, seed):
    rng = np.random.default_rng(seed)
    by_reader = collections.defaultdict(list)
    for p in files:
        m = re.search(r"clean_fullband/read_speech/.*reader_(\d+)_\d+\.wav$", p)
        if m:
            by_reader["r" + m.group(1)].append(p)
    need = max(clips_target, clips_babble)
    cands = sorted(r for r, v in by_reader.items() if len(v) >= need)
    cands = list(rng.permutation(cands)[: 4 * (n_target + n_babble)])
    print(f"read speech: {len(by_reader)} readers, {len(cands)} candidates screened for gender (F0)")

    firsts = fetch_many([sorted(by_reader[r])[0] for r in cands])
    gender = {}
    for r, path in zip(cands, firsts):
        if path:
            f0 = median_f0(read48(path))
            if np.isfinite(f0):
                gender[r] = ("F" if f0 > 165 else "M", round(f0, 1))
    fem = [r for r in cands if r in gender and gender[r][0] == "F"]
    mal = [r for r in cands if r in gender and gender[r][0] == "M"]
    nt_f, nb_f = n_target // 2, n_babble // 2
    target = fem[:nt_f] + mal[: n_target - nt_f]
    babble = fem[nt_f: nt_f + nb_f] + mal[n_target - nt_f: n_target - nt_f + n_babble - nb_f]
    print(f"  F0-based gender: {len(fem)} female, {len(mal)} male -> "
          f"{len(target)} target readers, {len(babble)} babble readers")

    def cut(reader, n_clips, root):
        clips = list(rng.permutation(sorted(by_reader[reader]))[:n_clips])
        out = os.path.join(root, reader)
        os.makedirs(out, exist_ok=True)
        k = 0
        for path in fetch_many(clips):
            if not path:
                continue
            x = read48(path)
            n = len(x) // 3
            for part in range(3):                 # 31 s clip -> three ~10 s utterances
                seg = x[part * n:(part + 1) * n]
                if np.sqrt(np.mean(seg ** 2)) < 1e-3:
                    continue
                sf.write(os.path.join(out, f"{reader}_{k:03d}.wav"), seg, C.SAMPLE_RATE, subtype="PCM_16")
                k += 1
        return k

    total = sum(cut(r, clips_target, C.CLEAN_DIR) for r in target)
    print(f"  clean speech: {total} utterances from {len(target)} readers -> {C.CLEAN_DIR}")
    babble_root = os.path.join(WORK, "babble_speakers")
    for r in babble:
        cut(r, clips_babble, babble_root)
    with open(os.path.join("datasets", "speakers.json"), "w", encoding="utf-8") as f:
        json.dump({"source": "DNS Challenge full-band read speech (48 kHz)",
                   "target_speakers": {r: gender[r][0] for r in target},
                   "babble_speakers": {r: gender[r][0] for r in babble},
                   "median_f0_hz": {r: gender[r][1] for r in target + babble}}, f, indent=1)
    return babble_root


# ------------------------------------------------------------------- noise
def build_noise(files, max_clips, max_seconds, seed):
    rng = np.random.default_rng(seed)
    sources = {}
    for name, prefix in NOISE_CLASSES.items():
        clips = [p for p in files if re.search(rf"noise_fullband/{re.escape(prefix)}_Freesound_validated_\d+_\d+\.wav$", p)]
        src_id = lambda p: int(re.search(r"_validated_(\d+)_", p).group(1))
        ids = sorted({src_id(p) for p in clips})
        ids = list(rng.permutation(ids))                  # random source recordings ...
        chosen, n = [], 0
        for i in ids:                                     # ... whole recordings only (all their segments)
            seg = sorted(p for p in clips if src_id(p) == i)
            chosen += seg
            n += len(seg)
            if n >= max_clips:
                break
        local = fetch_many(chosen)
        parts, used = [], []
        fade = int(0.01 * C.SAMPLE_RATE)
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        for p, l in zip(chosen, local):               # order = source recording order
            if not l:
                continue
            x = read48(l)
            active = np.abs(x) > 1e-4
            rms = np.sqrt(np.mean(x[active] ** 2)) if active.any() else 0
            if len(x) < C.SAMPLE_RATE // 2 or rms < 1e-4:
                continue
            x = 0.05 * x / rms                            # equal level per clip
            x[:fade] *= ramp
            x[-fade:] *= ramp[::-1]
            parts.append(x)
            used.append(os.path.basename(p))
            if sum(len(q) for q in parts) >= max_seconds * C.SAMPLE_RATE:
                break
        if not parts:
            print(f"  {name}: no usable clips")
            continue
        y = np.concatenate(parts)[: int(max_seconds * C.SAMPLE_RATE)]
        y = 0.5 * y / (np.max(np.abs(y)) + 1e-9)
        out = os.path.join(C.NOISE_DIR, name)
        os.makedirs(out, exist_ok=True)
        sf.write(os.path.join(out, f"{name}.wav"), y, C.SAMPLE_RATE, subtype="PCM_16")
        sources[name] = used
        n_rec = len({src_id(u) for u in used})
        print(f"  {name:14s}: {len(used):4d} clips from {n_rec} recordings, {len(y) / C.SAMPLE_RATE / 60:.1f} min")
    with open(os.path.join("datasets", "noise_sources.json"), "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=1)


# ------------------------------------------------------- synthetic stationary
def make_stationary(speech_root, n_files, seconds, seed):
    """Stationary reference noises used throughout audiology / hearing-aid research:
    white noise, pink (1/f) noise and speech-shaped noise (SSN) whose long-term spectrum equals the
    long-term average speech spectrum (LTASS) of the reserved babble readers (never target speakers)."""
    from scipy.signal import welch
    rng = np.random.default_rng(seed + 99)
    n = int(seconds * C.SAMPLE_RATE)
    freqs = np.fft.rfftfreq(n, 1 / C.SAMPLE_RATE)
    psd = None
    for r in sorted(os.listdir(speech_root)):
        for fn in sorted(os.listdir(os.path.join(speech_root, r)))[:6]:
            x, _ = sf.read(os.path.join(speech_root, r, fn), dtype="float32")
            f, p = welch(x, fs=C.SAMPLE_RATE, nperseg=4096)
            psd = p if psd is None else psd + p
    ltass = np.sqrt(np.interp(freqs, f, psd))
    shapes = {"white_noise": np.ones_like(freqs),
              "pink_noise": 1 / np.sqrt(np.maximum(freqs, 20.0)),
              "speech_shaped_noise": ltass}
    for name, mag in shapes.items():
        out = os.path.join(C.NOISE_DIR, name)
        os.makedirs(out, exist_ok=True)
        for k in range(n_files):
            spec = np.fft.rfft(rng.standard_normal(n)) * mag
            spec[0] = 0
            y = np.fft.irfft(spec, n)
            y = 0.25 * y / (np.max(np.abs(y)) + 1e-9)
            sf.write(os.path.join(out, f"{name}_{k:02d}.wav"), y.astype(np.float32), C.SAMPLE_RATE, subtype="PCM_16")
        print(f"  {name:20s}: {n_files} x {seconds:.0f} s (synthetic, stationary)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speakers", type=int, default=60, help="target readers (half female, half male)")
    ap.add_argument("--babble-speakers", type=int, default=12)
    ap.add_argument("--clips-per-speaker", type=int, default=15, help="31 s clips per target reader (x3 utterances)")
    ap.add_argument("--clips-per-babble-speaker", type=int, default=8)
    ap.add_argument("--noise-clips", type=int, default=400, help="max clips downloaded per noise class")
    ap.add_argument("--noise-seconds", type=float, default=600, help="length of each noise-class recording")
    ap.add_argument("--babble-talkers", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--babble-files", type=int, default=5)
    ap.add_argument("--babble-seconds", type=float, default=120)
    ap.add_argument("--skip-speech", action="store_true")
    ap.add_argument("--skip-noise", action="store_true")
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(C.NOISE_DIR, exist_ok=True)
    files = file_list()
    print(f"DNS full-band file list: {len(files)} files")
    if not args.skip_speech:
        babble_root = build_speech(files, args.speakers, args.babble_speakers, args.clips_per_speaker,
                                   args.clips_per_babble_speaker, C.SEED)
        for n in args.babble_talkers:
            make_babble(babble_root, n, args.babble_files, args.babble_seconds, C.SEED)
    if not args.skip_noise:
        print("noise classes:")
        build_noise(files, args.noise_clips, args.noise_seconds, C.SEED)
        make_stationary(os.path.join(WORK, "babble_speakers"), args.babble_files, args.babble_seconds, C.SEED)
    types = sorted(d for d in os.listdir(C.NOISE_DIR) if os.path.isdir(os.path.join(C.NOISE_DIR, d)))
    print(f"\nnoise types now in {C.NOISE_DIR}: {len(types)}\n  " + ", ".join(types))
    print("optional: add DEMAND environments with  python step0_get_data.py --skip-vctk --demand-envs ...")
    print("next: python step1_prepare_data.py")


if __name__ == "__main__":
    main()
