"""
STEP 0 - download and organise the thesis data (run once).

What it builds:

  datasets/clean/<speaker>/*.flac          VCTK 0.92, 48 kHz, mic1, gender-balanced speakers
  datasets/noise/<environment>/ch01.wav    DEMAND, 48 kHz, 17 real-world environments
  datasets/noise/babble_<N>talkers/*.wav   synthetic babble made from VCTK speakers that are
                                           NOT used as target speakers (no overlap)
  datasets/noise/<type>/*.wav              optional: MS-SNSD noise types (16 kHz), --mssnsd

Examples:
  python step0_get_data.py                       # everything with default settings
  python step0_get_data.py --vctk-zip D:/data/VCTK-Corpus-0.92.zip   # use a zip you downloaded yourself
  python step0_get_data.py --vctk-zip clean_trainset_28spk_wav.zip clean_testset_wav.zip   # smaller 48 kHz VCTK subset
  python step0_get_data.py --demand-envs PCAFETER PRESTO STRAFFIC TCAR DLIVING
  python step0_get_data.py --skip-vctk           # only noise

Disk: the VCTK zip is about 11 GB (deleted after extraction unless --keep-downloads);
DEMAND 48 kHz zips are 250-380 MB each (only channel 1 is kept, about 29 MB per environment).
"""
import _env  # noqa: F401  (Run button: uses the project environment automatically)
import argparse
import collections
import io
import json
import os
import random
import re
import shutil
import sys
import urllib.request
import zipfile

import numpy as np
import soundfile as sf

import config as C

DOWNLOADS = "downloads"
VCTK_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
VCTK_PAGE = "https://datashare.ed.ac.uk/handle/10283/3443"
DEMAND_URL = "https://zenodo.org/records/1227121/files/{env}_48k.zip?download=1"
DEMAND_ENVS = ["DKITCHEN", "DLIVING", "DWASHING", "NFIELD", "NPARK", "NRIVER", "OHALLWAY", "OMEETING",
               "OOFFICE", "PCAFETER", "PRESTO", "PSTATION", "SCAFE", "SPSQUARE", "STRAFFIC", "TBUS",
               "TCAR", "TMETRO"]
MSSNSD_API = "https://api.github.com/repos/microsoft/MS-SNSD/contents/{folder}"
MSSNSD_TYPES = ["Babble", "AirConditioner", "Typing", "VacuumCleaner", "CopyMachine", "Munching",
                "NeighborSpeaking", "ShuttingDoor", "Traffic", "Restaurant", "Cafe", "Car"]


# ------------------------------------------------------------------ helpers
def download(url, dest):
    """Streaming download with resume support."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.exists(dest) and not os.path.exists(dest + ".part"):
        print(f"  already downloaded: {dest}")
        return dest
    part = dest + ".part"
    done = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "thesis-data-script"})
    if done:
        req.add_header("Range", f"bytes={done}-")
    with urllib.request.urlopen(req, timeout=60) as r:
        total = r.headers.get("Content-Length")
        total = int(total) + done if total else None
        if done and r.status != 206:          # server ignored the range request
            done = 0
        mode = "ab" if done else "wb"
        with open(part, mode) as f:
            got = done
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total:
                    sys.stdout.write(f"\r  {os.path.basename(dest)}: {got / 1e6:8.1f} / {total / 1e6:.1f} MB")
                    sys.stdout.flush()
    print()
    os.replace(part, dest)
    return dest


# --------------------------------------------------------------------- VCTK
def read_speaker_genders(zf):
    """speaker id -> 'F' / 'M' from speaker-info.txt (if present in the zip)."""
    names = [n for n in zf.namelist() if n.lower().endswith("speaker-info.txt")]
    genders = {}
    if names:
        text = zf.read(names[0]).decode("utf-8", errors="ignore")
        for line in text.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 3 and parts[2].upper() in ("F", "M"):
                sid = parts[0] if parts[0].startswith("p") else "p" + parts[0]
                genders[sid] = parts[2].upper()
    return genders


def vctk_index(zfs):
    """speaker -> sorted list of (zip, member) mic1 audio files.
    Works for VCTK 0.92 (wav48_silence_trimmed/p225/p225_001_mic1.flac) and for the 48 kHz
    VCTK-derived Valentini set (clean_trainset_28spk_wav/p226_001.wav)."""
    table = collections.defaultdict(list)
    for zf in zfs:
        for n in zf.namelist():
            low = n.lower()
            if not low.endswith((".flac", ".wav")) or "mic2" in low or "__macosx" in low:
                continue
            m = re.search(r"(?:^|/)(p\d{3}|s\d+)_\d+[^/]*$", n)
            if m:
                table[m.group(1)].append((zf, n))
    return {k: sorted(v, key=lambda t: t[1]) for k, v in table.items()}


def estimate_gender(zf, members):
    """'F' / 'M' from the median F0 of a few utterances (used when speaker-info.txt is absent)."""
    import librosa
    f0s = []
    for m in members[:4]:
        with zf.open(m) as fh:
            x, sr = sf.read(io.BytesIO(fh.read()), dtype="float32")
        x = librosa.resample(x if x.ndim == 1 else x.mean(1), orig_sr=sr, target_sr=16000)
        f0, voiced, _ = librosa.pyin(x, fmin=60, fmax=400, sr=16000)
        f0s.extend(f0[voiced & ~np.isnan(f0)])
    return "F" if f0s and np.median(f0s) > 165 else "M"


def pick_speakers(index, genders, n_target, n_babble, seed):
    """Disjoint, gender-balanced target and babble speaker sets."""
    rng = random.Random(seed)
    spk = sorted(s for s in index if len(index[s]) >= 50)
    if len(spk) < n_target + n_babble:              # small corpus: keep >= 8 babble talkers, rest are targets
        n_babble = min(n_babble, max(8, len(spk) // 4), len(spk) // 2)
        n_target = len(spk) - n_babble
    by_g = collections.defaultdict(list)
    for s in spk:
        by_g[genders.get(s, "?")].append(s)
    for g in by_g:
        rng.shuffle(by_g[g])
    target, babble = [], []
    if "F" in by_g and "M" in by_g:
        target = by_g["F"][:n_target // 2] + by_g["M"][:n_target - n_target // 2]
        rest_f, rest_m = by_g["F"][n_target // 2:], by_g["M"][n_target - n_target // 2:]
        babble = rest_f[:n_babble // 2] + rest_m[:n_babble - n_babble // 2]
    else:
        pool = sum(by_g.values(), [])
        target, babble = pool[:n_target], pool[n_target:n_target + n_babble]
    return sorted(target), sorted(babble)


def extract_vctk(zip_paths, n_target, n_babble, max_utts, seed):
    zfs = [zipfile.ZipFile(p) for p in zip_paths]
    index = vctk_index(zfs)
    if not index:
        raise SystemExit("No VCTK audio found in the zip(s) - expected VCTK 0.92 or the 48 kHz Valentini clean sets.")
    genders = {}
    for zf in zfs:
        genders.update(read_speaker_genders(zf))
    if not genders:
        print("speaker-info.txt not found: estimating speaker gender from F0 ...")
        genders = {s: estimate_gender(v[0][0], [m for _, m in v]) for s, v in index.items()}
    target, babble = pick_speakers(index, genders, n_target, n_babble, seed)
    print(f"VCTK: {len(index)} speakers in zip -> {len(target)} target speakers, {len(babble)} babble speakers")

    def extract(speakers, root):
        rng = random.Random(seed)
        for s in speakers:
            files = list(index[s])
            rng.shuffle(files)
            out = os.path.join(root, s)
            os.makedirs(out, exist_ok=True)
            for zf, m in files[:max_utts]:
                with zf.open(m) as src, open(os.path.join(out, os.path.basename(m)), "wb") as dst:
                    shutil.copyfileobj(src, dst)

    extract(target, C.CLEAN_DIR)
    babble_root = os.path.join(DOWNLOADS, "babble_speakers")
    extract(babble, babble_root)
    info = {"target_speakers": {s: genders.get(s, "?") for s in target},
            "babble_speakers": {s: genders.get(s, "?") for s in babble}}
    with open(os.path.join("datasets", "speakers.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    return babble_root


# ------------------------------------------------------------------ babble
def make_babble(speaker_root, n_talkers, n_files, seconds, seed):
    """Sum n_talkers independent streams of speech (equal RMS) -> babble noise at 48 kHz."""
    import librosa
    rng = np.random.default_rng(seed + n_talkers)
    speakers = sorted(os.listdir(speaker_root))
    if len(speakers) < n_talkers:
        print(f"  only {len(speakers)} babble speakers available, need {n_talkers} - skipped")
        return
    out_dir = os.path.join(C.NOISE_DIR, f"babble_{n_talkers}talkers")
    os.makedirs(out_dir, exist_ok=True)
    n = int(seconds * C.SAMPLE_RATE)
    for k in range(n_files):
        mix = np.zeros(n, dtype=np.float64)
        for s in rng.choice(speakers, n_talkers, replace=False):
            files = sorted(os.listdir(os.path.join(speaker_root, s)))
            stream = []
            total = 0
            while total < n:
                x, sr = sf.read(os.path.join(speaker_root, s, files[rng.integers(len(files))]), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                if sr != C.SAMPLE_RATE:
                    x = librosa.resample(x, orig_sr=sr, target_sr=C.SAMPLE_RATE)
                stream.append(x)
                total += len(x)
            stream = np.concatenate(stream)[:n]
            start = rng.integers(0, n)                 # random circular offset: talkers start at different points
            stream = np.roll(stream, start)
            mix += stream / (np.sqrt(np.mean(stream ** 2)) + 1e-9)
        mix = 0.5 * mix / (np.max(np.abs(mix)) + 1e-9)
        sf.write(os.path.join(out_dir, f"babble{n_talkers}_{k:02d}.wav"), mix.astype(np.float32), C.SAMPLE_RATE)
    print(f"  babble with {n_talkers} talkers: {n_files} x {seconds:.0f} s -> {out_dir}")


# ------------------------------------------------------------------ DEMAND
def get_demand(envs, keep):
    for env in envs:
        out_dir = os.path.join(C.NOISE_DIR, env.lower())
        if os.path.exists(os.path.join(out_dir, "ch01.wav")):
            print(f"  DEMAND {env}: already present")
            continue
        zpath = download(DEMAND_URL.format(env=env), os.path.join(DOWNLOADS, f"{env}_48k.zip"))
        with zipfile.ZipFile(zpath) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith("ch01.wav")]
            if not members:
                print(f"  DEMAND {env}: ch01.wav not found in zip - skipped")
                continue
            os.makedirs(out_dir, exist_ok=True)
            with zf.open(members[0]) as src, open(os.path.join(out_dir, "ch01.wav"), "wb") as dst:
                shutil.copyfileobj(src, dst)
        if not keep:
            os.remove(zpath)
        print(f"  DEMAND {env}: ok")


# ------------------------------------------------------------------ MS-SNSD
def get_mssnsd(types):
    for folder in ("noise_train", "noise_test"):
        with urllib.request.urlopen(urllib.request.Request(
                MSSNSD_API.format(folder=folder), headers={"User-Agent": "thesis-data-script"}), timeout=60) as r:
            listing = json.load(r)
        for item in listing:
            name = item["name"]
            t = re.sub(r"_\d+\.wav$", "", name)
            if t in types and name.endswith(".wav"):
                out = os.path.join(C.NOISE_DIR, "mssnsd_" + t.lower(), name)
                if not os.path.exists(out):
                    download(item["download_url"], out)
    print("  MS-SNSD noise downloaded (16 kHz recordings: no content above 8 kHz)")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vctk-zip", nargs="+", help="already downloaded VCTK-Corpus-0.92.zip, or the 48 kHz "
                    "Valentini clean zips (clean_trainset_28spk_wav.zip clean_testset_wav.zip)")
    ap.add_argument("--skip-vctk", action="store_true")
    ap.add_argument("--speakers", type=int, default=40, help="target speakers (half female, half male)")
    ap.add_argument("--babble-speakers", type=int, default=12, help="speakers reserved for babble, never targets")
    ap.add_argument("--max-utts", type=int, default=120, help="utterances kept per target speaker")
    ap.add_argument("--babble-talkers", type=int, nargs="+", default=[4, 8], help="talkers per babble type")
    ap.add_argument("--babble-files", type=int, default=5)
    ap.add_argument("--babble-seconds", type=float, default=120)
    ap.add_argument("--demand-envs", nargs="*", default=DEMAND_ENVS)
    ap.add_argument("--skip-demand", action="store_true")
    ap.add_argument("--mssnsd", action="store_true", help="also download MS-SNSD noise types (16 kHz)")
    ap.add_argument("--keep-downloads", action="store_true")
    args = ap.parse_args()
    os.makedirs(DOWNLOADS, exist_ok=True)

    if not args.skip_vctk:
        zip_paths = args.vctk_zip
        if not zip_paths:
            zip_path = os.path.join(DOWNLOADS, "VCTK-Corpus-0.92.zip")
            zip_paths = [zip_path]
            try:
                print("Downloading VCTK 0.92 (about 11 GB, resumable if interrupted) ...")
                download(VCTK_URL, zip_path)
            except Exception as e:
                raise SystemExit(f"\nVCTK download failed ({e}).\nDownload the zip manually from {VCTK_PAGE}\n"
                                 f"and run:  python step0_get_data.py --vctk-zip <path to zip>")
        babble_root = extract_vctk(zip_paths, args.speakers, args.babble_speakers, args.max_utts, C.SEED)
        for n in args.babble_talkers:
            make_babble(babble_root, n, args.babble_files, args.babble_seconds, C.SEED)
        if not args.keep_downloads and not args.vctk_zip:
            os.remove(zip_path)

    if not args.skip_demand:
        print("Downloading DEMAND (48 kHz, channel 1 kept) ...")
        get_demand(args.demand_envs, args.keep_downloads)

    if args.mssnsd:
        print("Downloading MS-SNSD noise types ...")
        get_mssnsd(MSSNSD_TYPES)

    os.makedirs(C.NOISE_DIR, exist_ok=True)
    types = sorted(d for d in os.listdir(C.NOISE_DIR) if os.path.isdir(os.path.join(C.NOISE_DIR, d)))
    print(f"\nnoise types now in {C.NOISE_DIR}: {len(types)}\n  " + ", ".join(types))
    print("next: python step1_prepare_data.py")


if __name__ == "__main__":
    main()
