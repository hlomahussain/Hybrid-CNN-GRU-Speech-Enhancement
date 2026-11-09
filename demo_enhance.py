"""
DEMO - enhance any recording with a trained model (no dataset needed).

    python demo_enhance.py noisy.wav                         -> noisy_enhanced_hybrid.wav
    python demo_enhance.py noisy.wav --model cnn_only
    python demo_enhance.py --clean clean.wav --noise babble  --snr 5
          mixes a clean file with a test noise at 5 dB, enhances it with all three networks,
          saves every signal and prints PESQ / STOI / ESTOI / SI-SDR for each

Any sample rate / channel count is accepted (converted to 48 kHz mono).
"""
import _env  # noqa: F401  (Run button: uses the project environment automatically)
import argparse
import glob
import os

import numpy as np
import soundfile as sf

import config as C
from audio_utils import load_audio, mix_at_snr, compute_metrics_many
from step3_evaluate import make_runner, enhance


def find_noise(name):
    hits = sorted(glob.glob(os.path.join(C.NOISE_DIR, f"*{name}*", "*.*")))
    hits = [h for h in hits if h.lower().endswith((".wav", ".flac"))]
    if not hits:
        types = sorted(os.listdir(C.NOISE_DIR)) if os.path.isdir(C.NOISE_DIR) else []
        raise SystemExit(f"no noise matching '{name}' in {C.NOISE_DIR}. Available: {types}")
    return hits[0]


def main():
    from models import load_trained
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("noisy", nargs="?", help="noisy recording to enhance")
    ap.add_argument("--model", default="hybrid", choices=list(C.MODEL_NAMES))
    ap.add_argument("--clean", help="clean speech file (with --noise: build a test mixture and score it)")
    ap.add_argument("--noise", help="noise type name or part of it, e.g. babble_4, dliving, typing")
    ap.add_argument("--snr", type=float, default=5.0)
    ap.add_argument("--out", default="demo_output")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.noisy:
        model = load_trained(args.model)
        if model is None:
            raise SystemExit(f"{C.MODELS_DIR}/{args.model}.keras not found")
        x = load_audio(args.noisy)
        y = enhance(make_runner(model), x)
        out = os.path.join(args.out, os.path.splitext(os.path.basename(args.noisy))[0] + f"_enhanced_{args.model}.wav")
        sf.write(out, y, C.SAMPLE_RATE)
        print(f"enhanced -> {out}")
        return

    if not (args.clean and args.noise):
        raise SystemExit("give a noisy file, or --clean <file> --noise <type> [--snr dB]")
    clean = load_audio(args.clean, max_seconds=C.MAX_UTT_SECONDS)
    noise_path = find_noise(args.noise)
    noise = load_audio(noise_path)
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    # long noise recordings: take the segment from the test portion (last 20 %), never heard in training
    lo = 0
    if len(noise) >= 60 * C.SAMPLE_RATE:
        lo = int(0.8 * len(noise))
    start = np.random.default_rng(0).integers(lo, len(noise) - len(clean) + 1)
    clean, noisy = mix_at_snr(clean, noise[start:start + len(clean)], args.snr)
    outputs = {"noisy": noisy}
    for kind in C.MODEL_NAMES:
        model = load_trained(kind)
        if model is not None:
            outputs[kind] = enhance(make_runner(model), noisy)
    sf.write(os.path.join(args.out, "clean.wav"), clean, C.SAMPLE_RATE)
    for name, sig in outputs.items():
        sf.write(os.path.join(args.out, f"{name}.wav"), sig, C.SAMPLE_RATE)
    scores = compute_metrics_many(clean, outputs)
    print(f"\n{os.path.basename(args.clean)} + {os.path.basename(os.path.dirname(noise_path))} at {args.snr:g} dB SNR")
    print(f"{'signal':10s} {'PESQ':>6s} {'STOI':>6s} {'ESTOI':>6s} {'SI-SDR':>7s}")
    for name, m in scores.items():
        pq = "   n/a" if np.isnan(m["pesq"]) else f"{m['pesq']:6.3f}"
        print(f"{name:10s} {pq} {m['stoi']:6.3f} {m['estoi']:6.3f} {m['si_sdr']:7.2f}")
    print(f"\nall signals saved in {args.out}/")


if __name__ == "__main__":
    main()
