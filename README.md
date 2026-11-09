# Hybrid CNN-GRU Speech Enhancement for Digital Hearing Aids

Code and trained models for the thesis *Enhancing Speech Intelligibility in Digital Hearing Aids Using Advanced
Signal Processing Techniques*. The proposed model is a compact, causal 48 kHz network (2 Conv1D + 3 GRU layers,
80,038 parameters) that predicts 22 critical-band gains per 10 ms frame. It is compared with CNN-only and GRU-only
ablations, spectral subtraction, a Wiener filter and RNNoise.

## How to try it

1. Install **Python 3.11** from python.org (tick *Add python.exe to PATH*).
2. Open this folder in **VS Code**, open `RUN.py` and press **Run** (▶), or run `python RUN.py` in a terminal.
3. The first run installs the libraries automatically (TensorFlow, librosa, ...; 10–40 minutes). Later runs start at once.
4. Pressing **Run** then does everything automatically:
   - prints every results table of Chapter 4 (Tables 4.1 to 4.11, computed from the saved result files), and
   - runs a live test: a sentence from an unseen speaker is mixed with keyboard typing noise at 5 dB, enhanced by
     the three networks, and PESQ, STOI, ESTOI and SI-SDR are printed.

For more options run `python RUN.py --menu`:

| Option | What it does |
| --- | --- |
| 1 | Quick demo (typing noise, 5 dB) |
| 2 | Demo with any of the 2 test speakers, 16 noise types and 5 SNRs |
| 3 | Enhances a noisy recording: the included example, or any `.wav` file |
| 4 | Shows all the thesis results tables |
| 5 | Draws the thesis figures from the saved results and shows them (`FIGURES.py`) |

All audio produced by the demo is saved in `demo_output/` (clean, noisy and the output of each network) so it can be
listened to. A single sentence can rank the networks differently from the thesis averages, which are
measured over 1,600 mixtures (option 4). Without the C++ build tools on Windows the `pesq` package may not install; PESQ is then shown as n/a
and every other metric still works.

Command-line use:

```
python demo_enhance.py examples/noisy_example_babble_5dB.wav --model hybrid
python demo_enhance.py --clean datasets/clean/r02673/r02673_010.wav --noise typing --snr 0
```

## Contents

| File / folder | Content |
| --- | --- |
| `RUN.py` | One-click launcher (environment setup, all results tables, live test) |
| `step4_plots.py` | Draws every thesis figure from the saved result files into `results/figures/` (300 dpi) |
| `FIGURES.py` | Runs `step4_plots.py` and shows the figures in a window (Next / Previous buttons) |
| `RESULTS.py` | Prints Tables 4.1 to 4.11 of the thesis from the files in `results/` and `work/` |
| `demo_enhance.py` | Enhances a recording, or builds and scores one test mixture |
| `config.py` | All settings (framing, features, bands, network sizes, training) |
| `audio_utils.py` | STFT with Vorbis window, log-Mel features, ideal band gains, metrics |
| `models.py` | Hybrid CNN-GRU, CNN-only and GRU-only networks |
| `baselines.py` | Spectral subtraction and Wiener filter |
| `dataset.py` | Mixing and training-data generation |
| `step0_*.py` → `step3_evaluate.py` | Full pipeline: data, preparation and NSI, training, evaluation |
| `models/` | The three trained networks |
| `work/` | Speaker and noise splits, input normalisation, test manifest, data report |
| `results/` | Thesis result tables (CSV and `tables.md`), training histories, and the audio of the Fig. 4.6 example |
| `datasets/` | Demo data only: 2 test sentences (unseen speakers) and 20 s of the test portion of each of the 16 noise types |
| `examples/` | A ready noisy recording (female speaker + babble, 5 dB) |

The full 48 kHz dataset (3.3 GB) is not included. `step0_get_data_dns.py` rebuilds it from the public DNS Challenge
corpus; steps 0–3 then reproduce the training and the full evaluation (1,600 test mixtures, 8 methods).
