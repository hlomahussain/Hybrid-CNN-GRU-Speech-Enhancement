"""
Central configuration for all experiments.
Edit values here only; every script imports from this file.

Quick sanity run (small data, few epochs) without editing anything:
    Windows:  set SE_QUICK=1
    Linux:    export SE_QUICK=1
"""
import os

# ------------------------------------------------------------------ signal
SAMPLE_RATE = 48000          # full-band
WINDOW_SIZE = 960            # 20 ms
HOP_LENGTH = 480             # 10 ms hop (50 % overlap)
N_MELS = 128                 # network input: log-Mel bins
NUM_BANDS = 22               # network output: critical-band gains
TIME_STEPS = 100             # frames per training segment (1 s)

# 22 critical-band edges (Hz) for 48 kHz, as used in the proposal
BARK_EDGES = [0, 200, 400, 600, 800, 1000, 1200, 1480, 1760, 2120,
              2480, 2920, 3440, 4040, 4800, 5800, 7000, 8500, 10500,
              13500, 16500, 20000, 24000]

# -------------------------------------------------------------------- data
CLEAN_DIR = os.path.join("datasets", "clean")   # any depth; top-level sub-folder = speaker
NOISE_DIR = os.path.join("datasets", "noise")   # noise/<type>/*.wav  or  noise/<type>_<n>.wav

# Noise types that are NEVER used for training, only for testing
# (generalisation to unseen noise). Names = noise type names, lower-case.
UNSEEN_NOISE_TYPES = ["pink_noise", "copy_machine",                       # test-only noise types
                      "dkitchen", "dliving", "dwashing", "npark", "ooffice",   # DEMAND (only the test part is shipped)
                      "pcafeter", "straffic", "tcar"]                         # missing names are ignored

# Optional: restrict to these noise types (empty list = use every type found)
USE_NOISE_TYPES = []

CLEAN_SPLIT = (0.80, 0.10, 0.10)   # train / val / test   (by speaker when possible)
NOISE_SPLIT = (0.70, 0.10, 0.20)   # train / val / test   (by time inside every noise file)

TRAIN_SNR_RANGE = (0.0, 20.0)      # dB, uniform
TEST_SNRS = [0, 5, 10, 15, 20]     # dB

N_TRAIN_SEGMENTS = 30000           # 1-second segments  (~8.3 h of noisy speech)
N_VAL_SEGMENTS = 3000
N_TEST_UTTS = 20                   # test utterances; each is mixed with every noise type and SNR

# Noise stationarity classification (thesis section "Classification of noise by stationarity").
# Non-stationarity index (NSI) = median over frequency bands (up to 8.5 kHz) of the standard deviation
# over time of the band level in dB, after 100 ms smoothing. Measured on the training part of each noise.
STATIONARITY_THRESHOLDS_DB = (2.5, 5.0)   # NSI < 2.5 stationary | 2.5-5 quasi-stationary | >= 5 non-stationary
NOISE_CATEGORY_OVERRIDES = {}             # e.g. {"dwashing": "non-stationary"} to override the measured class
MAX_UTT_SECONDS = 10.0             # longer utterances are cropped
MIN_UTT_SECONDS = 1.5

SEED = 1234

# ---------------------------------------------------------------- training
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 8
LOSS_GAMMA = 0.5                   # perceptual loss exponent (RNNoise eq. 8)
GRU_ACTIVATION = "relu"            # as in the proposal / RNNoise
CONV_PADDING = "causal"            # causal = no look-ahead (real-time); "same" = 2 frames look-ahead

# ------------------------------------------------------------------- paths
WORK_DIR = "work"                  # features, splits, test manifest
MODELS_DIR = "models"
RESULTS_DIR = os.environ.get("SE_RESULTS_DIR", "results")   # RUN.py option 6 writes to results_rerun

# -------------------------------------------------------------- evaluation
PESQ_SR = 16000                    # PESQ wide-band is defined at 16 kHz
N_AUDIO_EXAMPLES = 2               # utterances saved as wav for listening (per noise type, at EXAMPLE_SNR)
EXAMPLE_SNR = 5
N_JOBS = max(1, min(6, (os.cpu_count() or 2) - 1))   # parallel processes for PESQ / STOI (1 = no multiprocessing)

# --------------------------------------------------------------- quick run
QUICK = os.environ.get("SE_QUICK", "0") == "1"
if QUICK:
    N_TRAIN_SEGMENTS = 1500
    N_VAL_SEGMENTS = 300
    N_TEST_UTTS = 4
    EPOCHS = 3
    TEST_SNRS = [0, 10, 20]
    WORK_DIR, MODELS_DIR, RESULTS_DIR = "work_quick", "models_quick", "results_quick"

# Model registry: key -> display name (order = order in tables and figures)
MODEL_NAMES = {
    "hybrid": "Hybrid CNN-GRU (proposed)",
    "cnn_only": "CNN-only",
    "gru_only": "GRU-only",
}
