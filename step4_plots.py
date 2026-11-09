"""
STEP 4 - thesis figures from the files written by steps 2 and 3 (nothing is typed by hand).

Outputs (in results/figures/, 300 dpi PNG):
  fig_training_curves.png        training vs validation loss per model
  fig_<metric>_vs_snr.png        each method vs input SNR (mean over noise types)
  fig_<metric>_by_noise.png      PESQ / STOI vs SNR, one panel per noise type (ordered by stationarity)
  fig_<metric>_vs_snr_by_category.png   stationary / quasi-stationary / non-stationary panels
  fig_improvement_by_noise.png   PESQ / STOI improvement over the noisy input per noise type
  fig_improvement_by_category.png       improvement per stationarity class, all methods
  fig_spectrograms_<example>.png clean / noisy / enhanced spectrograms with the measured scores

Run:  python step4_plots.py
"""
import _env  # noqa: F401  (Run button: uses the project environment automatically)
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

import config as C

FIG = os.path.join(C.RESULTS_DIR, "figures")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# method -> (colour, marker, line style). Colour follows the method everywhere; the marker is a second cue (print / colour-blind).
STYLE = {
    C.MODEL_NAMES["hybrid"]:      ("#2a78d6", "o", "-"),
    C.MODEL_NAMES["cnn_only"]:    ("#eb6834", "s", "-"),
    C.MODEL_NAMES["gru_only"]:    ("#1baf7a", "^", "-"),
    "RNNoise (official library)": ("#eda100", "D", "-"),
    "Wiener filter":              ("#e87ba4", "v", "-"),
    "Spectral Subtraction":       ("#008300", "P", "-"),
    "Noisy (unprocessed)":        ("#8a8984", "X", "-"),
    "Oracle 22-band gains":       ("#3b3a37", "*", ":"),
}
LABEL = {"pesq": "PESQ (wide-band)", "stoi": "STOI", "estoi": "ESTOI", "si_sdr": "SI-SDR (dB)"}

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
})


CATS = ["stationary", "quasi-stationary", "non-stationary"]


def noise_order(df):
    """Noise types sorted from most stationary to least stationary (by measured NSI)."""
    path = os.path.join(C.WORK_DIR, "noise_stationarity.csv")
    types = list(df["noise_type"].unique())
    if os.path.exists(path):
        stat = pd.read_csv(path).set_index("noise_type")
        return sorted(types, key=lambda t: stat["nsi_db"].get(t, 1e9)), stat
    return sorted(types), None


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


def draw_lines(ax, table, methods):
    for m in methods:
        if m not in table.index:
            continue
        c, mk, ls = STYLE[m]
        lw = 2.6 if m == C.MODEL_NAMES["hybrid"] else 1.8
        ax.plot(table.columns, table.loc[m], color=c, marker=mk, linestyle=ls, linewidth=lw, markersize=7,
                markeredgecolor="white", markeredgewidth=1.2, label=m, zorder=3 if lw > 2 else 2)
    ax.set_xticks(list(table.columns))


def training_curves():
    files = [(k, os.path.join(C.RESULTS_DIR, f"history_{k}.csv")) for k in C.MODEL_NAMES]
    files = [(k, f) for k, f in files if os.path.exists(f)]
    if not files:
        return
    fig, axes = plt.subplots(1, len(files), figsize=(4.6 * len(files), 3.6), sharey=True, squeeze=False)
    for ax, (k, f) in zip(axes[0], files):
        h = pd.read_csv(f)
        ax.plot(h["epoch"] + 1, h["loss"], color="#2a78d6", linewidth=2, label="training")
        ax.plot(h["epoch"] + 1, h["val_loss"], color="#eb6834", linewidth=2, label="validation")
        best = h["val_loss"].idxmin()
        ax.plot(h["epoch"][best] + 1, h["val_loss"][best], "o", color="#eb6834", markersize=8,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(f"best: {h['val_loss'][best]:.4f}", (h["epoch"][best] + 1, h["val_loss"][best]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=MUTED)
        ax.set_title(C.MODEL_NAMES[k], fontsize=11)
        ax.set_xlabel("Epoch")
    axes[0][0].set_ylabel("Perceptual loss")
    axes[0][-1].legend(loc="upper right")
    save(fig, "fig_training_curves.png")


def metric_vs_snr(df, methods):
    for m in LABEL:
        t = df.pivot_table(index="method", columns="snr", values=m, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        draw_lines(ax, t, methods)
        ax.set_xlabel("Input SNR (dB)")
        ax.set_ylabel(LABEL[m])
        ax.set_title(f"{LABEL[m]} vs input SNR (mean over {df['noise_type'].nunique()} noise types)", fontsize=11)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
        save(fig, f"fig_{m}_vs_snr.png")


def metric_by_noise(df, methods, metric):
    types, stat = noise_order(df)
    cols = min(4, len(types))
    rows = int(np.ceil(len(types) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.0 * rows), sharex=True, sharey=True, squeeze=False)
    for ax, nt in zip(axes.ravel(), types):
        sub = df[df["noise_type"] == nt]
        draw_lines(ax, sub.pivot_table(index="method", columns="snr", values=metric, aggfunc="mean"), methods)
        title = nt
        if stat is not None and nt in stat.index:
            title += f"\n{stat.loc[nt, 'category']}, NSI {stat.loc[nt, 'nsi_db']:.1f} dB"
        if not bool(sub["seen"].iloc[0]):
            title += " (unseen)"
        ax.set_title(title, fontsize=9)
    for i in range(len(types), rows * cols):
        axes.ravel()[i].set_visible(False)
        above = axes[rows - 2, i % cols] if rows > 1 else None
        if above is not None:
            above.tick_params(labelbottom=True)
            above.set_xlabel("Input SNR (dB)")
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Input SNR (dB)")
    for ax in axes[:, 0]:
        ax.set_ylabel(LABEL[metric])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.tight_layout()
    save(fig, f"fig_{metric}_by_noise.png")


def improvement_by_noise(df):
    noisy = "Noisy (unprocessed)"
    nets = [C.MODEL_NAMES[k] for k in ("hybrid", "cnn_only", "gru_only") if C.MODEL_NAMES[k] in set(df["method"])]
    if not nets:
        return
    types, stat = noise_order(df)
    fig, axes = plt.subplots(2, 1, figsize=(min(22, 0.75 * len(types) + 4), 7.5), sharex=True)
    for ax, m in zip(axes, ("pesq", "stoi")):
        t = df.pivot_table(index="noise_type", columns="method", values=m, aggfunc="mean").reindex(types)
        width = 0.8 / len(nets)
        for i, name in enumerate(nets):
            x = np.arange(len(types)) + (i - (len(nets) - 1) / 2) * width
            ax.bar(x, t[name] - t[noisy], width=width * 0.88, color=STYLE[name][0], label=name, zorder=2)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(f"{LABEL[m]} gain")
    short = {"stationary": "S", "quasi-stationary": "Q", "non-stationary": "N"}
    labels = types if stat is None else [f"{t}\n({short.get(stat['category'].get(t, '?'), '?')})" for t in types]
    axes[1].set_xticks(np.arange(len(types)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_title("Improvement over the noisy input per noise type, ordered by NSI (S = stationary, Q = quasi-stationary, N = non-stationary)",
                      fontsize=11, loc="left")
    axes[0].legend(loc="upper right", ncol=len(nets), fontsize=9)
    save(fig, "fig_improvement_by_noise.png")


def improvement_by_category(df):
    if "category" not in df.columns:
        return
    noisy, oracle = "Noisy (unprocessed)", "Oracle 22-band gains"
    cats = [c for c in CATS if c in set(df["category"])]
    methods = [m for m in STYLE if m in set(df["method"]) and m not in (noisy, oracle)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, m in zip(axes, ("pesq", "stoi")):
        t = df.pivot_table(index="category", columns="method", values=m, aggfunc="mean").reindex(cats)
        width = 0.8 / len(methods)
        for i, name in enumerate(methods):
            x = np.arange(len(cats)) + (i - (len(methods) - 1) / 2) * width
            ax.bar(x, t[name] - t[noisy], width=width * 0.88, color=STYLE[name][0], label=name, zorder=2)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xticks(np.arange(len(cats)))
        ax.set_xticklabels(cats)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(f"{LABEL[m]} improvement over noisy input")
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.05, 1.02), ncol=3, fontsize=9)
    save(fig, "fig_improvement_by_category.png")


def metric_vs_snr_by_category(df, methods, metric):
    if "category" not in df.columns:
        return
    cats = [c for c in CATS if c in set(df["category"])]
    fig, axes = plt.subplots(1, len(cats), figsize=(4.6 * len(cats), 3.8), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], cats):
        sub = df[df["category"] == c]
        draw_lines(ax, sub.pivot_table(index="method", columns="snr", values=metric, aggfunc="mean"), methods)
        ax.set_title(f"{c} noise ({sub['noise_type'].nunique()} types)", fontsize=10)
        ax.set_xlabel("Input SNR (dB)")
    axes[0][0].set_ylabel(LABEL[metric])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.tight_layout()
    save(fig, f"fig_{metric}_vs_snr_by_category.png")


def spectrograms(df):
    dirs = sorted(glob.glob(os.path.join(C.RESULTS_DIR, "audio", "*")))
    if not dirs:
        return
    first_utt = min(int(os.path.basename(d)[3:5]) for d in dirs)
    wanted = None
    if "category" in df.columns:                              # one per category (median NSI) + all babble types
        _, stat = noise_order(df)
        if stat is not None:
            wanted = {t for t in stat.index if "babble" in t}
            for c in CATS:
                g = stat[stat["category"] == c].sort_values("nsi_db")
                if len(g):
                    wanted.add(g.index[len(g) // 2])
    for d in dirs:
        tag = os.path.basename(d)                               # utt00_babble_8talkers_5dB
        utt, snr = int(tag[3:5]), int(tag.rsplit("_", 1)[1][:-2])
        nt = tag[6:].rsplit("_", 1)[0]
        if utt != first_utt or (wanted is not None and nt not in wanted):
            continue
        scores = df[(df["utt"] == utt) & (df["noise_type"] == nt) & (df["snr"] == snr)].set_index("method")
        panels = [("Clean reference", "clean.wav", None), ("Noisy input", "Noisy.wav", "Noisy (unprocessed)")]
        for k in ("cnn_only", "gru_only", "hybrid"):
            name = C.MODEL_NAMES[k]
            panels.append((name, name.split(" (")[0].replace(" ", "_") + ".wav", name))
        found = []
        for title, fname, method in panels:                       # .wav, or the same name as .flac
            for f in (fname, fname[:-4] + ".flac"):
                if os.path.exists(os.path.join(d, f)):
                    found.append((title, f, method))
                    break
        panels = found
        if len(panels) < 2:
            continue
        fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.3 * len(panels)), sharex=True)
        panels[0] = (f"Clean reference   ({nt} noise, {snr} dB SNR)",) + panels[0][1:]
        ref = None
        for ax, (title, fname, method) in zip(np.atleast_1d(axes), panels):
            x, sr = sf.read(os.path.join(d, fname), dtype="float32")
            D = np.abs(librosa.stft(x, n_fft=1024, hop_length=256))
            ref = ref or D.max()
            img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=ref, top_db=80), sr=sr, hop_length=256,
                                           x_axis="time", y_axis="hz", ax=ax, cmap="magma", vmin=-80, vmax=0)
            if method is not None and method in scores.index:
                s = scores.loc[method]
                title += f"   PESQ {s['pesq']:.2f} | STOI {s['stoi']:.3f} | SI-SDR {s['si_sdr']:.1f} dB"
            ax.set_title(title, fontsize=10, loc="left")
            ax.set_ylabel("Hz")
            ax.grid(False)
            ax.label_outer()
        fig.colorbar(img, ax=np.atleast_1d(axes).tolist(), format="%+2.0f dB", pad=0.01, fraction=0.03)
        save(fig, f"fig_spectrograms_{tag}.png")


def main():
    training_curves()
    raw = os.path.join(C.RESULTS_DIR, "results_raw.csv")
    if not os.path.exists(raw):
        print("results_raw.csv not found - run step3_evaluate.py first")
        return
    df = pd.read_csv(raw)
    methods = [m for m in STYLE if m in set(df["method"])]
    metric_vs_snr(df, methods)
    for m in ("pesq", "stoi"):
        metric_by_noise(df, methods, m)
        metric_vs_snr_by_category(df, methods, m)
    improvement_by_noise(df)
    improvement_by_category(df)
    spectrograms(df)


if __name__ == "__main__":
    main()
    if os.environ.get("SE_NO_VIEWER") != "1" and os.path.isdir(FIG):
        import subprocess, sys
        subprocess.call([sys.executable, "FIGURES.py", "--view-only"])   # open the figure window
