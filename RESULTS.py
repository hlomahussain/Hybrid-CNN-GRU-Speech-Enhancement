"""
RESULTS.py - prints every results table of Chapter 4 (Tables 4.1 to 4.11), numbered as in the thesis,
straight from the saved result files. Nothing is retyped by hand: every number is read or computed
from work/ and results/.

Press Run in VS Code, or:  python RESULTS.py
"""
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parent

if __name__ == "__main__" and os.environ.get("RUN_NO_BOOTSTRAP") != "1":
    from RUN import DONE, VPY, in_project_env, setup
    if not in_project_env():
        if not DONE.exists():
            setup()
        sys.exit(subprocess.call([str(VPY), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=ROOT))

import re

import numpy as np
import pandas as pd

R, W = ROOT / "results", ROOT / "work"
HYB, CNN, GRU = "Hybrid CNN-GRU (proposed)", "CNN-only", "GRU-only"
NOISY, ORACLE, RNN = "Noisy (unprocessed)", "Oracle 22-band gains", "RNNoise (official library)"
ORDER = [NOISY, "Spectral Subtraction", "Wiener filter", RNN, CNN, GRU, HYB, ORACLE]
SHOW = {RNN: "RNNoise (pre-trained)", ORACLE: "Oracle 22-band gains (upper bound)",
        "Spectral Subtraction": "Spectral subtraction"}
NOISE_NAME = {
    "white_noise": ("White noise", "Generated"), "pink_noise": ("Pink noise", "Generated"),
    "speech_shaped_noise": ("Speech-shaped noise", "Generated"), "dkitchen": ("Kitchen (DKITCHEN)", "DEMAND"),
    "fan": ("Fan", "DNS / Freesound"), "ooffice": ("Office (OOFFICE)", "DEMAND"), "npark": ("Park (NPARK)", "DEMAND"),
    "babble_8talkers": ("Babble, 8 talkers", "Generated"), "dwashing": ("Washing machine (DWASHING)", "DEMAND"),
    "dliving": ("Living room (DLIVING)", "DEMAND"), "babble_4talkers": ("Babble, 4 talkers", "Generated"),
    "squeaky_chair": ("Squeaky chair", "DNS / Freesound"), "munching": ("Munching", "DNS / Freesound"),
    "door": ("Door", "DNS / Freesound"), "copy_machine": ("Copy machine", "DNS / Freesound"),
    "typing": ("Typing", "DNS / Freesound"),
}


def title(text):
    print("\n" + "=" * 100)
    print(" " + text)
    print("=" * 100)


def show(df):
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 40):
        print(df.to_string())


def names(index):
    return [SHOW.get(m, m) for m in index]


def fmt(x, d):
    """Round half up (as in the thesis tables), e.g. 0.355 -> 0.36."""
    from decimal import Decimal, ROUND_HALF_UP
    v = Decimal(str(round(float(x), 6))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return f"{v:.{d}f}".replace("-0." + "0" * d, "0." + "0" * d)


def sign(x, d=2):
    t = fmt(x, d)
    return t if t.startswith("-") or float(t) == 0 else "+" + t


def main():
    raw = pd.read_csv(R / "results_raw.csv")
    methods = [m for m in ORDER if m in set(raw["method"])]

    # 4.1 speech partitions (from the data report written by step 1)
    title("Table 4.1. Speech partitions.")
    rows = []
    for line in (W / "data_report.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s+(train|val|test)\s*:\s*(\d+) files,\s*(\d+) speakers?, ([\d.]+) h", line)
        if m:
            name = {"train": "Training", "val": "Validation", "test": "Test"}[m.group(1)]
            rows.append(dict(Partition=name, Speakers=int(m.group(3)), Utterances=int(m.group(2)),
                             **{"Duration (h)": float(m.group(4))}))
    show(pd.DataFrame(rows).set_index("Partition"))

    # 4.2 noise types and NSI
    title("Table 4.2. Noise types, measured NSI and stationarity class.")
    st = pd.read_csv(W / "noise_stationarity.csv").sort_values("nsi_db")
    t = pd.DataFrame({
        "Noise type": [NOISE_NAME.get(n, (n, ""))[0] for n in st["noise_type"]],
        "Source": [NOISE_NAME.get(n, ("", ""))[1] for n in st["noise_type"]],
        "NSI (dB)": st["nsi_db"].map(lambda v: f"{v:.2f}").values,
        "Class": st["category"].str.capitalize().values,
        "Used in training": np.where(st["seen"].astype(bool), "Yes", "No (unseen)"),
    }).set_index("Noise type")
    show(t)

    # 4.3 networks
    title("Table 4.3. The three networks compared.")
    cx = pd.read_csv(R / "complexity.csv").set_index("model")
    arch = {HYB: "2 causal Conv1D (32, 64) + 3 GRU (24, 48, 96) + dense (22)",
            CNN: "4 causal Conv1D (32, 64, 96, 128) + dense (22)",
            GRU: "3 GRU (24, 48, 96) + dense (22)"}
    show(pd.DataFrame({"Architecture": [arch[m] for m in (HYB, CNN, GRU)],
                       "Trainable parameters": [f"{int(cx.loc[m, 'trainable_params']):,}" for m in (HYB, CNN, GRU)]},
                      index=[HYB, CNN + " (ablation)", GRU + " (ablation)"]))

    # 4.4 training
    title("Table 4.4. Training length and best validation loss.")
    rows = []
    for key, name in (("hybrid", "Hybrid CNN-GRU"), ("gru_only", "GRU-only"), ("cnn_only", "CNN-only")):
        h = pd.read_csv(R / f"history_{key}.csv")
        b = int(h["val_loss"].idxmin())
        rows.append({"Model": name, "Epochs run": len(h), "Best epoch": b + 1,
                     "Best validation loss": f"{h['val_loss'][b]:.4f}"})
    show(pd.DataFrame(rows).set_index("Model"))

    # 4.5 overall
    title("Table 4.5. Mean scores over 1,600 test mixtures (+/- half-width of the 95% confidence interval).")
    ov = pd.read_csv(R / "table_overall.csv", index_col=0).reindex(methods)
    t = pd.DataFrame(index=names(ov.index))
    for m, lab, d in (("pesq", "PESQ", 3), ("stoi", "STOI", 3), ("estoi", "ESTOI", 3), ("si_sdr", "SI-SDR (dB)", 2)):
        t[lab] = [f"{fmt(a, d)} +/- {fmt(b, d)}" for a, b in zip(ov[f"{m}_mean"], ov[f"{m}_ci95"])]
    show(t)

    # 4.6 SI-SDR vs SNR
    title("Table 4.6. SI-SDR (dB) versus input SNR.")
    t = raw.pivot_table(index="method", columns="snr", values="si_sdr", aggfunc="mean")
    t = t.reindex([m for m in (NOISY, "Wiener filter", RNN, GRU, CNN, HYB) if m in t.index]).round(1)
    t.index, t.columns = names(t.index), [f"{c} dB" for c in t.columns]
    show(t)
    for m, lab in (("pesq", "PESQ"), ("stoi", "STOI")):
        t = raw.pivot_table(index="method", columns="snr", values=m, aggfunc="mean").reindex(methods).round(3)
        t.index, t.columns = names(t.index), [f"{c} dB" for c in t.columns]
        title(f"(Figures 4.2 / 4.3) {lab} versus input SNR.")
        show(t)

    # 4.7 SI-SDR improvement per stationarity class
    title("Table 4.7. SI-SDR improvement (dB) over the noisy input per stationarity class.")
    g = raw.pivot_table(index="method", columns="category", values="si_sdr", aggfunc="mean")
    g = (g - g.loc[NOISY]).reindex([m for m in methods if m not in (NOISY, ORACLE)])
    g = g[["stationary", "quasi-stationary", "non-stationary"]].map(sign)
    g.index, g.columns = names(g.index), ["Stationary", "Quasi-stationary", "Non-stationary"]
    show(g)
    for m, lab, d in (("pesq", "PESQ", 2), ("stoi", "STOI", 3)):
        g = raw.pivot_table(index="method", columns="category", values=m, aggfunc="mean")
        g = (g - g.loc[NOISY]).reindex([x for x in methods if x not in (NOISY, ORACLE)])
        g = g[["stationary", "quasi-stationary", "non-stationary"]].map(lambda v: sign(v, d))
        g.index, g.columns = names(g.index), ["Stationary", "Quasi-stationary", "Non-stationary"]
        title(f"(Figure 4.4) {lab} improvement over the noisy input per stationarity class.")
        show(g)

    # 4.8 babble
    title("Table 4.8. Results for multi-talker babble, mean over 5 SNRs and 20 utterances.")
    b = raw[raw["noise_type"].isin(["babble_4talkers", "babble_8talkers"])]
    t = b.pivot_table(index="method", columns="noise_type", values=["pesq", "stoi", "si_sdr"], aggfunc="mean")
    t = t.reindex([m for m in (NOISY, "Wiener filter", RNN, CNN, GRU, HYB) if m in t.index])
    out = pd.DataFrame(index=names(t.index))
    for nt, lab in (("babble_4talkers", "4-talker"), ("babble_8talkers", "8-talker")):
        out[f"{lab} PESQ"] = t[("pesq", nt)].round(3).values
        out[f"{lab} STOI"] = t[("stoi", nt)].round(3).values
        out[f"{lab} SI-SDR"] = t[("si_sdr", nt)].round(2).values
    show(out)

    # 4.9 seen vs unseen gains
    title("Table 4.9. Improvement over the noisy input on seen and unseen noise types.")
    su = pd.read_csv(R / "table_seen_unseen.csv", index_col=0)          # written by step 3
    su.columns = pd.MultiIndex.from_tuples([tuple(c.rsplit("_", 1)) for c in su.columns])
    t = su - su.loc[NOISY]
    t = t.reindex([m for m in ("Wiener filter", RNN, CNN, GRU, HYB) if m in t.index])
    out = pd.DataFrame(index=names(t.index))
    for m, lab, d in (("pesq", "PESQ gain", 2), ("stoi", "STOI gain", 3), ("si_sdr", "SI-SDR gain", 2)):
        for s in ("seen", "unseen"):
            out[f"{lab}, {s}"] = t[(m, s)].map(lambda v: sign(v, d)).values
    show(out)
    hp = t.loc[HYB, ("pesq", "unseen")] / t.loc[HYB, ("pesq", "seen")] * 100
    cp = t.loc[CNN, ("pesq", "unseen")] / t.loc[CNN, ("pesq", "seen")] * 100
    print(f"\n Share of the PESQ gain kept on unseen noise: Hybrid {hp:.0f}%, CNN-only {cp:.0f}%")

    # 4.10 significance and wins (320 conditions = 20 utterances x 16 noise types, averaged over SNR)
    title("Table 4.10. Hybrid CNN-GRU minus each comparison method "
          "(Wins = share of the 320 conditions; Holm-corrected p-values).")
    sig = pd.read_csv(R / "significance.csv")
    cond = {m: raw.pivot_table(index=["utt", "noise_type"], columns="method", values=m, aggfunc="mean")
            for m in ("pesq", "stoi", "estoi", "si_sdr")}
    rows = []
    for other in (CNN, GRU, RNN):
        for m, lab in (("pesq", "PESQ"), ("stoi", "STOI"), ("estoi", "ESTOI"), ("si_sdr", "SI-SDR")):
            s = sig[(sig["compared_with"] == other) & (sig["metric"] == m)].iloc[0]
            d = cond[m][HYB] - cond[m][other]
            p = s["p_holm_conditions"]
            sigtxt = "No" if p >= 0.05 else ("Yes" if s["mean_difference"] > 0 or m == "si_sdr" and d.mean() > 0
                                             else f"Yes ({SHOW.get(other, other).split(' (')[0]} better)")
            rows.append({"Compared with": SHOW.get(other, other).split(" (")[0], "Metric": lab,
                         "Mean difference": (sign(s["mean_difference"], 2) + " dB") if m == "si_sdr"
                         else sign(s["mean_difference"], 3),
                         "Wins (%)": f"{(d > 0).mean() * 100:.1f}",
                         "p (Holm)": "< 0.001" if p < 0.001 else f"{p:.3f}", "Significant": sigtxt})
    show(pd.DataFrame(rows).set_index(["Compared with", "Metric"]))

    # 4.11 complexity
    title("Table 4.11. Model size and processing speed.")
    t = cx.reindex([CNN, GRU, HYB])
    show(pd.DataFrame({"Trainable parameters": [f"{int(v):,}" for v in t["trainable_params"]],
                       "Weights, float32 (kB)": t["float32_weights_kB"].round(0).astype(int).values,
                       "Real-time factor": t["real_time_factor"].values,
                       "Time per 10 ms frame (ms)": t["ms_per_10ms_frame"].round(2).values}, index=t.index))
    print(f"\n Hybrid: {1 / t.loc[HYB, 'real_time_factor']:.0f} x faster than real time")

    # per noise type (Section 4.6, Figure 4.5)
    for m, lab in (("pesq", "PESQ"), ("stoi", "STOI"), ("si_sdr", "SI-SDR (dB)")):
        t = raw.pivot_table(index="noise_type", columns="method", values=m, aggfunc="mean")
        t = t.reindex(list(st["noise_type"]))[[x for x in methods if x != ORACLE]].round(3 if m != "si_sdr" else 2)
        t.columns = [SHOW.get(c, c).split(" (")[0] for c in t.columns]
        title(f"(Section 4.6 / Figure 4.5) {lab} per noise type, ordered by NSI.")
        show(t)

    print("\nAll tables (more detail, standard deviations): results\\tables.md     Figures: results\\figures")


if __name__ == "__main__":
    main()
