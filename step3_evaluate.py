"""
STEP 3 - evaluate every method on the same held-out test mixtures.

Outputs (in results/):
  results_raw.csv        one row per (mixture, method): PESQ, STOI, ESTOI, SI-SDR
  table_<metric>_by_snr.csv / table_<metric>_by_noise.csv
  table_overall.csv      mean +- std per method and improvement over the noisy input
  table_seen_unseen.csv  seen vs unseen noise types (if UNSEEN_NOISE_TYPES is set)
  table_<metric>_by_category.csv   stationary / quasi-stationary / non-stationary noise
  table_gain_by_category.csv       improvement over the noisy input per stationarity category
  significance.csv       paired Wilcoxon test: proposed model vs every other method
                         (per mixture, and per utterance x noise condition with Holm correction)
  complexity.csv         parameters, model size, real-time factor
  tables.md              all tables in Markdown, ready to copy
  audio/                 a few enhanced examples for listening

Run:  python step3_evaluate.py
"""
import _env  # noqa: F401  (Run button: uses the project environment automatically)
import argparse
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import config as C
import baselines
from audio_utils import stft, logmel_features, ideal_band_gains, apply_band_gains, compute_metrics_many
from dataset import make_test_mixture

METRICS = ["pesq", "stoi", "estoi", "si_sdr"]
PROPOSED = C.MODEL_NAMES["hybrid"]
NOISY, ORACLE = "Noisy (unprocessed)", "Oracle 22-band gains"


def make_runner(model):
    import tensorflow as tf

    @tf.function(input_signature=[tf.TensorSpec([1, None, C.N_MELS], tf.float32)], reduce_retracing=True)
    def run(x):
        return model(x, training=False)
    return run


def enhance(run, noisy):
    S = stft(noisy)
    gains = run(logmel_features(S)[None])[0].numpy()
    return apply_band_gains(S, gains, len(noisy))


def measure_rtf(run, seconds=10.0, repeats=3):
    x = np.random.default_rng(0).standard_normal(int(seconds * C.SAMPLE_RATE)).astype(np.float32) * 0.05
    enhance(run, x)                                   # warm-up
    best = min(_timed(lambda: enhance(run, x)) for _ in range(repeats))
    return best / seconds


def _timed(fn):
    t = time.perf_counter(); fn(); return time.perf_counter() - t


def main():
    from models import load_trained, count_params        # imports TensorFlow (kept out of worker processes)
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="keep results/results_raw.csv and evaluate only mixtures not in it (e.g. after step1b)")
    args = ap.parse_args()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    manifest = pd.read_csv(os.path.join(C.WORK_DIR, "test_manifest.csv"))
    raw_path = os.path.join(C.RESULTS_DIR, "results_raw.csv")
    old = None
    if args.resume and os.path.exists(raw_path):
        old = pd.read_csv(raw_path).drop(columns=["category"], errors="ignore")
        print(f"resume: {old['mixture'].nunique()} mixtures already evaluated")

    # ------------------------------------------------------------- methods
    methods = {NOISY: lambda clean, noisy: noisy,
               "Spectral Subtraction": lambda clean, noisy: baselines.spectral_subtraction(noisy),
               "Wiener filter": lambda clean, noisy: baselines.wiener_filter(noisy)}
    if baselines.rnnoise_available():
        methods["RNNoise (official library)"] = lambda clean, noisy: baselines.rnnoise(noisy)
    else:
        print("pyrnnoise not installed -> RNNoise baseline skipped  (pip install pyrnnoise)")

    complexity = []
    for kind in ["gru_only", "cnn_only", "hybrid"]:
        model = load_trained(kind)
        if model is None:
            print(f"models/{kind}.keras not found -> skipped")
            continue
        run = make_runner(model)
        methods[C.MODEL_NAMES[kind]] = (lambda clean, noisy, run=run: enhance(run, noisy))
        trainable, total = count_params(model)
        rtf = measure_rtf(run)
        complexity.append(dict(model=C.MODEL_NAMES[kind], trainable_params=trainable, total_params=total,
                               size_MB=round(os.path.getsize(os.path.join(C.MODELS_DIR, f"{kind}.keras")) / 1e6, 2),
                               float32_weights_kB=round(total * 4 / 1024, 1),
                               real_time_factor=round(rtf, 4),
                               ms_per_10ms_frame=round(rtf * 10, 3)))
    methods[ORACLE] = lambda clean, noisy: apply_band_gains(
        stft(noisy), ideal_band_gains(stft(clean), stft(noisy)), len(noisy))

    # ---------------------------------------------------------- evaluation
    rows, pending = [], []
    example_utts = set(sorted(manifest["utt"].unique())[:C.N_AUDIO_EXAMPLES])
    example_snr = min(C.TEST_SNRS, key=lambda v: abs(v - C.EXAMPLE_SNR))
    pool = ProcessPoolExecutor(max_workers=C.N_JOBS) if C.N_JOBS > 1 else None

    def collect(meta, scores):
        for name, m in scores.items():
            rows.append(dict(**meta, method=name, **m))

    todo = manifest if old is None else manifest[~manifest.index.isin(set(old["mixture"]))]
    for idx, row in tqdm(todo.iterrows(), total=len(todo), desc="evaluating"):
        clean, noisy = make_test_mixture(row)
        outputs = {name: fn(clean, noisy) for name, fn in methods.items()}
        if row["utt"] in example_utts and row["snr"] == example_snr:
            d = os.path.join(C.RESULTS_DIR, "audio", f"utt{row['utt']:02d}_{row['noise_type']}_{row['snr']}dB")
            os.makedirs(d, exist_ok=True)
            sf.write(os.path.join(d, "clean.wav"), clean, C.SAMPLE_RATE)
            for name, est in outputs.items():
                sf.write(os.path.join(d, name.split(" (")[0].replace(" ", "_") + ".wav"), est, C.SAMPLE_RATE)
        meta = dict(mixture=idx, utt=row["utt"], noise_type=row["noise_type"], snr=row["snr"], seen=row["seen"])
        if pool is None:
            collect(meta, compute_metrics_many(clean, outputs))
        else:
            pending.append((meta, pool.submit(compute_metrics_many, clean, outputs)))
            while len(pending) > 4 * C.N_JOBS:           # keep memory bounded
                collect(pending[0][0], pending.pop(0)[1].result())
    for meta, fut in pending:
        collect(meta, fut.result())
    if pool is not None:
        pool.shutdown()
    df = pd.DataFrame(rows)
    if old is not None:
        df = pd.concat([old, df], ignore_index=True).sort_values(["mixture"], kind="stable")
    stat_path = os.path.join(C.WORK_DIR, "noise_stationarity.csv")
    stat = pd.read_csv(stat_path) if os.path.exists(stat_path) else None
    if stat is not None:
        df["category"] = df["noise_type"].map(dict(zip(stat["noise_type"], stat["category"]))).fillna("unknown")
    df.to_csv(raw_path, index=False)

    # -------------------------------------------------------------- tables
    order = list(methods)
    md = ["# Results\n", f"Test set: {manifest['utt'].nunique()} unseen utterances x "
          f"{manifest['noise_type'].nunique()} noise types x SNR {sorted(manifest['snr'].unique())} dB "
          f"= {len(manifest)} mixtures.\n"]

    def emit(title, table, fname, digits=3):
        table.round(digits).to_csv(os.path.join(C.RESULTS_DIR, fname))
        md.append(f"\n## {title}\n\n" + table.round(digits).to_markdown() + "\n")

    if stat is not None:
        emit("Noise types and stationarity class (non-stationarity index, dB)",
             stat.set_index("noise_type"), "table_noise_stationarity.csv", digits=2)

    overall = df.groupby("method")[METRICS].agg(["mean", "std", "count"]).reindex(order)
    for m in METRICS:
        overall[(m, "ci95")] = 1.96 * overall[(m, "std")] / np.sqrt(overall[(m, "count")])
    overall = overall.drop(columns=[(m, "count") for m in METRICS])
    overall.columns = [f"{m}_{s}" for m, s in overall.columns]
    for m in METRICS:
        overall[f"{m}_gain_vs_noisy"] = overall[f"{m}_mean"] - overall.loc[NOISY, f"{m}_mean"]
    overall = overall[[f"{m}_{s}" for m in METRICS for s in ("mean", "std", "ci95", "gain_vs_noisy")]]
    emit("Overall (mean, std, 95% confidence interval half-width, improvement over noisy input)",
         overall, "table_overall.csv")

    if stat is not None:
        cats = [c for c in ("stationary", "quasi-stationary", "non-stationary") if c in set(df["category"])]
        for m in METRICS:
            t = df.pivot_table(index="method", columns="category", values=m, aggfunc="mean").reindex(order)
            emit(f"{m.upper()} by noise stationarity class", t[cats], f"table_{m}_by_category.csv")
        g = df.pivot_table(index="method", columns="category", values=["pesq", "stoi", "si_sdr"], aggfunc="mean")
        g = g - g.loc[NOISY]
        g = g.reindex(order).drop(index=[NOISY])
        g = g[[(m, c) for m in ("pesq", "stoi", "si_sdr") for c in cats]]
        g.columns = [f"{m}_gain_{c}" for m, c in g.columns]
        emit("Improvement over the noisy input by stationarity class", g, "table_gain_by_category.csv")

    for m in METRICS:
        t = df.pivot_table(index="method", columns="snr", values=m, aggfunc="mean").reindex(order)
        t["mean"] = df.groupby("method")[m].mean()
        emit(f"{m.upper()} by input SNR (dB)", t, f"table_{m}_by_snr.csv")
        t = df.pivot_table(index="method", columns="noise_type", values=m, aggfunc="mean").reindex(order)
        emit(f"{m.upper()} by noise type", t, f"table_{m}_by_noise.csv")

    if (~df["seen"].astype(bool)).any():
        t = df.assign(noise=np.where(df["seen"].astype(bool), "seen", "unseen")).pivot_table(
            index="method", columns="noise", values=METRICS, aggfunc="mean").reindex(order)
        t.columns = [f"{m}_{s}" for m, s in t.columns]
        emit("Seen vs unseen noise types", t, "table_seen_unseen.csv")

    # paired significance test: proposed vs each other method on the same mixtures
    if PROPOSED in methods:
        from scipy.stats import wilcoxon
        wide = {m: df.pivot(index="mixture", columns="method", values=m) for m in METRICS}
        # conservative version: mixtures of one (utterance, noise type) share speech and noise, so they
        # are averaged over SNR first -> one independent-ish value per condition
        cond = {m: df.pivot_table(index=["utt", "noise_type"], columns="method", values=m, aggfunc="mean")
                for m in METRICS}

        def test(a, b):
            ok = a.notna() & b.notna()
            diff = (a - b)[ok]
            p = wilcoxon(diff).pvalue if np.any(diff != 0) and len(diff) >= 10 else np.nan
            return diff, int(ok.sum()), p

        sig = []
        for other in order:
            if other in (PROPOSED,):
                continue
            for m in METRICS:
                diff, n, p = test(wide[m][PROPOSED], wide[m][other])
                _, n_c, p_c = test(cond[m][PROPOSED], cond[m][other])
                sig.append(dict(compared_with=other, metric=m, n=n,
                                mean_difference=round(float(diff.mean()), 4), p_value=p,
                                significant_at_0_05=bool(p < 0.05) if p == p else None,
                                n_conditions=n_c, p_value_conditions=p_c))
        sig = pd.DataFrame(sig)
        # Holm-Bonferroni correction over all comparisons (condition-level p-values)
        pc = sig["p_value_conditions"].fillna(1.0).to_numpy()
        order_p = np.argsort(pc)
        adj = np.empty_like(pc)
        running = 0.0
        for rank, i in enumerate(order_p):
            running = max(running, min(1.0, (len(pc) - rank) * pc[i]))
            adj[i] = running
        sig["p_holm_conditions"] = adj
        sig["significant_holm_0_05"] = adj < 0.05
        sig.to_csv(os.path.join(C.RESULTS_DIR, "significance.csv"), index=False)
        md.append("\n## Paired Wilcoxon signed-rank test: proposed model minus other method\n\n"
                  + sig.to_markdown(index=False) + "\n")

    if complexity:
        cx = pd.DataFrame(complexity)
        cx.to_csv(os.path.join(C.RESULTS_DIR, "complexity.csv"), index=False)
        md.append("\n## Complexity (this machine, CPU)\n\n" + cx.to_markdown(index=False)
                  + "\n\nreal_time_factor = processing time / audio duration (below 1 = faster than real time).\n")

    with open(os.path.join(C.RESULTS_DIR, "tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n" + overall[[f"{m}_mean" for m in METRICS]].round(3).to_string())
    print(f"\nall tables -> {C.RESULTS_DIR}/tables.md")


if __name__ == "__main__":
    main()
