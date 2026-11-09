"""
STEP 1b (optional) - add new noise types to the TEST set only, after the models are trained.

Every noise type found in datasets/noise that is not yet in work/test_manifest.csv is added as an
UNSEEN type: same test utterances, same SNRs, noise taken from the test part of the recording.
Its stationarity index is measured and appended to work/noise_stationarity.csv.
Training data and models are not touched.

Typical use: DEMAND environments that finished downloading after training started.
    python step0_get_data.py --skip-vctk --demand-envs TCAR STRAFFIC
    python step1b_add_test_noise.py
    python step3_evaluate.py --resume      (evaluates only the new mixtures)
    python step4_plots.py
"""
import _env  # noqa: F401  (Run button: uses the project environment automatically)
import json
import os

import numpy as np
import pandas as pd
import soundfile as sf

import config as C
from dataset import scan_noise, pick_noise
from step1_prepare_data import measure_stationarity


def main():
    man_path = os.path.join(C.WORK_DIR, "test_manifest.csv")
    stat_path = os.path.join(C.WORK_DIR, "noise_stationarity.csv")
    manifest = pd.read_csv(man_path)
    noise_table = scan_noise()
    new = [t for t in noise_table if t not in set(manifest["noise_type"])]
    if not new:
        print("no new noise types - nothing to do")
        return
    rng = np.random.default_rng(C.SEED + 17)
    utts = manifest.groupby("utt")["clean"].first()
    rows = []
    for t in new:
        for u, cf in utts.items():
            dur = min(sf.info(cf).duration, C.MAX_UTT_SECONDS)
            path, off, rd = pick_noise(noise_table, t, "test", dur, rng)
            for snr in sorted(manifest["snr"].unique()):
                rows.append(dict(utt=u, clean=cf, noise_type=t, noise_file=path, noise_offset=round(off, 4),
                                 noise_read=round(rd, 4), snr=snr, seen=False))
    manifest = pd.concat([manifest, pd.DataFrame(rows)], ignore_index=True)
    manifest.to_csv(man_path, index=False)

    stat_new = measure_stationarity({t: noise_table[t] for t in new}, new)
    stat = pd.concat([pd.read_csv(stat_path), stat_new]) if os.path.exists(stat_path) else stat_new
    stat.sort_values("nsi_db").to_csv(stat_path, index=False)

    sp_path = os.path.join(C.WORK_DIR, "splits.json")
    if os.path.exists(sp_path):
        with open(sp_path, encoding="utf-8") as f:
            sp = json.load(f)
        sp["unseen_noise"] = sorted(set(sp.get("unseen_noise", [])) | set(new))
        with open(sp_path, "w", encoding="utf-8") as f:
            json.dump(sp, f, indent=1)
    with open(os.path.join(C.WORK_DIR, "data_report.txt"), "a", encoding="utf-8") as f:
        f.write("\nAdded after training as UNSEEN test-only noise types (step1b):\n")
        for _, r in stat_new.iterrows():
            f.write(f"  {r.noise_type:22s} NSI {r.nsi_db:6.2f} dB   {r.category:17s} UNSEEN\n")
    print(stat_new.to_string(index=False))
    print(f"added {len(rows)} test mixtures for {new}; run: python step3_evaluate.py --resume")


if __name__ == "__main__":
    main()
