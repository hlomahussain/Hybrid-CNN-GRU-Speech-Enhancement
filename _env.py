"""
_env.py - makes every script work when you press Run in VS Code, whatever Python is selected.

Imported at the top of each script. If the script was started with a Python that is not the
project environment (~/.venvs/hybrid_cnn_gru), it installs the environment once (same as RUN.py)
and restarts the script inside it. It also moves to the project folder so relative paths work.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _has(pattern):
    return any(ROOT.glob(pattern))


# Scripts that rebuild what is already finished: pressing Run on them by mistake must not
# re-download the data, retrain for hours or overwrite the thesis results.
GUARDS = {
    "step0_get_data.py": ("datasets/clean/*/*", "the datasets are already in the 'datasets' folder"),
    "step0_get_data_dns.py": ("datasets/clean/*/*", "the datasets are already in the 'datasets' folder"),
    "step1_prepare_data.py": ("work/*", "the prepared data is already in the 'work' folder"),
    "step1b_add_test_noise.py": ("work/*", "the test set is already in the 'work' folder"),
    "step2_train.py": ("models/*.keras", "the three trained models are already in the 'models' folder"),
    "step3_evaluate.py": (os.path.join(os.environ.get("SE_RESULTS_DIR", "results"), "results_raw.csv"),
                          "the thesis results are already in the 'results' folder"),
}


def guard():
    name = Path(sys.argv[0]).name
    if "--force" in sys.argv:
        sys.argv.remove("--force")
        return
    if name in GUARDS and "--resume" not in sys.argv and _has(GUARDS[name][0]):
        print("=" * 72)
        print(f" {name}: nothing to do - {GUARDS[name][1]}.")
        print(" This step was already completed for the thesis; running it again would redo hours of work.")
        print(" The file can still be opened and read: it is the exact code that produced the thesis results.")
        print(" To see the results press Run on  RUN.py.")
        print(f" (To really run it again: python {name} --force)")
        print("=" * 72)
        sys.exit(0)


def ensure():
    os.chdir(ROOT)
    guard()
    if os.environ.get("RUN_NO_BOOTSTRAP") == "1":
        return
    from RUN import DONE, VPY, in_project_env, setup
    if in_project_env():
        return
    if not DONE.exists():
        setup()
    script = Path(sys.argv[0]).resolve()
    sys.exit(subprocess.call([str(VPY), str(script), *sys.argv[1:]], cwd=ROOT))


if __name__ != "__main__":
    ensure()
