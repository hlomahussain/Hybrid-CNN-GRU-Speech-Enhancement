"""
RUN.py - one-click entry point (open this file in VS Code and press Run / the play button).

First run: creates a private Python environment and installs every library automatically
(about 0.5-1 GB download, 10-40 minutes depending on the connection). Later runs start at once.
Then it prints all the Chapter 4 results tables (RESULTS.py) and runs one live enhancement automatically.
`python RUN.py --menu` opens a menu with more options (other noise types and SNRs, your own recording).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow start-up messages
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

ROOT = Path(__file__).resolve().parent
VENV = Path.home() / ".venvs" / "hybrid_cnn_gru"
VPY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
DONE = VENV / ".setup_done"


# ------------------------------------------------------------------ setup
def run(cmd, **kw):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd], **kw)


def setup():
    v = sys.version_info
    if not ((3, 10) <= (v.major, v.minor) <= (3, 13)):
        sys.exit(f"Python {v.major}.{v.minor} is not supported by TensorFlow. Install Python 3.11 from python.org "
                 "(tick 'Add python.exe to PATH'), select it in VS Code (bottom-right corner) and press Run again.")
    print("=" * 70)
    print(" First run: installing the libraries (TensorFlow, librosa, ...).")
    print(" This needs an internet connection and takes 10-40 minutes. Please wait.")
    print("=" * 70, flush=True)
    if not VPY.exists():
        if run([sys.executable, "-m", "venv", VENV]) != 0:
            sys.exit("Could not create the Python environment.")
    run([VPY, "-m", "pip", "install", "--upgrade", "pip"])
    if run([VPY, "-m", "pip", "install", "-r", ROOT / "requirements.txt"]) != 0:
        sys.exit("\nInstallation failed (see the red lines above). Check the internet connection and press Run again:"
                 " the download continues where it stopped.")
    if run([VPY, "-m", "pip", "install", "pesq"]) != 0:
        print("\nNOTE: the 'pesq' metric could not be installed (on Windows it needs 'Microsoft C++ Build Tools').")
        print("Everything works without it; PESQ is shown as n/a in new measurements.")
        print("The thesis results in the results folder already contain PESQ.\n")
    vs = ROOT / ".vscode"
    vs.mkdir(exist_ok=True)
    settings = {"python.defaultInterpreterPath": str(VPY), "python.terminal.activateEnvironment": True}
    (vs / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    DONE.write_text("ok", encoding="utf-8")
    print("\nSetup finished.\n", flush=True)


def in_project_env():
    try:
        return VPY.exists() and Path(sys.executable).resolve() == VPY.resolve()
    except OSError:
        return False


# ------------------------------------------------------------------- menu
def py(*args, env=None):
    return subprocess.call([sys.executable, *args], cwd=ROOT, env=env)


def open_folder(path):
    path = ROOT / path
    print(f"Folder: {path}")
    if os.name == "nt" and path.exists():
        os.startfile(path)


def ask(prompt, default=""):
    v = input(f"{prompt} [{default}]: ").strip().strip('"').strip("'")
    return v or default


def noise_types():
    d = ROOT / "datasets" / "noise"
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


def show_results():
    """All results tables of Chapter 4 (Tables 4.1-4.11), printed by RESULTS.py from the saved result files."""
    if py("RESULTS.py") == 0:
        return
    import pandas as pd
    pd.set_option("display.width", 160)
    t = pd.read_csv(ROOT / "results" / "table_overall.csv", index_col=0)
    cols = [c for c in ("pesq_mean", "stoi_mean", "estoi_mean", "si_sdr_mean") if c in t]
    print("\nThesis result (Table 4.5): mean over the 1,600 test mixtures\n")
    print(t[cols].rename(columns=lambda c: c.replace("_mean", "").upper()).round(3).to_string())
    print("\nAll result tables: results\\tables.md")


CLEAN_FILES = {"1": ("male speaker", ROOT / "datasets" / "clean" / "r02673" / "r02673_010.wav"),
               "2": ("female speaker", ROOT / "datasets" / "clean" / "r05490" / "r05490_010.wav")}
EXAMPLE = ROOT / "examples" / "noisy_example_babble_5dB.wav"


def menu():
    os.chdir(ROOT)
    while True:
        print("""
==================  Hybrid CNN-GRU speech enhancement  ==================
 1  Quick demo: keyboard typing noise at 5 dB SNR (about 1 minute)
 2  Demo with another speaker / noise type / SNR (16 noise types)
 3  Enhance a noisy recording (.wav) - the included example or your own
 4  Show all the thesis results tables (Tables 4.1-4.11)
 5  Draw the thesis figures from the results and show them (about 1 minute)
 0  Exit
=========================================================================""")
        c = input("Choose: ").strip()
        if c == "1":
            py("demo_enhance.py", "--clean", CLEAN_FILES["1"][1], "--noise", "typing", "--snr", "5")
            open_folder("demo_output")
        elif c == "2":
            for k, (name, _) in CLEAN_FILES.items():
                print(f"  {k}  {name}")
            spk = CLEAN_FILES.get(ask("Speaker number", "1"), CLEAN_FILES["1"])[1]
            types = noise_types()
            for i, t in enumerate(types, 1):
                print(f"  {i:2d}  {t}")
            k = ask("Noise number", "1")
            nt = types[int(k) - 1] if k.isdigit() and 0 < int(k) <= len(types) else k
            snr = ask("SNR in dB (0, 5, 10, 15, 20)", "5")
            py("demo_enhance.py", "--clean", spk, "--noise", nt, "--snr", snr)
            open_folder("demo_output")
        elif c == "3":
            f = ask("Full path of the .wav file (Enter = included example)", str(EXAMPLE))
            if f and Path(f).exists():
                for m in ("hybrid", "cnn_only", "gru_only"):
                    py("demo_enhance.py", f, "--model", m)
                open_folder("demo_output")
            else:
                print("File not found.")
        elif c == "4":
            show_results()
        elif c == "5":
            py("FIGURES.py")
        elif c == "0":
            break


def auto():
    """Default when Run is pressed: thesis results table + one live enhancement, no input needed."""
    os.chdir(ROOT)
    show_results()
    print("\n" + "=" * 73)
    print(" Live test: a sentence from an unseen speaker + keyboard typing noise at 5 dB,")
    print(" enhanced by the three trained networks (about 1 minute)")
    print("=" * 73, flush=True)
    py("demo_enhance.py", "--clean", CLEAN_FILES["1"][1], "--noise", "typing", "--snr", "5")
    open_folder("demo_output")
    print("\nListen to demo_output\\noisy.wav and demo_output\\hybrid.wav to hear the difference.")
    print("Thesis figures (drawn from the results):  python FIGURES.py")
    print("More options (other noise types, SNRs, your own recording):  python RUN.py --menu")


if __name__ == "__main__":
    if os.environ.get("RUN_NO_BOOTSTRAP") == "1" or in_project_env():
        menu() if "--menu" in sys.argv else auto()
    else:
        if not DONE.exists():
            setup()
        sys.exit(subprocess.call([str(VPY), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=ROOT))
