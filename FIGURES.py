"""
FIGURES.py - press Run (play button) in VS Code: draws all thesis figures from the saved results
and shows them one by one in a window.

    Right arrow / Space : next figure        Left arrow : previous figure
    Close the window    : finish

The figures are also saved as image files in results/figures (ready for the thesis).
Uses the same Python environment as RUN.py (installed automatically on the first run).
"""
import os
import subprocess
import sys
from pathlib import Path

from RUN import DONE, VPY, in_project_env, setup

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "results" / "figures"
ORDER = ["fig_training_curves", "fig_pesq_vs_snr", "fig_stoi_vs_snr", "fig_estoi_vs_snr", "fig_si_sdr_vs_snr",
         "fig_improvement_by_category", "fig_improvement_by_noise", "fig_pesq_vs_snr_by_category",
         "fig_stoi_vs_snr_by_category", "fig_pesq_by_noise", "fig_stoi_by_noise", "fig_spectrograms"]


THESIS_FIG = {"fig_training_curves": "4.1", "fig_pesq_vs_snr": "4.2", "fig_stoi_vs_snr": "4.3",
              "fig_improvement_by_category": "4.4", "fig_improvement_by_noise": "4.5",
              "fig_spectrograms_utt00_babble_4talkers_5dB": "4.6"}


def figure_files():
    files = [p for p in FIG_DIR.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    rank = lambda p: next((i for i, k in enumerate(ORDER) if p.stem.startswith(k)), len(ORDER))
    return sorted(files, key=lambda p: (rank(p), p.name))


def viewer(files):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.widgets import Button

    for k in ("keymap.back", "keymap.forward", "keymap.home", "keymap.save", "keymap.fullscreen"):
        plt.rcParams[k] = []          # free the arrow keys for this viewer
    state = {"i": 0}
    fig = plt.figure(figsize=(14, 8))
    try:
        fig.canvas.manager.set_window_title("Hybrid CNN-GRU - thesis figures")
    except Exception:
        pass

    ax = fig.add_axes([0.01, 0.09, 0.98, 0.85])
    b_prev = Button(fig.add_axes([0.30, 0.015, 0.16, 0.06]), "<  Previous")
    b_next = Button(fig.add_axes([0.54, 0.015, 0.16, 0.06]), "Next  >")

    def draw():
        ax.clear()
        ax.imshow(mpimg.imread(files[state["i"]]))
        ax.axis("off")
        stem = files[state["i"]].stem
        tag = f"  =  Thesis Fig. {THESIS_FIG[stem]}" if stem in THESIS_FIG else "  (additional figure)"
        fig.suptitle(f"Figure {state['i'] + 1} of {len(files)}:  {stem}{tag}      "
                     "(click Next / Previous, or use the arrow keys)", fontsize=11)
        fig.canvas.draw_idle()

    def step(d):
        state["i"] = (state["i"] + d) % len(files)
        draw()

    def key(event):
        if event.key in ("right", " ", "down", "pagedown", "enter", "n"):
            step(1)
        elif event.key in ("left", "up", "pageup", "backspace", "p"):
            step(-1)

    b_next.on_clicked(lambda e: step(1))
    b_prev.on_clicked(lambda e: step(-1))
    fig.canvas.mpl_connect("key_press_event", key)
    draw()
    plt.show()


def main():
    os.chdir(ROOT)
    if "--view-only" not in sys.argv:
        print("Drawing the thesis figures from the saved results (about 1 minute) ...", flush=True)
        env = {**os.environ, "SE_NO_VIEWER": "1"}
        if subprocess.call([sys.executable, "step4_plots.py"], cwd=ROOT, env=env) != 0:
            print("Could not redraw the figures; showing the saved ones instead.")
    files = figure_files()
    if not files:
        sys.exit(f"No figures found in {FIG_DIR}")
    print(f"\n{len(files)} figures in {FIG_DIR}")
    print("A window opens now: right arrow / space = next, left arrow = previous, close the window to finish.")
    viewer(files)


if __name__ == "__main__":
    if os.environ.get("RUN_NO_BOOTSTRAP") == "1" or in_project_env():
        main()
    else:
        if not DONE.exists():
            setup()
        sys.exit(subprocess.call([str(VPY), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=ROOT))
