"""
Shadow-test figure: one episode, all decision points.

Top panel:  per-step min-max band of the 10 candidate energies (band width IS the spread)
            + the executed candidate (always #0 in shadow mode) as a line.
            The 10 candidates are fresh samples each step, so they are drawn as a band,
            not as 10 series (they have no identity across steps).
Bottom panel: relative spread (max-min)/mean per step vs the 2% gate.
Both panels: vertical markers where each subtask completed.

Also prints the per-completion markdown table (spread / best_idx / executed rank / gated
at the decision step that produced each completion).

Run in the openpi venv on the eval box:
  cd /workspace/polaris/third_party/openpi
  uv run python /workspace/polaris/experiments/jepa_verifier/plot_shadow_episode.py \
      --log /workspace/polaris/runs/armA_energies.jsonl --ic 9 --episode 4 \
      --out /workspace/polaris/experiments/jepa_verifier/figs/shadow_ic9_ep4
"""
import argparse
import json
import statistics as st
from pathlib import Path

NAMES = ["reach_ice_cream", "reach_grapes", "lift_ice_cream",
         "lift_grapes", "inside_ice_cream_bowl", "inside_grapes_bowl"]
SHORT = ["reach ice", "reach grapes", "lift ice", "lift grapes", "ice in bowl", "grapes in bowl"]

INK = "#374151"        # primary text
MUTED = "#6b7280"      # secondary
GRID = "#e5e7eb"
BAND = "#93c5fd"       # light blue fill
LINE = "#1d4ed8"       # executed-candidate line
MARKER = "#9ca3af"     # completion verticals


def episodes_of_ic(log_path, ic):
    rs = [r for r in (json.loads(l) for l in open(log_path)) if r["ic"] == ic]
    eps, cur, prev = [], [], -1
    for r in rs:
        nd = sum(r["done"])
        if cur and (nd < prev or (prev > 0 and nd == 0)):
            eps.append(cur)
            cur = []
        cur.append(r)
        prev = nd
    if cur:
        eps.append(cur)
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--ic", type=int, required=True)
    ap.add_argument("--episode", type=int, required=True, help="episode ordinal within this IC")
    ap.add_argument("--gate", type=float, default=0.02)
    ap.add_argument("--out", default="shadow_episode")
    args = ap.parse_args()

    ep = episodes_of_ic(args.log, args.ic)[args.episode]
    steps = list(range(len(ep)))
    emin = [min(r["energies"]) for r in ep]
    emax = [max(r["energies"]) for r in ep]
    e0 = [r["energies"][0] for r in ep]
    spread = [r["spread"] * 100 for r in ep]

    comp = {}
    for i, r in enumerate(ep):
        for j in range(6):
            if r["done"][j] and j not in comp:
                comp[j] = i

    # ---- markdown table ----
    print(f"IC {args.ic}, episode {args.episode}: {len(ep)} decision steps\n")
    print("| subtask | completed at step | spread at deciding step | best_idx | executed(0) rank | gated? |")
    print("|---|---|---|---|---|---|")
    for j, t in sorted(comp.items(), key=lambda kv: kv[1]):
        d = ep[max(0, t - 1)]
        rank0 = sorted(range(10), key=lambda q: d["energies"][q]).index(0) + 1
        print(f"| {NAMES[j]} | {t} | {d['spread'] * 100:.2f}% | {d['best_idx']} "
              f"| {rank0}/10 | {'yes' if d['gated'] else 'no'} |")
    spr = [r["spread"] for r in ep]
    print(f"\nsummary: median spread {st.median(spr) * 100:.2f}%, max {max(spr) * 100:.2f}%, "
          f"steps over {args.gate:.0%} gate: {sum(s > args.gate for s in spr)}/{len(spr)}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 12, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    })
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.12},
    )

    ax.fill_between(steps, emin, emax, color=BAND, alpha=0.55, linewidth=0,
                    label="range of the 10 candidates (min-max)")
    ax.plot(steps, e0, color=LINE, lw=2, label="executed candidate (always #0 in shadow)")
    ax.set_ylabel("V-JEPA energy to active subtask goal")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="upper right", fontsize=10)

    ax2.plot(steps, spread, color=LINE, lw=2)
    ax2.axhline(args.gate * 100, color=MUTED, ls="--", lw=1.5)
    ax2.text(len(ep) - 1, args.gate * 100, f"  {args.gate:.0%} gate", color=MUTED,
             va="bottom", ha="right", fontsize=10)
    ax2.set_ylabel("candidate spread\n(max−min)/mean  [%]")
    ax2.set_xlabel("decision step  (1 step = one 16-action chunk ≈ 0.53 s)")
    ax2.set_ylim(0, max(max(spread) * 1.25, args.gate * 100 * 1.6))
    ax2.grid(axis="y", color=GRID, lw=0.8)

    ytop = max(emax)
    for j, t in sorted(comp.items(), key=lambda kv: kv[1]):
        for a in (ax, ax2):
            a.axvline(t, color=MARKER, ls=":", lw=1.2, zorder=0)
        ax.text(t, ytop, " " + SHORT[j], rotation=90, va="top", ha="right",
                fontsize=9, color=INK)

    fig.suptitle(
        f"Shadow test, IC {args.ic} episode {args.episode} (successful): "
        "the 10 candidates are indistinguishable at every decision point,\n"
        "including the six decisions that actually completed a subtask (dotted lines)",
        fontsize=12.5,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.out}.png", dpi=160, bbox_inches="tight")
    print(f"\nsaved {args.out}.png")


if __name__ == "__main__":
    main()
