"""
Action-contrast test: is the predictor blind to actions, or were pi0.5's candidates
just too similar?

Part B (candidate diversity, from the action log alone):
  - How different are pi0.5's 10 sampled candidates, in commanded EE displacement?
  - Per step: Spearman correlation between pairwise action distance (|net displacement
    difference|) and pairwise energy difference. If the predictor responds to actions,
    more-different actions must get more-different energies.

Part A (predictor sensitivity, one state):
  Encode one real frame; from the SAME pose, roll the predictor on:
    - the 10 REAL logged pi0.5 candidates,
    - SYNTHETIC max-contrast actions (zero, +/-x, +/-y, +/-z at 0.05 m/step, +/-yaw),
  and compare pairwise predicted-future L1 against the real frame-to-frame L1 in the
  same video (the scale that real 0.5-1 s world change produces).

Interpretation:
  synthetic contrast ~ candidate level  ->  predictor blind to actions (loophole closed)
  synthetic contrast >> candidate level ->  candidate diversity was the bottleneck

Prereqs: rerun 1-2 episodes with the server started with VJEPA_LOG_ACTIONS=1 (and a fresh
VJEPA_ENERGY_LOG); pass that log + the episode video here.

Run in the openpi venv on the eval box:
  cd /workspace/polaris/third_party/openpi
  uv run python /workspace/polaris/experiments/jepa_verifier/action_contrast.py \
      --log /workspace/polaris/runs/action_log.jsonl \
      --video /workspace/polaris/runs/pi05/action_probe/episode_0.mp4 \
      --ckpt /workspace/polaris/third_party/vjepa2/checkpoints/vjepa2-ac-vitg.pt \
      --episode 0 --step 10 \
      --out /workspace/polaris/experiments/jepa_verifier/figs/action_contrast
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
import torchvision.transforms as T

from vjepa2 import rollout  # also provides compute_new_pose via its module namespace


# ---------- shared helpers (same preprocessing as the policy) ----------

def build_transform():
    return T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def encode_frame(encoder, transform, frame_rgb, tokens, device):
    t = transform(frame_rgb)
    clip = np.expand_dims(np.stack([t, t], axis=0), axis=0)
    tensor = torch.from_numpy(clip).float().permute(0, 2, 1, 3, 4).to(device)
    with torch.inference_mode():
        h = encoder(tensor)[:, -tokens:, :]
        return F.layer_norm(h, (h.size(-1),))


def read_base_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(np.ascontiguousarray(rgb[:, : rgb.shape[1] // 2, :]))
    cap.release()
    return frames


def episodes_with_actions(log_path):
    recs = [json.loads(l) for l in open(log_path) if "cand_ee_actions" in l]
    eps, cur, prev = [], [], -1
    for r in recs:
        nd = sum(r["done"])
        if cur and (nd < prev or (prev > 0 and nd == 0)):
            eps.append(cur)
            cur = []
        cur.append(r)
        prev = nd
    if cur:
        eps.append(cur)
    return eps


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def pairwise_future_l1(z_hats):
    """z_hats: (S, tokens, D) -> (S, S) matrix of mean-abs distances."""
    S = z_hats.shape[0]
    m = np.zeros((S, S))
    for i in range(S):
        for j in range(i + 1, S):
            m[i, j] = m[j, i] = F.l1_loss(z_hats[i], z_hats[j]).item()
    return m


def propagate_states(pose0, actions):
    """pose0 (7,), actions (S,T,7) torch -> states (S,T+1,7) via compute_new_pose."""
    S, Tn, _ = actions.shape
    states = [pose0.view(1, 1, 7).repeat(S, 1, 1)]
    for t in range(Tn):
        nxt = rollout.compute_new_pose(states[-1][:, -1:], actions[:, t:t + 1])
        states.append(nxt)
    return torch.cat(states, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="jsonl written with VJEPA_LOG_ACTIONS=1")
    ap.add_argument("--video", required=True, help="episode video matching that log")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episode", type=int, default=0, help="episode ordinal within the log")
    ap.add_argument("--step", type=int, default=10, help="decision step for Part A")
    ap.add_argument("--out", default="action_contrast")
    args = ap.parse_args()

    eps = episodes_with_actions(args.log)
    ep = eps[args.episode]
    print(f"log: {len(eps)} episodes with actions; using episode {args.episode} ({len(ep)} steps)\n")

    # ================= Part B: candidate diversity & action-energy correlation ===========
    disp_pairs_all, corr_per_step = [], []
    net_disp_all = []
    for r in ep:
        A = np.asarray(r["cand_ee_actions"])            # (10, 4, 7)
        D = A[:, :, :3].sum(axis=1)                     # net commanded displacement (10, 3) [m]
        net_disp_all.extend(np.linalg.norm(D, axis=1))
        E = np.asarray(r["energies"])
        d_act, d_e = [], []
        for i in range(10):
            for j in range(i + 1, 10):
                d_act.append(np.linalg.norm(D[i] - D[j]))
                d_e.append(abs(E[i] - E[j]))
        disp_pairs_all.extend(d_act)
        corr_per_step.append(spearman(np.array(d_act), np.array(d_e)))

    disp_pairs_all = np.array(disp_pairs_all) * 100     # cm
    corr = np.array([c for c in corr_per_step if not np.isnan(c)])
    print("== Part B: pi0.5 candidate diversity (net EE displacement over the ~1.07 s window) ==")
    print(f"  per-candidate net displacement:  median {np.median(net_disp_all)*100:.2f} cm,  "
          f"p90 {np.percentile(net_disp_all, 90)*100:.2f} cm")
    print(f"  pairwise displacement DIFFERENCE between candidates: "
          f"median {np.median(disp_pairs_all):.2f} cm,  p90 {np.percentile(disp_pairs_all, 90):.2f} cm")
    print(f"  Spearman(action distance, energy distance) per step: "
          f"median {np.median(corr):+.3f},  IQR [{np.percentile(corr,25):+.3f}, {np.percentile(corr,75):+.3f}]"
          f"  (n={len(corr)} steps; ~0 = energies ignore how different the actions are)")

    # ================= Part A: predictor sensitivity at one state ========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant", pretrained=False)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    strip = lambda sd: {k.replace("module.", "", 1): v for k, v in sd.items()}
    encoder.load_state_dict(strip(ckpt["encoder"]))
    predictor.load_state_dict(strip(ckpt["predictor"]))
    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()
    tokens = int((256 // encoder.patch_size) ** 2)
    transform = build_transform()

    frames = read_base_frames(args.video)
    s = min(args.step, len(frames) - 3, len(ep) - 1)
    z = encode_frame(encoder, transform, frames[s], tokens, device)
    z_next1 = encode_frame(encoder, transform, frames[s + 1], tokens, device)
    z_next2 = encode_frame(encoder, transform, frames[s + 2], tokens, device)
    real_1 = F.l1_loss(z[0], z_next1[0]).item()
    real_2 = F.l1_loss(z[0], z_next2[0]).item()

    rec = ep[s]
    pose0 = torch.tensor(rec["ee_pose0"], dtype=torch.float32, device=device)

    # real logged candidates at this step
    cand_actions = torch.tensor(rec["cand_ee_actions"], dtype=torch.float32, device=device)  # (10,4,7)
    cand_states = propagate_states(pose0, cand_actions)
    with torch.inference_mode():
        z_cand = rollout.forward_actions(z, predictor, cand_states, cand_actions)
    m_cand = pairwise_future_l1(z_cand)
    cand_pairs = m_cand[np.triu_indices(10, k=1)]

    # synthetic max-contrast actions: 4 steps of 0.05 m (official per-step clip)
    def seq(dx=0.0, dy=0.0, dz=0.0, dyaw=0.0):
        a = torch.zeros(1, 4, 7, device=device)
        a[0, :, 0], a[0, :, 1], a[0, :, 2], a[0, :, 5] = dx, dy, dz, dyaw
        return a

    synth = {
        "zero": seq(),
        "+x": seq(dx=0.05), "-x": seq(dx=-0.05),
        "+y": seq(dy=0.05), "-y": seq(dy=-0.05),
        "+z": seq(dz=0.05), "-z": seq(dz=-0.05),
        "+yaw": seq(dyaw=0.15), "-yaw": seq(dyaw=-0.15),
    }
    labels = list(synth)
    syn_actions = torch.cat([synth[k] for k in labels], dim=0)     # (S,4,7)
    syn_states = propagate_states(pose0, syn_actions)
    with torch.inference_mode():
        z_syn = rollout.forward_actions(z, predictor, syn_states, syn_actions)
    m_syn = pairwise_future_l1(z_syn)

    iz = labels.index("zero")
    opp = [("+x", "-x"), ("+y", "-y"), ("+z", "-z"), ("+yaw", "-yaw")]
    opp_vals = [m_syn[labels.index(a), labels.index(b)] for a, b in opp]
    zero_vs = {k: m_syn[iz, labels.index(k)] for k in labels if k != "zero"}
    z0 = z_syn[iz:iz + 1]
    zero_drift = F.l1_loss(z0[0], z[0, :tokens]).item()

    print(f"\n== Part A: predictor sensitivity at step {s} (all numbers = latent L1, same units) ==")
    print(f"  REAL world change,  1 step  (~0.53 s): {real_1:.4f}")
    print(f"  REAL world change,  2 steps (~1.07 s): {real_2:.4f}   <- scale a working predictor should reach")
    print(f"  predicted futures, 10 REAL pi0.5 candidates: median pair {np.median(cand_pairs):.4f}, "
          f"max pair {cand_pairs.max():.4f}")
    print(f"  predicted futures, OPPOSITE max-magnitude actions (40 cm apart): "
          f"{', '.join(f'{a}vs{b}={v:.4f}' for (a, b), v in zip(opp, opp_vals))}")
    print(f"  zero-action future vs current frame (does it 'advance time'?): {zero_drift:.4f}")
    print(f"  zero-action vs moving actions: "
          + ", ".join(f"{k}={v:.4f}" for k, v in list(zero_vs.items())[:4]))

    verdict = ("predictor BLIND to actions (synthetic ~ candidates)"
               if max(opp_vals) < 2.5 * np.median(cand_pairs)
               else "predictor RESPONDS to actions -> candidate diversity was the bottleneck")
    print(f"\n  heuristic verdict: {verdict}")

    # ---- figure: horizontal bars, one hue, direct labels ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, BAR, HILITE = "#374151", "#6b7280", "#93c5fd", "#1d4ed8"
    rows = [
        ("real change, 1.07 s (2 frames)", real_2, HILITE),
        ("real change, 0.53 s (1 frame)", real_1, HILITE),
        ("synthetic: +x vs -x (40 cm apart)", opp_vals[0], BAR),
        ("synthetic: +z vs -z (40 cm apart)", opp_vals[2], BAR),
        ("synthetic: zero vs +x (20 cm apart)", zero_vs["+x"], BAR),
        ("10 real pi0.5 candidates (max pair)", float(cand_pairs.max()), BAR),
        ("10 real pi0.5 candidates (median pair)", float(np.median(cand_pairs)), BAR),
    ]
    plt.rcParams.update({"font.size": 12, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    names = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    cols = [r[2] for r in rows][::-1]
    bars = ax.barh(names, vals, color=cols, height=0.62)
    for b, v in zip(bars, vals):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.4f}", va="center", fontsize=11, color=INK)
    ax.set_xlabel("latent L1 distance between (predicted) frames")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("How much do predicted futures differ, vs how much the real world changes?\n"
                 f"(one state: episode {args.episode}, step {s})", fontsize=12)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=160)
    print(f"\nsaved {args.out}.png")


if __name__ == "__main__":
    main()
