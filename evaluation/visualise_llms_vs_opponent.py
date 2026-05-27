#!/usr/bin/env python3
"""
Visualize all LLM negotiation sessions against a given opponent in a given domain.

Reads session logs from session_logs/{domain}/{opponent}_vs_{llm}.xlsx and produces
a two-subplot figure that overlays every LLM's offers (one color per LLM) in both
the LLM's own utility space and the opponent's utility space.

Adapted from evaluation/visualise.py.
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


SESSION_LOGS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "session_logs",
)


_LLM_COLOR_MAP = {
    "GPT-4o":         "#4DAF4A",  # green
    "GPT-5.5":        "#FF7F00",  # orange
    "Gemini Pro 3.1": "#F781BF",  # pink
    "Opus 4.7":       "#A65628",  # brown
}

_LLM_MARKER_MAP = {
    "GPT-4o":         "o",
    "GPT-5.5":        "s",
    "Gemini Pro 3.1": "^",
    "Opus 4.7":       "D",
}

_NO_DEAL_COLOR = "#E41A1C"  # red — reserved for the no-deal marker


def _pretty_llm_name(raw: str) -> str:
    """Map a raw LLM filename token to a friendly alias used in the legend."""
    name = raw[len("LLM-"):] if raw.startswith("LLM-") else raw
    name = name.lstrip("~").lower()
    if "gpt-4o" in name:
        return "GPT-4o"
    if "gpt-5" in name:
        return "GPT-5.5"
    if "gemini" in name:
        return "Gemini Pro 3.1"
    if "claude" in name or "anthropic" in name or "opus" in name or "sonnet" in name:
        return "Opus 4.7"
    return raw  # unknown — surface the raw id so it can be added to the map


def _load_session(path: str):
    """Load a single session Excel file and extract LLM/opponent offer tracks.

    File convention: {Opponent}_vs_{LLM}.xlsx — so Agent A is the opponent,
    Agent B is the LLM under test.
    """
    df = pd.read_excel(path, sheet_name="Session")

    llm_times, llm_self_util, llm_opp_util = [], [], []
    opp_times, opp_self_util, opp_in_llm_util = [], [], []

    for _, row in df.iterrows():
        if row["Action"] == "Accept":
            continue
        t = float(row["Time"])
        ua = float(row["AgentAUtility"])  # opponent utility
        ub = float(row["AgentBUtility"])  # LLM utility
        if row["Who"] == "B":  # LLM's own offer
            llm_times.append(t)
            llm_self_util.append(ub)
            llm_opp_util.append(ua)
        elif row["Who"] == "A":  # opponent's offer
            opp_times.append(t)
            opp_self_util.append(ua)
            opp_in_llm_util.append(ub)

    accept = df[df["Action"] == "Accept"]
    if not accept.empty:
        last = accept.iloc[-1]
        final = {
            "agreed": True,
            "time": float(last["Time"]),
            "llm_util": float(last["AgentBUtility"]),
            "opp_util": float(last["AgentAUtility"]),
        }
    else:
        final = {"agreed": False, "time": 1.0, "llm_util": 0.0, "opp_util": 0.0}

    return {
        "rounds": len(df),
        "llm_times": llm_times,
        "llm_self_util": llm_self_util,
        "llm_opp_util": llm_opp_util,
        "opp_times": opp_times,
        "opp_self_util": opp_self_util,
        "opp_in_llm_util": opp_in_llm_util,
        "final": final,
    }


def visualise_llms(domain: str, opponent: str, output_dir: str = None,
                   smooth_window: int = 5) -> None:
    domain_dir = os.path.join(SESSION_LOGS_ROOT, domain)
    if not os.path.isdir(domain_dir):
        raise FileNotFoundError(f"Domain directory not found: {domain_dir}")

    prefix = f"{opponent}_vs_"
    matches = sorted(
        f for f in os.listdir(domain_dir)
        if f.startswith(prefix) and f.endswith(".xlsx")
    )
    if not matches:
        raise FileNotFoundError(
            f"No sessions found for opponent '{opponent}' in {domain_dir}"
        )

    llm_sessions = []
    for fname in matches:
        llm_raw = fname[len(prefix):-len(".xlsx")]
        if not llm_raw.startswith("LLM-"):
            continue  # only LLM opponents
        path = os.path.join(domain_dir, fname)
        print(f"📖 Loading {fname}")
        data = _load_session(path)
        llm_sessions.append((_pretty_llm_name(llm_raw), data))

    if not llm_sessions:
        raise RuntimeError(
            f"No LLM session files matched prefix '{prefix}' in {domain_dir}"
        )

    fallback_palette = [
        "#377EB8",  # blue
        "#984EA3",  # purple
        "#17BECF",  # cyan
        "#7F7F7F",  # gray
        "#BCBD22",  # olive
    ]
    fallback_markers = ["v", "p", "h", "P", "X", "<", ">"]
    colors, markers = [], []
    fb_idx = 0
    for llm_name, _ in llm_sessions:
        if llm_name in _LLM_COLOR_MAP:
            colors.append(_LLM_COLOR_MAP[llm_name])
            markers.append(_LLM_MARKER_MAP[llm_name])
        else:
            colors.append(fallback_palette[fb_idx % len(fallback_palette)])
            markers.append(fallback_markers[fb_idx % len(fallback_markers)])
            fb_idx += 1

    fig, ax1 = plt.subplots(figsize=(10, 6.8))

    def _sorted_by_time(times, ys):
        if not times:
            return [], []
        order = np.argsort(times)
        return np.asarray(times)[order], np.asarray(ys)[order]

    all_times, all_utils = [], []
    for _, data in llm_sessions:
        all_times.extend(data["llm_times"])
        all_utils.extend(data["llm_self_util"])
        if data["final"]["agreed"]:
            all_times.append(data["final"]["time"])
            all_utils.append(data["final"]["llm_util"])

    for (llm_name, data), color, marker in zip(llm_sessions, colors, markers):
        final = data["final"]
        agreed = final["agreed"]
        label = llm_name

        t_llm, y_llm = _sorted_by_time(data["llm_times"], data["llm_self_util"])
        if smooth_window > 1 and len(y_llm) >= 2:
            y_line = (
                pd.Series(y_llm)
                .rolling(window=smooth_window, min_periods=1, center=True)
                .mean()
                .to_numpy()
            )
        else:
            y_line = np.asarray(y_llm)
        ax1.plot(t_llm, y_line, color=color, alpha=0.95,
                 linewidth=2.8, zorder=4)
        if len(t_llm) >= 2:
            n_markers = min(4, len(t_llm))
            idxs = np.linspace(0, len(t_llm) - 1, n_markers + 2, dtype=int)[1:-1]
            ax1.scatter(np.asarray(t_llm)[idxs], np.asarray(y_line)[idxs],
                        marker=marker, c=[color], s=260,
                        edgecolors="white", linewidths=1.5, zorder=4.5)
        if len(t_llm) > 0:
            last_t, last_y = t_llm[-1], y_line[-1]
            if agreed:
                ax1.scatter([last_t], [last_y],
                            c=[color], s=900,
                            marker=r"$\mathbf{\checkmark}$",
                            linewidths=2.5, zorder=5)
            else:
                ax1.scatter([last_t], [last_y],
                            c=[_NO_DEAL_COLOR], s=420, marker="X",
                            edgecolors="black", linewidth=2.0, zorder=6)

    ax1.set_xlabel("Negotiation Time", fontsize=26, fontweight="bold")
    ax1.set_ylabel("Self Utility", fontsize=26, fontweight="bold")

    t_min, t_max = min(all_times), max(all_times)
    u_min, u_max = min(all_utils), max(all_utils)
    t_pad_left = max(0.02, (t_max - t_min) * 0.05)
    t_pad_right = max(0.02, (t_max - t_min) * 0.03)
    u_pad = max(0.02, (u_max - u_min) * 0.08)
    ax1.set_xlim(max(0.0, t_min - t_pad_left), t_max + t_pad_right)
    ax1.set_ylim(max(0.0, u_min - u_pad), min(1.0, u_max + u_pad))

    ax1.tick_params(labelsize=20)
    ax1.grid(True, alpha=0.25)

    llm_handles = [
        plt.Line2D([0], [0], marker=marker, linestyle="-",
                   color=color,
                   markersize=14 if marker in ("o", "^") else 12,
                   linewidth=2.5,
                   markeredgecolor="none", label=llm_name)
        for (llm_name, _), color, marker in zip(llm_sessions, colors, markers)
    ]
    fig.legend(handles=llm_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.88), ncol=len(llm_handles),
               fontsize=24, frameon=False,
               handletextpad=0.6, columnspacing=1.1)

    plt.tight_layout(rect=[0, 0, 1, 0.88])

    if output_dir is None:
        output_dir = os.path.join("plots", "llms_vs_opponent", domain)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"all_LLMs_vs_{opponent}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n✅ Saved: {out_path}")
    print(f"   LLMs plotted: {len(llm_sessions)}")
    for name, data in llm_sessions:
        f = data["final"]
        print(
            f"   • {name}: rounds={data['rounds']}, "
            f"final LLM={f['llm_util']:.3f}, opp={f['opp_util']:.3f}, "
            f"{'agreed' if f['agreed'] else 'no deal'}"
        )


def _parse_args():
    p = argparse.ArgumentParser(
        description="Visualize all LLMs against a given opponent in a given domain.",
    )
    p.add_argument("opponent", help="Opponent agent name, e.g. Adversary-aggression, TRAD-Boulware")
    p.add_argument("domain", help="Domain directory under session_logs/, e.g. domain5")
    p.add_argument("--output-dir", default=None, help="Where to save the PNG")
    p.add_argument("--smooth-window", type=int, default=5,
                   help="Centered rolling-mean window for the LLM utility line (1 = no smoothing)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    visualise_llms(args.domain, args.opponent, args.output_dir,
                   smooth_window=args.smooth_window)
