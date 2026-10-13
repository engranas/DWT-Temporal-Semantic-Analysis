"""Figures: summary of the main results and the case-study trajectory."""

import os

import matplotlib.pyplot as plt
import numpy as np


def summary_figure(users, baselines, events, path):
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    x = np.arange(2)

    e = users[["A2_energy", "D2_energy", "D1_energy"]].median()
    ax[0].bar(["A2", "D2", "D1"], e.values, color=["#9aa5b1", "#3b82c4", "#e07b39"])
    ax[0].set_title("(a) Median centered energy")

    med = baselines.groupby("band")[["rho_adjacent", "rho_mmd"]].median().loc[["D1", "D2"]]
    ax[1].bar(x - .18, med["rho_adjacent"], .36, label="Adjacent cosine")
    ax[1].bar(x + .18, med["rho_mmd"], .36, label="RBF-MMD")
    ax[1].set_xticks(x, ["D1", "D2"])
    ax[1].set_ylim(0, 1.25 * med.values.max())
    ax[1].legend(fontsize=8, ncol=2, loc="upper center")
    ax[1].set_title("(b) Median Spearman ρ")

    for a, metric, title in [(ax[2], "gain_coherence", "(c) Phrase coherence"),
                             (ax[3], "grounding", "(d) Out-of-window grounding")]:
        m = events.groupby(["band", "event_type"])[metric].median().unstack().loc[["D1", "D2"]]
        a.bar(x - .18, m["low_gain_control"], .36, label="Control", color="#9aa5b1")
        a.bar(x + .18, m["high_gain"], .36, label="High gain", color="#3b82c4")
        a.set_xticks(x, ["D1", "D2"])
        a.set_title(title)
        a.set_ylim(0, 1.25 * m.values.max())
        a.legend(fontsize=8, ncol=2, loc="upper center")
        for i, (lo, hi) in enumerate(zip(m["low_gain_control"], m["high_gain"])):
            a.text(i - .18, lo, f"{lo:.3f}", ha="center", va="bottom", fontsize=8)
            a.text(i + .18, hi, f"{hi:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    return fig


def case_study_figure(boundaries, user, path):
    b = boundaries[boundaries.user_code == user]

    def mm(s):
        return (s - s.min()) / (s.max() - s.min())

    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax[0].plot(b.boundary, mm(b.D2_score), "o-", label="D2 boundary score")
    ax[0].plot(b.boundary, mm(b.mmd), "s--", label="RBF-MMD")
    ax[1].plot(b.boundary, mm(b.D1_score), "o-", label="D1 boundary score")
    ax[1].plot(b.boundary, mm(b.adjacent_cosine_change), "s--", label="Adjacent cosine change")
    for a in ax:
        a.legend(fontsize=8)
        a.set_ylabel("min-max normalized")
    ax[1].set_xlabel("Boundary between 14-day bins")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    return fig
