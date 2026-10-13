"""Summary statistics reported in the paper.

Paired comparisons use the Wilcoxon signed-rank test on user-level values
(event measures are summarized by their median within user, scale and event
type). Tests with a pre-specified direction are one-sided; the two-sided
p-value is reported alongside. Condition comparisons use the two-sided
Mann-Whitney U test with Holm correction across energy components.
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu, wilcoxon


def _paired(a, b, alternative, decimals):
    m = pd.concat([a, b], axis=1).dropna().round(decimals)
    x, y = m.iloc[:, 0], m.iloc[:, 1]
    return wilcoxon(x, y, alternative=alternative).pvalue, wilcoxon(x, y).pvalue, len(m)


def _holm(pvalues):
    pv = pd.Series(pvalues).sort_values()
    adj = np.minimum(1, np.maximum.accumulate(pv.values * np.arange(len(pv), 0, -1)))
    return pd.Series(adj, index=pv.index)


def summary_statistics(res, events, dist, cfg):
    users, gain, base, shuffle, bounds = (res["users"], res["gain"], res["baselines"],
                                          res["shuffle"], res["boundaries"])
    rows = []

    def add(analysis, measure, value, two_sided=None, n=None):
        rows.append({"analysis": analysis, "measure": measure, "value": value,
                     "two_sided_p": two_sided, "n": n})

    # Sample
    add("Sample", "users (BPD / bipolar)",
        f"{(users.community == 'BPD').sum()} / {(users.community == 'bipolar').sum()}")
    add("Sample", "trajectory length in bins (min-max)", f"{users.n_bins.min()}-{users.n_bins.max()}")

    # Energy
    for b in ["A2", "D2", "D1"]:
        add("Energy", f"median {b} energy", users[f"{b}_energy"].median(), n=len(users))
    add("Energy", "median detail energy (D1+D2)", users["detail_energy"].median(), n=len(users))

    # Semantic gain by scale
    ug = gain.groupby(["user_code", "band"])["semantic_gain"].mean().unstack()
    add("Semantic gain", "median user-level gain D1", ug["D1"].median())
    add("Semantic gain", "median user-level gain D2", ug["D2"].median())
    p1, p2, n = _paired(ug["D1"], ug["D2"], "greater", cfg.rank_decimals)
    add("Semantic gain", "Wilcoxon D1 > D2, one-sided p", p1, p2, n)

    # Condition comparison (exploratory)
    cond = {}
    for b in ["A2", "D2", "D1"]:
        x = users.loc[users.community == "BPD", f"{b}_energy"]
        y = users.loc[users.community == "bipolar", f"{b}_energy"]
        cond[b] = mannwhitneyu(x, y, alternative="two-sided").pvalue
        add("Condition comparison", f"{b} energy median BPD / bipolar",
            f"{x.median():.3f} / {y.median():.3f}")
        add("Condition comparison", f"{b} energy, Mann-Whitney p", cond[b], n=len(users))
    holm = _holm(cond)
    for b in ["A2", "D2", "D1"]:
        add("Condition comparison", f"{b} energy, Holm-adjusted p", holm[b])

    # Embedding-based change measures
    med = base.groupby("band")[["rho_mmd", "rho_adjacent", "top3_overlap"]].median()
    for band in ["D1", "D2"]:
        add("Change measures", f"median rho {band} vs adjacent cosine change", med.loc[band, "rho_adjacent"])
        add("Change measures", f"median rho {band} vs RBF-MMD", med.loc[band, "rho_mmd"])
        add("Change measures", f"median top-3 overlap {band} vs RBF-MMD", med.loc[band, "top3_overlap"])
    nb = base[base.band == "D1"]["n_boundaries"]
    add("Change measures", "boundaries per user (min / median / max)", f"{nb.min()} / {nb.median():.0f} / {nb.max()}")

    # Chronological structure
    add("Chronological structure", "mean detail energy, observed", shuffle.observed_detail_energy.mean())
    add("Chronological structure", "mean detail energy, shuffled", shuffle.shuffle_mean.mean())
    add("Chronological structure", "median within-user difference", shuffle.difference.median())
    lower = int((shuffle.difference < 0).sum())
    add("Chronological structure", "users with lower observed energy", f"{lower}/{len(shuffle)}")
    p1, p2, n = _paired(shuffle.observed_detail_energy, shuffle.shuffle_mean, "less", cfg.rank_decimals)
    add("Chronological structure", "Wilcoxon observed < shuffled, one-sided p", p1, p2, n)
    add("Chronological structure", "sign test, one-sided p",
        binomtest(lower, len(shuffle), alternative="greater").pvalue,
        binomtest(lower, len(shuffle)).pvalue, len(shuffle))
    sig = shuffle[shuffle.q_bh < .05]
    add("Chronological structure", "users significant after BH-FDR (BPD / bipolar)",
        f"{len(sig)} ({(sig.community == 'BPD').sum()} / {(sig.community == 'bipolar').sum()})")

    # Cross-scale distinctiveness
    perm = dist["permutation"]
    add("Cross-scale distinctiveness", "mean D1-D2 profile distance, observed", perm["observed_distance"], n=perm["n_users"])
    add("Cross-scale distinctiveness", "mean D1-D2 profile distance, permuted", perm["null_distance"])
    add("Cross-scale distinctiveness", "permutation p", perm["p"])
    add("Cross-scale distinctiveness", "median Jaccard D1-D2", dist["users"].phrase_jaccard.median())
    add("Cross-scale distinctiveness", "median lexical distinctiveness", dist["users"].phrase_distinctiveness.median())

    # High-gain events vs. matched controls
    def event_tests(ev, label):
        add(label, "events (high-gain / control)",
            f"{(ev.event_type == 'high_gain').sum()} / {(ev.event_type == 'low_gain_control').sum()}")
        per_user = ev.groupby(["user_code", "band", "event_type"])[
            ["gain_coherence", "grounding", "semantic_shift"]].median()
        for metric in ["gain_coherence", "grounding", "semantic_shift"]:
            for band in sorted(ev.band.unique()):
                sub = ev[ev.band == band]
                add(label, f"{band} {metric}: median control",
                    sub.loc[sub.event_type == "low_gain_control", metric].median())
                add(label, f"{band} {metric}: median high-gain",
                    sub.loc[sub.event_type == "high_gain", metric].median())
                u = per_user[metric].xs(band, level="band").unstack()
                p1, p2, n = _paired(u["high_gain"], u["low_gain_control"], "greater", cfg.rank_decimals)
                add(label, f"{band} {metric}: Wilcoxon high > control, one-sided p", p1, p2, n)

    event_tests(events, "High-gain vs control")
    event_tests(events[(events.band == "D2") & ~events.duplicate_d2_pair], "High-gain vs control (unique D2 events)")

    # Case study
    case_user = cfg.case_study_user
    if case_user is None or case_user not in set(users.user_code):
        cand = users[(users.community == "BPD") & (users.n_bins == 16)].user_code
        d2 = base[(base.band == "D2") & base.user_code.isin(cand)].set_index("user_code")
        case_user = d2["rho_mmd"].idxmax()
    cs = base[base.user_code == case_user].set_index("band")
    top = (events[(events.user_code == case_user) & (events.band == "D2") & (events.event_type == "high_gain")]
           .sort_values("mean_semantic_gain", ascending=False).iloc[0])
    add("Case study", "user", case_user)
    add("Case study", "D2 vs RBF-MMD rho", cs.loc["D2", "rho_mmd"], n=int(cs.loc["D2", "n_boundaries"]))
    add("Case study", "D2 vs RBF-MMD p", cs.loc["D2", "p_mmd"])
    add("Case study", "D1 vs adjacent cosine change rho", cs.loc["D1", "rho_adjacent"])
    add("Case study", "maximum RBF-MMD", bounds[bounds.user_code == case_user].mmd.max())
    add("Case study", "highest-gain D2 event: mean gain", top.mean_semantic_gain)
    add("Case study", "highest-gain D2 event: coherence", top.gain_coherence)

    table = pd.DataFrame(rows)
    for col in ["value", "two_sided_p"]:
        table[col] = table[col].map(_fmt)
    table["n"] = table["n"].map(lambda v: "" if v is None or pd.isna(v) else str(int(v)))
    return table, case_user


def _fmt(v):
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{v:.3g}" if 0 < abs(v) < 1e-3 else f"{v:.3f}"
    return str(v)
