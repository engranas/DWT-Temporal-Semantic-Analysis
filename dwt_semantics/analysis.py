"""Per-user analyses: energy, semantic gain, change measures, permutation tests and events."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .trajectory import (band_components, cosine_distance, dwt, energy_fractions,
                         normalize_rows, normalize_vector)

BANDS = {"D2": ("R0", "R1"), "D1": ("R1", "R2")}  # (before, after) reconstruction


# ---------------------------------------------------------------- helpers

def support(t, band, n):
    """Time block of the DWT coefficient that generates time point t."""
    block = {"D1": 2, "D2": 4}[band]
    start = t // block * block
    return start, min(start + block, n)


def out_of_window_positions(user_posts, traj, t, band, cfg):
    """Positions of a user's posts outside the support of t (plus buffer)."""
    s, e = support(t, band, len(traj["bin_ids"]))
    lo = traj["bin_ids"][s:e].min() - cfg.exclusion_buffer_bins
    hi = traj["bin_ids"][s:e].max() + cfg.exclusion_buffer_bins
    b = user_posts["bin_idx_global"]
    return np.where(((b < lo) | (b > hi)).to_numpy())[0]


def phrases_in(rows, phr):
    if len(rows) == 0:
        return np.array([], dtype=int)
    ids = np.unique(phr["occurrence"][rows].nonzero()[1])
    return np.intersect1d(ids, phr["valid_ids"])


def adjacent_change(E):
    E = normalize_rows(E)
    return 1.0 - np.sum(E[:-1] * E[1:], axis=1)


def _sq_dist(X, Y):
    return np.sum(X ** 2, 1)[:, None] + np.sum(Y ** 2, 1)[None, :] - 2 * X @ Y.T


def mmd_scores(E, half_window):
    """Biased RBF-MMD^2 between the bins before and after each boundary."""
    d = _sq_dist(E, E)[np.triu_indices(len(E), k=1)]
    d = d[d > 1e-12]
    gamma = 1.0 / (2.0 * np.median(d)) if len(d) else 1.0

    def k(X, Y):
        return np.exp(-gamma * _sq_dist(X, Y))

    scores = np.full(len(E) - 1, np.nan)
    w = half_window
    for b in range(w, len(E) - w + 1):
        L, R = E[b - w: b], E[b: b + w]
        scores[b - 1] = k(L, L).mean() + k(R, R).mean() - 2 * k(L, R).mean()
    return scores


def boundary_score(component):
    m = np.linalg.norm(component, axis=1)
    return (m[:-1] + m[1:]) / 2.0


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


# ---------------------------------------------------------------- trajectories

def analyze_users(df, trajectories, post_emb, phr, cfg, rng):
    """Energy, semantic gain, change measures and chronological shuffle test."""
    user_rows, gain_rows, boundary_rows, baseline_rows, shuffle_rows = [], [], [], [], []

    for (community, user), traj in trajectories.items():
        g = df[df["user_code"] == user].sort_values("dt")
        rows = g.index.to_numpy()
        E = traj["embeddings"]
        n = len(E)
        comp = band_components(E, cfg)
        assert np.allclose(comp["R2"], E), "perfect reconstruction failed"

        # Centered energy fractions
        centered = E - E.mean(axis=0)
        energy = energy_fractions(dwt(centered, cfg))
        user_rows.append({"community": community, "user_code": user, "n_bins": n,
                          "observed_bins": int(traj["observed"].sum()),
                          "interpolated_bins": int(traj["imputed"].sum()),
                          "A2_energy": energy["A2"], "D2_energy": energy["D2"],
                          "D1_energy": energy["D1"], "detail_energy": energy["D2"] + energy["D1"]})

        # Semantic gain with out-of-window phrases
        for band, (before_key, after_key) in BANDS.items():
            for t in range(n):
                positions = out_of_window_positions(g, traj, t, band, cfg)
                ids = phrases_in(rows[positions], phr) if len(positions) else np.array([], dtype=int)
                if len(ids) == 0:
                    continue
                sim_before = phr["embeddings"][ids] @ normalize_vector(comp[before_key][t])
                sim_after = phr["embeddings"][ids] @ normalize_vector(comp[after_key][t])
                gain = sim_after - sim_before
                positive = np.where(gain > 0)[0]
                top = positive[np.argsort(gain[positive])[::-1]][: cfg.top_phrases_per_time]
                for rank, i in enumerate(top, 1):
                    gain_rows.append({"community": community, "user_code": user, "band": band,
                                      "time_index": t, "rank": rank, "phrase": phr["phrases"][ids[i]],
                                      "phrase_id": int(ids[i]), "similarity_before": sim_before[i],
                                      "similarity_after": sim_after[i], "semantic_gain": gain[i]})

        # DWT detail magnitude vs. embedding-based change measures
        mmd, adj = mmd_scores(E, cfg.mmd_half_window), adjacent_change(E)
        scores = {"D1": boundary_score(comp["D1"]), "D2": boundary_score(comp["D2"])}
        for b in range(n - 1):
            boundary_rows.append({"community": community, "user_code": user, "boundary": b + 1,
                                  "adjacent_cosine_change": adj[b], "mmd": mmd[b],
                                  "D1_score": scores["D1"][b], "D2_score": scores["D2"][b]})
        for band, s_dwt in scores.items():
            valid = np.isfinite(mmd) & np.isfinite(s_dwt)
            row = {"community": community, "user_code": user, "band": band,
                   "n_boundaries": int(valid.sum())}
            if valid.sum() >= 3:
                idx = np.where(valid)[0]
                r_dwt, r_mmd, r_adj = (np.round(v[valid], cfg.rank_decimals) for v in (s_dwt, mmd, adj))
                row["rho_mmd"], row["p_mmd"] = spearmanr(r_dwt, r_mmd)
                row["rho_adjacent"], row["p_adjacent"] = spearmanr(r_dwt, r_adj)
                k = min(3, valid.sum())
                top_dwt = set(idx[np.argsort(s_dwt[valid])[::-1][:k]])
                top_mmd = set(idx[np.argsort(mmd[valid])[::-1][:k]])
                row["top3_overlap"] = len(top_dwt & top_mmd) / k
            baseline_rows.append(row)

        # Chronological shuffle test
        observed_detail = energy["D1"] + energy["D2"]
        null = []
        for _ in range(cfg.n_order_shuffles):
            e_shuffled = energy_fractions(dwt(centered[rng.permutation(n)], cfg))
            null.append(e_shuffled["D1"] + e_shuffled["D2"])
        null = np.array(null)
        p = (1 + np.sum(np.abs(null - null.mean()) >= abs(observed_detail - null.mean()))) \
            / (cfg.n_order_shuffles + 1)
        shuffle_rows.append({"community": community, "user_code": user,
                             "observed_detail_energy": observed_detail, "shuffle_mean": null.mean(),
                             "difference": observed_detail - null.mean(), "empirical_p": p})

    shuffle_df = pd.DataFrame(shuffle_rows)
    shuffle_df["q_bh"] = bh_fdr(shuffle_df["empirical_p"])
    return {"users": pd.DataFrame(user_rows), "gain": pd.DataFrame(gain_rows),
            "boundaries": pd.DataFrame(boundary_rows), "baselines": pd.DataFrame(baseline_rows),
            "shuffle": shuffle_df}


# ---------------------------------------------------------------- distinctiveness

def band_distinctiveness(gain_df, phr, cfg, rng):
    """D1 vs. D2 phrase profiles: Jaccard, profile distance, permutation test."""
    summary = (gain_df.groupby(["community", "user_code", "band", "phrase", "phrase_id"])
               .agg(mean_gain=("semantic_gain", "mean"), max_gain=("semantic_gain", "max"),
                    times_selected=("semantic_gain", "size"))
               .reset_index())
    top = pd.concat(
        [grp.sort_values(["mean_gain", "times_selected"], ascending=[False, False])
            .head(cfg.top_phrases_per_user_band)
         for _, grp in summary.groupby(["community", "user_code", "band"])],
        ignore_index=True)

    emb = phr["embeddings"]

    def profile(ids):
        return normalize_vector(emb[ids].mean(axis=0))

    rows, band_sets = [], {}
    for user in top["user_code"].unique():
        d1 = top[(top["user_code"] == user) & (top["band"] == "D1")]
        d2 = top[(top["user_code"] == user) & (top["band"] == "D2")]
        if len(d1) == 0 or len(d2) == 0:
            continue
        ids1, ids2 = d1["phrase_id"].to_numpy(int), d2["phrase_id"].to_numpy(int)
        s1, s2 = set(d1["phrase"]), set(d2["phrase"])
        jaccard = len(s1 & s2) / len(s1 | s2)
        rows.append({"user_code": user,
                     "semantic_cosine_distance": cosine_distance(profile(ids1), profile(ids2)),
                     "phrase_jaccard": jaccard, "phrase_distinctiveness": 1 - jaccard})
        if len(ids1) >= 3 and len(ids2) >= 3:
            band_sets[user] = (ids1, ids2)

    observed = float(np.mean([cosine_distance(profile(a), profile(b)) for a, b in band_sets.values()]))
    null = []
    for _ in range(cfg.n_band_permutations):
        per_user = []
        for ids1, ids2 in band_sets.values():
            V = np.vstack([emb[ids1], emb[ids2]])
            perm = rng.permutation(len(V))
            per_user.append(cosine_distance(normalize_vector(V[perm[:len(ids1)]].mean(0)),
                                            normalize_vector(V[perm[len(ids1):]].mean(0))))
        null.append(np.mean(per_user))
    null = np.array(null)
    p = (1 + np.sum(null >= observed)) / (cfg.n_band_permutations + 1)

    return {"summary": summary, "users": pd.DataFrame(rows),
            "permutation": {"n_users": len(band_sets), "observed_distance": observed,
                            "null_distance": float(null.mean()), "p": float(p)}}


# ---------------------------------------------------------------- events

def _coherence(ids, emb):
    ids = sorted(set(int(i) for i in ids))
    if len(ids) < 2:
        return np.nan
    V = normalize_rows(emb[ids])
    return float((V @ V.T)[np.triu_indices(len(ids), k=1)].mean())


def _top_phrases(vec, ids, phr, k):
    sims = normalize_rows(phr["embeddings"][ids]) @ normalize_vector(vec)
    return [phr["phrases"][ids[i]] for i in np.argsort(sims)[::-1][:k]]


def gain_events(df, trajectories, post_emb, gain_df, phr, cfg):
    """High-gain events and matched low-gain controls for each user and scale.

    For a trajectory of T bins, the k = min(events_per_band, T // 2) time points
    with the highest mean gain are high-gain events and the k lowest are controls.
    """
    rows_out = []
    for (community, user), traj in trajectories.items():
        g = df[(df["community"] == community) & (df["user_code"] == user)].sort_values("dt")
        rows = g.index.to_numpy()
        E_posts = post_emb[rows]
        comp = band_components(traj["embeddings"], cfg)
        for band, (before_key, after_key) in BANDS.items():
            before_m, after_m = comp[before_key], comp[after_key]
            ug = gain_df[(gain_df["user_code"] == user) & (gain_df["band"] == band)]
            if ug.empty:
                continue
            time_scores = ug.groupby("time_index")["semantic_gain"].mean().sort_values(ascending=False)
            k = min(cfg.events_per_band, len(time_scores) // 2)
            if k == 0:
                continue
            high = time_scores.head(k).index.tolist()
            low = time_scores.drop(index=high).sort_values().head(k).index.tolist()
            for event_type, times in (("high_gain", high), ("low_gain_control", low)):
                for t in times:
                    positions = out_of_window_positions(g, traj, t, band, cfg)
                    ids = list(dict.fromkeys(int(i) for i in phrases_in(rows[positions], phr))) \
                        if len(positions) else []
                    if not ids:
                        continue
                    ev = (ug[(ug["time_index"] == t) & ug["phrase_id"].isin(ids)]
                          .sort_values("semantic_gain", ascending=False)
                          .drop_duplicates("phrase_id").head(cfg.top_phrases_per_time))
                    if ev.empty:
                        continue
                    gained_ids = list(dict.fromkeys(ev["phrase_id"].astype(int)))
                    query = normalize_vector(normalize_rows(phr["embeddings"][gained_ids]).mean(axis=0))
                    sims = normalize_rows(E_posts[positions]) @ query
                    best = np.argsort(sims)[::-1][: cfg.evidence_posts]
                    row = {"community": community, "user_code": user, "band": band,
                           "event_type": event_type, "time_index": int(t),
                           "calendar_bin": int(traj["bin_ids"][t]),
                           "mean_semantic_gain": float(time_scores.loc[t]),
                           "semantic_shift": cosine_distance(before_m[t], after_m[t]),
                           "gain_coherence": _coherence(gained_ids, phr["embeddings"]),
                           "before_phrases": " | ".join(_top_phrases(before_m[t], ids, phr, cfg.top_phrases_per_time)),
                           "gained_phrases": " | ".join(ev["phrase"].astype(str)),
                           "after_phrases": " | ".join(_top_phrases(after_m[t], ids, phr, cfg.top_phrases_per_time))}
                    for j in range(cfg.evidence_posts):
                        has = j < len(best)
                        row[f"evidence_{j+1}_similarity"] = float(sims[best[j]]) if has else np.nan
                        row[f"evidence_{j+1}_text"] = g.iloc[positions[best[j]]]["clean_text"] if has else ""
                    rows_out.append(row)

    events = (pd.DataFrame(rows_out)
              .sort_values(["community", "user_code", "band", "event_type", "time_index"])
              .reset_index(drop=True))
    events["event_id"] = [f"KG{i:04d}" for i in range(1, len(events) + 1)]
    # Out-of-window grounding: mean similarity to the closest out-of-window posts.
    events["grounding"] = events[[f"evidence_{j+1}_similarity" for j in range(cfg.evidence_posts)]].mean(axis=1)
    # With a level-2 Haar transform, R0 and R1 are constant within each pair of
    # bins, so the two time points of a pair yield identical D2 events.
    events["duplicate_d2_pair"] = (events["band"].eq("D2")
                                   & events.assign(pair=events.time_index // 2)
                                   .duplicated(["user_code", "band", "event_type", "pair"]))
    return events


def annotation_packet(events, cfg):
    """Blinded, shuffled packet for human evaluation (unique events only) and its key."""
    text_cols = ["event_id", "before_phrases", "gained_phrases", "after_phrases"] + \
                [f"evidence_{j+1}_text" for j in range(cfg.evidence_posts)]
    packet = events.loc[~events["duplicate_d2_pair"], text_cols].copy()
    for col in ["meaningful_change", "new_information", "coherent_theme", "interpretable",
                "theme_label", "confidence", "notes", "annotator_id"]:
        packet[col] = ""
    packet = packet.sample(frac=1, random_state=cfg.annotation_seed).reset_index(drop=True)
    key = events.drop(columns=[c for c in text_cols if c != "event_id"])
    return packet, key
