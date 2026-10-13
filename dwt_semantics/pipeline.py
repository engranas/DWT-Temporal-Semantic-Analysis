"""End-to-end pipeline: corpus -> trajectories -> analyses -> tables and figures."""

import os

import numpy as np
import pandas as pd

from . import analysis, embed, plots, stats
from .config import Config
from .data import load_corpus
from .trajectory import build_trajectory


def run(data_path, output_dir="outputs", cfg=None, model=None, progress=True):
    cfg = cfg or Config()
    rng = np.random.default_rng(cfg.seed)
    results_dir = os.path.join(output_dir, "results")
    private_dir = os.path.join(output_dir, "private")
    fig_dir = os.path.join(output_dir, "figures")
    for d in (results_dir, private_dir, fig_dir):
        os.makedirs(d, exist_ok=True)

    # 1. Corpus and sample
    df, attrition = load_corpus(data_path, cfg)

    # 2. Embeddings and phrase dictionary
    model = model or embed.load_encoder(cfg)
    post_emb = embed.encode(model, df["clean_text"], cfg, progress)
    phr = embed.build_phrase_dictionary(df["clean_text"], model, cfg, progress)

    # 3. Calendar trajectories
    trajectories = {}
    for (community, user), g in df.groupby(["community", "user_code"]):
        traj = build_trajectory(g, post_emb[g.index.to_numpy()], cfg)
        if traj is not None:
            trajectories[(community, user)] = traj
    final_users = [u for _, u in trajectories]
    attrition.append((f"Contiguous trajectory >= {cfg.min_segment_bins} bins",
                      int(df["user_code"].isin(final_users).sum()), len(trajectories)))
    attrition_df = pd.DataFrame(attrition, columns=["stage", "posts", "users"])

    # 4. Analyses (the random generator is consumed in this fixed order)
    res = analysis.analyze_users(df, trajectories, post_emb, phr, cfg, rng)
    dist = analysis.band_distinctiveness(res["gain"], phr, cfg, rng)
    events = analysis.gain_events(df, trajectories, post_emb, res["gain"], phr, cfg)
    packet, key = analysis.annotation_packet(events, cfg)

    # 5. Statistics and figures
    table, case_user = stats.summary_statistics(res, events, dist, cfg)
    plots.summary_figure(res["users"], res["baselines"], events, os.path.join(fig_dir, "results_summary.png"))
    plots.case_study_figure(res["boundaries"], case_user, os.path.join(fig_dir, "case_study.png"))

    # 6. Outputs: aggregate (publishable) vs. per-user / text-bearing (private)
    table.to_csv(os.path.join(results_dir, "statistics.csv"), index=False)
    attrition_df.to_csv(os.path.join(results_dir, "attrition.csv"), index=False)
    (dist["summary"].groupby(["band", "phrase"])
     .agg(users=("user_code", "nunique"), mean_gain=("mean_gain", "mean"), selections=("times_selected", "sum"))
     .reset_index().sort_values(["band", "users", "mean_gain"], ascending=[True, False, False])
     .to_csv(os.path.join(results_dir, "phrase_gain_by_band.csv"), index=False))

    private = {"user_metrics.csv": res["users"], "semantic_gain.csv": res["gain"],
               "boundaries.csv": res["boundaries"], "dwt_vs_change_measures.csv": res["baselines"],
               "chronological_shuffle.csv": res["shuffle"], "band_distinctiveness.csv": dist["users"],
               "events.csv": events, "annotation_packet_blinded.csv": packet, "annotation_key.csv": key}
    for name, frame in private.items():
        frame.to_csv(os.path.join(private_dir, name), index=False)

    return {"attrition": attrition_df, "statistics": table, **res, "distinctiveness": dist,
            "events": events, "case_user": case_user}
