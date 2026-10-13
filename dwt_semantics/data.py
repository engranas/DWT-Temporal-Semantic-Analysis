"""Corpus loading, filtering and calendar binning."""

import pandas as pd

from .text import clean_text, user_hash

REQUIRED_COLUMNS = ["community", "created_at", "text", "user_screen_name"]


def load_corpus(path, cfg):
    """Load the corpus and apply the sample-construction filters.

    Returns the eligible posts (sorted by user and time, with a global
    calendar-bin index) and an attrition table.
    """
    raw = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    attrition = [("Full corpus", len(raw), raw["user_screen_name"].nunique())]

    df = raw[raw["community"].isin(cfg.groups)].copy()
    attrition.append(("BPD- and bipolar-related communities", len(df), df["user_screen_name"].nunique()))

    df["dt"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["dt", "text", "user_screen_name"])
    df["user_screen_name"] = df["user_screen_name"].astype(str).str.lower().str.strip()
    df = df.drop_duplicates(subset=["user_screen_name", "created_at", "text"])
    attrition.append(("Valid timestamp, de-duplicated", len(df), df["user_screen_name"].nunique()))

    n_communities = df.groupby("user_screen_name")["community"].nunique()
    df = df[~df["user_screen_name"].isin(n_communities[n_communities > 1].index)].copy()
    attrition.append(("Cross-community users removed", len(df), df["user_screen_name"].nunique()))

    df["user_code"] = df["user_screen_name"].map(user_hash)
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() >= 3].copy()
    attrition.append(("Non-empty after cleaning", len(df), df["user_code"].nunique()))

    # Global calendar bins anchored at the earliest post in the corpus.
    timestamps = df["dt"].dt.tz_convert(None)
    origin = timestamps.min().normalize()
    df["bin_idx_global"] = ((timestamps - origin).dt.days // cfg.bin_days).astype(int)
    df = df.sort_values(["user_code", "dt"]).reset_index(drop=True)

    stats = df.groupby(["community", "user_code"]).agg(
        n=("dt", "size"), bins=("bin_idx_global", "nunique"))
    eligible = stats[(stats["n"] >= cfg.min_posts) & (stats["bins"] >= cfg.min_active_bins)].index
    keys = pd.MultiIndex.from_frame(df[["community", "user_code"]])
    df = df[keys.isin(eligible)].reset_index(drop=True)
    attrition.append((f">= {cfg.min_posts} posts and >= {cfg.min_active_bins} active bins",
                      len(df), df["user_code"].nunique()))

    return df, attrition
