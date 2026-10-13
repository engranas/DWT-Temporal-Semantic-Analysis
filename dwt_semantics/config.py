"""Experiment configuration. Defaults are the settings used in the paper."""

from dataclasses import dataclass


@dataclass
class Config:
    # Data
    groups: tuple = ("BPD", "bipolar")

    # Sentence encoder
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    batch_size: int = 64

    # Calendar trajectories
    bin_days: int = 14               # one semantic state = 14 days
    min_posts: int = 20              # user eligibility
    min_active_bins: int = 6         # user eligibility
    min_segment_bins: int = 8        # minimum contiguous trajectory (~16 weeks)
    max_interpolation_gap: int = 1   # interpolate at most one missing bin

    # Wavelet: level-2 Haar -> A2 (>8 weeks), D2 (~8 weeks), D1 (~4 weeks)
    wavelet: str = "haar"
    level: int = 2

    # Phrase dictionary (1-3-grams)
    phrase_min_df: int = 3
    phrase_max_df: float = 0.65
    max_phrases: int = 6000

    # Semantic gain and out-of-window restriction
    top_phrases_per_time: int = 8
    top_phrases_per_user_band: int = 30
    exclusion_buffer_bins: int = 1

    # Embedding-based change measures
    mmd_half_window: int = 2         # 2 bins before vs. 2 bins after a boundary

    # Permutation tests
    n_order_shuffles: int = 1000
    n_band_permutations: int = 1000

    # High-gain events vs. matched low-gain controls
    events_per_band: int = 5
    evidence_posts: int = 3

    # Ranking precision: values are rounded before rank-based statistics so that
    # mathematically tied quantities stay tied regardless of hardware.
    rank_decimals: int = 6

    # Reproducibility
    seed: int = 42
    annotation_seed: int = 2026

    # Case-study user shown in the paper (None = 16-bin BPD user with the
    # strongest D2-MMD correlation).
    case_study_user: str = "U_9aa3f03e41"
