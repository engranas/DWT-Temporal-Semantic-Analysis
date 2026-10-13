"""Calendar-time semantic trajectories and the discrete wavelet transform."""

import numpy as np
import pywt


def normalize_vector(x):
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x)
    return np.zeros_like(x) if norm < 1e-12 else x / norm


def normalize_rows(X):
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return X / norms


def cosine_distance(a, b):
    return 1.0 - float(np.dot(normalize_vector(a), normalize_vector(b)))


def fill_short_gaps(X, observed, max_gap):
    """Linearly interpolate interior runs of at most `max_gap` missing bins."""
    X, n = X.copy(), len(X)
    imputed = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if observed[i]:
            i += 1
            continue
        start = i
        while i < n and not observed[i]:
            i += 1
        gap = i - start
        if 0 < start and i < n and gap <= max_gap:
            for k in range(gap):
                alpha = (k + 1) / (gap + 1)
                X[start + k] = (1 - alpha) * X[start - 1] + alpha * X[i]
                imputed[start + k] = True
    return X, imputed


def longest_run(valid):
    best, start = None, None
    for i, v in enumerate(list(valid) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def build_trajectory(user_posts, post_embeddings, cfg):
    """Build a user's calendar-time trajectory.

    Post embeddings are averaged within each calendar bin and L2-normalized;
    single missing bins are interpolated; the longest contiguous segment is
    kept and trimmed to a multiple of 2**level bins, choosing the window with
    the most observed bins. Returns None if the result is shorter than
    `min_segment_bins`.
    """
    bin_ids = user_posts["bin_idx_global"].to_numpy()
    bins = np.arange(bin_ids.min(), bin_ids.max() + 1)
    X = np.full((len(bins), post_embeddings.shape[1]), np.nan)
    observed = np.zeros(len(bins), dtype=bool)
    for i, b in enumerate(bins):
        rows = np.where(bin_ids == b)[0]
        if len(rows):
            X[i] = normalize_vector(post_embeddings[rows].mean(axis=0))
            observed[i] = True

    X, imputed = fill_short_gaps(X, observed, cfg.max_interpolation_gap)
    segment = longest_run(~np.isnan(X).any(axis=1))
    if segment is None:
        return None
    s, e = segment
    block = 2 ** cfg.level
    usable = (e - s) // block * block
    if usable < cfg.min_segment_bins:
        return None
    scores = [observed[s + o: s + o + usable].sum() for o in range(e - s - usable + 1)]
    s += int(np.argmax(scores))
    keep = slice(s, s + usable)
    return {"embeddings": normalize_rows(X[keep]), "bin_ids": bins[keep],
            "observed": observed[keep], "imputed": imputed[keep]}


def dwt(E, cfg):
    return pywt.wavedec(E, cfg.wavelet, mode="periodization", level=cfg.level, axis=0)


def band_components(E, cfg):
    """Reconstruct each band in embedding space.

    R0 = A2, R1 = A2 + D2, R2 = A2 + D2 + D1 (= E).
    """
    coeffs = dwt(E, cfg)

    def only(k):
        kept = [c if i == k else np.zeros_like(c) for i, c in enumerate(coeffs)]
        return pywt.waverec(kept, cfg.wavelet, mode="periodization", axis=0)[: len(E)]

    A2, D2, D1 = only(0), only(1), only(2)
    return {"A2": A2, "D2": D2, "D1": D1, "R0": A2, "R1": A2 + D2, "R2": A2 + D2 + D1}


def energy_fractions(coeffs):
    energies = np.array([np.sum(c ** 2) for c in coeffs])
    total = energies.sum()
    fractions = energies / total if total > 0 else np.zeros_like(energies)
    return dict(zip(["A2", "D2", "D1"], fractions))
