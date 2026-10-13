# Temporal Semantic Analysis of Arabic Mental-Health Discourse using Discrete Wavelet Transform

Official code for the MultiPsyche 2026 paper (AACL-IJCNLP 2026 workshop):

> Rana Aref Salama, Amal Abdullah Alqahtani, Mona T. Diab, and Abdou Youssef. 2026.
> *Temporal Semantic Analysis of Arabic Mental-Health Discourse using Discrete Wavelet Transform.*

The framework models each user's posting history as a calendar-time semantic trajectory, applies the Discrete Wavelet Transform (DWT) along the time axis to separate finer and broader semantic variation, and interprets each temporal scale through **semantic gain**: the change in a phrase's similarity to the reconstructed trajectory when that scale's detail is added.

## Method overview

1. **Trajectories.** Posts are embedded with `paraphrase-multilingual-mpnet-base-v2` and averaged within 14-day calendar bins. Single missing bins are interpolated, and the longest contiguous segment (at least 8 bins, trimmed to a multiple of 4) is kept.
2. **DWT.** A level-2 Haar DWT along time gives A2 (more than 8 weeks), D2 (about 8 weeks) and D1 (about 4 weeks). These are reconstructed progressively: R0 = A2, R1 = A2 + D2, R2 = A2 + D2 + D1.
3. **Semantic gain.** gain_D2(w, t) = cos(w, R1[t]) − cos(w, R0[t]) and gain_D1(w, t) = cos(w, R2[t]) − cos(w, R1[t]). Candidate phrases (1–3-grams) must occur in the user's posts **outside** the time support of t, with a one-bin buffer.
4. **Evaluation.**
   - Agreement with adjacent-bin cosine change and RBF-MMD.
   - A chronological permutation test.
   - D1/D2 phrase distinctiveness (Jaccard overlap, profile distance and a permutation test).
   - High-gain events vs. matched low-gain controls (phrase coherence, representation shift and out-of-window grounding).

The temporal components operate on sentence embeddings and are language-agnostic. The Arabic-specific parts live in `dwt_semantics/text.py`:
- condition-name masking that covers Arabic morphological variants;
- an Arabic-script-aware tokenizer;
- an Arabic stopword list.

## Repository structure

```
dwt_semantics/
  config.py       experiment settings (defaults = paper settings)
  text.py         cleaning, condition masking, Arabic stopwords, user hashing
  data.py         corpus loading, filtering, calendar binning
  embed.py        sentence embeddings and phrase dictionary
  trajectory.py   calendar trajectories, DWT, progressive reconstruction, energy
  analysis.py     semantic gain, change measures, permutation tests, events
  stats.py        summary statistics
  plots.py        figures
  pipeline.py     end-to-end run
run_pipeline.py   command-line entry point
notebooks/reproduce.ipynb   Colab notebook
```

## Installation

```bash
git clone https://github.com/<ORG>/<REPO>.git
cd <REPO>
pip install -r requirements.txt
```

Tested with Python 3.10 to 3.13.

## Data

The analysis uses the Arabic mental-health discourse corpus of Alqahtani et al. (2026), collected from condition-specific Arabic-language X Communities. In line with X's terms of service, **the corpus is not distributed with this repository**. To request access, contact the corpus authors.

Expected input: `tweets_preprocessed_9582.csv`, with the columns `community`, `created_at`, `text` and `user_screen_name`.

## Reproducing the results

```bash
python run_pipeline.py --data data/tweets_preprocessed_9582.csv --out outputs
```

To run in Colab, open `notebooks/reproduce.ipynb`. A full run takes about 5 minutes on a GPU (Colab T4) or 10–15 minutes on a CPU; most of that time goes to embedding.

### Outputs

| Path | Contents |
|---|---|
| `outputs/results/statistics.csv` | All reported statistics |
| `outputs/results/attrition.csv` | Posts and users retained at each filtering step |
| `outputs/results/phrase_gain_by_band.csv` | Phrase-level semantic gain aggregated over users |
| `outputs/figures/results_summary.png` | Energy, change-measure correlations, coherence and grounding |
| `outputs/figures/case_study.png` | Boundary scores for the case-study user |
| `outputs/private/` | Per-user tables, event tables with post text, and a blinded annotation packet. **Do not share; git-ignored.** |

### Statistical conventions

- **Paired comparisons:** Wilcoxon signed-rank tests on user-level values. Event measures are summarized by their median within each user, scale and event type.
- **One-sided tests:** tests with a pre-specified direction (high-gain > control, D1 > D2 gain, observed < shuffled energy) are one-sided. The two-sided p-value is reported alongside.
- **Condition comparisons:** two-sided Mann–Whitney U tests with Holm correction. Per-user shuffle tests use Benjamini–Hochberg correction.
- **Out-of-window grounding:** the mean cosine similarity between the centroid of an event's gained phrases and the three most similar posts outside the event window.
- **Duplicate D2 events:** with a level-2 Haar transform, the D2 reconstruction is constant within each pair of bins, so paired D2 events are identical. `statistics.csv` reports results both with all events and with unique D2 events only.

### Reproducibility

- **Random seeds:** all random procedures are seeded (`seed = 42`, annotation shuffle `2026`). The single random generator is used in a fixed order: chronological shuffles first, then the D1/D2 permutations.
- **Rounding before ranking:** values are rounded to 6 decimals before rank-based statistics. Some quantities are tied by construction (for example, the cosine change into and out of an interpolated bin). Rounding keeps those ties exact, so results don't depend on floating-point differences between GPU and CPU or across library versions.

### Configuration

Every setting is in `dwt_semantics/config.py`. Common variations can be set from the command line:

```bash
python run_pipeline.py --data ... --bin-days 21           # temporal resolution
python run_pipeline.py --data ... --wavelet db2           # wavelet family
python run_pipeline.py --data ... --model sentence-transformers/LaBSE   # encoder
```

## Ethical considerations

- **Community labels:** these are discourse contexts, not clinical diagnoses. The framework does not infer diagnosis, severity or individual mental-health status, and must not be used for clinical decisions or to profile or monitor individuals.
- **User identifiers:** these are hashed before analysis. Hashes of known usernames can be recomputed, so the per-user outputs in `outputs/private/` must not be shared.

## Citation

```bibtex
@inproceedings{salama-etal-2026-temporal,
  title     = {Temporal Semantic Analysis of Arabic Mental-Health Discourse using Discrete Wavelet Transform},
  author    = {Salama, Rana Aref and Alqahtani, Amal Abdullah and Diab, Mona T. and Youssef, Abdou},
  booktitle = {Proceedings of the First Workshop on Multimodal, Multilingual, and Multicultural Mental Health and Psychotherapy (MultiPsyche)},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`). The license covers the code only, not the corpus.
