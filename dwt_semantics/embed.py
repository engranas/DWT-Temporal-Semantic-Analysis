"""Post embeddings and the corpus phrase dictionary."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from .text import ARABIC_STOPWORDS, CONDITION_REGEX, TOKEN_PATTERN


def load_encoder(cfg):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(cfg.model_name)


def encode(model, texts, cfg, progress=True):
    return model.encode(list(texts), normalize_embeddings=True, batch_size=cfg.batch_size,
                        show_progress_bar=progress, convert_to_numpy=True)


def build_phrase_dictionary(texts, model, cfg, progress=True):
    """1-3-gram phrase dictionary, its occurrence matrix and phrase embeddings.

    Phrases containing a condition mention are excluded from interpretation.
    """
    vectorizer = CountVectorizer(
        ngram_range=(1, 3), min_df=cfg.phrase_min_df, max_df=cfg.phrase_max_df,
        max_features=cfg.max_phrases, stop_words=list(ARABIC_STOPWORDS),
        token_pattern=TOKEN_PATTERN,
    )
    occurrence = vectorizer.fit_transform(texts).tocsr()
    phrases = np.asarray(vectorizer.get_feature_names_out())
    valid_ids = np.where([CONDITION_REGEX.search(p) is None for p in phrases])[0]
    embeddings = encode(model, phrases, cfg, progress)
    return {"phrases": phrases, "occurrence": occurrence,
            "valid_ids": valid_ids, "embeddings": embeddings}
