"""Text cleaning, condition masking and user anonymization."""

import hashlib
import re

# Condition mentions (English and Arabic morphological/orthographic variants)
# are masked so that the community label cannot be read directly off the text.
CONDITION_REGEX = re.compile(
    r"\bbpd\b|\bbipolar\b|\bbipolar disorder\b|\bborderline personality disorder\b|\bborderline\b|"
    r"اضطراب\s*الشخصي[ةه]?\s*الحدي[ةه]?|اضطراب\s*الشخصية\s*الحدية|الشخصي[ةه]?\s*الحدي[ةه]?|"
    r"الشخصية\s*الحدية|الحدي[ةه]|الحدّية|الحدّي|"
    r"اضطراب\s*ثنائي\s*القطب|ثنائي\s*القطب|ثنائية\s*القطب|ثنايي\s*القطب",
    flags=re.IGNORECASE,
)

# Small, conservative Arabic stopword list (MSA and frequent dialectal forms).
ARABIC_STOPWORDS = {
    "في", "من", "على", "الى", "إلى", "عن", "ما", "مع", "هذا", "هذه", "هو", "هي", "هم",
    "انا", "أنا", "انت", "أنت", "ان", "إن", "أن", "كان", "كانت", "يكون", "كل", "او", "أو",
    "ثم", "لكن", "لما", "ليه", "كيف", "اي", "أي", "بس", "عشان", "اللي", "الي", "والله",
    "الله", "مرة", "شي", "شيء", "كذا", "حتى", "بعد", "قبل", "اذا", "إذا", "ولا", "مو",
    "مش", "فيه", "فيها", "عندي", "عندك", "احنا", "نحن", "كنت", "صار", "يصير", "لان",
    "لأن", "انه", "إنه", "انها", "إنها",
}

# Arabic-script-aware token pattern: letter sequences of length >= 2.
TOKEN_PATTERN = r"(?u)\b[^\W\d_]{2,}\b"


def clean_text(text):
    text = CONDITION_REGEX.sub(" ", str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\[(USER|URL|CONDITION)\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s؀-ۿ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def user_hash(name):
    return "U_" + hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:10]
