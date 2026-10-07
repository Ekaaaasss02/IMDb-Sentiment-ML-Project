import re
import html
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def ensure_nltk_resources():
    """Download the small NLTK resources needed by this project if absent."""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)


ensure_nltk_resources()

_STOP_WORDS = set(stopwords.words("english"))
_STEMMER = PorterStemmer()


def clean_text(text: str, remove_stopwords: bool = True, stemming: bool = True) -> str:
    """
    Clean one IMDb review.

    The paper describes removing punctuation, line breaks, numbers and
    stop words, lower-casing, and normalizing words to their roots.
    """
    text = "" if text is None else str(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)       # HTML tags
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)      # punctuation/numbers
    tokens = text.split()

    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOP_WORDS]

    if stemming:
        tokens = [_STEMMER.stem(t) for t in tokens]

    return " ".join(tokens)


def clean_corpus(texts: Iterable[str]):
    return [clean_text(x) for x in texts]
