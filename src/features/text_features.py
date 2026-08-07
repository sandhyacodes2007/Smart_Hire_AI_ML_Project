"""
Text cleaning and TF-IDF feature helpers, shared by the classifier
and the recommender so both work on identically preprocessed text.
"""
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config

_URL_RE = re.compile(r"http\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip urls/emails/punctuation/digits, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def build_vectorizer(max_features=None, ngram_range=None) -> TfidfVectorizer:
    """Create a fresh TF-IDF vectorizer using the project defaults."""
    return TfidfVectorizer(
        max_features=max_features or config.TFIDF_MAX_FEATURES,
        ngram_range=ngram_range or config.TFIDF_NGRAM_RANGE,
        stop_words="english",
        sublinear_tf=True,
    )


def save_vectorizer(vectorizer, path):
    joblib.dump(vectorizer, path)


def load_vectorizer(path):
    return joblib.load(path)