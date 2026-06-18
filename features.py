"""
features.py — Feature extraction for the Phishing Email Detection Model

Combines two feature sources into one sklearn Pipeline:
  1. TF-IDF text features (word + bigram frequencies)
  2. Handcrafted numeric features (URL count, suspicious keywords,
     punctuation patterns, IP-based links, generic greetings, etc.)

The combined Pipeline is what gets trained, evaluated, and saved —
so predict.py only needs to load one object to go from raw email
text straight to a prediction.
"""

import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

# Keywords commonly seen in phishing / urgency-driven scam emails
SUSPICIOUS_KEYWORDS = [
    "verify", "suspended", "urgent", "immediately", "click here", "confirm",
    "password", "account", "billing", "update your", "act now", "limited time",
    "winner", "congratulations", "claim your", "free", "gift card", "refund",
    "security alert", "unauthorized", "locked", "expire", "final notice",
    "re-confirm", "validate", "restricted",
]

URL_REGEX = re.compile(r"https?://[^\s]+")
IP_URL_REGEX = re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
GENERIC_GREETING_REGEX = re.compile(
    r"^(dear customer|dear user|dear account holder|dear valued customer)",
    re.IGNORECASE,
)


def _extract_one(text: str) -> list:
    """Compute handcrafted numeric features for a single email's text."""
    text = text or ""
    lower = text.lower()
    words = text.split()

    urls = URL_REGEX.findall(text)
    num_links = len(urls)
    has_ip_url = 1 if IP_URL_REGEX.search(text) else 0
    suspicious_tld = 1 if any(
        d in lower for d in [".tk", ".xyz", ".ga", ".cn", ".info", ".ru"]
    ) else 0

    num_suspicious_keywords = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in lower)
    num_exclamations = text.count("!")
    num_dollar_signs = text.count("$")
    text_length = len(text)
    num_words = len(words)
    avg_word_length = (sum(len(w) for w in words) / num_words) if num_words else 0
    num_uppercase_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    has_generic_greeting = 1 if GENERIC_GREETING_REGEX.search(text.strip()) else 0

    return [
        num_links,
        has_ip_url,
        suspicious_tld,
        num_suspicious_keywords,
        num_exclamations,
        num_dollar_signs,
        text_length,
        num_words,
        avg_word_length,
        num_uppercase_words,
        has_generic_greeting,
    ]


FEATURE_NAMES = [
    "num_links", "has_ip_url", "suspicious_tld", "num_suspicious_keywords",
    "num_exclamations", "num_dollar_signs", "text_length", "num_words",
    "avg_word_length", "num_uppercase_words", "has_generic_greeting",
]


class HandcraftedFeaturesExtractor(BaseEstimator, TransformerMixin):
    """sklearn-compatible transformer: raw email text -> numeric feature matrix."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([_extract_one(t) for t in X], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)


def build_pipeline_features() -> FeatureUnion:
    """Combined TF-IDF + handcrafted-feature extractor (no classifier)."""
    return FeatureUnion([
        ("tfidf", TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )),
        ("handcrafted", Pipeline([
            ("extract", HandcraftedFeaturesExtractor()),
            ("scale", StandardScaler()),
        ])),
    ])
