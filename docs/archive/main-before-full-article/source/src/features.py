"""Feature engineering for the classical TF-IDF models.

Logistic regression and LinearSVC share one text pipeline: word and
bigram TF-IDF, then the classifier. ``src.sklearn_models`` builds that
pipeline here, and ``src.train`` trains those models, so the vectorizer
settings stay in one place.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.config import TFIDF_MIN_DF, TFIDF_NGRAM_RANGE


def make_tfidf_vectorizer() -> TfidfVectorizer:
    """Word and bigram TF-IDF used by both classical models."""
    return TfidfVectorizer(ngram_range=TFIDF_NGRAM_RANGE, min_df=TFIDF_MIN_DF)


def build_tfidf_pipeline(classifier) -> Pipeline:
    """TF-IDF vectorizer followed by the given sklearn classifier."""
    return Pipeline(
        steps=[
            ("tfidf", make_tfidf_vectorizer()),
            ("clf", classifier),
        ]
    )
