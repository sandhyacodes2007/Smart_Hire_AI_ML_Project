"""
Job recommender: matches a resume against the job corpus using
TF-IDF + cosine similarity.
"""
import joblib                                              # to load saved model files
import pandas as pd                                        # to load/handle the jobs DataFrame
from sklearn.metrics.pairwise import cosine_similarity     # similarity scoring

from src import config                                     # project paths/settings
from src.features.text_features import clean_text          # text cleaning function

# Load these ONCE when the module is imported, not every time the function runs —
# loading from disk is slow, so we don't want to repeat it per resume
_vectorizer = joblib.load(config.MODELS_DIR / "tfidf_vectorizer.pkl")
_jobs = pd.read_csv(config.JOB_CORPUS_FILE)
_job_tfidf = _vectorizer.transform(_jobs["clean_text"])


def recommend_jobs(resume_text, top_n=5):
    """
    Given raw resume text, return the top_n most similar job postings.
    """
    cleaned = clean_text(resume_text)                       # apply same cleaning as training data
    vec = _vectorizer.transform([cleaned])                  # convert resume to TF-IDF vector
    scores = cosine_similarity(vec, _job_tfidf).flatten()   # similarity vs every job
    idx = scores.argsort()[::-1][:top_n]                    # top N indices, highest first

    result = _jobs.iloc[idx][["Job Title", "Key Skills"]].copy()  # pull matching job rows
    result["score"] = scores[idx]                            # attach similarity scores
    return result.reset_index(drop=True)                     # clean up index for display