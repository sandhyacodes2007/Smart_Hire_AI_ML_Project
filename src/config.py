from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RESUME_DATASET_FILE = RAW_DIR / "resume_dataset.csv"
NAUKRI_DATASET_FILE = RAW_DIR / "naukri_jobs.csv"

JOB_CORPUS_FILE = INTERIM_DIR / "job_corpus.csv"
JOBS_CLEAN_FILE = PROCESSED_DIR / "jobs_clean.csv"
RESUMES_CLEAN_FILE = PROCESSED_DIR / "resumes_clean.csv"

TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

for d in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)