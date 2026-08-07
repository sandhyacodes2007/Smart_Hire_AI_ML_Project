"""
Cleans the raw Resume and Naukri CSVs and writes model-ready files.
Run from the project root:
    python -m src.data.preprocess
"""
import pandas as pd

from src import config
from src.features.text_features import clean_text


def build_resumes_clean():
    df = pd.read_csv(config.RESUME_DATASET_FILE)
    df = df.dropna(subset=["Resume_str", "Category"]).copy()
    df["clean_text"] = df["Resume_str"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]

    out = df[["ID", "Category", "clean_text"]].rename(columns={"clean_text": "Resume"})
    out["Resume_str"] = df["Resume_str"]

    out.to_csv(config.RESUMES_CLEAN_FILE, index=False)
    print(f"Saved {len(out)} cleaned resumes -> {config.RESUMES_CLEAN_FILE}")
    return out


def build_job_corpus():
    df = pd.read_csv(config.NAUKRI_DATASET_FILE)
    text_cols = ["Job Title", "Key Skills", "Role Category", "Functional Area", "Industry", "Role"]
    present_cols = [c for c in text_cols if c in df.columns]
    df = df.dropna(subset=["Job Title"]).copy()

    combined = df[present_cols].fillna("").agg(" ".join, axis=1)
    df["clean_text"] = combined.apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]

    df = df.rename(columns={"Uniq Id": "job_id"})
    keep_cols = ["job_id", "Job Title", "Key Skills", "Role Category",
                 "Functional Area", "Industry", "Role", "clean_text"]
    out = df[[c for c in keep_cols if c in df.columns]]

    out.to_csv(config.JOB_CORPUS_FILE, index=False)
    print(f"Saved {len(out)} jobs -> {config.JOB_CORPUS_FILE}")
    return out


if __name__ == "__main__":
    build_resumes_clean()
    build_job_corpus()
    print("\nDone. Check data/processed/resumes_clean.csv and data/interim/job_corpus.csv")