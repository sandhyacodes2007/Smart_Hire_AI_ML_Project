"""
Downloads the datasets SmartHire needs into data/raw/.
Requires a Kaggle API token already set up on this machine.
"""
import shutil
import zipfile

from src import config

DATASETS = [
    ("snehaanbhawal/resume-dataset", config.RESUME_DATASET_FILE),
    ("promptcloud/jobs-on-naukricom", config.NAUKRI_DATASET_FILE),
]


def download_dataset(slug: str, dest_path):
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {slug} ...")
    tmp_dir = config.RAW_DIR / "_tmp_download"
    tmp_dir.mkdir(exist_ok=True)

    api.dataset_download_files(slug, path=str(tmp_dir), unzip=False)

    zips = list(tmp_dir.glob("*.zip"))
    if not zips:
        print(f"  No zip found for {slug} — check the dataset slug is correct.")
        return
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(tmp_dir)

    csvs = list(tmp_dir.glob("*.csv")) + list(tmp_dir.glob("**/*.csv"))
    if not csvs:
        print(f"  No CSV found after extracting {slug}.")
        return

    shutil.copy(csvs[0], dest_path)
    print(f"  Saved -> {dest_path}")

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    for slug, dest_path in DATASETS:
        if dest_path.exists():
            print(f"{dest_path.name} already exists, skipping.")
            continue
        try:
            download_dataset(slug, dest_path)
        except Exception as e:
            print(f"Failed to download {slug}: {e}")
            print("  Check the dataset slug on kaggle.com and your kaggle.json credentials.")

    print("\nDone. Check data/raw/ for the downloaded files.")