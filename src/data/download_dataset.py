from pathlib import Path
import shutil

import kagglehub


DATASET_HANDLE = "dollyprajapati182/balanced-raf-db-dataset-7575-grayscale"
TARGET_FOLDER = "data/raw/balanced-raf-db-dataset-7575-grayscale"


def copy_dataset(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    destination = project_root / TARGET_FOLDER
    source = Path(kagglehub.dataset_download(DATASET_HANDLE))

    print(f"Dataset cache path: {source}")
    print(f"Copying dataset to: {destination}")

    copy_dataset(source, destination)

    print("Dataset copied successfully.")


if __name__ == "__main__":
    main()
