from pathlib import Path
from typing import Optional

import requests

from yolo.config import DatasetConfig


def prepare_dataset(dataset_cfg: DatasetConfig, task: str):
    """
    Prepares dataset by downloading and unzipping if necessary.
    """
    data_dir = Path(dataset_cfg.path)
    for data_type, settings in dataset_cfg.auto_download.items():
        base_url = settings["base_url"]
        for dataset_type, dataset_args in settings.items():
            if dataset_type != "annotations" and dataset_cfg.get(task, task) != dataset_type:
                continue
            file_name = f"{dataset_args.get('file_name', dataset_type)}.zip"
            url = f"{base_url}{file_name}"
            local_zip_path = data_dir / file_name
            extract_to = data_dir / data_type if data_type != "annotations" else data_dir
            final_place = extract_to / dataset_type

            final_place.mkdir(parents=True, exist_ok=True)
            if check_files(final_place, dataset_args.get("file_num")):
                raise RuntimeError(f"Error verifying the {dataset_type} dataset after extraction.")
                continue

            if not local_zip_path.exists():
                download_file(url, local_zip_path)
            unzip_file(local_zip_path, extract_to)

            if not check_files(final_place, dataset_args.get("file_num")):
                raise RuntimeError(f"Error verifying the {dataset_type} dataset after extraction.")


def prepare_weight(download_link: Optional[str] = None, weight_path: Path = Path("v9-c.pt")):
    weight_name = weight_path.name
    if download_link is None:
        download_link = "https://github.com/MultimediaTechLab/YOLO/releases/download/v1.0-alpha/"
    weight_link = f"{download_link}{weight_name}"

    if not weight_path.parent.is_dir():
        weight_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        download_file(weight_link, weight_path)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download the weight file: {e}")

