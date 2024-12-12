import pandas as pd
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import logging
from pprint import pprint

logging.basicConfig(level=logging.INFO)


def extract_url_genus_from_record(record, taxonomy):

    url = record["image"].replace(
        "lilablobssc.blob.core.windows.net/",
        "lilawildlife.blob.core.windows.net/lila-wildlife/",
    )

    genus_id = record["annotations"]["taxonomy"][0]["genus"]
    genus_id = int(genus_id) if genus_id is not None else None
    
    if genus_id is not None:
        genus_name = taxonomy["genus"].int2str(genus_id)
    else:
        genus_name = "None"

    return url, genus_name

def main():
    cn2tax_df = pd.read_json(
        "https://huggingface.co/datasets/society-ethics/lila_camera_traps/raw/main/data/common_names_to_tax.json",
        lines=True,
    )
    cn2tax_df = cn2tax_df.set_index("common_name")
    

    dataset = load_dataset(
        "society-ethics/lila_camera_traps", "Orinoquia Camera Traps", split="train"
    )
    
    taxonomy = dataset.features["annotations"].feature["taxonomy"]

    # get 100 random items from the dataset
    sample = dataset.shuffle().select(range(100))    

    # create a list of dictionaries with the genus and url of each record
    for record in sample:
        url, genus = extract_url_genus_from_record(record, taxonomy)
        print("Genus:", genus, "URL:", url)


if __name__ == "__main__":
    main()