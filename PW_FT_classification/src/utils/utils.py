import os
import pandas as pd
import cv2
import supervision as sv
from PIL import Image 
import numpy as np
from pathlib import Path

def save_crop_images(results, output_dir, original_csv_path, input_dir, overwrite=False):
    """
    Save cropped images based on the detection bounding boxes.

    Args:
        results (list):
            Detection results containing image ID and detections.
        output_dir (str):
            Directory to save the cropped images.
        original_csv_path (str):
            Path to the original CSV file.
        overwrite (bool):
            Whether overwriting existing image folders. Default to False.
    Return:
        new_csv_path (str):
            Path to the new CSV file.
    """
    assert isinstance(results, list)

    # Read the original CSV file
    original_df = pd.read_csv(original_csv_path)
    original_df['path'] = original_df['path'].apply(os.path.normpath)

    # Prepare a list to store new records for the new CSV
    new_records = []
    for entry in results:
        # Process the data if the name of the file is in the dataframe
        if os.path.normpath(entry["img_id"]).endswith(tuple(original_df['path'].values)):
            for i, (xyxy, _, _, cat, _, _) in enumerate(entry["detections"]):
                cropped_img = sv.crop_image(
                    image=np.array(Image.open(entry["img_id"]).convert("RGB")), xyxy=xyxy
                )
                new_img_name = "{}_{}_{}".format(
                        int(cat), i, os.path.basename(entry["img_id"]))
                
                img_id_parts=Path(entry["img_id"]).parts
                last_input_dir=Path(input_dir).parts[-1]
                relative_dir=Path(*img_id_parts[img_id_parts.index(last_input_dir)+1:-1])
                full_output_dir = os.path.join(output_dir, relative_dir)
                os.makedirs(full_output_dir, exist_ok=True)

                with sv.ImageSink(target_dir_path=full_output_dir, overwrite=overwrite) as sink:
                    sink.save_image(
                        image=cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR),
                        image_name=new_img_name
                        ),
                
                # Save the crop into a new csv
                image_name = entry['img_id']
                
                classification_id = original_df[original_df['path'].str.endswith(image_name.split(os.sep)[-1])]['classification'].values[0]
                classification_name = original_df[original_df['path'].str.endswith(image_name.split(os.sep)[-1])]['label'].values[0]
                # Add record to the new CSV data
                new_records.append({
                'path': os.path.join(relative_dir, new_img_name),
                'classification': classification_id,
                'label': classification_name
                })

    # Create a DataFrame from the new records
    new_df = pd.DataFrame(new_records)

    # Define the path for the new CSV file
    new_file_name = "{}_cropped.csv".format(original_csv_path.split(os.sep)[-1].split('.')[0])
    new_csv_path = os.path.join(output_dir, new_file_name)

    # Save the new DataFrame to CSV
    new_df.to_csv(new_csv_path, index=False)

    return new_csv_path

