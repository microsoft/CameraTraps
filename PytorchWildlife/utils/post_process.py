# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Post-processing functions."""

import os
import numpy as np
import json
import cv2
from PIL import Image
import supervision as sv
import shutil
from pathlib import Path

__all__ = [
    "save_detection_images",
    "save_detection_images_dots",
    "save_crop_images",
    "save_detection_json",
    "save_detection_json_as_dots",
    "save_detection_classification_json",
    "save_detection_timelapse_json",
    "save_detection_classification_timelapse_json",
    "detection_folder_separation"
]


def save_detection_images(results, output_dir, input_dir = None, overwrite=False):
    """
    Save detected images with bounding boxes and labels annotated.

    Args:
        results (list or dict):
            Detection results containing image ID, detections, and labels.
        output_dir (str):
            Directory to save the annotated images.
        input_dir (str):
            Directory containing the input images. Default to None.
        overwrite (bool):
            Whether overwriting existing image folders. Default to False.
    """
    box_annotator = sv.BoxAnnotator(thickness=4)
    lab_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_thickness=4, text_scale=2)
    os.makedirs(output_dir, exist_ok=True)

    with sv.ImageSink(target_dir_path=output_dir, overwrite=overwrite) as sink: 
        if isinstance(results, list):
            for entry in results:
                annotated_img = lab_annotator.annotate(
                    scene=box_annotator.annotate(
                        scene=np.array(Image.open(entry["img_id"]).convert("RGB")),
                        detections=entry["detections"],
                    ),
                    detections=entry["detections"],
                    labels=entry["labels"],
                )
                if input_dir:
                    relative_path = os.path.relpath(entry["img_id"], input_dir)
                    save_path = os.path.join(output_dir, relative_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True) 
                    image_name = relative_path 
                else:
                    image_name = os.path.basename(entry["img_id"])
                sink.save_image(
                    image=cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR), image_name=image_name
                )
        else:
            annotated_img = lab_annotator.annotate(
                scene=box_annotator.annotate(
                    scene=np.array(Image.open(results["img_id"]).convert("RGB")),
                    detections=results["detections"],
                ),
                detections=results["detections"],
                labels=results["labels"],
            )

            sink.save_image(
                image=cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR), image_name=os.path.basename(results["img_id"])
            )

def save_detection_images_dots(results, output_dir, input_dir = None, overwrite=False):
    """
    Save detected images with bounding boxes and labels annotated.

    Args:
        results (list or dict):
            Detection results containing image ID, detections, and labels.
        output_dir (str):
            Directory to save the annotated images.
        input_dir (str):
            Directory containing the input images. Default to None.
        overwrite (bool):
            Whether overwriting existing image folders. Default to False.
    """
    dot_annotator = sv.DotAnnotator(radius=6)  
    lab_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_RIGHT)   
    os.makedirs(output_dir, exist_ok=True)
    
    with sv.ImageSink(target_dir_path=output_dir, overwrite=overwrite) as sink:
        if isinstance(results, list):
            for entry in results:
                annotated_img = lab_annotator.annotate(
                    scene=dot_annotator.annotate(
                        scene=np.array(Image.open(entry["img_id"]).convert("RGB")),
                        detections=entry["detections"],
                    ),
                    detections=entry["detections"],
                    labels=entry["labels"],
                )
                if input_dir:
                    relative_path = os.path.relpath(entry["img_id"], input_dir)
                    save_path = os.path.join(output_dir, relative_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True) 
                    image_name = relative_path 
                else:
                    image_name = os.path.basename(entry["img_id"])
                sink.save_image(
                    image=cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR), image_name=image_name
                )
        else:
            annotated_img = lab_annotator.annotate(
                scene=dot_annotator.annotate(
                    scene=np.array(Image.open(results["img_id"]).convert("RGB")),
                    detections=results["detections"],
                ),
                detections=results["detections"],
                labels=results["labels"],
            )
            sink.save_image(
                image=cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR), image_name=os.path.basename(results["img_id"])
            )


# !!! Output paths need to be optimized !!!
def save_crop_images(results, output_dir, input_dir = None, overwrite=False):
    """
    Save cropped images based on the detection bounding boxes.

    Args:
        results (list):
            Detection results containing image ID and detections.
        output_dir (str):
            Directory to save the cropped images.
        input_dir (str):
            Directory containing the input images. Default to None.
        overwrite (bool):
            Whether overwriting existing image folders. Default to False.
    """

    os.makedirs(output_dir, exist_ok=True)

    with sv.ImageSink(target_dir_path=output_dir, overwrite=overwrite) as sink:
        if isinstance(results, list):
            for entry in results:
                for i, (xyxy, cat) in enumerate(zip(entry["detections"].xyxy, entry["detections"].class_id)):
                    cropped_img = sv.crop_image(
                        image=np.array(Image.open(entry["img_id"]).convert("RGB")), xyxy=xyxy
                    )
                    if input_dir:
                        relative_path = os.path.relpath(entry["img_id"], input_dir)
                        save_path = os.path.join(output_dir, relative_path)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True) 
                        image_name = os.path.join(os.path.dirname(relative_path), "{}_{}_{}".format(int(cat), i, os.path.basename(entry["img_id"])))
                    else:
                        image_name = "{}_{}_{}".format(int(cat), i, os.path.basename(entry["img_id"]))
                    sink.save_image(
                        image=cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR),
                        image_name=image_name,
                    )
        else:
            for i, (xyxy, cat) in enumerate(zip(results["detections"].xyxy, results["detections"].class_id)):
                cropped_img = sv.crop_image(
                    image=np.array(Image.open(results["img_id"]).convert("RGB")), xyxy=xyxy
                )
                sink.save_image(
                    image=cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR),
                    image_name="{}_{}_{}".format(int(cat), i, os.path.basename(results["img_id"]),
                ))

def save_detection_json(det_results, output_dir, categories=None, exclude_category_ids=[], exclude_file_path=None):
    """
    Save detection results to a JSON file.

    Args:
        results (list):
            Detection results containing image ID, bounding boxes, category, and confidence.
        output_dir (str):
            Path to save the output JSON file.
        categories (list, optional):
            List of categories for detected objects. Defaults to None.
        exclude_category_ids (list, optional):
            List of category IDs to exclude from the output. Defaults to []. Category IDs can be found in the definition of each models.
        exclude_file_path (str, optional):
            We can exclude the some path sections from the image ID. Defaults to None.
    """
    json_results = {"annotations": [], "categories": categories}

    for det_r in det_results:

        # Category filtering
        img_id = det_r["img_id"]
        category = det_r["detections"].class_id

        bbox = det_r["detections"].xyxy.astype(int)[~np.isin(category, exclude_category_ids)]
        confidence =  det_r["detections"].confidence[~np.isin(category, exclude_category_ids)]
        category = category[~np.isin(category, exclude_category_ids)]

        # if not all([x in exclude_category_ids for x in category]):
        json_results["annotations"].append(
            {
                "img_id": img_id.replace(exclude_file_path + os.sep, '') if exclude_file_path else img_id,
                "bbox": bbox.tolist(),
                "category": category.tolist(),
                "confidence": confidence.tolist(),
            }
        )

    with open(output_dir, "w") as f:
        json.dump(json_results, f, indent=4)

def save_detection_json_as_dots(det_results, output_dir, categories=None, exclude_category_ids=[], exclude_file_path=None):
    """
    Save detection results to a JSON file in dots format.

    Args:
        results (list):
            Detection results containing image ID, bounding boxes, category, and confidence.
        output_dir (str):
            Path to save the output JSON file.
        categories (list, optional):
            List of categories for detected objects. Defaults to None.
        exclude_category_ids (list, optional):
            List of category IDs to exclude from the output. Defaults to []. Category IDs can be found in the definition of each models.
        exclude_file_path (str, optional):
            We can exclude the some path sections from the image ID. Defaults to None.
    """
    json_results = {"annotations": [], "categories": categories}

    for det_r in det_results:

        # Category filtering
        img_id = det_r["img_id"]
        category = det_r["detections"].class_id

        bbox = det_r["detections"].xyxy.astype(int)[~np.isin(category, exclude_category_ids)]
        dot = np.array([[np.mean(row[::2]), np.mean(row[1::2])] for row in bbox])
        confidence =  det_r["detections"].confidence[~np.isin(category, exclude_category_ids)]
        category = category[~np.isin(category, exclude_category_ids)]

        # if not all([x in exclude_category_ids for x in category]):
        json_results["annotations"].append(
            {
                "img_id": img_id.replace(exclude_file_path + os.sep, '') if exclude_file_path else img_id,
                "dot": dot.tolist(),
                "category": category.tolist(),
                "confidence": confidence.tolist(),
            }
        )

    with open(output_dir, "w") as f:
        json.dump(json_results, f, indent=4)


def save_detection_timelapse_json(
    det_results, output_dir, categories=None,
    exclude_category_ids=[], exclude_file_path=None, info={"detector": "megadetector_v5"}
    ):
    """
    Save detection results to a JSON file.

    Args:
        results (list):
            Detection results containing image ID, bounding boxes, category, and confidence.
        output_dir (str):
            Path to save the output JSON file.
        categories (list, optional):
            List of categories for detected objects. Defaults to None.
        exclude_category_ids (list, optional):
            List of category IDs to exclude from the output. Defaults to []. Category IDs can be found in the definition of each models.
        exclude_file_path (str, optional):
            Some time, Timelapse has path issues. We can exclude the some path sections from the image ID. Defaults to None.
        detector (dict, optional):
            Default Timelapse info. Defaults to {"detector": "megadetector_v5}.
    """

    json_results = {
        "info": info,
        "detection_categories": categories,
        "images": []
    }

    for det_r in det_results:

        img_id = det_r["img_id"]
        category_id_list = det_r["detections"].class_id

        bbox_list = det_r["detections"].xyxy.astype(int)[~np.isin(category_id_list, exclude_category_ids)]
        confidence_list =  det_r["detections"].confidence[~np.isin(category_id_list, exclude_category_ids)]
        normalized_bbox_list = np.array(det_r["normalized_coords"])[~np.isin(category_id_list, exclude_category_ids)]
        category_id_list = category_id_list[~np.isin(category_id_list, exclude_category_ids)]

        # if not all([x in exclude_category_ids for x in category_id_list]):
        image_annotations = {
            "file": img_id.replace(exclude_file_path + os.sep, '') if exclude_file_path else img_id,
            "max_detection_conf": float(max(confidence_list)) if len(confidence_list) > 0 else '',
            "detections": []
        }
        for i in range(len(bbox_list)):
            normalized_bbox = [float(y) for y in normalized_bbox_list[i]]
            detection = {
                "category": str(category_id_list[i]),
                "conf": float(confidence_list[i]),
                "bbox": [normalized_bbox[0], normalized_bbox[1], normalized_bbox[2]-normalized_bbox[0], normalized_bbox[3]-normalized_bbox[1]],
                "classifications": []
            }

            image_annotations["detections"].append(detection)

        json_results["images"].append(image_annotations)

    with open(output_dir, "w") as f:
        json.dump(json_results, f, indent=4)


def save_detection_classification_json(
    det_results, clf_results, output_path, det_categories=None, clf_categories=None, exclude_file_path=None
):
    """
    Save classification results to a JSON file.

    Args:
        det_results (list):
            Detection results containing image ID, bounding boxes, detection category, and confidence.
        clf_results (list):
            classification results containing image ID, classification category, and confidence.
        output_dir (str):
            Path to save the output JSON file.
        det_categories (list, optional):
            List of categories for detected objects. Defaults to None.
        clf_categories (list, optional):
            List of categories for classified objects. Defaults to None.
        exclude_file_path (str, optional):
            We can exclude the some path sections from the image ID. Defaults to None.
    """

    json_results = {
        "annotations": [],
        "det_categories": det_categories,
        "clf_categories": clf_categories,
    }

    with open(output_path, "w") as f:
        counter = 0
        for det_r in det_results:
            clf_categories = []
            clf_confidence = []
            for i in range(counter, len(clf_results)):
                clf_r = clf_results[i]
                if clf_r["img_id"] == det_r["img_id"]:
                    clf_categories.append(clf_r["class_id"])
                    clf_confidence.append(clf_r["confidence"])
                    counter += 1
                else:
                    break

            json_results["annotations"].append(
                {
                    "img_id": str(det_r["img_id"]).replace(exclude_file_path + os.sep, '') if exclude_file_path else str(det_r["img_id"]),
                    "bbox": [
                        [int(x) for x in sublist]
                        for sublist in det_r["detections"].xyxy.astype(int).tolist()
                    ],
                    "det_category": [
                        int(x) for x in det_r["detections"].class_id.tolist()
                    ],
                    "det_confidence": [
                        float(x) for x in det_r["detections"].confidence.tolist()
                    ],
                    "clf_category": [int(x) for x in clf_categories],
                    "clf_confidence": [float(x) for x in clf_confidence],
                }
            )
        json.dump(json_results, f, indent=4)


def save_detection_classification_timelapse_json(
    det_results, clf_results, output_path, det_categories=None, clf_categories=None,
    exclude_file_path=None, info={"detector": "megadetector_v5"}
):
    """
    Save detection and classification results to a JSON file in the specified format.

    Args:
        det_results (list):
            Detection results containing image ID, bounding boxes, detection category, and confidence.
        clf_results (list):
            Classification results containing image ID, classification category, and confidence.
        output_path (str):
            Path to save the output JSON file.
        det_categories (dict, optional):
            Dictionary of categories for detected objects. Defaults to None.
        clf_categories (dict, optional):
            Dictionary of categories for classified objects. Defaults to None.
        exclude_file_path (str, optional):
            We can exclude the some path sections from the image ID. Defaults to None.
    """
    json_results = {
        "info": info,
        "detection_categories": det_categories,
        "classification_categories": clf_categories,
        "images": []
    }

    for det_r in det_results:
        image_annotations = {
            "file": str(det_r["img_id"]).replace(exclude_file_path + os.sep, '') if exclude_file_path else str(det_r["img_id"]),
            "max_detection_conf": float(max(det_r["detections"].confidence)) if len(det_r["detections"].confidence) > 0 else '',
            "detections": []
        }

        for i in range(len(det_r["detections"])):
            det = det_r["detections"][i]
            normalized_bbox = [float(y) for y in det_r["normalized_coords"][i]]
            detection = {
                "category": str(det.class_id[0]),
                "conf": float(det.confidence[0]),
                "bbox": [normalized_bbox[0], normalized_bbox[1], normalized_bbox[2]-normalized_bbox[0], normalized_bbox[3]-normalized_bbox[1]],
                "classifications": []
            }

            # Find classifications for this detection
            for clf_r in clf_results:
                if clf_r["img_id"] == det_r["img_id"]:
                    detection["classifications"].append([str(clf_r["class_id"]), float(clf_r["confidence"])])

            image_annotations["detections"].append(detection)

        json_results["images"].append(image_annotations)

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=4)


def detection_folder_separation(json_file, img_path, destination_path, confidence_threshold):
    """
    Processes detection data from a JSON file to sort images into 'Animal' or 'No_animal' directories
    based on detection categories and confidence levels.

    This function reads a JSON formatted file containing annotations of image detections.
    Each image is checked for detections with category '0' and a confidence level above the specified
    threshold. If such detections are found, the image is categorized under 'Animal'. Images without
    any category '0' detections above the threshold, including those with no detections at all, are 
    categorized under 'No_animal'.

    Parameters:
    - json_file (str): Path to the JSON file containing detection data.
    - destination_path (str): Base path where 'Animal' and 'No_animal' folders will be created
                              and into which images will be sorted and copied.
    - source_images_directory (str): Path to the directory containing the source images to be processed.
    - confidence_threshold (float): The confidence threshold to consider a detection as valid.

    Effects:
    - Reads from the specified `json_file`.
    - Copies files from `source_images_directory` to either `destination_path/Animal` or
      `destination_path/No_animal` based on the detection data and confidence level.

    Note:
    - The function assumes that the JSON file structure includes keys 'annotations', each containing
      'img_id', 'bbox', 'category', and 'confidence'. It does not handle missing keys or unexpected
      JSON structures and may raise an exception in such cases.
    - Directories `Animal` and `No_animal` are created if they do not already exist.
    - Images are copied, not moved; original images remain in the source directory.
    """

    # Load JSON data from the file
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Ensure the destination directories exist
    os.makedirs(destination_path, exist_ok=True)
    animal_path = os.path.join(destination_path, "Animal")
    no_animal_path = os.path.join(destination_path, "No_animal")
    os.makedirs(animal_path, exist_ok=True)
    os.makedirs(no_animal_path, exist_ok=True)
    
    # Process each image detection
    i = 0
    for item in data['annotations']:
        i+=1
        img_id = item['img_id']
        categories = item['category']
        confidences = item['confidence']
        
        # Check if there is any category '0' with confidence above the threshold
        file_targeted_for_animal = False
        for category, confidence in zip(categories, confidences):
            if category == 0 and confidence > confidence_threshold:
                file_targeted_for_animal = True
                break
        
        if file_targeted_for_animal:
            target_folder = animal_path
        else:
            target_folder = no_animal_path
        
        # Construct the source and destination file paths
        src_file_path = os.path.join(img_path, img_id)
        dest_file_path = os.path.join(target_folder, os.path.basename(img_id))
        
        # Copy the file to the appropriate directory
        shutil.copy(src_file_path, dest_file_path)

    return "{} files were successfully separated".format(i)
