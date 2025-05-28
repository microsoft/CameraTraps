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
import csv

__all__ = [
    "save_detection_images",
    "save_detection_images_dots",
    "save_crop_images",
    "save_detection_json",
    "save_detection_json_as_dots",
    "save_detection_classification_json",
    "save_detection_timelapse_json",
    "save_detection_classification_timelapse_json",
    "save_detection_classification_csv_dwc",
    "detection_folder_separation",
    "detection_classification_folder_separation",
]


def save_detection_images(results, output_dir, input_dir=None, overwrite=False):
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
    box_annotator = sv.BoxAnnotator(
        thickness=4)

    os.makedirs(output_dir, exist_ok=True)

    with sv.ImageSink(target_dir_path=output_dir, overwrite=overwrite) as sink: 
        all_results = results if isinstance(results, list) else [results]

        for entry in all_results:

            img = np.array(
                Image.open(entry["img_id"]).convert("RGB")
            )
            detections = entry["detections"]
            frame = box_annotator.annotate(scene=img, detections=detections)

            img_h, img_w = frame.shape[:2]

            font      = cv2.FONT_HERSHEY_DUPLEX
            text_color = (255,255,255)       

            # Dynamic text sizing based on image height
            reference_height = 1280.0
            scale = (img_h / reference_height) * 2.0

            # thickness & padding proportional too
            thickness = max(1, int(round(scale)))
            padding   = max(2, int(round(scale * 2)))

            for box, label, class_id in zip(detections.xyxy, entry["labels"], entry["detections"].class_id):
                bg_color   = box_annotator.color.by_idx(class_id).as_bgr()
                x1, y1, x2, y2 = box.astype(int)

                # measure text
                (tw, th), baseline = cv2.getTextSize(
                    label, font, scale, thickness
                )
                text_block_h = th + baseline + 2*padding

                # **vertical** placement: above if room, else below
                if y1 - text_block_h >= 0:
                    # rectangle sits flush on the box edge
                    rect_top    = y1 - text_block_h
                    rect_bottom = y1
                    # baseline for the text
                    ty = y1 - padding - baseline
                else:
                    # fall-back to drawing below
                    rect_top    = y2
                    rect_bottom = y2 + text_block_h
                    # baseline for the text
                    ty = y2 + th + padding

                # **horizontal** start at left edge of box
                tx = x1

                # clamp so text_rect never leaves left edge
                if tx - padding < 0:
                    tx = padding

                # clamp so text_rect never leaves right edge
                if tx + tw + padding > img_w:
                    tx = img_w - tw - padding

                # draw background rectangle
                cv2.rectangle(
                    frame,
                    (tx - padding, rect_top),
                    (tx + tw + padding, rect_bottom),
                    bg_color,
                    thickness=-1  # filled
                )
                # draw text itself
                cv2.putText(
                    frame,
                    label,
                    (tx, ty),
                    font,
                    scale,
                    text_color,
                    thickness
                )

            if input_dir:
                rel = os.path.relpath(entry["img_id"], input_dir)
                save_name = rel
                os.makedirs(os.path.dirname(os.path.join(output_dir, rel)), exist_ok=True)
            else:
                save_name = os.path.basename(entry["img_id"])

            sink.save_image(
                image=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                image_name=save_name
            )

def save_detection_images_dots(results, output_dir, input_dir=None, overwrite=False):
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
            for i, entry in enumerate(results):
                if "img_id" in entry:
                    scene = np.array(Image.open(entry["img_id"]).convert("RGB"))
                    image_name = os.path.basename(entry["img_id"])
                else:
                    scene = entry["img"]
                    image_name = f"output_image_{i}.jpg" # default name if no image id is provided

                annotated_img = lab_annotator.annotate(
                    scene=dot_annotator.annotate(
                        scene=scene,
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
                sink.save_image(
                    image=cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR), image_name=image_name
                )
        else:
            if "img_id" in results:
                scene = np.array(Image.open(results["img_id"]).convert("RGB"))
                image_name = os.path.basename(results["img_id"])
            else:
                scene = results["img"]
                image_name = "output_image.jpg" # default name if no image id is provided
            
            annotated_img = lab_annotator.annotate(
                scene=dot_annotator.annotate(
                    scene=scene,
                    detections=results["detections"],
                ),
                detections=results["detections"],
                labels=results["labels"],
            )   
            sink.save_image(
                image=cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR), image_name=image_name
            )


# !!! Output paths need to be optimized !!!
def save_crop_images(results, output_dir, input_dir=None, overwrite=False):
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
        det_results (list):
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
        det_results (list):
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
        det_results (list):
            Detection results containing image ID, bounding boxes, category, and confidence.
        output_dir (str):
            Path to save the output JSON file.
        categories (list, optional):
            List of categories for detected objects. Defaults to None.
        exclude_category_ids (list, optional):
            List of category IDs to exclude from the output. Defaults to []. Category IDs can be found in the definition of each models.
        exclude_file_path (str, optional):
            Some time, Timelapse has path issues. We can exclude the some path sections from the image ID. Defaults to None.
        info (dict, optional):
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
        output_path (str):
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


def save_detection_classification_csv_dwc(
    det_results,
    clf_results,
    output_path,
    model_name,
    det_categories=None,
    clf_categories=None,
    exclude_file_path=None
):
    """
    Save detection and classification results in Darwin Core (DwC)-like CSV format.

    Args:
        det_results (list): Detection results with image ID, bounding boxes, class IDs, and confidence.
        clf_results (list): Classification results with image ID, class ID, and confidence.
        output_path (str): Path to save the output CSV file.
        det_categories (list, optional): List of detection category names.
        clf_categories (list, optional): List of classification category names.
        exclude_file_path (str, optional): If provided, strip this prefix from image paths.
    """

    headers = [
        "occurrenceID", "eventID", "scientificName",
        "occurrenceRemarks", "recordedBy", "identifiedBy",
        "classificationConfidence"
    ]

    rows = []
    detection_id = 1

    # Index classification results by img_id
    clf_dict = {}
    for clf in clf_results:
        clf_dict.setdefault(clf["img_id"], []).append(clf)

    for det_r in det_results:
        img_path = str(det_r["img_id"])
        display_path = img_path.replace(exclude_file_path + os.sep, '') if exclude_file_path else img_path

        detections = det_r["detections"]
        bboxes = detections.xyxy.astype(int)
        det_classes = detections.class_id
        det_confidences = detections.confidence

        # Get classification results for this image
        clf_entries = clf_dict.get(det_r["img_id"], [])

        for i in range(len(bboxes)):
            cat_id = det_classes[i]
            cat_name = det_categories[cat_id] if det_categories else f"Category_{cat_id}"
            box = bboxes[i]
            confidence = det_confidences[i]

            occurrence_remarks = f"BBox(xmin={box[0]}, ymin={box[1]}, xmax={box[2]}, ymax={box[3]})"

            # Use the first classification result, or leave blank if none
            if clf_entries:
                clf_label = clf_entries[0]["class_id"]
                clf_name = clf_categories[clf_label] if clf_categories else f"Class_{clf_label}"
                clf_conf = clf_entries[0]["confidence"]
            else:
                clf_name = ""
                clf_conf = ""

            rows.append({
                "occurrenceID": f"det-{detection_id}",
                "eventID": display_path,
                "scientificName": clf_name,
                "occurrenceRemarks": occurrence_remarks,
                "identifiedBy": model_name,
                "classificationConfidence": f"{clf_conf:.2f}" if clf_conf != "" else ""
            })

            detection_id += 1

    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


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
        dest_file_path = os.path.join(target_folder, os.path.dirname(img_id))
        os.makedirs(dest_file_path, exist_ok=True)
        
        # Copy the file to the appropriate directory
        shutil.copy(src_file_path, dest_file_path)

    return "{} files were successfully separated".format(i)

def detection_classification_folder_separation(json_file, img_path, destination_path, det_conf_threshold, clf_conf_threshold, overwrite=False):
    """
    Processes detection and classification data from a JSON file and saves images with annotated
    bounding boxes into structured directories based on classification categories and confidence levels.
    If a detection has an associated classification confidence ≥ `clf_conf_threshold`, the label
    shown on the image will be the classification category name (e.g., "Leptotila").
    If classification is not available or below threshold, the detection category name is used instead (e.g., "animal").
    Confidence values are shown alongside labels

    Parameters:
    - json_file (str): Path to the JSON file containing detection and classification data.
    - img_path (str): Path to the directory containing the source images.
    - destination_path (str): Base path to save annotated images into structured folders.
    - det_conf_threshold (float): Threshold for detection filtering.
    - clf_conf_threshold (float): Threshold for classification filtering.
    - overwrite (bool): If True, overwrite existing output folders/images.

    Returns:
    - str: Summary of processed files.
    """

    # Load detection data
    with open(json_file, 'r') as file:
        data = json.load(file)

    det_map = data.get("det_categories", {})
    clf_map = data.get("clf_categories", {})

    # Prepare output roots
    animal_root = os.path.join(destination_path, 'Animal')
    no_animal_root = os.path.join(destination_path, 'No_animal')
    for root in (animal_root, no_animal_root):
        if overwrite and os.path.exists(root):
            # Clear folder
            for dirpath, _, filenames in os.walk(root):
                for file in filenames:
                    os.remove(os.path.join(dirpath, file))
        os.makedirs(root, exist_ok=True)

    # Box Annotator
    box_annotator = sv.BoxAnnotator(thickness=4)

    processed = 0
    for ann in data.get('annotations', []):
        img_id = ann['img_id']
        det_cats = ann['det_category']
        det_confs = ann['det_confidence']
        bboxes   = np.array(ann['bbox'], dtype=float)
        clf_cats = ann.get('clf_category', [])
        clf_confs= ann.get('clf_confidence', [])

        # decide root folder by animal presence
        has_animal = any(c==0 and conf >= det_conf_threshold for c, conf in zip(det_cats, det_confs))
        root_folder = animal_root if has_animal else no_animal_root

        # classification subdirs
        valid_clfs = [clf_map.get(str(c), 'Unknown') for c, conf in zip(clf_cats, clf_confs) if conf >= clf_conf_threshold]
        clf_dirs = valid_clfs or ['Unknown']

        # load image
        try:
            img = np.array(Image.open(os.path.join(img_path, img_id)).convert('RGB'))
        except Exception as e:
            print(f"Skipping {img_id}: {e}")
            continue

        # annotate boxes first
        detections = sv.Detections(xyxy=bboxes, class_id=np.array(det_cats, dtype=int))
        frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        h, w = frame.shape[:2]

        # dynamic text parameters
        font = cv2.FONT_HERSHEY_DUPLEX
        ref_h = 1280.0
        base_scale = 2.0
        scale = max(0.5, (h / ref_h) * base_scale)
        thickness = max(1, int(round(scale)))
        padding = max(2, int(round(scale * 2)))
        text_color = (255,255,255)

        # draw labels per detection
        for idx, (box, cat_id, det_conf) in enumerate(zip(bboxes, det_cats, det_confs)):
            if det_conf < det_conf_threshold:
                continue
            # choose label
            cls_label = clf_map.get(str(clf_cats[idx]), det_map.get(str(cat_id), 'Unknown'))
            cls_conf  = clf_confs[idx] if idx < len(clf_confs) else None
            label = f"{cls_label} {cls_conf:.2f}" if cls_conf is not None else cls_label

            x1, y1, x2, y2 = map(int, box)
            # measure text size
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
            block_h = th + baseline + 2*padding

            # vertical placement
            if y1 - block_h >= 0:
                top = y1 - block_h
                bottom = y1
                text_y = y1 - padding
            else:
                top = y2
                bottom = y2 + block_h
                text_y = y2 + th + padding

            # horizontal placement
            text_x = x1
            if text_x - padding < 0:
                text_x = padding
            if text_x + tw + padding > w:
                text_x = w - tw - padding

            # background rect
            color = box_annotator.color.by_idx(int(cat_id)).as_bgr()
            cv2.rectangle(frame, (text_x - padding, top), (text_x + tw + padding, bottom), color, -1)
            # text
            cv2.putText(frame, label, (text_x, text_y), font, scale, text_color, thickness)

        # save per classification folder
        for clf_dir in clf_dirs:
            dest = os.path.join(root_folder, clf_dir)
            os.makedirs(dest, exist_ok=True)
            out_path = os.path.join(dest, os.path.basename(img_id))
            cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        processed += 1

    return f"{processed} files were successfully separated"
