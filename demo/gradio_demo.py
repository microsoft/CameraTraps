# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Gradio Demo for image detection"""

# Importing necessary basic libraries and modules
import os

# PyTorch imports 
import torch

# Importing the model, dataset, transformations and utility functions from PytorchWildlife
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils
 
# Importing basic libraries
import shutil
import time
from PIL import Image
import supervision as sv
import gradio as gr
from zipfile import ZipFile
from torch.utils.data import DataLoader
import numpy as np
import ast

# Importing the models, dataset, transformations, and utility functions from PytorchWildlife
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife.data import datasets as pw_data 

# Importing the library for video processing
from tqdm import tqdm
from typing import Callable
from supervision import VideoInfo, VideoSink, get_video_frames_generator
import json
# Setting the device to use for computations ('cuda' indicates GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Initializing a supervision box annotator for visualizing detections
dot_annotator = sv.DotAnnotator(radius=6)
box_annotator = sv.BoxAnnotator(thickness=4)
lab_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_thickness=4, text_scale=2)
# Create a temp folder
os.makedirs(os.path.join("..","temp"), exist_ok=True) # ASK: Why do we need this?

# Initializing the detection and classification models
detection_model = None
classification_model = None
    
# Defining functions for different detection scenarios
def load_models(det, version, clf, wpath=None, wclass=None):

    global detection_model, classification_model
    if det != "None":
        if det == "HerdNet General":
            detection_model = pw_detection.HerdNet(device=DEVICE)
        elif det == "HerdNet Ennedi":
            detection_model = pw_detection.HerdNet(device=DEVICE, version="ennedi")
        else:
            detection_model = pw_detection.__dict__[det](device=DEVICE, pretrained=True, version=version)
    else:
        detection_model = None
        return "NO MODEL LOADED!!"

    if clf != "None":
        # Create an exception for custom weights
        if clf == "CustomWeights":
            if (wpath is not None) and (wclass is not None): 
                wclass = ast.literal_eval(wclass)
                classification_model = pw_classification.__dict__[clf](weights=wpath, class_names=wclass, device=DEVICE)
        else:
            classification_model = pw_classification.__dict__[clf](device=DEVICE, pretrained=True)
    else:
        classification_model = None

    return "Loaded Detector: {}. Version: {}. Loaded Classifier: {}".format(det, version, clf)


def single_image_detection(input_img, det_conf_thres, clf_conf_thres, img_index=None):
    """Performs detection on a single image and returns an annotated image.

    Args:
        input_img (PIL.Image): Input image in PIL.Image format defaulted by Gradio.
        det_conf_thres (float): Confidence threshold for detection.
        clf_conf_thres (float): Confidence threshold for classification.
        img_index: Image index identifier.
    Returns:
        annotated_img (PIL.Image.Image): Annotated image with bounding box instances.
    """

    input_img = np.array(input_img)
    # If the detection model is HerdNet, use dot annotator, else use box annotator
    if detection_model.__class__.__name__.__contains__("HerdNet"):
        annotator = dot_annotator
        # Herdnet receives both clf and det confidence thresholds
        results_det = detection_model.single_image_detection(input_img,
                                                             img_path=img_index,
                                                             det_conf_thres=det_conf_thres,
                                                             clf_conf_thres=clf_conf_thres)
    else:
        annotator = box_annotator
        results_det = detection_model.single_image_detection(input_img,
                                                             img_path=img_index,
                                                             det_conf_thres = det_conf_thres)
    
    if classification_model is not None:
        labels = []
        for xyxy, det_id in zip(results_det["detections"].xyxy, results_det["detections"].class_id):
            # Only run classifier when detection class is animal
            if det_id == 0:
                cropped_image = sv.crop_image(image=input_img, xyxy=xyxy)
                results_clf = classification_model.single_image_classification(cropped_image)
                labels.append("{} {:.2f}".format(results_clf["prediction"] if results_clf["confidence"] > clf_conf_thres else "Unknown",
                                                 results_clf["confidence"]))
            else:
                labels = results_det["labels"]
    else:
        labels = results_det["labels"]
    annotated_img = lab_annotator.annotate(
        scene=annotator.annotate(
            scene=input_img,
            detections=results_det["detections"],
        ),
        detections=results_det["detections"],
        labels=labels,
    )
    return annotated_img

def batch_detection(zip_file, timelapse, det_conf_thres):
    """Perform detection on a batch of images from a zip file and return path to results JSON.
    
    Args:
        zip_file (File): Zip file containing images.
        det_conf_thres (float): Confidence threshold for detection.
        timelapse (boolean): Flag to output JSON for timelapse.
        clf_conf_thres (float): Confidence threshold for classification.

    Returns:
        json_save_path (str): Path to the JSON file containing detection results.
    """
    # Clean the temp folder if it contains files
    extract_path = os.path.join("..","temp","zip_upload")
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    os.makedirs(extract_path)

    json_save_path = os.path.join(extract_path, "results.json")
    with ZipFile(zip_file.name) as zfile:
        zfile.extractall(extract_path)
        # Check the contents of the extracted folder
        extracted_files = os.listdir(extract_path)
        
    if len(extracted_files) == 1 and os.path.isdir(os.path.join(extract_path, extracted_files[0])):
        tgt_folder_path = os.path.join(extract_path, extracted_files[0])
    else:
        tgt_folder_path = extract_path
    # If the detection model is HerdNet set batch_size to 1
    if detection_model.__class__.__name__.__contains__("HerdNet"):
        det_results = detection_model.batch_image_detection(tgt_folder_path, batch_size=1, det_conf_thres=det_conf_thres, id_strip=tgt_folder_path) 
    else:
        det_results = detection_model.batch_image_detection(tgt_folder_path, batch_size=16, det_conf_thres=det_conf_thres, id_strip=tgt_folder_path)

    if classification_model is not None:
        clf_dataset = pw_data.DetectionCrops(
            det_results,
            transform=pw_trans.Classification_Inference_Transform(target_size=224),
            path_head=tgt_folder_path
        )
        clf_loader = DataLoader(clf_dataset, batch_size=32, shuffle=False, 
                                pin_memory=True, num_workers=4, drop_last=False)
        clf_results = classification_model.batch_image_classification(clf_loader, id_strip=tgt_folder_path)
        if timelapse:
            json_save_path = json_save_path.replace(".json", "_timelapse.json")
            pw_utils.save_detection_classification_timelapse_json(det_results=det_results,
                                                        clf_results=clf_results,
                                                        det_categories=detection_model.CLASS_NAMES,
                                                        clf_categories=classification_model.CLASS_NAMES,
                                                        output_path=json_save_path)
        else:
            pw_utils.save_detection_classification_json(det_results=det_results,
                                                        clf_results=clf_results,
                                                        det_categories=detection_model.CLASS_NAMES,
                                                        clf_categories=classification_model.CLASS_NAMES,
                                                        output_path=json_save_path)
    else:
        if timelapse:
            json_save_path = json_save_path.replace(".json", "_timelapse.json")
            pw_utils.save_detection_timelapse_json(det_results, json_save_path, categories=detection_model.CLASS_NAMES)
        elif detection_model.__class__.__name__.__contains__("HerdNet"):
            pw_utils.save_detection_json_as_dots(det_results, json_save_path, categories=detection_model.CLASS_NAMES)
        else: 
            pw_utils.save_detection_json(det_results, json_save_path, categories=detection_model.CLASS_NAMES)

    return json_save_path

def batch_path_detection(tgt_folder_path, det_conf_thres):
    """Perform detection on a batch of images from a zip file and return path to results JSON.
    
    Args:
        tgt_folder_path (str): path to the folder containing the images.
        det_conf_thres (float): Confidence threshold for detection.
    Returns:
        json_save_path (str): Path to the JSON file containing detection results.
    """

    json_save_path = os.path.join(tgt_folder_path, "results.json")
    det_results = detection_model.batch_image_detection(tgt_folder_path, det_conf_thres=det_conf_thres, id_strip=tgt_folder_path)
    if detection_model.__class__.__name__.__contains__("HerdNet"):
        pw_utils.save_detection_json_as_dots(det_results, json_save_path, categories=detection_model.CLASS_NAMES)
    else:
        pw_utils.save_detection_json(det_results, json_save_path, categories=detection_model.CLASS_NAMES)

    return json_save_path

def process_video_timelapse(
    source_path: str,
    target_path: str,
    callback: Callable[[np.ndarray, int], np.ndarray],
    target_fps: int = 1,
    codec: str = "mp4v",
    detection_categories: dict = {0: 'animal', 1: 'person', 2: 'vehicle'},
    clf_categories: dict = {0: 'opossum', 1: 'other'}
) -> None:
    """
    Process a video frame-by-frame, applying a callback function to each frame and saving the results 
    to a new video. This version includes a progress bar and allows codec selection.
    
    Args:
        source_path (str): 
            Path to the source video file.
        target_path (str): 
            Path to save the processed video.
        callback (Callable[[np.ndarray, int], np.ndarray]): 
            A function that takes a video frame and its index as input and returns the processed frame.
        codec (str, optional): 
            Codec used to encode the processed video. Default is "avc1".
    """
    source_video_info = VideoInfo.from_video_path(video_path=source_path)
    
    if source_video_info.fps > target_fps:
        stride = int(source_video_info.fps / target_fps)
        source_video_info.fps = target_fps
    else:
        stride = 1

    json_results = {
        "info": {"detector": "MDV6-yolov10-e"},
        "detection_categories": detection_categories,
        "classification_categories": clf_categories,
        "images": []
    }
    detections = []
    with VideoSink(target_path=target_path, video_info=source_video_info, codec=codec) as sink:
        with tqdm(total=int(source_video_info.total_frames / stride)) as pbar: 
            for index, frame in enumerate(
                get_video_frames_generator(source_path=source_path, stride=stride)
            ):
                result_frame = callback(frame, index)
                detections.append(result_frame)
                pbar.update(1)
    image_info = {"file": os.path.basename(target_path), "frame_rate":target_fps, "detections": detections}
    json_results["images"].append(image_info)

    # Save the json
    json_path = target_path.replace(".{}".format(target_path.split(".")[-1]), "_detection.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=4)
    print(f"JSON results saved to {json_path}")
    return json_path

def video_detection(video, det_conf_thres, clf_conf_thres, target_fps, codec, timelapse):
    """Perform detection on a video and return path to processed video.
    
    Args:
        video (str): Video source path.
        det_conf_thres (float): Confidence threshold for detection.
        clf_conf_thres (float): Confidence threshold for classification.

    """
    if not timelapse:
        def callback(frame, index):
            annotated_frame = single_image_detection(frame,
                                                    img_index=index,
                                                    det_conf_thres=det_conf_thres,
                                                    clf_conf_thres=clf_conf_thres)
            return annotated_frame 
        
        target_path = os.path.join("..","temp","video_detection.mp4")
        pw_utils.process_video(source_path=video, target_path=target_path,
                            callback=callback, target_fps=int(target_fps), codec=codec)
        return target_path
    else:
        def callback(frame: np.ndarray, index: int) -> np.ndarray:
            """
            Callback function to process each video frame for detection and classification.
            
            Parameters:
            - frame (np.ndarray): Video frame as a numpy array.
            - index (int): Frame index.
            
            Returns:
            annotated_frame (np.ndarray): Annotated video frame.
            """
            
            results_det = detection_model.single_image_detection(frame, img_path=index)
            labels = []
            normalized_coords = []
            classifications = []
            frame_width, frame_height = frame.shape[1], frame.shape[0]
            for xyxy in results_det["detections"].xyxy:
                x_min, y_min, x_max, y_max = xyxy
                cropped_image = sv.crop_image(image=frame, xyxy=xyxy)
                results_clf = classification_model.single_image_classification(cropped_image)
                labels.append("{} {:.2f}".format(results_clf["prediction"], results_clf["confidence"]))
                norm_bbox = [
                    x_min / frame_width,  # Normalize x_min
                    y_min / frame_height, # Normalize y_min
                    (x_max - x_min) / frame_width,  # Normalize width
                    (y_max - y_min) / frame_height  # Normalize height
                ]
                classifications.append([str(results_clf["class_id"]), float(results_clf["confidence"])])
                normalized_coords.append(norm_bbox)

            annotation = {
            "category": [str(i) for i in results_det["detections"].class_id],
            "conf": [float(j) for j in results_det["detections"].confidence],
            "bbox": normalized_coords,
            "classifications": classifications,
            "frame_number": index
            }

            return annotation
        process_video_timelapse(source_path=video, target_path=target_path, callback=callback, target_fps=int(target_fps), detection_categories=detection_model.CLASS_NAMES, clf_categories=classification_model.CLASS_NAMES)

# Building Gradio UI

with gr.Blocks() as demo:
    gr.Markdown("# Pytorch-Wildlife Demo.")
    with gr.Row():
        det_drop = gr.Dropdown(
            ["None", "MegaDetectorV5", "MegaDetectorV6", "HerdNet General", "HerdNet Ennedi"],
            label="Detection model",
            info="Will add more detection models!",
            value="None" # Default 
        )
        det_version = gr.Dropdown(  
            ["None"],  
            label="Model version",  
            info="Select the version of the model",
            value="None",
        )
    
    with gr.Column():
        clf_drop = gr.Dropdown(
            ["None", "AI4GOpossum", "AI4GAmazonRainforest", "AI4GSnapshotSerengeti", "CustomWeights"],
            interactive=True,
            label="Classification model",
            info="Will add more classification models!",
            visible=False,
            value="None"
        )
        custom_weights_path = gr.Textbox(label="Custom Weights Path", visible=False, interactive=True, placeholder="./weights/my_weight.pt")
        custom_weights_class = gr.Textbox(label="Custom Weights Class", visible=False, interactive=True, placeholder="{1:'ocelot', 2:'cow', 3:'bear'}")
        load_but = gr.Button("Load Models!")
        load_out = gr.Text("NO MODEL LOADED!!", label="Loaded models:")
   
    def update_ui_elements(det_model):  
        if det_model == "MegaDetectorV6":  
            return gr.Dropdown(choices=["MDV6-yolov9-c", "MDV6-yolov9-e", "MDV6-yolov10-c", "MDV6-yolov10-e", "MDV6-rtdetr-c"], interactive=True, label="Model version", value="MDV6-yolov9e"), gr.update(visible=True)  
        elif det_model == "MegaDetectorV5":  
            return gr.Dropdown(choices=["a", "b"], interactive=True, label="Model version", value="a"), gr.update(visible=True)
        else:
            return gr.Dropdown(choices=["None"], interactive=True, label="Model version", value="None"), gr.update(value="None", visible=False) 
    
    det_drop.change(update_ui_elements, det_drop, [det_version, clf_drop])

    def toggle_textboxes(model):
        if model == "CustomWeights":
            return gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False)
    
    clf_drop.change(
        toggle_textboxes,
        clf_drop,
        [custom_weights_path, custom_weights_class]
    )

    with gr.Tab("Single Image Process"):
        with gr.Row():
            with gr.Column():
                sgl_in = gr.Image(type="pil")
                sgl_conf_sl_det = gr.Slider(0, 1, label="Detection Confidence Threshold", value=0.2)
                sgl_conf_sl_clf = gr.Slider(0, 1, label="Classification Confidence Threshold", value=0.7, visible=True)
            sgl_out = gr.Image() 
        sgl_but = gr.Button("Detect Animals!")
    with gr.Tab("Folder Separation"):
        with gr.Row():
            with gr.Column():
                inp_path = gr.Textbox(label="Input path", placeholder="./data/")
                out_path = gr.Textbox(label="Output path", placeholder="./output/")
                bth_conf_fs = gr.Slider(0, 1, label="Detection Confidence Threshold", value=0.2)
                process_btn = gr.Button("Process Files")
                bth_out2 = gr.File(label="Detection Results JSON.", height=200)
                with gr.Column():
                    process_files_button = gr.Button("Separate files")
                    process_result = gr.Text("Click on 'Separate files' once you see the JSON file", label="Separated files:")
                    process_btn.click(batch_path_detection, inputs=[inp_path, bth_conf_fs], outputs=bth_out2)
                    process_files_button.click(pw_utils.detection_folder_separation, inputs=[bth_out2, inp_path, out_path, bth_conf_fs], outputs=process_result)
    with gr.Tab("Batch Image Process"):
        with gr.Row():
            with gr.Column():
                bth_in = gr.File(label="Upload zip file.")
                # The timelapse checkbox is only visible when the detection model is not HerdNet
                chck_timelapse = gr.Checkbox(label="Generate timelapse JSON", visible=False)
                bth_conf_sl = gr.Slider(0, 1, label="Detection Confidence Threshold", value=0.2)
            bth_out = gr.File(label="Detection Results JSON.", height=200)
        bth_but = gr.Button("Detect Animals!")
    with gr.Tab("Single Video Process"):
        with gr.Row():
            with gr.Column():
                vid_in = gr.Video(label="Upload a video.")
                vid_conf_sl_det = gr.Slider(0, 1, label="Detection Confidence Threshold", value=0.2)
                vid_conf_sl_clf = gr.Slider(0, 1, label="Classification Confidence Threshold", value=0.7)
                chck_timelapse_video = gr.Checkbox(label="Generate timelapse JSON")
                vid_fr = gr.Dropdown([5, 10, 30], label="Output video framerate", value=30)
                vid_enc = gr.Dropdown(
                    ["mp4v", "avc1"],
                    label="Video encoder",
                    info="mp4v is default, av1c is faster (needs conda install opencv)",
                    value="mp4v"
                    )
                vid_timelapse_out = gr.File(label="Detection Results JSON.", height=200)
            vid_out = gr.Video()
            
        vid_but = gr.Button("Detect Animals!")
        
    # Show timelapsed checkbox only when detection model is not HerdNet
    det_drop.change(
        lambda model: gr.update(visible=True) if "HerdNet" not in model else gr.update(visible=False),
        det_drop,
        [chck_timelapse]
    )

    load_but.click(load_models, inputs=[det_drop, det_version, clf_drop, custom_weights_path, custom_weights_class], outputs=load_out)
    sgl_but.click(single_image_detection, inputs=[sgl_in, sgl_conf_sl_det, sgl_conf_sl_clf], outputs=sgl_out)
    bth_but.click(batch_detection, inputs=[bth_in, chck_timelapse, bth_conf_sl], outputs=bth_out)
    if chck_timelapse_video:
        vid_but.click(video_detection, inputs=[vid_in, vid_conf_sl_det, vid_conf_sl_clf, vid_fr, vid_enc, chck_timelapse_video], outputs=vid_timelapse_out)
    else:
        vid_but.click(video_detection, inputs=[vid_in, vid_conf_sl_det, vid_conf_sl_clf, vid_fr, vid_enc, chck_timelapse_video], outputs=vid_out)

if __name__ == "__main__":
    demo.launch(share=True)
