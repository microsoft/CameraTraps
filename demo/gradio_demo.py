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
def load_models(det, clf, wpath=None, wclass=None):

    global detection_model, classification_model
    if det != "None":
        if det == "HerdNet General":
            detection_model = pw_detection.HerdNet(device=DEVICE)
        elif det == "HerdNet Ennedi":
            detection_model = pw_detection.HerdNet(device=DEVICE, dataset="ennedi")
        elif det == "MegaDetectorV6":
            detection_model = pw_detection.__dict__[det](device=DEVICE, pretrained=True)
        else:
            detection_model = pw_detection.__dict__[det](device=DEVICE, pretrained=True)

    if clf != "None":
        # Create an exception for custom weights
        if clf == "CustomWeights":
            if (wpath is not None) and (wclass is not None): 
                wclass = ast.literal_eval(wclass)
                classification_model = pw_classification.__dict__[clf](weights=wpath, class_names=wclass, device=DEVICE)
        else:
            classification_model = pw_classification.__dict__[clf](device=DEVICE, pretrained=True)

    return "Loaded Detector: {}. Loaded Classifier: {}".format(det, clf)


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


def video_detection(video, det_conf_thres, clf_conf_thres, target_fps, codec):
    """Perform detection on a video and return path to processed video.
    
    Args:
        video (str): Video source path.
        det_conf_thres (float): Confidence threshold for detection.
        clf_conf_thres (float): Confidence threshold for classification.

    """
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
        clf_drop = gr.Dropdown(
            ["None", "AI4GOpossum", "AI4GAmazonRainforest", "AI4GSnapshotSerengeti", "CustomWeights"],
            interactive=True,
            label="Classification model",
            info="Will add more classification models!",
            value="None"
        )
    with gr.Column():
        custom_weights_path = gr.Textbox(label="Custom Weights Path", visible=False, interactive=True, placeholder="./weights/my_weight.pt")
        custom_weights_class = gr.Textbox(label="Custom Weights Class", visible=False, interactive=True, placeholder="{1:'ocelot', 2:'cow', 3:'bear'}")
        load_but = gr.Button("Load Models!")
        load_out = gr.Text("NO MODEL LOADED!!", label="Loaded models:")
    
    def update_ui_elements(det_model):  
        if "HerdNet" in det_model: # Disable all the classification model dropdown because HerdNet does not require a classification model apart
            return gr.Dropdown(choices=["None"], interactive=True, label="Classification model", value="None")
        else:
            return gr.Dropdown(choices=["None", "AI4GOpossum", "AI4GAmazonRainforest", "AI4GAmazonRainforest_v2", "AI4GSnapshotSerengeti", "CustomWeights"], interactive=True, label="Classification model", value="None")

    det_drop.change(update_ui_elements, det_drop, [clf_drop])

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
                vid_fr = gr.Dropdown([5, 10, 30], label="Output video framerate", value=30)
                vid_enc = gr.Dropdown(
                    ["mp4v", "avc1"],
                    label="Video encoder",
                    info="mp4v is default, av1c is faster (needs conda install opencv)",
                    value="mp4v"
                    )
            vid_out = gr.Video()
        vid_but = gr.Button("Detect Animals!")
        
    # Show timelapsed checkbox only when detection model is not HerdNet
    det_drop.change(
        lambda model: gr.update(visible=True) if "HerdNet" not in model else gr.update(visible=False),
        det_drop,
        [chck_timelapse]
    )

    load_but.click(load_models, inputs=[det_drop, clf_drop, custom_weights_path, custom_weights_class], outputs=load_out)
    sgl_but.click(single_image_detection, inputs=[sgl_in, sgl_conf_sl_det, sgl_conf_sl_clf], outputs=sgl_out)
    bth_but.click(batch_detection, inputs=[bth_in, chck_timelapse, bth_conf_sl], outputs=bth_out)
    vid_but.click(video_detection, inputs=[vid_in, vid_conf_sl_det, vid_conf_sl_clf, vid_fr, vid_enc], outputs=vid_out)

if __name__ == "__main__":
    demo.launch(share=True)
