# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Gradio Demo for Herdnet image detection"""

# Importing necessary basic libraries and modules
import os
os.environ['WANDB_MODE'] = 'disabled'  

# PyTorch imports 
import torch

# Importing the model, dataset, transformations and utility functions from PytorchWildlife
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models.detection.herdnet.animaloc import models as animaloc_models
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
box_annotator = sv.DotAnnotator(radius=6)
lab_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_RIGHT)
# Create a temp folder
os.makedirs(os.path.join("..","temp"), exist_ok=True) # ASK: Why do we need this?

# Initializing the detection and classification models
detection_model = None
classification_model = None
    
# Defining functions for different detection scenarios
def load_models(det, clf, wpath=None, wclass=None):

    global detection_model, classification_model
    breakpoint()
    #detection_model = pw_detection.__dict__[det](device=DEVICE, pretrained=True)
    if det == "HerdNet":
        num_classes = 7
        weights_path = "/home/v-ichaconsil/ssdprivate/PytorchWildlife/CameraTraps/PytorchWildlife/models/detection/herdnet/weights/20220413_HerdNet_General_dataset_2022.pth" # TODO: What if someone else wants to run this?
        model = animaloc_models.HerdNet(num_classes=num_classes, pretrained=False) # Architecture of the model
        model = animaloc_models.LossWrapper(model, []) # Model wrapper
        #detection_model = pw_detection.herdnet.HerdNet(weights=weights_path, device=DEVICE, model=model)
        detection_model = pw_detection.__dict__[det](weights=weights_path, device=DEVICE, model=model)
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
        det_conf_thre (float): Confidence threshold for detection.
        clf_conf_thre (float): Confidence threshold for classification.
        img_index: Image index identifier.
    Returns:
        annotated_img (PIL.Image.Image): Annotated image with bounding box instances.
    """
    # trans_img = trans_det(input_img)
    input_img = np.array(input_img)
    
    # If the detection model is HerdNet, do not pass conf_thres
    if detection_model.__class__.__name__ == "HerdNet":
        results_det = detection_model.single_image_detection(input_img,
                                                             img_path=img_index)
    else:
        results_det = detection_model.single_image_detection(input_img,
                                                         img_path=img_index,
                                                         conf_thres=det_conf_thres)

    
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
        scene=box_annotator.annotate(
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
        det_conf_thre (float): Confidence threshold for detection.
        timelapse (boolean): Flag to output JSON for timelapse.
        clf_conf_thre (float): Confidence threshold for classification.

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
    # If the detection model is HerdNet, do not pass conf_thres and also set batch_size to 1
    if detection_model.__class__.__name__ == "HerdNet":
        det_results = detection_model.batch_image_detection(tgt_folder_path, batch_size=1, id_strip=tgt_folder_path) # TODO: Do we want default JPG extension?
    else:
        det_results = detection_model.batch_image_detection(tgt_folder_path, batch_size=16, conf_thres=det_conf_thres, id_strip=tgt_folder_path)

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
            pw_utils.save_detection_timelapse_json(det_results, json_save_path, categories=detection_model.CLASS_NAMES)
        else: # TODO: Right now the JSON saves the detections as bbox for Herdnet
            pw_utils.save_detection_json(det_results, json_save_path, categories=detection_model.CLASS_NAMES)

    return json_save_path

def batch_path_detection(tgt_folder_path, det_conf_thres):
    """Perform detection on a batch of images from a zip file and return path to results JSON.
    
    Args:
        tgt_folder_path (str): path to the folder containing the images.
        det_conf_thre (float): Confidence threshold for detection.
    Returns:
        json_save_path (str): Path to the JSON file containing detection results.
    """
    json_save_path = os.path.join(tgt_folder_path, "results.json")
    det_dataset = pw_data.DetectionImageFolder(tgt_folder_path, transform=trans_det) # FIXME: trans_det is not defined
    det_loader = DataLoader(det_dataset, batch_size=32, shuffle=False, 
                            pin_memory=True, num_workers=2, drop_last=False)
    det_results = detection_model.batch_image_detection(det_loader, conf_thres=det_conf_thres, id_strip=tgt_folder_path)
    pw_utils.save_detection_json(det_results, json_save_path, categories=detection_model.CLASS_NAMES)

    return json_save_path


def video_detection(video, det_conf_thres, clf_conf_thres, target_fps, codec):
    """Perform detection on a video and return path to processed video.
    
    Args:
        video (str): Video source path.
        det_conf_thre (float): Confidence threshold for detection.
        clf_conf_thre (float): Confidence threshold for classification.

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
            ["MegaDetectorV5", "HerdNet"],
            label="Detection model",
            info="Will add more detection models!",
            value="MegaDetectorV5" # Default detection model
        )
        clf_drop = gr.Dropdown(
            ["None", "AI4GOpossum", "AI4GAmazonRainforest", "AI4GSnapshotSerengeti", "CustomWeights"],
            label="Classification model",
            info="Will add more classification models!",
            value="None"
        )
    with gr.Column():
        custom_weights_path = gr.Textbox(label="Custom Weights Path", visible=False, interactive=True, placeholder="./weights/my_weight.pt")
        custom_weights_class = gr.Textbox(label="Custom Weights Class", visible=False, interactive=True, placeholder="{1:'ocelot', 2:'cow', 3:'bear'}")
        load_but = gr.Button("Load Models!")
        load_out = gr.Text("NO MODEL LOADED!!", label="Loaded models:")
    
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
                sgl_conf_sl_clf = gr.Slider(0, 1, label="Classification Confidence Threshold", value=0.7)
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
                chck_timelapse = gr.Checkbox(label="Timelapse Output", info="Output JSON for timelapse.")
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

    load_but.click(load_models, inputs=[det_drop, clf_drop, custom_weights_path, custom_weights_class], outputs=load_out)
    sgl_but.click(single_image_detection, inputs=[sgl_in, sgl_conf_sl_det, sgl_conf_sl_clf], outputs=sgl_out)
    bth_but.click(batch_detection, inputs=[bth_in, chck_timelapse, bth_conf_sl], outputs=bth_out)
    vid_but.click(video_detection, inputs=[vid_in, vid_conf_sl_det, vid_conf_sl_clf, vid_fr, vid_enc], outputs=vid_out)

if __name__ == "__main__":
    demo.launch(share=True)