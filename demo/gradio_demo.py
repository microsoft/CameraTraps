# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Demo using Gradio interface"""

#%% 
# Importing basic libraries
import os
import time
from PIL import Image
import supervision as sv
import gradio as gr
from zipfile import ZipFile
import torch
from torch.utils.data import DataLoader
# %%
from torchvision import transforms
from PIL import Image

import torch
from PIL import Image
import numpy as np


#%% 
# Importing the models, dataset, transformations, and utility functions from PytorchWildlife
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife.data import datasets as pw_data 
from PytorchWildlife import utils as pw_utils

#%% 
# Setting the device to use for computations ('cuda' indicates GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Initializing a supervision box annotator for visualizing detections
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
# Create a temp folder
os.makedirs("../temp/", exist_ok=True)

# Initializing the detection and classification models
detection_model = None
classification_model = None
    
# Defining transformations for detection and classification
trans_det = None
trans_clf = None

#%% Defining functions for different detection scenarios
def load_models(det, clf):

    global detection_model, classification_model, trans_det, trans_clf, trans_det_2

    detection_model = pw_detection.__dict__[det](device=DEVICE, pretrained=True)
    if clf != "None":
        classification_model = pw_classification.__dict__[clf](device=DEVICE, pretrained=True)

    trans_det = pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
                                          stride=detection_model.STRIDE)
    trans_clf = pw_trans.Classification_Inference_Transform(target_size=224)

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
    trans_img = trans_det(input_img)
    input_img = np.array(input_img)
    
    results_det = detection_model.single_image_detection(trans_img,
                                                         input_img.shape,
                                                         img_path=img_index,
                                                         conf_thres=det_conf_thres)

    
    if classification_model is not None:
        labels = []
        for xyxy, det_id in zip(results_det["detections"].xyxy, results_det["detections"].class_id):
            # Only run classifier when detection class is animal
            if det_id == 0:
                cropped_image = sv.crop_image(image=input_img, xyxy=xyxy)
                results_clf = classification_model.single_image_classification(trans_clf(Image.fromarray(cropped_image)))
                labels.append("{} {:.2f}".format(results_clf["prediction"] if results_clf["confidence"] > clf_conf_thres else "Unknown",
                                                 results_clf["confidence"]))
            else:
                labels = results_det["labels"]
    else:
        labels = results_det["labels"]
    annotated_img = box_annotator.annotate(scene=input_img, detections=results_det["detections"], labels=labels)
    return annotated_img


def batch_detection(zip_file, det_conf_thres):
    """Perform detection on a batch of images from a zip file and return path to results JSON.
    
    Args:
        zip_file (File): Zip file containing images.
        det_conf_thre (float): Confidence threshold for detection.
        clf_conf_thre (float): Confidence threshold for classification.

    Returns:
        json_save_path (str): Path to the JSON file containing detection results.
    """
    extract_path = "../temp/zip_upload"
    os.makedirs(extract_path, exist_ok=True)
    json_save_path = os.path.join(extract_path, "results.json")
    with ZipFile(zip_file.name) as zfile:
        zfile.extractall(extract_path)
    tgt_folder_path = os.path.join(extract_path, zip_file.name.rsplit('/', 1)[1].rstrip(".zip"))
    det_dataset = pw_data.DetectionImageFolder(tgt_folder_path, transform=trans_det)
    det_loader = DataLoader(det_dataset, batch_size=32, shuffle=False, 
                            pin_memory=True, num_workers=4, drop_last=False)
    det_results = detection_model.batch_image_detection(det_loader, conf_thres=det_conf_thres, id_strip=tgt_folder_path)

    if classification_model is not None:
        clf_dataset = pw_data.DetectionCrops(
            det_results,
            transform=pw_trans.Classification_Inference_Transform(target_size=224),
            path_head=tgt_folder_path
        )
        clf_loader = DataLoader(clf_dataset, batch_size=32, shuffle=False, 
                                pin_memory=True, num_workers=4, drop_last=False)
        clf_results = classification_model.batch_image_classification(clf_loader, id_strip=tgt_folder_path)
        pw_utils.save_detection_classification_json(det_results=det_results,
                                                    clf_results=clf_results,
                                                    det_categories=detection_model.CLASS_NAMES,
                                                    clf_categories=classification_model.CLASS_NAMES,
                                                    output_path=json_save_path)
    else:
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
    
    target_path = "../temp/video_detection.mp4"
    pw_utils.process_video(source_path=video, target_path=target_path,
                           callback=callback, target_fps=int(target_fps), codec=codec)
    return target_path

#%% Building Gradio UI

with gr.Blocks() as demo:
    gr.Markdown("# Pytorch-Wildlife Demo.")
    with gr.Row():
        det_drop = gr.Dropdown(
            ["MegaDetectorV5"],
            label="Detection model",
            info="Will add more detection models!",
            value="MegaDetectorV5"
        )
        clf_drop = gr.Dropdown(
            ["None", "AI4GOpossum", "AI4GAmazonRainforest"],
            label="Classification model",
            info="Will add more classification models!",
            value="None"
        )
    with gr.Column():
        load_but = gr.Button("Load Models!")
        load_out = gr.Text("NO MODEL LOADED!!", label="Loaded models:")
    with gr.Tab("Single Image Process"):
        with gr.Row():
            with gr.Column():
                sgl_in = gr.Image(type="pil")
                sgl_conf_sl_det = gr.Slider(0, 1, label="Detection Confidence Threshold", value=0.2)
                sgl_conf_sl_clf = gr.Slider(0, 1, label="Classification Confidence Threshold", value=0.7)
            sgl_out = gr.Image() 
        sgl_but = gr.Button("Detect Animals!")
    with gr.Tab("Batch Image Process"):
        with gr.Row():
            with gr.Column():
                bth_in = gr.File(label="Upload zip file.")
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

    load_but.click(load_models, inputs=[det_drop, clf_drop], outputs=load_out)
    sgl_but.click(single_image_detection, inputs=[sgl_in, sgl_conf_sl_det, sgl_conf_sl_clf], outputs=sgl_out)
    bth_but.click(batch_detection, inputs=[bth_in, bth_conf_sl], outputs=bth_out)
    vid_but.click(video_detection, inputs=[vid_in, vid_conf_sl_det, vid_conf_sl_clf, vid_fr, vid_enc], outputs=vid_out)

if __name__ == "__main__":
    demo.launch(share=True)
