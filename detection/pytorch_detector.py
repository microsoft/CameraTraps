"""
Module to run MegaDetector v5, a PyTorch YOLOv5 (Ultralytics) animal detection model,
on images.
"""

#%% Imports
import sys

import torch
import numpy as np

from detection.run_detector import CONF_DIGITS, COORD_DIGITS, FAILURE_INFER
import ct_utils

try:
    # import pre- and post-processing functions from the YOLOv5 repo https://github.com/ultralytics/yolov5
    from utils.general import non_max_suppression, scale_coords, xyxy2xywh
    from utils.augmentations import letterbox
except ModuleNotFoundError as e:
    raise ModuleNotFoundError('Could not import YOLOv5 functions.')

print(f'Using PyTorch version {torch.__version__}')


#%% Classes

class PTDetector:

    IMAGE_SIZE = 1280  # image size used in training
    STRIDE = 64

    def __init__(self, model_path: str, force_cpu: bool = False):
        if torch.cuda.is_available() and not force_cpu:
            self.device = torch.device('cuda:0')
        else:
            self.device = 'cpu'
        self.model = PTDetector._load_model(model_path, self.device)
        if (self.device != 'cpu') and torch.cuda.is_available():
            print('Sending model to GPU')
            self.model.to(self.device)

    @staticmethod
    def _load_model(model_pt_path, device):
        checkpoint = torch.load(model_pt_path, map_location=device)
        model = checkpoint['model'].float().fuse().eval()  # FP32 model
        return model

    def generate_detections_one_image(self, img_original, image_id, detection_threshold):
        """Apply the detector to an image.

        Args:
            img_original: the PIL Image object with EXIF rotation taken into account
            image_id: a path to identify the image; will be in the "file" field of the output object
            detection_threshold: confidence above which to include the detection proposal

        Returns:
        A dict with the following fields, see the 'images' key in https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
            - 'file' (always present)
            - 'max_detection_conf'
            - 'detections', which is a list of detection objects containing keys 'category', 'conf' and 'bbox'
            - 'failure'
        """

        result = {
            'file': image_id
        }
        detections = []
        max_conf = 0.0

        try:
            img_original = np.asarray(img_original)

            # padded resize
            img = letterbox(img_original, new_shape=PTDetector.IMAGE_SIZE,
                                 stride=PTDetector.STRIDE, auto=True)[0]  # JIT requires auto=False
            img = img.transpose((2, 0, 1))  # HWC to CHW; PIL Image is RGB already
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            img = img.to(self.device)
            img = img.float()
            img /= 255

            if len(img.shape) == 3:  # always true for now, TODO add inference using larger batch size
                img = torch.unsqueeze(img, 0)

            pred: list = self.model(img)[0]

            # NMS
            pred = non_max_suppression(prediction=pred, conf_thres=detection_threshold)

            # format detections/bounding boxes
            gn = torch.tensor(img_original.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            for det in pred:
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_original.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        # normalized center-x, center-y, width and height
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

                        api_box = ct_utils.convert_yolo_to_xywh(xywh)

                        conf = ct_utils.truncate_float(conf.tolist(), precision=CONF_DIGITS)

                        # MegaDetector output format's categories start at 1, but this model's start at 0
                        cls = int(cls.tolist()) + 1
                        if cls not in (1, 2, 3):
                            raise KeyError(f'{cls} is not a valid class.')

                        detections.append({
                            'category': str(cls),
                            'conf': conf,
                            'bbox': ct_utils.truncate_float_array(api_box, precision=COORD_DIGITS)
                        })
                        max_conf = max(max_conf, conf)

        except Exception as e:
            result['failure'] = FAILURE_INFER
            print('PTDetector: image {} failed during inference: {}'.format(image_id, str(e)))

        result['max_detection_conf'] = max_conf
        result['detections'] = detections

        return result


if __name__ == '__main__':
    # for testing

    import visualization.visualization_utils as viz_utils

    model_file = "<path to the model .pt file>"
    im_file = "test_images/test_images/island_conservation_camera_traps_palau_cam10a_cam10a12122018_palau_cam10a12122018_20181108_174532_rcnx1035.jpg"

    detector = PTDetector(model_file)
    image = viz_utils.load_image(im_file)

    res = detector.generate_detections_one_image(image, im_file, detection_threshold=0.2)
