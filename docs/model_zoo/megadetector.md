# :mag: Model Zoo and Release Schedules

|Models|Version Names|Licence|Release|Parameters (M)|mAR (Animal Class)|mAP50 (All Classes)|
|---|---|---|---|---|---|---|
|[MegaDetectorV5a](https://zenodo.org/records/15398270/files/md_v5a.0.0.pt?download=1)|a|AGPL-3.0|Released|139.9|81.7|92.0|
|[MegaDetectorV5b](https://zenodo.org/records/15398270/files/MegaDetector_v5b.0.0.pt?download=1)|b|AGPL-3.0|Released|139.9|80.9|90.1|
|[MegaDetectorV6-Ultralytics-YoloV9-Compact](https://zenodo.org/records/15398270/files/MDV6-yolov9-c.pt?download=1)|MDV6-yolov9-c|AGPL-3.0|Released|25.5|78.4|87.9|
|[MegaDetectorV6-Ultralytics-YoloV9-Extra](https://zenodo.org/records/15398270/files/MDV6-yolov9-e-1280.pt?download=1)|MDV6-yolov9-e|AGPL-3.0|Released|58.1|82.1|88.6|
|[MegaDetectorV6-Ultralytics-YoloV10-Compact](https://zenodo.org/records/15398270/files/MDV6-yolov10-c.pt?download=1) (even smaller and no NMS)|MDV6-yolov10-c|AGPL-3.0|Released|2.3|76.8|87.2|
|[MegaDetectorV6-Ultralytics-YoloV10-Extra](https://zenodo.org/records/15398270/files/MDV6-yolov10-e-1280.pt?download=1) (extra large model and no NMS)|MDV6-yolov10-e|AGPL-3.0|Released|29.5|82.8|92.8|
|[MegaDetectorV6-Ultralytics-RtDetr-Compact](https://zenodo.org/records/15398270/files/MDV6-rtdetr-c.pt?download=1)|MDV6-rtdetr-c|AGPL-3.0|Released|31.9|81.6|89.9|
|MegaDetectorV6-Ultralytics-YoloV11-Compact|-|AGPL-3.0|Will Not Release|2.6|76.0|87.6|
|MegaDetectorV6-Ultralytics-YoloV11-Extra|-|AGPL-3.0|Will Not Release|56.9|81.2|92.3|
|[MegaDetectorV6-MIT-YoloV9-Compact](https://zenodo.org/records/15398270/files/MDV6-mit-yolov9-c.ckpt?download=1)|MDV6-mit-yolov9-c|MIT|Released|9.7|74.8|87.6|
|[MegaDetectorV6-MIT-YoloV9-Extra](https://zenodo.org/records/15398270/files/MDV6-mit-yolov9-e.ckpt?download=1)|MDV6-mit-yolov9-e|MIT|Released|51|76.1|71.5|
|[MegaDetectorV6-Apache-RTDetr-Compact](https://zenodo.org/records/15398270/files/MDV6-apa-rtdetr-c.pth?download=1)|MDV6-apa-rtdetr-c|Apache|Released|20|81.1|91.0|
|[MegaDetectorV6-Apache-RTDetr-Extra](https://zenodo.org/records/15398270/files/MDV6-apa-rtdetr-e.pth?download=1)|MDV6-apa-rtdetr-e|Apache|Released|76|82.9|94.1|
|MegaDetector-Overhead|-|MIT|Mid 2025|-|||
|MegaDetector-Bioacoustics|-|MIT|Late 2025|-|||

<!-- |Models|Version Names|Licence|Release|Parameters (M)|mAP<sup>val<br>50-95|Animal Recall|
|---|---|---|---|---|---|---|
|MegaDetectorV5|-|AGPL-3.0|Released|121|74.7|74.9|
|MegaDetectorV6-Ultralytics-YoloV9-Compact|MDV6-yolov9-c|AGPL-3.0|Released|25.5|73.8|82.6|
|MegaDetectorV6-Ultralytics-YoloV9-Extra|MDV6-yolov9-e|AGPL-3.0|Released|58.1|80.2|87.1|
|MegaDetectorV6-Ultralytics-YoloV10-Compact (even smaller and no NMS)|MDV6-yolov10-c|AGPL-3.0|Released|2.3|71.8|78.8|
|MegaDetectorV6-Ultralytics-YoloV10-Extra (extra large model and no NMS)|MDV6-yolov10-c|AGPL-3.0|Released|29.5|79.9|85.2|
|MegaDetectorV6-Ultralytics-RtDetr-Compact|MDV6-redetr-c|AGPL-3.0|Released|31.9|73.9|83.4|
|MegaDetectorV6-Ultralytics-YoloV11-Compact|-|AGPL-3.0|Will Not Release|2.6|71.9|79.8|
|MegaDetectorV6-Ultralytics-YoloV11-Extra|-|AGPL-3.0|Will Not Release|56.9|79.3|86.0|
|MegaDetectorV6-MIT-YoloV9-Compact|MDV6-mit-yolov9-c|MIT|MDV6-mit-yolov9-c|February 2025|9.7|73.84|-|
|MegaDetectorV6-MIT-YoloV9-Extra|MDV6-mit-yolov9-c|MIT|February 2025|51|Training|Training|
|MegaDetectorV6-Apache-RTDetr-Compact|MDV6-apa-redetr-c|Apache|February 2025|20|76.3|-|
|MegaDetectorV6-Apache-RTDetr-Extra|MDV6-apa-redetr-c|Apache|February 2025|76|80.8|-| -->

> [!TIP]
> We are specifically reporting `Animal Recall` as our primary performance metric, even though it is not commonly used in traditional object detection studies, which typically focus on balancing overall model performance. For MegaDetector, our goal is to optimize for animal recall—in other words, minimizing false negative detections of animals or, more simply, ensuring our model misses as few animals as possible. While this may result in a higher false positive rate, we rely on downstream classification models to further filter the detected objects. We believe this approach is more practical for real-world animal monitoring scenarios.

>[!TIP]
>Some models, such as MegaDetectorV6, HerdNet, and AI4G-Amazon, have different versions, and they are loaded by their corresponding version names. Here is an example: `detection_model = pw_detection.MegaDetectorV6(version="MDV6-yolov10-e")`.
