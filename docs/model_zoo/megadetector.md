# :mag: Model Zoo and Release Schedules

|Models|Version Names|Licence|Release|Parameters (M)|
|---|---|---|---|---|
|MegaDetectorV5|-|AGPL-3.0|Released|121|
|MegaDetectorV6-Ultralytics-YoloV9-Compact|MDV6-yolov9-c|AGPL-3.0|Released|25.5|
|MegaDetectorV6-Ultralytics-YoloV9-Extra|MDV6-yolov9-e|AGPL-3.0|Released|58.1|
|MegaDetectorV6-Ultralytics-YoloV10-Compact (even smaller and no NMS)|MDV6-yolov10-c|AGPL-3.0|Released|2.3|
|MegaDetectorV6-Ultralytics-YoloV10-Extra (extra large model and no NMS)|MDV6-yolov10-c|AGPL-3.0|Released|29.5|
|MegaDetectorV6-Ultralytics-RtDetr-Compact|MDV6-redetr-c|AGPL-3.0|Released|31.9|
|MegaDetectorV6-Ultralytics-YoloV11-Compact|-|AGPL-3.0|Will Not Release|2.6|
|MegaDetectorV6-Ultralytics-YoloV11-Extra|-|AGPL-3.0|Will Not Release|56.9|
|MegaDetectorV6-MIT-YoloV9-Compact|MDV6-mit-yolov9-c|MIT|Training|9.7|
|MegaDetectorV6-MIT-YoloV9-Extra|MDV6-mit-yolov9-c|MIT|Training|51|
|MegaDetectorV6-Apache-RTDetr-Compact|MDV6-apa-redetr-c|Apache|Training|20|
|MegaDetectorV6-Apache-RTDetr-Extra|MDV6-apa-redetr-c|Apache|Training|76|
|MegaDetector-Overhead|-|MIT|Mid 2025|-|
|MegaDetector-Bioacoustics|-|MIT|Late 2025|-|

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
