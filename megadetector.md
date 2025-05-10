# :racing_car::dash::dash: MegaDetectorV6: SMALLER, FASTER, BETTER!  
We have officially released our 6th version of MegaDetector, MegaDetectorV6! In the next generation of MegaDetector, we are focusing on computational efficiency, performance, modernizing of model architectures, and licensing. We have trained multiple new models using different model architectures, including Yolo-v9, Yolo-v10, and RT-Detr for maximum user flexibility. We have a [rolling release schedule](#mag-model-zoo-and-release-schedules) for different versions of MegaDetectorV6.

>[!NOTE]
> - Following our initial release, we’ve been delighted to see so many people explore our new models. We’d like to extend our heartfelt thanks to everyone who has shown interest in our latest models—your support means a great deal to us!
> - That said, we’ve received a number of feedback comments highlighting a discrepancy between the reported performance (particularly MDV5) and the actual performance observed. We are actively investigating this issue and have identified a potential error or corruption in the validation data we used. For the time being, we’ll remove our current performance numbers from the model zoo for now to avoid confusion.
> - We sincerely apologize for any confusion or inconvenience this may have caused. Our team is working diligently to address this matter, and we will update our experiments—and potentially retrain the model if data corruption is confirmed—as soon as possible. Thank you for your patience and understanding!



MegaDetectorV6 models are based on architectures optimized for performance and low-budget devices. For example, the MegaDetectorV6-Ultralytics-YoloV10-Compact (MDV6-yolov10-c) model only have ***2% of the parameters*** of the previous MegaDetectorV5 and still exhibits comparable animal recall on our validation datasets. 

<!-- In the following figure, we can see the Performance to Parameter metric of each released MegaDetector model. All of the V6 models, extra large or compact, have at least 50% less parameters compared to MegaDetectorV5 but with much higher animal detection performance. -->

<!-- ![image](assets/ParamPerf.png) -->

<!-- >[!TIP] -->
<!-- >From now on, we encourage our users to use MegaDetectorV6 as their default animal detection model and choose whichever model that fits the project needs. To reduce potential confusion, we have also standardized the model names into MDV6-Compact and MDV6-Extra for two model sizes using the same architecture. Learn how to use MegaDetectorV6 in our [image demo](demo/image_detection_demo_v6.ipynb) and [video demo](demo/video_detection_demo_v6.ipynb). -->


## :bangbang: Model licensing 

The **Pytorch-Wildlife** package is under MIT, however some of the models in the model zoo are not. For example, MegaDetectorV5, which is trained using the Ultralytics package, a package under AGPL-3.0, and is not for closed-source commercial uses if they are using updated 'ultralytics' packages. 

There may be a confusion because YOLOv5 was initially released before the establishment of the AGPL-3.0 license. According to the official [Ultralytics-Yolov5](https://github.com/ultralytics/yolov5) package, it is under AGPL-3.0 now, and the maintainers have discussed how their licensing policy has evolved over time in their issues section. 

<!-- We aim to prevent any confusion or potential issues for our users. -->

<!-- > [!IMPORTANT]
> THIS IS TRUE TO ALL EXISTING MEGADETECTORV5 MODELS IN ALL EXISTING FORKS THAT ARE TRAINED USING YOLOV5, AN ULTRALYTICS-DEVELOPED MODEL. -->

We want to make Pytorch-Wildlife a platform where different models with different licenses can be hosted and want to enable different use cases. To reduce user confusions, in our [model zoo](#mag-model-zoo-and-release-schedules) section, we list all existing and planned future models in our model zoo, their corresponding license, and release schedules. 

In addition, since the **Pytorch-Wildlife** package is under MIT, all the utility functions, including data pre-/post-processing functions and model fine-tuning functions in this packages are under MIT as well.

## :mag: Model Zoo and Release Schedules

### MegaDetectors 
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

 