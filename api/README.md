# Camera Trap Image Processing APIs

Though most of our users either use the MegaDetector model directly or work with us to run MegaDetector on the cloud, we also package useful components developed in the Camera Traps project into APIs that users can operate (on the cloud or on local computers) to process camera trap images in a variety of scenarios. This folder contains the source code of the APIs and documentation on how to set them up.


## Detector

Our animal detection model ([MegaDetector](https://github.com/Microsoft/CameraTraps#megadetector)) trained on camera trap images from a variety of ecosystems can be served via two APIs, one for real-time applications or small batches of test images (synchronous API), and one for processing large collections of images (batch processing API). These APIs can be adapted to deploy any algorithms or models &ndash; see our tutorial in the [AI for Earth API Framework](https://github.com/Microsoft/AIforEarth-API-Development) repo.


### Synchronous API

This API&rsquo;s `/detect` endpoint processes up to 8 images at a time, and optionally returns copies of the input images annotated with the detection bounding boxes. This API powers the [demo](../demo) web app.

To build the API, first download the [MegaDetector](https://github.com/Microsoft/CameraTraps#megadetector) model file to `detector_synchronous/api/animal_detection_api/model`.


### Batch processing API

This API runs the detector on up to two million images in one request using [Azure Batch](https://azure.microsoft.com/en-us/services/batch/). To use this API the input images need to be copied to Azure [Blob Storage](https://azure.microsoft.com/en-us/services/storage/blobs/). Please see the [user guide](./batch_processing/README.md) and get in touch with us if you&rsquo;re interested in standing up your own instane of the batch processing API. 

The [batch_processing](batch_processing) folder includes the source for the API itself, tools for working with the results the API generates, and support for integrating our API output with other tools.

