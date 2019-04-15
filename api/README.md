# Camera Traps APIs

We package useful components developed in the Camera Traps project into APIs and host them using the [AI for Earth API Framework](https://github.com/Microsoft/AIforEarth-API-Development) on Azure. This folder contains the source code of the APIs and their documentation. To see the APIs in action or start using them for your own applications, visit our APIs [homepage](https://www.microsoft.com/en-us/ai/ai-for-earth-apis?activetab=pivot1%3aprimaryr3) to check out our demo apps and request a product key.


## Detector

Our animal detection model ([MegaDetector](https://github.com/Microsoft/CameraTraps#megadetector)) trained on camera trap images from a variety of ecosystems is exposed through two APIs, one for real-time applications or small batches of test images (synchronous API), and one for processing large collections of images (batch processing API). These APIs can be adapted to deploy any algorithms or models - see our tutorial in the [AI for Earth API Framework](https://github.com/Microsoft/AIforEarth-API-Development) repo.


### Synchronous API

The `/detect` endpoint here processes up to 8 images at a time, and optionally returns copies of the input images annotated with the detection bounding boxes. This API powers the [demo](../demo) web app.

To build the API, first download the model file from the [MegaDetector](https://github.com/Microsoft/CameraTraps#megadetector) section to `detector_synchronous/api/animal_detection_api/model`.


### Batch processing API

This API runs the detector on up to 2 million images in one request using [Azure Machine Learning Service](https://azure.microsoft.com/en-us/services/machine-learning-service/)'s _Managed Compute_ functionality, formerly known as Batch AI. To use this API the input images need to be copied to Azure [Blob Storage](https://azure.microsoft.com/en-us/services/storage/blobs/). Please see the [user guide](./detector_batch_processing/README.md) and get in touch with us if you're interested in processing camera trap images this way. 

Upcoming improvements:
- [ ] Adapt `runserver.py` to use the newest version of the AI4E API Framework
- [ ] Process a sample of `sample_n` images from all input images
- [ ] Allow the job status monitoring thread to be re-started
- [ ] More checks on the input container and image list SAS keys
