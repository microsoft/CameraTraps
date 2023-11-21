# Camera Trap Image Processing APIs

Though most of our users either use the MegaDetector model directly or work with us to run MegaDetector on the cloud, we also package useful components developed in the Camera Traps project into APIs that users can operate (on the cloud or on local computers) to process camera trap images in a variety of scenarios. This folder contains the source code of the APIs and documentation on how to set them up.


### Synchronous API

This API is intended for real-time scenarios where a small number of images are processed at a time and latency is a priority.  See documentation [here](synchronous).

### Batch processing API

This API runs the detector on lots of images (typically millions) and distributes the work over potentially many nodes using [Azure Batch](https://azure.microsoft.com/en-us/services/batch/). See documentation [here](batch_processing).

