## MegaDetector batch processing workflow integration

This folder contains information about ways to use MegaDetector output files in various workflows.  Specifically...

### Timelapse2

[Timelapse2](http://saul.cpsc.ucalgary.ca/timelapse/) can read the results produced by the [MegaDetector batch processing API](https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing) and/or [run_tf_detector_batch.py](https://github.com/microsoft/CameraTraps/blob/master/detection/run_tf_detector_batch.py), as well as the species classification results produced by our [classification pipeline](https://github.com/microsoft/CameraTraps/tree/master/classification).  For information about how to use this feature, see [timelapse.md](timelapse.md), but mostly see the section in the Timelapse manual called "Automatic Image Recognition".  If you're a Timelapse user, you may also want to check out our [guide to configuring Azure virtual machines](remote_desktop.md) to run Timelapse in the cloud, which can make it easier to split annotation workloads across your team.

### eMammal

A [standalone application](https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing/integration/eMammal) is available to transfer MegaDetector results from our .json results format into the [eMammal desktop client](https://emammal.si.edu/eyes-wildlife/content/downloading-desktop-application).  Many eMammal users also work with our results by splitting images into separate folders for animal/empty/vehicle/person using [this script](https://github.com/microsoft/CameraTraps/blob/master/api/batch_processing/postprocessing/separate_detections_into_folders.py), then either discarding the blanks or creating separate deployments for animal/empty/human.

### digiKam

[Python tools](digiKam/README.md) (which can be run with a GUI) to transfer MegaDetector results from our .json results format into XMP image metadata, specifically for use with[digiKam](https://www.digikam.org/).

### Data preparation

For any of these use cases, you may also want to check out our [Camera Trap JSON Manager App](https://github.com/microsoft/CameraTraps/blob/master/api/batch_processing/postprocessing/CameraTrapJsonManagerApp.md), which can help you split/modify our .json results files to break into smaller projects, adjust relative paths, etc.

If you use any of these tools &ndash; or if we're missing an important one &ndash; <a href="mailto:cameratraps@lila.science">email us</a>!


