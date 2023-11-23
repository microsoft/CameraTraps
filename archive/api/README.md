# Announcement

At the core of our mission is the desire to create a harmonious space where conservation scientists from all over the globe can unite, share, and grow. We are expanding the CameraTraps repo to introduce **Pytorch-Wildlife**, a Collaborative Deep Learning Framework for Conservation, where researchers can come together to share and use datasets and deep learning architectures for wildlife conservation.
 
We've been inspired by the potential and capabilities of Megadetector, and we deeply value its contributions to the community. **As we forge ahead with Pytorch-Wildlife, under which Megadetector now resides, please know that we remain committed to supporting, maintaining, and developing Megadetector, ensuring its continued relevance, expansion, and utility.**

To use the newest version of MegaDetector with all the exisitng functionatlities, you can use our newly developed [user interface](#explore-pytorch-wildlife-and-megadetector-with-our-user-interface) or simply load the model with **Pytorch-Wildlife** and the weights will be automatically downloaded:

```python
from PytorchWildlife.models import detection as pw_detection
detection_model = pw_detection.MegaDetectorV5()
```

If you'd like to learn more about **Pytorch-Wildlife**, please continue reading.

For those interested in accessing the previous MegaDetector repository, which utilizes the same `MegaDetector v5` model weights and was primarily developed by Dan Morris during his time at Microsoft, please visit the [archive](./archive) directory, or you can visit this [forked repository](https://github.com/agentmorris/MegaDetector/tree/main) that Dan Morris is actively maintaining.
 
**If you have any questions regarding MegaDetector and Pytorch-Wildlife, please <a href="mailto:zhongqimiao@microsoft.com">email us</a>!**

# Camera Trap Image Processing APIs

Though most of our users either use the MegaDetector model directly or work with us to run MegaDetector on the cloud, we also package useful components developed in the Camera Traps project into APIs that users can operate (on the cloud or on local computers) to process camera trap images in a variety of scenarios. This folder contains the source code of the APIs and documentation on how to set them up.


### Synchronous API

This API is intended for real-time scenarios where a small number of images are processed at a time and latency is a priority.  See documentation [here](synchronous).

### Batch processing API

This API runs the detector on lots of images (typically millions) and distributes the work over potentially many nodes using [Azure Batch](https://azure.microsoft.com/en-us/services/batch/). See documentation [here](batch_processing).

