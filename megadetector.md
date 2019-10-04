## MegaDetector overview

Conservation biologists invest a huge amount of time reviewing camera trap images, and &ndash; even worse &ndash; a huge fraction of that time is spent reviewing images they aren&rsquo;t interested in.  This primarily includes empty images, but for many projects, images of people and vehicles are also &ldquo;noise&rdquo;, or at least need to be handled separately from animals.

<i>Machine learning can accelerate this process, letting biologists spend their time on the images that matter.</i>

To this end, this page hosts a model we&rsquo;ve trained to detect animals, people, and vehicles in camera trap images, using several hundred thousand bounding boxes from a variety of ecosystems.  It does not identify animals, it just finds them.  The current model is based on Faster-RCNN with an InceptionResNetv2 base network, and was trained with the TensorFlow Object Detection API.  We use this model as our first stage for classifier training and inference.

## Downloading the model(s)

### MegaDetector v3, 2019.05.30

#### Release notes

In addition to incorporating additional data, this release adds a preliminary &ldquo;human&rdquo; class.  Our animal training data is still far more comprehensive than our humans-in-camera-traps data, so if you&rsquo;re interested in using our detector but find that it works better on animals than people, stay tuned.

#### Download links

- [Frozen model (.pb)](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v3.pb)
- [TFODAPI config file](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v3.config)
- [Last checkpoint (for resuming training)](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v3_checkpoint.zip)
- [TensorFlow SavedModel](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/saved_model_normalized_megadetector_v3_tf19.zip) (inputs in tf [common image format](https://www.tensorflow.org/hub/common_signatures/images#image_input), default output signature)
- [Tensorflow SavedModel for TFServing](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/saved_model_megadetector_v3_tf19.zip) (inputs in uint8 format, serving_default output signature)

### MegaDetector v2, 2018

#### Release notes

First MegaDetector release!

#### Download links

- [Frozen model (.pb)](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v2.pb)
- [TFODAPI config file](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v2.config)
- [Last checkpoint (for resuming training)](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v2_checkpoint.zip)


## Using the models 

We provide three ways to apply this model to new images:

- To &ldquo;test drive&rdquo; this model on small sets of images and get super-satisfying visual output, we provide [run_tf_detector.py](https://github.com/Microsoft/CameraTraps/blob/master/detection/run_tf_detector.py), an example script for invoking this detector on new images.  This script doesn&rsquo;t depend on anything else in our repo, so you can download it and give it a try.  [Let us know](mailto:cameratraps@microsoft.com) how it works on your images!
- To apply this model to larger image sets on a single machine, we recommend a slightly different script, [run_tf_detector_batch](https://github.com/Microsoft/CameraTraps/blob/master/detection/run_tf_detector_batch.py).  This outputs data in the same format as our [batch processing API](https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing), so you can leverage all of our post-processing tools.
- Speaking of which, when we process loads of images from collaborators, we use our [batch processing API](https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing), which we can make available externally on request.  [Email us](mailto:cameratraps@microsoft.com) for more information.

If you use the MegaDetector in a publication, please cite: 
```
@article{beery2019efficient,
  title={Efficient Pipeline for Camera Trap Image Review},
  author={Beery, Sara and Morris, Dan and Yang, Siyu},
  journal={arXiv preprint arXiv:1907.06772},
  year={2019}
}
```

## Tell me more about why detectors are a good first step for camera trap images

Can do!  See these &ldquo;<a href="http://dmorris.net/misc/cameratraps/ai4e_camera_traps_overview">slides</a>&rdquo;.

## Pretty picture

Here&rsquo;s a &ldquo;teaser&rdquo; image of what detector output looks like:

![alt text](images/detector_example.jpg "Red bounding box on fox")

Image credit University of Washington.


## Mesmerizing video

Here&rsquo;s a neat <a href="http://dolphinvm.westus2.cloudapp.azure.com/video/detector_video.html">video</a> of our v2 detector running in a variety of ecosystems, on locations unseen during training.

<a href="http://dolphinvm.westus2.cloudapp.azure.com/video/detector_video.html"><img width=600 src="http://dolphinvm.westus2.cloudapp.azure.com/video/mvideo.jpg"></a><br/>

Image credit <a href="https://emammal.si.edu/">eMammal</a>.


## Can you share the training data?

This model is trained on bounding boxes from a variety of ecosystems, and many of the images we use in training are not publicly-shareable for license reasons.  We do train in part on bounding boxes from two public data sets:

- [Caltech Camera Traps](http://lila.science/datasets/caltech-camera-traps)
- [Snapshot Serengeti](http://lila.science/datasets/snapshot-serengeti)

...so if our detector performs really well on those data sets, that&rsquo;s great, but it&rsquo;s a little bit cheating, because we haven&rsquo;t published the set of locations from those data sets that we use during training.
