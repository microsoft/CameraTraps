# Overview

This repo contains the tools for training, running, and evaluating detectors and classifiers for images collected from motion-triggered camera traps.  The core functionality provided is:

- Data parsing from frequently-used camera trap metadata formats into a common format
- Training and evaluation of detectors, particularly [MegaDetector](megadetector.md), an object detection model that does a pretty good job finding animals, people, and vehicles (and therefore is pretty good at finding empty images) in a variety of terrestrial ecosystems
- A [batch processing API](https://github.com/microsoft/CameraTraps/tree/main/api/batch_processing) that runs MegaDetector on large image collections, to accelerate population surveys
- A [real-time API](https://github.com/microsoft/CameraTraps/tree/main/api/synchronous) that runs MegaDetector (and some species classifiers) synchronously, primarily to support anti-poaching scenarios (e.g. see this [blog post](https://customers.microsoft.com/en-us/story/1384184517929343083-wildlife-protection-solutions-nonprofit-ai-for-earth) describing how this API supports [Wildlife Protection Solutions](https://wildlifeprotectionsolutions.org/))
- A [Web-based demo](https://github.com/microsoft/CameraTraps/tree/main/demo) that calls our real-time API
- Training and evaluation of species-level classifiers for specific data sets
- Miscellaneous useful tools for manipulating camera trap data
- Research experiments we're doing around camera trap data (i.e., some directories are highly experimental and you should take them with a grain of salt)

This repo is maintained by folks at [Ecologize](http://ecologize.org/) and folks in the [Microsoft AI for Earth](http://aka.ms/aiforearth) program who like looking at pictures of animals.  We want to support conservation, of course, but we also really like looking at pictures of animals.


# What's MegaDetector all about?

The main model that we train and run using tools in this repo is [MegaDetector](megadetector.md), an object detection model that identifies animals, people, and vehicles in camera trap images.  This model is trained on several hundred thousand bounding boxes from a variety of ecosystems.  Lots more information &ndash; including download links and instructions for running the model &ndash; is available on the [MegaDetector page](megadetector.md).

Here's a "teaser" image of what detector output looks like:

![Red bounding box on fox](images/detector_example.jpg)

Image credit University of Washington.


# How do I get started?

If you're just considering the use of AI in your workflow, and aren't even sure yet whether MegaDetector would be useful to you, we recommend reading [this page](collaborations.md) first.

If you're already familiar with MegaDetector and you're ready to run it on your data (and you have some familiarity with running Python code), see the [MegaDetector README](megadetector.md) for instructions on downloading and running MegaDetector.


# Who is using MegaDetector?

We work with ecologists all over the world to help them spend less time annotating images and more time thinking about conservation.  You can read a little more about how this works on our [getting started with MegaDetector](collaborations.md) page.

Here are a few of the organizations that have used MegaDetector... we're only listing organizations who (a) we know about and (b) have kindly given us permission to refer to them here, so if you're using MegaDetector or other tools from this repo and would like to be added to this list, <a href="mailto:cameratraps@lila.science">email us</a>!

* Idaho Department of Fish and Game
* San Diego Zoo Global
* University of Washington Quantitative Ecology Lab
* University of Idaho
* Borderlands Research Institute at Sul Ross State University
* Parks Canada
* Australian Wildlife Conservancy
* Lab of Dr. Bilal Habib at the Wildlife Institute of India
* Royal Society for the Protection of Birds (RSPB)
* Wildlife Protection Solutions
* Island Conservation
* Synthetaic
* School of Natural Sciences, University of Tasmania
* Arizona Department of Environmental Quality
* Wildlife Research, Oregon Department of Fish and Wildlife
* National Wildlife Refuge System, Southwest Region, US Fish and Wildlife
* Mammal Spatial Ecology and Conservation Lab at Washington State University
* Point No Point Treaty Council
* SPEA (Portuguese Society for the Study of Birds)
* Ghost Cat Analytics
* EcoLogic Consultants Ltd.
* Smithsonian Northern Great Plains Program
* Federal University of Amapá, Ecology and Conservation of Amazonian Vertebrates Research Group
* Hamaarag, The Steinhardt Museum of Natural History, Tel Aviv University
* Czech University of Life Sciences Prague
* Ramat Hanadiv Nature Park, Israel
* TU Berlin, Department of Ecology
* DC Cat Count, led by the Humane Rescue Alliance
* Center for Biodiversity and Conservation at the American Museum of Natural History
* Camelot
* Graeme Shannon's Research Group at Bangor University 
* Snapshot USA
* University of British Columbia Wildlife Coexistence Lab
* Michigan Department of Natural Resources, Wildlife Division
* Serra dos Órgãos National Park / ICMBio
* McLoughlin Lab in Population Ecology, University of Saskatchewan
* Upper Yellowstone Watershed Group
* Blumstein lab, UCLA
* National Park Service Santa Monica Mountains Recreation Area
* Conservation X Labs
* The Nature Conservancy in Wyoming
* Seattle Urban Carnivore Project
* Road Ecology Center, University of California, Davis
* [TrapTagger](https://wildeyeconservation.org/trap-tagger-about/)
* Blackbird Environmental
* UNSW Sydney
* Taronga Conservation Society
* Myall Lakes Dingo Project
* Irvine Ranch Conservancy ([story](https://www.ocregister.com/2022/03/30/ai-software-is-helping-researchers-focus-on-learning-about-ocs-wild-animals/))
* SUMHAL, Estación Biológica de Doñana
* Capitol Reef National Park and Utah Valley University
* University of Victoria Applied Conservation Macro Ecology (ACME) Lab 
* Université du Québec en Outaouais Institut des Science de la Forêt Tempérée (ISFORT)

# Data

This repo does not directly host camera trap data, but we work with our collaborators to make data and annotations available whenever possible on [lila.science](http://lila.science).


# Contact

For questions about this repo, contact [cameratraps@lila.science](mailto:cameratraps@lila.science).


# Contents

This repo is organized into the following folders...


## api

Code for hosting our models as an API, either for synchronous operation (e.g. for real-time inference or for our Web-based demo) or as a batch process (for large biodiversity surveys).  The synchronous API in this folder does a bunch of fancy load-balancing stuff, in comparison to...


## api-flask-redis

Code for a simplified synchronous API that runs as a single-node Flask app.


## classification

Experimental code for training species classifiers on new data sets, generally trained on MegaDetector crops.  Currently the main pipeline described in this folder relies on a large database of labeled images that is not publicly available; therefore, this folder is not yet set up to facilitate training of your own classifiers.  However, it is useful for <i>users</i> of the classifiers that we train, and contains some useful starting points if you are going to take a "DIY" approach to training classifiers on cropped images.  

All that said, here's another "teaser image" of what you get at the end of training and running a classifier:

<img src="images/warthog_classifications.jpg" width="700">


## data_management

Code for:

* Converting frequently-used metadata formats to [COCO Camera Traps](https://github.com/Microsoft/CameraTraps/blob/main/data_management/README.md#coco-cameratraps-format) format
* Creating, visualizing, and  editing COCO Camera Traps .json databases


## demo

Source for the Web-based demo of our MegaDetector model.


## detection

Code for training, running, and evaluating MegaDetector.


## research

Ongoing research projects that use this repository in one way or another; as of the time I'm editing this README, there are projects in this folder around active learning and the use of simulated environments for training data augmentation.


## sandbox

Random things that don't fit in any other directory.  For example:

* A not-super-useful but super-duper-satisfying and mostly-successful attempt to use OCR to pull metadata out of image pixels in a fairly generic way, to handle those pesky cases when image metadata is lost.
* Experimental postprocessing scripts that were built for a single use case


## taxonomy-mapping

Code to facilitate mapping data-set-specific categories (e.g. "lion", which means very different things in Idaho vs. South Africa) to a standard taxonomy.


## test-images

A handful of images from LILA that facilitate testing and debugging.


## visualization

Shared tools for visualizing images with ground truth and/or predicted annotations.


# Gratuitous pretty camera trap picture

![Bird flying above water](images/nacti.jpg)

Image credit USDA, from the [NACTI](http://lila.science/datasets/nacti) data set.


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [cla.microsoft.com](https://cla.microsoft.com).

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License

This repository is licensed with the [MIT license](https://github.com/Microsoft/dotnet/blob/main/LICENSE).
