![image](https://microsoft.github.io/CameraTraps/assets/Pytorch_Banner_transparentbk.png)

<div align="center"> 
<font size="6"> A Collaborative Deep Learning Framework for Conservation </font>
<br>
<hr>
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://img.shields.io/pypi/v/PytorchWildlife?color=limegreen" /></a> 
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://static.pepy.tech/badge/pytorchwildlife" /></a> 
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://img.shields.io/pypi/pyversions/PytorchWildlife" /></a> 
<a href="https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" /></a>
<a href="https://colab.research.google.com/drive/1rjqHrTMzEHkMualr4vB55dQWCsCKMNXi?usp=sharing"><img src="https://img.shields.io/badge/Colab-Demo-blue?logo=GoogleColab" /></a>
<!-- <a href="https://colab.research.google.com/drive/16-OjFVQ6nopuP-gfqofYBBY00oIgbcr1?usp=sharing"><img src="https://img.shields.io/badge/Colab-Video detection-blue?logo=GoogleColab" /></a> -->
<a href="https://cameratraps.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/read-docs-yellow?logo=ReadtheDocs" /></a>
<a href="https://github.com/microsoft/CameraTraps/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/PytorchWildlife" /></a>
<a href="https://discord.gg/TeEVxzaYtm"><img src="https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=Discord" /></a>
<br><br>
</div>

## 🐾 Introduction

At the core of our mission is the desire to create a harmonious space where conservation scientists from all over the globe can unite. Where they're able to share, grow, use datasets and deep learning architectures for wildlife conservation.
We've been inspired by the potential and capabilities of Megadetector, and we deeply value its contributions to the community. As we forge ahead with Pytorch-Wildlife, under which Megadetector now resides, please know that we remain committed to supporting, maintaining, and developing Megadetector, ensuring its continued relevance, expansion, and utility.

Pytorch-Wildlife is pip installable:
```
pip install PytorchWildlife
```

To use the newest version of MegaDetector with all the existing functionalities, you can use our [Hugging Face interface](https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife) or simply load the model with **Pytorch-Wildlife**. The weights will be automatically downloaded:
```python
from PytorchWildlife.models import detection as pw_detection
detection_model = pw_detection.MegaDetectorV6()
```

For those interested in accessing the previous MegaDetector repository, which utilizes the same `MegaDetectorV5` model weights and was primarily developed by Dan Morris during his time at Microsoft, please visit the [archive](https://github.com/microsoft/CameraTraps/blob/main/archive) directory, or you can visit this [forked repository](https://github.com/agentmorris/MegaDetector/tree/main) that Dan Morris is actively maintaining.

>[!TIP]
>If you have any questions regarding MegaDetector and Pytorch-Wildlife, please [email us](zhongqimiao@microsoft.com) or join us in our discord channel: [![](https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=PytorchWildife)](https://discord.gg/TeEVxzaYtm)


## 📣 Announcements

### 🎉🎉🎉 Pytorch-Wildlife Version 1.1.0 is out!
- MegaDetectorV6 is finally out! Please refer to our [next section](#racing_cardashdash-megadetectorv6-smaller-better-and-faster) and our [release notes](https://github.com/microsoft/CameraTraps/releases/tag/pw_v1.1.0) for more details! 
- We have incorporated a point-based overhead animal detection model into our model zoo called [HerdNet (Delplanque et al. 2022)](https://www.sciencedirect.com/science/article/pii/S092427162300031X?via%3Dihub). Two model weights are incorporated in this release, `HerdNet-general` (their default weights) and `HerdNet-ennedi` (their model trained on Ennedi 2019 datasets). More details can be found [here](PytorchWildlife/models/detection/herdnet/Herdnet.md) and in their original [repo](https://github.com/Alexandre-Delplanque/HerdNet). This is the first third-party model in Pytorch-Wildlife and the foundation of our expansion to overhead/aerial animal detection and classification. Please see our [HerdNet demo](demo/image_detection_demo_herdnet.ipynb) on how to use it!
- You can now load custom weights you fine-tuned on your own datasets using the [finetuning module](PW_FT_classification) directly in the Pytorch-Wildlife pipeline! Please see the [demo](demo/custom_weight_loading_v6.ipynb) on how to do it. You can also load it in our Gradio app!
- You can now automatically separate your image detections into folders based on detection results! Please see our [folder separation demo](demo/image_separation_demo_v6.ipynb) on how to do it. You can also test it in our Gradio demo!
- We have also simplified the batch detection pipeline. Now we do not need to define pytorch datasets and dataloaders specifically. Please make sure to change your code and check our [release notes](https://github.com/microsoft/CameraTraps/releases/tag/pw_v1.1.0) and our [new demo](demo/image_demo.py#58) for more details.


<details>
<summary><font size="3">👉 Click for more updates</font></summary>
  <li> Issues [#523](https://github.com/microsoft/CameraTraps/issues/523), [#524](https://github.com/microsoft/CameraTraps/issues/524) and [#526](https://github.com/microsoft/CameraTraps/issues/526) have been solved!
  <li> PyTorchWildlife is now compatible with Supervision 0.23+ and Python 3.10+!
  <li> CUDA 12.x compatibility. <br>
</details>

### :racing_car::dash::dash: MegaDetectorV6: SMALLER, BETTER, and FASTER!  

After a few months of public beta testing, we are finally ready to officially release our 6th version of MegaDetector, MegaDetectorV6! In the next generation of MegaDetector, we are focusing on computational efficiency, performance, mordernizing of model architectures, and licensing. We have trained multiple new models using different model architectures, including Yolo-v9, Yolo-v11, and RT-Detr for maximum user flexibility. We have a [rolling release schedule](#mag-model-zoo-and-release-schedules) for different versions of MegaDetectorV6, and in the first step, we are releasing the compact version of MegaDetectorV6 with Yolo-v9 (MDv6-ultralytics-yolov9-compact, MDv6-c in short). From now on, we encourage our users to use MegaDetectorV6 as their default animal detection model.

This MDv6-c model is optimized for performance and low-budget devices. It has only ***one-sixth (SMALLER)*** of the parameters of the previous MegaDetectorV5 and exhibits ***12% higher recall (BETTER)*** on animal detection in our validation datasets. In other words, MDv6-c has significantly fewer false negatives when detecting animals, making it a more robust animal detection model than MegaDetectorV5. Furthermore, one of our testers reported that the speed of MDv6-c is at least ***5 times FASTER*** than MegaDetectorV5 on their datasets.

|Models|Parameters|Precision|Recall|
|---|---|---|---|
|MegaDetectorV5|121M|0.96|0.73|
|MegaDetectroV6-c|22M|0.92|0.85|

Learn how to use MegaDetectorV6 in our [image demo](demo/image_detection_demo_v6.ipynb) and [video demo](demo/video_detection_demo_v6.ipynb).

### :bangbang: Model licensing  `(IMPORTANT!!)`

The **Pytorch-Wildlife** package is under MIT, however some of the models in the model zoo are not. For example, MegaDetectorV5, which is trained using the Ultralytics package, is under AGPL-3.0, and is not for closed-source comercial uses.
> [!IMPORTANT]
> THIS IS TRUE TO ALL EXISTING MEGADETECTORV5 MODELS IN ALL EXISTING FORKS THAT ARE TRAINED USING YOLOV5, AN ULTRALYTICS-DEVELOPED MODEL.

We want to make Pytorch-Wildlife a platform where different models with different licenses can be hosted and want to enable different usecases. To reduce user confusions, in our [model zoo](#mag-model-zoo) section, we list all existing and planed future models in our model zoo, their corresponding license, and release schedules. 

In addition, since the **Pytorch-Wildlife** package is under MIT, all the utility functions, including data pre-/post-processing functions and model fine-tuning functions in this packages are under MIT as well.

### :mag: Model Zoo and Release Schedules

#### Detection models
|Models|Licence|Release|
|---|---|---|
|MegaDetectorV5|AGPL-3.0|Released|
|MegaDetectroV6-Ultralytics-YoloV9-Compact|AGPL-3.0|Released|
|HerdNet-general|CC BY-NC-SA-4.0|Released|
|HerdNet-ennedi|CC BY-NC-SA-4.0|Released|
|MegaDetectroV6-Ultralytics-YoloV9-Extra|AGPL-3.0|November 2024|
|MegaDetectroV6-Ultralytics-YoloV10-Compact (even smaller and no NMS)|AGPL-3.0|November 2024|
|MegaDetectroV6-Ultralytics-YoloV10-Extra (extra large model and no NMS)|AGPL-3.0|November 2024|
|MegaDetectroV6-MIT-YoloV9-Compact|MIT|December 2024|
|MegaDetectroV6-MIT-YoloV9-Extra|MIT|December 2024|
|MegaDetectroV6-Ultralytics-YoloV11-Compact (better performance)|AGPL-3.0|December 2024|
|MegaDetectroV6-Ultralytics-YoloV11-Extra (better performance)|AGPL-3.0|December 2024|
|MegaDetectroV6-Apache-RTDetr-Compact|Apache|January 2025|
|MegaDetectroV6-Apache-RTDetr-Extra|Apache|January 2025|

#### Classification models
|Models|Licence|Release|
|---|---|---|
|AI4G-Oppossum|MIT|Released|
|AI4G-Amazon|MIT|Released|
|AI4G-Serengeti|MIT|Released|


## 👋 Welcome to Pytorch-Wildlife

**PyTorch-Wildlife** is a platform to create, modify, and share powerful AI conservation models. These models can be used for a variety of applications, including camera trap images, overhead images, underwater images, or bioacoustics. Your engagement with our work is greatly appreciated, and we eagerly await any feedback you may have.


The **Pytorch-Wildlife** library allows users to directly load the `MegaDetector` model weights for animal detection. We've fully refactored our codebase, prioritizing ease of use in model deployment and expansion. In addition to `MegaDetector`, **Pytorch-Wildlife** also accommodates a range of classification weights, such as those derived from the Amazon Rainforest dataset and the Opossum classification dataset. Explore the codebase and functionalities of **Pytorch-Wildlife** through our interactive [HuggingFace web app](https://huggingface.co/spaces/AndresHdzC/pytorch-wildlife) or local [demos and notebooks](https://github.com/microsoft/CameraTraps/tree/main/demo), designed to showcase the practical applications of our enhancements at [PyTorchWildlife](https://github.com/microsoft/CameraTraps/blob/main/INSTALLATION.md). You can find more information in our [documentation](https://cameratraps.readthedocs.io/en/latest/).

👇 Here is a brief example on how to perform detection and classification on a single image using `PyTorch-wildlife`
```python
import numpy as np
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification

img = np.random.randn(3, 1280, 1280)

# Detection
detection_model = pw_detection.MegaDetectorV6() # Model weights are automatically downloaded.
detection_result = detection_model.single_image_detection(img)

#Classification
classification_model = pw_classification.AI4GAmazonRainforest() # Model weights are automatically downloaded.
classification_results = classification_model.single_image_classification(img)
```

## ⚙️ Install Pytorch-Wildlife
```
pip install PytorchWildlife
```
Please refer to our [installation guide](https://github.com/microsoft/CameraTraps/blob/main/INSTALLATION.md) for more installation information.

## 🕵️ Explore Pytorch-Wildlife and MegaDetector with our Demo User Interface

If you want to directly try **Pytorch-Wildlife** with the AI models available, including `MegaDetector`, you can use our [**Gradio** interface](https://github.com/microsoft/CameraTraps/tree/main/demo). This interface allows users to directly load the `MegaDetector` model weights for animal detection. In addition, **Pytorch-Wildlife** also has two classification models in our initial version. One is trained from an Amazon Rainforest camera trap dataset and the other from a Galapagos opossum classification dataset (more details of these datasets will be published soon). To start, please follow the [installation instructions](https://github.com/microsoft/CameraTraps/blob/main/INSTALLATION.md) on how to run the Gradio interface! We also provide multiple [**Jupyter** notebooks](https://github.com/microsoft/CameraTraps/tree/main/demo) for demonstration.

![image](https://microsoft.github.io/CameraTraps/assets/gradio_UI.png)


## 🛠️ Core Features
   What are the core components of Pytorch-Wildlife?
![Pytorch-core-diagram](https://microsoft.github.io/CameraTraps/assets/Pytorch_Wildlife_core_figure.jpg)


### 🌐 Unified Framework:
  Pytorch-Wildlife integrates **four pivotal elements:**

▪ Machine Learning Models<br>
▪ Pre-trained Weights<br>
▪ Datasets<br>
▪ Utilities<br>

### 👷 Our work:
  In the provided graph, boxes outlined in red represent elements that will be added and remained fixed, while those in blue will be part of our development.


### 🚀 Inaugural Model:
  We're kickstarting with YOLO as our first available model, complemented by pre-trained weights from `MegaDetector`. We have `MegaDetectorV5`, which is the same `MegaDetector v5` model from the previous repository, and many different versions of `MegaDetectorV6` for different usecases.


### 📚 Expandable Repository:
  As we move forward, our platform will welcome new models and pre-trained weights for camera traps and bioacoustic analysis. We're excited to host contributions from global researchers through a dedicated submission platform.


### 📊 Datasets from LILA:
  Pytorch-Wildlife will also incorporate the vast datasets hosted on LILA, making it a treasure trove for conservation research.


### 🧰 Versatile Utilities:
  Our set of utilities spans from visualization tools to task-specific utilities, many inherited from Megadetector.


### 💻 User Interface Flexibility:
  While we provide a foundational user interface, our platform is designed to inspire. We encourage researchers to craft and share their unique interfaces, and we'll list both existing and new UIs from other collaborators for the community's benefit.


Let's shape the future of wildlife research, together! 🙌

## 🖼️ Examples

### Image detection using `MegaDetector`
<img src="https://microsoft.github.io/CameraTraps/assets/animal_det_1.JPG" alt="animal_det_1" width="400"/><br>
*Credits to Universidad de los Andes, Colombia.*

### Image classification with `MegaDetector` and `AI4GAmazonRainforest`
<img src="https://microsoft.github.io/CameraTraps/assets/animal_clas_1.png" alt="animal_clas_1" width="500"/><br>
*Credits to Universidad de los Andes, Colombia.*

### Opossum ID with `MegaDetector` and `AI4GOpossum`
<img src="https://microsoft.github.io/CameraTraps/assets/opossum_det.png" alt="opossum_det" width="500"/><br>
*Credits to the Agency for Regulation and Control of Biosecurity and Quarantine for Galápagos (ABG), Ecuador.*

## 🔥 Future highlights
- [ ] A detection model fine-tuning module to fine-tune your own detection model for Pytorch-Wildlife.
- [ ] Direct LILA connection for more training/validation data.
- [ ] More pretrained detection and classification models to expand the current model zoo.

To check the full version of the roadmap with completed tasks and long term goals, please click [here!](roadmaps.md).

## 🤜🤛 Collaboration with EcoAssist!
We are thrilled to announce our collaboration with [EcoAssist](https://addaxdatascience.com/ecoassist/#spp-models)---a powerful user interface software that enables users to directly load models from the PyTorch-Wildlife model zoo for image analysis on local computers. With EcoAssist, you can now utilize MegaDetectorV5 and the classification models---AI4GAmazonRainforest and AI4GOpossum---for automatic animal detection and identification, alongside a comprehensive suite of pre- and post-processing tools. This partnership aims to enhance the overall user experience with PyTorch-Wildlife models for a general audience. We will work closely to bring more features together for more efficient and effective wildlife analysis in the future.


## :fountain_pen: Cite us!
We have recently published a [summary paper on Pytorch-Wildlife](https://arxiv.org/abs/2405.12930). The paper has been accepted as an oral presentation at the [CV4Animals workshop](https://www.cv4animals.com/) at this CVPR 2024. Please feel free to cite us!

```
@misc{hernandez2024pytorchwildlife,
      title={Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation}, 
      author={Andres Hernandez and Zhongqi Miao and Luisa Vargas and Sara Beery and Rahul Dodhia and Juan Lavista},
      year={2024},
      eprint={2405.12930},
      archivePrefix={arXiv},
}
```

Also, don't forget to cite our original paper for MegaDetector: 

```
@misc{beery2019efficient,
      title={Efficient Pipeline for Camera Trap Image Review},
      author={Sara Beery and Dan Morris and Siyu Yang},
      year={2019}
      eprint={1907.06772},
      archivePrefix={arXiv},
}
```

## 🤝 Contributing
This project is open to your ideas and contributions. If you want to submit a pull request, we'll have some guidelines available soon.

We have adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [us](zhongqimiao@microsoft.com) with any additional questions or comments.

## License
This repository is licensed with the [MIT license](https://github.com/Microsoft/dotnet/blob/main/LICENSE).


## 👥 Existing Collaborators

The extensive collaborative efforts of Megadetector have genuinely inspired us, and we deeply value its significant contributions to the community. As we continue to advance with Pytorch-Wildlife, our commitment to delivering technical support to our existing partners on MegaDetector remains the same.

Here we list a few of the organizations that have used MegaDetector. We're only listing organizations who have given us permission to refer to them here or have posted publicly about their use of MegaDetector.

<details>
<summary><font size="3">👉 Full list of organizations</font></summary>

(Newly Added) [TerrOïko](https://www.terroiko.fr/) ([OCAPI platform](https://www.terroiko.fr/ocapi))

[Arizona Department of Environmental Quality](http://azdeq.gov/)

[Blackbird Environmental](https://blackbirdenv.com/)

[Camelot](https://camelotproject.org/)

[Canadian Parks and Wilderness Society (CPAWS) Northern Alberta Chapter](https://cpawsnab.org/)

[Conservation X Labs](https://conservationxlabs.com/)

[Czech University of Life Sciences Prague](https://www.czu.cz/en)

[EcoLogic Consultants Ltd.](https://www.consult-ecologic.com/)

[Estación Biológica de Doñana](http://www.ebd.csic.es/inicio)

[Idaho Department of Fish and Game](https://idfg.idaho.gov/)

[Island Conservation](https://www.islandconservation.org/)

[Myall Lakes Dingo Project](https://carnivorecoexistence.info/myall-lakes-dingo-project/)

[Point No Point Treaty Council](https://pnptc.org/)

[Ramat Hanadiv Nature Park](https://www.ramat-hanadiv.org.il/en/)

[SPEA (Portuguese Society for the Study of Birds)](https://spea.pt/en/)

[Synthetaic](https://www.synthetaic.com/)

[Taronga Conservation Society](https://taronga.org.au/)

[The Nature Conservancy in Wyoming](https://www.nature.org/en-us/about-us/where-we-work/united-states/wyoming/)

[TrapTagger](https://wildeyeconservation.org/trap-tagger-about/)

[Upper Yellowstone Watershed Group](https://www.upperyellowstone.org/)

[Applied Conservation Macro Ecology Lab](http://www.acmelab.ca/), University of Victoria

[Banff National Park Resource Conservation](https://www.pc.gc.ca/en/pn-np/ab/banff/nature/conservation), Parks Canada(https://www.pc.gc.ca/en/pn-np/ab/banff/nature/conservation)

[Blumstein Lab](https://blumsteinlab.eeb.ucla.edu/), UCLA

[Borderlands Research Institute](https://bri.sulross.edu/), Sul Ross State University

[Capitol Reef National Park](https://www.nps.gov/care/index.htm) / Utah Valley University

[Center for Biodiversity and Conservation](https://www.amnh.org/research/center-for-biodiversity-conservation), American Museum of Natural History

[Centre for Ecosystem Science](https://www.unsw.edu.au/research/), UNSW Sydney

[Cross-Cultural Ecology Lab](https://crossculturalecology.net/), Macquarie University

[DC Cat Count](https://hub.dccatcount.org/), led by the Humane Rescue Alliance

[Department of Fish and Wildlife Sciences](https://www.uidaho.edu/cnr/departments/fish-and-wildlife-sciences), University of Idaho

[Department of Wildlife Ecology and Conservation](https://wec.ifas.ufl.edu/), University of Florida

[Ecology and Conservation of Amazonian Vertebrates Research Group](https://www.researchgate.net/lab/Fernanda-Michalski-Lab-4), Federal University of Amapá

[Gola Forest Programma](https://www.rspb.org.uk/our-work/conservation/projects/scientific-support-for-the-gola-forest-programme/), Royal Society for the Protection of Birds (RSPB)

[Graeme Shannon's Research Group](https://wildliferesearch.co.uk/group-1), Bangor University

[Hamaarag](https://hamaarag.org.il/), The Steinhardt Museum of Natural History, Tel Aviv University

[Institut des Science de la Forêt Tempérée (ISFORT)](https://isfort.uqo.ca/), Université du Québec en Outaouais

[Lab of Dr. Bilal Habib](https://bhlab.in/about), the Wildlife Institute of India

[Mammal Spatial Ecology and Conservation Lab](https://labs.wsu.edu/dthornton/), Washington State University

[McLoughlin Lab in Population Ecology](http://mcloughlinlab.ca/lab/), University of Saskatchewan

[National Wildlife Refuge System, Southwest Region](https://www.fws.gov/about/region/southwest), U.S. Fish & Wildlife Service

[Northern Great Plains Program](https://nationalzoo.si.edu/news/restoring-americas-prairie), Smithsonian

[Quantitative Ecology Lab](https://depts.washington.edu/sefsqel/), University of Washington

[Santa Monica Mountains Recreation Area](https://www.nps.gov/samo/index.htm), National Park Service

[Seattle Urban Carnivore Project](https://www.zoo.org/seattlecarnivores), Woodland Park Zoo

[Serra dos Órgãos National Park](https://www.icmbio.gov.br/parnaserradosorgaos/), ICMBio

[Snapshot USA](https://emammal.si.edu/snapshot-usa), Smithsonian

[Wildlife Coexistence Lab](https://wildlife.forestry.ubc.ca/), University of British Columbia

[Wildlife Research](https://www.dfw.state.or.us/wildlife/research/index.asp), Oregon Department of Fish and Wildlife

[Wildlife Division](https://www.michigan.gov/dnr/about/contact/wildlife), Michigan Department of Natural Resources

Department of Ecology, TU Berlin

Ghost Cat Analytics

Protected Areas Unit, Canadian Wildlife Service

[School of Natural Sciences](https://www.utas.edu.au/natural-sciences), University of Tasmania [(story)](https://www.utas.edu.au/about/news-and-stories/articles/2022/1204-innovative-camera-network-keeps-close-eye-on-tassie-wildlife)

[Kenai National Wildlife Refuge](https://www.fws.gov/refuge/kenai), U.S. Fish & Wildlife Service [(story)](https://www.peninsulaclarion.com/sports/refuge-notebook-new-technology-increases-efficiency-of-refuge-cameras/)

[Australian Wildlife Conservancy](https://www.australianwildlife.org/) [(blog](https://www.australianwildlife.org/cutting-edge-technology-delivering-efficiency-gains-in-conservation/), [blog)](https://www.australianwildlife.org/efficiency-gains-at-the-cutting-edge-of-technology/)

[Felidae Conservation Fund](https://felidaefund.org/) [(WildePod platform)](https://wildepod.org/) [(blog post)](https://abhaykashyap.com/blog/ai-powered-camera-trap-image-annotation-system/)

[Alberta Biodiversity Monitoring Institute (ABMI)](https://www.abmi.ca/home.html) [(WildTrax platform)](https://www.wildtrax.ca/) [(blog post)](https://wildcams.ca/blog/the-abmi-visits-the-zoo/)

[Shan Shui Conservation Center](http://en.shanshui.org/) [(blog post)](https://mp.weixin.qq.com/s/iOIQF3ckj0-rEG4yJgerYw?fbclid=IwAR0alwiWbe3udIcFvqqwm7y5qgr9hZpjr871FZIa-ErGUukZ7yJ3ZhgCevs) [(translated blog post)](https://mp-weixin-qq-com.translate.goog/s/iOIQF3ckj0-rEG4yJgerYw?fbclid=IwAR0alwiWbe3udIcFvqqwm7y5qgr9hZpjr871FZIa-ErGUukZ7yJ3ZhgCevs&_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp)

[Irvine Ranch Conservancy](http://www.irconservancy.org/) [(story)](https://www.ocregister.com/2022/03/30/ai-software-is-helping-researchers-focus-on-learning-about-ocs-wild-animals/)

[Wildlife Protection Solutions](https://wildlifeprotectionsolutions.org/) [(story](https://customers.microsoft.com/en-us/story/1384184517929343083-wildlife-protection-solutions-nonprofit-ai-for-earth), [story)](https://www.enterpriseai.news/2023/02/20/ai-helps-wildlife-protection-solutions-safeguard-endangered-species/)

[Road Ecology Center](https://roadecology.ucdavis.edu/), University of California, Davis [(Wildlife Observer Network platform)](https://wildlifeobserver.net/)

[The Nature Conservancy in California](https://www.nature.org/en-us/about-us/where-we-work/united-states/california/) [(Animl platform)](https://github.com/tnc-ca-geo/animl-frontend)

[San Diego Zoo Wildlife Alliance](https://science.sandiegozoo.org/) [(Animl R package)](https://github.com/conservationtechlab/animl)

</details><br>


>[!IMPORTANT]
>If you would like to be added to this list or have any questions regarding MegaDetector and Pytorch-Wildlife, please [email us](zhongqimiao@microsoft.com) or join us in our Discord channel: [![](https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=PytorchWildife)](https://discord.gg/TeEVxzaYtm)

