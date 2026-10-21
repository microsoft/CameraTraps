![A colorful banner illustrating various species of animals and plants in a natural environment, symbolizing biodiversity and the use of AI for conservation purposes.](https://zenodo.org/records/20044680/files/Biodiversity_Banner.png)

# Microsoft Biodiversity

**Open-source AI for biodiversity monitoring and conservation.**  
Microsoft AI for Good Lab — camera-trap detection, bioacoustic analysis, species classification, field deployment.

<div align="center">
<font size="6"> Open-source AI for camera traps, bioacoustics, and wildlife monitoring </font>
<br>
<hr>
<a href="https://github.com/microsoft/Biodiversity/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" /></a>
<a href="https://discord.gg/TeEVxzaYtm"><img src="https://img.shields.io/badge/Discord-Join_us-5865F2?logo=discord&logoColor=white" /></a>
<a href="https://microsoft.github.io/Biodiversity/"><img src="https://img.shields.io/badge/Docs-526CFE?logo=MaterialForMkDocs&logoColor=white" /></a>
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://static.pepy.tech/badge/pytorchwildlife" /></a>
<a href="https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" /></a>
<a href="https://colab.research.google.com/drive/1rjqHrTMzEHkMualr4vB55dQWCsCKMNXi?usp=sharing"><img src="https://img.shields.io/badge/Colab-Demo-blue?logo=GoogleColab" /></a>
<br><br>
</div>

## 📣 Announcements

### What we've been up to

Our journey started with **MegaDetector** — a camera-trap animal detection model that became a widely adopted tool in the conservation community. Building on that foundation, we created **PyTorch-Wildlife** as a unified platform to host all of our AI for biodiversity work, bringing together detection, classification, and eventually much more.

Over time, our scope grew well beyond camera-trap imagery. We now have active work in bioacoustics, overhead animal detection, and edge computing for remote field deployments. As the ecosystem expanded, it became clear that keeping everything inside a single repository was working against us. Code was harder to find, harder to maintain, and harder to extend.

So we made a deliberate decision: break the work into focused, dedicated repositories — one per project — where the code in each repo is concentrated, the ownership is clear, and future contributors know exactly where to go. This repository is the hub that ties them together. PyTorch-Wildlife now lives at [microsoft/Pytorch-Wildlife](https://github.com/microsoft/Pytorch-Wildlife), MegaDetector at [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector), and everything else is linked in the table below.

### PyTorch-Wildlife is becoming SPARROW-Playground

PyTorch-Wildlife is being fully revamped as **[SPARROW-Playground](https://github.com/microsoft/SPARROW-Playground)**, an inference-only playground for the SPARROW ecosystem. Inference moves from PyTorch to the [SPARROW-Engine](https://github.com/microsoft/SPARROW-Engine) Python package (`pip install sparrow-engine`), which gives direct access to the full model zoo without a deep-learning framework. Notebooks are rewritten around SPARROW-Engine and will add data ingestion from cellular (4G) cameras, FTPS cameras, and SPARROW devices.

- The `PytorchWildlife` PyPI package is no longer maintained. Its final release (1.3.1) warns on import and points to `sparrow-engine`; there will be no further releases.
- Fine-tuning has moved to [microsoft/MegaDetector-Classifier](https://github.com/microsoft/MegaDetector-Classifier).
- [microsoft/Pytorch-Wildlife](https://github.com/microsoft/Pytorch-Wildlife) will be archived. Its documentation pages, and the PyTorch-Wildlife pages on this site, will redirect to SPARROW-Playground.

#### Previous versions:
- [Release notes](https://microsoft.github.io/Biodiversity/releases/release_notes/)

## Projects

| Repo | What it is |
|---|---|
| [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) | AI model for detecting animals, people, and vehicles in camera-trap imagery — where it all started ([documentation](https://microsoft.github.io/MegaDetector/)) |
| [microsoft/MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic) | Bioacoustic AI for biodiversity monitoring — audio classification and species identification from sound |
| [microsoft/MegaDetector-Classifier](https://github.com/microsoft/MegaDetector-Classifier) | Camera-trap species classification fine-tuning — adapt classifiers to your own datasets and geographic regions |
| [microsoft/MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead) | Overhead imagery detection — point-based wildlife localization from aerial views |
| [microsoft/MegaDetector-Sonar](https://github.com/microsoft/MegaDetector-Sonar) | Sonar-based wildlife detection — processing and feature detection in sidescan sonar imagery |
| [microsoft/SPARROW-Playground](https://github.com/microsoft/SPARROW-Playground) | Inference playground for the SPARROW ecosystem, built on SPARROW-Engine — successor to PyTorch-Wildlife |
| [microsoft/SPARROW-Engine](https://github.com/microsoft/SPARROW-Engine) | Framework-free inference engine and Python package with the full conservation model zoo |
| [microsoft/Pytorch-Wildlife](https://github.com/microsoft/Pytorch-Wildlife) | *(being archived — see SPARROW-Playground)* The original PyTorch framework and model zoo for conservation AI |
| [microsoft/SPARROW](https://github.com/microsoft/SPARROW) | Solar-Powered Acoustic and Remote Recording Observation Watch — AI edge device for remote field deployments |

## Cite us

When citing work that uses any of the repositories under this umbrella, please cite:

- **Hernandez et al. 2024** — *Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation* — for any use of the PyTorch-Wildlife framework or models accessed through it
- **Beery, Morris, Yang 2019** — *Efficient Pipeline for Camera Trap Image Review* — for any use of MegaDetector specifically

A `citation.cff` file is included in this repository for automated citation tools.

## Contributing

We welcome community contributions. See our [Contribution Guidelines](https://microsoft.github.io/Biodiversity/contribute/#how-to-participate) for how to participate.

## Community

Have questions or want to connect with the team? Join us on Discord: [![Discord](https://img.shields.io/badge/Discord-Join_us-5865F2?logo=discord&logoColor=white)](https://discord.gg/TeEVxzaYtm)

A list of organizations using MegaDetector across global conservation work — six years of partnerships, from national parks to research universities to NGOs — is maintained on the [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) repository.

>[!IMPORTANT]
>If you would like to be added to this list or have any questions regarding MegaDetector and PyTorch-Wildlife, please [email us](zhongqimiao@microsoft.com) or join us in our Discord channel: [![](https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=PytorchWildife)](https://discord.gg/TeEVxzaYtm)

## About

Maintained by [Microsoft AI for Good Lab](https://www.microsoft.com/en-us/research/group/ai-for-good-research-lab/).