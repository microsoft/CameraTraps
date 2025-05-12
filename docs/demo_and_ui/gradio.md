
## 🕵️ Explore Pytorch-Wildlife and MegaDetector with our Demo User Interface

If you want to directly try **Pytorch-Wildlife** and the models in our [model zoo](../model_zoo/megadetector.md), you can use our **Gradio interface**. 

To start using the app locally:

```bash
python gradio_demo.py
```
The `gradio_demo.py` will launch a Gradio interface where you can:
- Perform Single Image Detection: Upload an image and set a confidence threshold to get detections.
- Perform Batch Image Detection: Upload a zip file containing multiple images to get detections in a JSON format.
- Perform Video Detection: Upload a video and get a processed video with detected animals. 

Or, you can also go to our [HuggingFace Page](https://huggingface.co/spaces/AndresHdzC/pytorch-wildlife) for some quick testing.

As a showcase platform, the gradio demo offers a hands-on experience with all the available features. However, it's important to note that this interface is primarily for demonstration purposes. While it is fully equipped to run all the features effectively, it may not be optimized for scenarios involving excessive data loads. We advise users to be mindful of this limitation when experimenting with large datasets.

Some browsers may not render processed videos due to unsupported codec. If that happens, please either use a newer version of browser or run the following for a `conda` version of `opencv` and choose `avc1` in the Video encoder drop down menu in the webapp (this might not work for MacOS):

```bash
pip uninstall opencv-python
conda install -c conda-forge opencv
```

![image](https://zenodo.org/records/15376499/files/gradio_UI.png)