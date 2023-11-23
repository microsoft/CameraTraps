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

## Mapping labels to a standard taxonomy (usually for new LILA datasets)

When a new .json file comes in and needs to be mapped to scientific names...

* Assuming this is a LILA dataset, edit the [LILA metadata file](http://lila.science/wp-content/uploads/2020/03/lila_sas_urls.txt) to include the new .json and dataset name.

* Assuming this is a LILA dataset, use get_lila_category_list.py to download the .json files for every LILA dataset.  This will produced a .json-formatted dictionary mapping each dataset to all of the categories it contains.

* Use map_new_lila_datasets.py to create a .csv file mapping each of those categories to a scientific name and taxonomy.  This will eventually become a subset of rows in the "master" .csv file.  This is a semi-automated process; it will look up common names against the iNat and GBIF taxonomies, with some heuristics to avoid simple problems (like making sure that "greater_kudu" matches "greater kudu", or that "black backed jackal" matches "black-backed jackal"), but you will need to fill in a few gaps manually.  I do this with three windows open: a .csv editor, Spyder (with the cell called "manual lookup" from this script open), and a browser.  Once you generate this .csv file, it's considered permanent, i.e., the cell that wrote it won't re-write it, so manually edit to your heart's content.

* Use preview_lila_taxonomy.py to produce an HTML file full of images that you can use to make sure that the matches were sensible; be particularly suspicious of anything that doesn't look like a mammal, bird, or reptile.  Go back and fix things in the .csv file.  This script/notebook also does a bunch of other consistency checking.

* When you are totally satisfied with that .csv file, manually append it to the "master" .csv file (lila-taxonomy-mapping.csv), which is currently in a private repository.  preview_lila_taxonomy can also be run against the master file.

* Check for errors (one more time) (this should be redundant with what's now included in preview_lila_taxonomy.py, but it can't hurt) by running:

    ```bash
    python taxonomy_mapping/taxonomy_csv_checker.py /path/to/taxonomy.csv
    ```
    
* Prepare the "release" taxonomy file (which removes a couple columns and removes unused rows) using prepare_lila_taxonomy_release.py .

* Use map_lila_categories.py to get a mapping of every LILA data set to the common taxonomy.

* The `visualize_taxonomy.ipynb` notebook demonstrates how to visualize the taxonomy hierarchy. It requires the *networkx* and *graphviz* Python packages.
