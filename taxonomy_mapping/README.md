When a new .json file comes in and needs to be mapped to scientific names...

* Assuming this is a LILA dataset, edit the [LILA metadata file](http://lila.science/wp-content/uploads/2020/03/lila_sas_urls.txt) to include the new .json and dataset name.

* Assuming this is a LILA dataset, use get_lila_category_list.py to download the .json files for every LILA dataset.  This will produced a .json-formatted dictionary mapping each dataset to all of the categories it contains.

* Use map_new_lila_datasets.py to create a .csv file mapping each of those categories to a scientific name and taxonomy.  This will eventually become a subset of rows in the "master" .csv file.  This is a semi-automated process; it will look up common names against the iNat and GBIF taxonomies, with some heuristics to avoid simple problems (like making sure that "greater_kudu" matches "greater kudu", or that "black backed jackal" matches "black-backed jackal"), but you will need to fill in a few gaps manually.  I do this with three windows open: a .csv editor, Spyder (with the cell called "manual lookup" from this script open), and a browser.  Once you generate this .csv file, it's considered permanent, i.e., the cell that wrote it won't re-write it, so manually edit to your heart's content.

* Use preview_lila_taxonomy.py to produce an HTML file full of images that you can use to make sure that the matches were sensible; be particularly suspicious of anything that doesn't look like a mammal, bird, or reptile.  Go back and fix things in the .csv file.  This script/notebook also does a bunch of other consistency checking.

* When you are totally satisfied with that .csv file, manually append it to the "master" .csv file (lila-taxonomy-mapping.csv), which is currently in a private repository

## TODO

* Check for errors by running

```bash
python taxonomy_mapping/taxonomy_csv_checker.py /path/to/taxonomy.csv
```

* Produce final output by running [...]

* The `visualize_taxonomy.ipynb` notebook demonstrates how to visualize the taxonomy hierarchy. It requires the *networkx* and *graphviz* Python packages.

