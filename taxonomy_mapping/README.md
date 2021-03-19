We generated a list of all the annotations in our universe; the scripts in this folder were used to (interactively) map them onto the GBIF and iNat taxonomies.


## Creating the Taxonomy CSV

Creating the taxonomy CSV file requires running 3 scripts.

1. Generate a spreadsheet of the class names within each desired dataset by querying MegaDB. These class names are the names provided directly by our partner organizations and may include abbreviations, e.g., "wtd" meaning "white-tailed deer."

    This is done by running the `taxonomy_mapping/species_by_dataset.py` script. The first time running this step may take a while. However, intermediary outputs are cached in JSON files for much faster future runs.

2. Because each partner organization uses their own naming scheme, we need to map the class names onto a common taxonomy. We use a combination of the [iNaturalist taxonomy](https://forum.inaturalist.org/t/how-to-download-taxa/3542) and the [Global Biodiversity Information Facility (GBIF) Backbone Taxonomy](https://www.gbif.org/dataset/d7dddbf4-2cf0-4f39-9b2a-bb099caae36c).

    This is done by running the `taxonomy_mapping/process_species_by_dataset.py` script. Note that this script is not meant to be run as a normal Python script but is instead intended to be run interactively.

3. Once the taxonomy CSV is generated, check for errors by running

```bash
python taxonomy_mapping/taxonomy_csv_checker.py /path/to/taxonomy.csv
```


## Visualize the Taxonomy Hierarchy

The `visualize_taxonomy.ipynb` notebook demonstrates how to visualize the taxonomy hierarchy. It requires the *networkx* and *graphviz* Python packages.
