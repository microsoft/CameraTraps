r"""Validates a classification label specification JSON file and optionally
queries MegaDB to find matching image files.

See README.md for an example of a classification label specification JSON file.

The validation step takes the classification label specification JSON file and
finds the dataset labels that belong to each classification label. It checks
that the following conditions hold:
1) Each classification label specification matches at least 1 dataset label.
2) If the classification label includes a taxonomical specification, then the
    taxa is actually a part of our master taxonomy.
3) If the 'prioritize' key is found for a given label, then the label must
    also have a 'max_count' key.
4) If --allow_multilabel=False, then no dataset label is included in more than
    one classification label.

If --output-dir <output_dir> is given, then we query MegaDB for images
that match the dataset labels identified during the validation step. We filter
out images that have unaccepted file extensions and images that don't actually
exist in Azure Blob Storage, and we record these bad images:
    <output_dir>/json_validator_bad_images.json
Finally, we output a JSON file with good image files and their attributes,
sorted by image filename:
    <output_dir>/queried_images.json

The output JSON file looks like:

{
    "caltech/cct_images/59f5fe2b-23d2-11e8-a6a3-ec086b02610b.jpg": {
        "dataset": "caltech",
        "location": 13,
        "class": "mountain_lion",  // class from dataset
        "label": ["monutain_lion"]  // labels to use in classifier
    },
    "caltech/cct_images/59f79901-23d2-11e8-a6a3-ec086b02610b.jpg": {
        "dataset": "caltech",
        "location": 13,
        "class": "mountain_lion",  // class from dataset
        "bbox": [{"category": "animal",
                  "bbox": [0, 0.347, 0.237, 0.257]}],
        "label": ["monutain_lion"]  // labels to use in classifier
    },
    ...
}

Example usage:

    python json_validator.py label_spec.json \
        $HOME/camera-traps-private/camera_trap_taxonomy_mapping.csv \
        --output-dir run --json-indent 2
"""
# allow forward references in typing annotations
from __future__ import annotations

import argparse
from concurrent import futures
import json
import os
import pprint
import random
import time
from typing import (Any, ClassVar, Container, Dict, Iterable, List, Mapping,
                    Optional, Set, Tuple, Union)

import pandas as pd
from tqdm import tqdm

from data_management.megadb import megadb_utils
import path_utils  # from ai4eutils
import sas_blob_utils  # from ai4eutils


class TaxonNode:
    r"""A node in a taxonomy tree, associated with a set of dataset labels.

    By default we support multiple parents for each TaxonNode because different
    taxonomies may have a different granularity of hierarchy. If the taxonomy
    was created from a mixture of different taxonomies, then we may see the
    following, for example:

        "eastern gray squirrel" (inat)     "squirrel" (gbif)
        ------------------------------     -----------------
    family:                        sciuridae
                                  /          \
    subfamily:          sciurinae             |  # skips subfamily
                                |             |
    tribe:               sciurini             |  # skips tribe
                                  \          /
    genus:                          sciurus
    """
    # class variables
    single_parent_only: ClassVar[bool] = False

    # instance variables
    level: str
    name: str
    ids: Set[Tuple[str, int]]
    children: List[TaxonNode]
    parents: List[TaxonNode]
    dataset_labels: Set[Tuple[str, str]]

    def __init__(self, level: str, name: str):
        """Initializes a TaxonNode."""
        self.level = level
        self.name = name
        self.ids = set()
        self.children = []
        self.parents = []
        self.dataset_labels = set()

    def __repr__(self):
        id_str = ', '.join(f'{source}={id}' for source, id in self.ids)
        return f'TaxonNode({id_str}, level={self.level}, name={self.name})'

    def add_id(self, source: str, taxon_id: int) -> None:
        assert source in ['gbif', 'inat', 'manual']
        self.ids.add((source, taxon_id))

    def add_parent(self, parent: TaxonNode) -> None:
        """Adds a TaxonNode to the list of parents of the current TaxonNode.

        Args:
            parent: TaxonNode, must be higher in the taxonomical hierarchy
        """
        if TaxonNode.single_parent_only and len(self.parents) > 0:
            assert len(self.parents) == 1
            assert self.parents[0] is parent, (
                f'self.parents: {self.parents}, new parent: {parent}')
            return
        if parent not in self.parents:
            self.parents.append(parent)
            parent.add_child(self)

    def add_child(self, child: TaxonNode) -> None:
        if child not in self.children:
            self.children.append(child)
            child.add_parent(self)

    def add_dataset_label(self, ds: str, ds_label: str) -> None:
        """
        Args:
            ds: str, name of dataset
            ds_label: str, name of label used by that dataset
        """
        self.dataset_labels.add((ds, ds_label))

    def get_dataset_labels(self,
                           include_datasets: Optional[Container[str]] = None
                           ) -> Set[Tuple[str, str]]:
        """Returns a set of all (ds, ds_label) tuples that belong to this taxon
        node or its descendants.

        Args:
            include_datasets: list of str, names of datasets to include
                if None, then all datasets are included

        Returns: set of (ds, ds_label) tuples
        """
        result = self.dataset_labels
        if include_datasets is not None:
            result = set(tup for tup in result if tup[0] in include_datasets)

        for child in self.children:
            result |= child.get_dataset_labels(include_datasets)
        return result

    @classmethod
    def lowest_common_ancestor(cls, nodes: Iterable[TaxonNode]
                               ) -> Optional[TaxonNode]:
        """Returns the lowest common ancestor (LCA) of a list or set of nodes.

        For each node in <nodes>, get the set of nodes on the path to the root.
        The LCA of <nodes> is certainly in the intersection of these sets.
        Iterate through the nodes in this set intersection, looking for a node
        such that none of its children is in this intersection. Given n nodes
        from a k-ary tree of height h, the algorithm runs in O((n + k)h).

        Returns: TaxonNode, the LCA if it exists, or None if no LCA exists
        """
        paths = []
        for node in nodes:
            # get path to root
            path = {node}
            remaining = list(node.parents)  # make a shallow copy
            while len(remaining) > 0:
                x = remaining.pop()
                if x not in path:
                    path.add(x)
                    remaining.extend(x.parents)
            paths.append(path)
        intersect = set.intersection(*paths)

        for node in intersect:
            if intersect.isdisjoint(node.children):
                return node
        return None


def main(label_spec_json_path: str,
         taxonomy_csv_path: str,
         allow_multilabel: bool = False,
         single_parent_taxonomy: bool = False,
         check_blob_exists: Union[bool, str] = False,
         output_dir: str = None,
         json_indent: Optional[int] = None,
         seed: int = 123) -> None:
    """Main function."""
    random.seed(seed)

    print('Building taxonomy hierarchy')
    taxonomy_df = pd.read_csv(taxonomy_csv_path)
    if single_parent_taxonomy:
        TaxonNode.single_parent_only = True
    taxonomy_dict, _ = build_taxonomy_dict(taxonomy_df)

    print('Validating input json')
    with open(label_spec_json_path, 'r') as f:
        input_js = json.load(f)
    label_to_inclusions = validate_json(
        input_js, taxonomy_dict, allow_multilabel=allow_multilabel)

    if output_dir is None:
        pprint.pprint(label_to_inclusions)
        return

    os.makedirs(output_dir, exist_ok=True)
    labels_path = os.path.join(output_dir, 'included_dataset_labels.txt')
    with open(labels_path, 'w') as f:
        pprint.pprint(label_to_inclusions, stream=f)

    # use MegaDB to generate list of images
    print('Generating output json')
    output_js = get_output_json(label_to_inclusions)

    # only keep images that:
    # 1) end in a supported file extension, and
    # 2) actually exist in Azure Blob Storage
    output_js = remove_bad_images(
        output_js, output_dir, check_blob_exists=check_blob_exists)

    # save label counts, pre-subsampling
    save_path = os.path.join(output_dir, 'image_counts_by_label_presample.json')
    with open(save_path, 'w') as f:
        image_counts_by_label = {
            label: len(filter_images(output_js, label))
            for label in sorted(input_js.keys())
        }
        json.dump(image_counts_by_label, f, indent=1)

    output_js = sample_with_priority(input_js, output_js)
    output_json_path = os.path.join(output_dir, 'queried_images.json')
    with open(output_json_path, 'w') as f:
        json.dump(output_js, f, indent=json_indent)

    # save label counts, post-subsampling
    save_path = os.path.join(output_dir, 'image_counts_by_label_sampled.json')
    with open(save_path, 'w') as f:
        image_counts_by_label = {
            label: len(filter_images(output_js, label))
            for label in sorted(input_js.keys())
        }
        json.dump(image_counts_by_label, f, indent=1)


def build_taxonomy_dict(
        taxonomy_df: pd.DataFrame
        ) -> Tuple[
            Dict[Tuple[str, str], TaxonNode],
            Dict[Tuple[str, str], TaxonNode]
        ]:
    """Creates a mapping from (taxon_level, taxon_name) to TaxonNodes, used for
    gathering all dataset labels associated with a given taxon.

    Args:
        taxonomy_df: pd.DataFrame, see taxonomy_mapping directory for more info

    Returns:
        taxon_to_node: dict, maps (taxon_level, taxon_name) to a TaxonNode
        label_to_node: dict, maps (dataset_name, dataset_label) to the lowest
            TaxonNode node in the tree that contains the label
    """
    taxon_to_node = {}  # maps (taxon_level, taxon_name) to a TaxonNode
    label_to_node = {}  # maps (dataset_name, dataset_label) to a TaxonNode
    for _, row in taxonomy_df.iterrows():
        ds = row['dataset_name']
        ds_label = row['query']
        taxa_ancestry = row['taxonomy_string']
        id_source = row['source']
        if pd.isna(taxa_ancestry):
            # taxonomy CSV rows without 'taxonomy_string' entries can only be
            # added to the JSON via the 'dataset_labels' key
            continue
        else:
            taxa_ancestry = eval(taxa_ancestry)  # pylint: disable=eval-used

        taxon_child: Optional[TaxonNode] = None
        for i, taxon in enumerate(taxa_ancestry):
            taxon_id, taxon_level, taxon_name, _ = taxon

            key = (taxon_level, taxon_name)
            if key not in taxon_to_node:
                node = TaxonNode(level=taxon_level, name=taxon_name)
                taxon_to_node[key] = node
            node = taxon_to_node[key]

            if taxon_child is not None:
                node.add_child(taxon_child)

            node.add_id(id_source, int(taxon_id))  # np.int64 -> int
            if i == 0:
                assert row['taxonomy_level'] == taxon_level, (
                    f'taxonomy CSV level: {row["taxonomy_level"]}, '
                    f'level from taxonomy_string: {taxon_level}')
                assert row['scientific_name'] == taxon_name
                node.add_dataset_label(ds, ds_label)
                label_to_node[(ds, ds_label)] = node

            taxon_child = node

    return taxon_to_node, label_to_node


def parse_taxa(taxa_dicts: List[Dict[str, str]],
               taxonomy_dict: Dict[Tuple[str, str], TaxonNode]
               ) -> Set[Tuple[str, str]]:
    """Gathers the dataset labels requested by a "taxa" specification.

    Args:
        taxa_dicts: list of dict, corresponds to the "taxa" key in JSON, e.g.,
            [
                {'level': 'family', 'name': 'cervidae', 'datasets': ['idfg']},
                {'level': 'genus',  'name': 'meleagris'}
            ]
        taxonomy_dict: dict, maps (taxon_level, taxon_name) to a TaxonNode

    Returns: set of (ds, ds_label), dataset labels requested by the taxa spec
    """
    results = set()
    for taxon in taxa_dicts:
        key = (taxon['level'], taxon['name'])
        datasets = taxon.get('datasets', None)
        results |= taxonomy_dict[key].get_dataset_labels(datasets)
    return results


def parse_spec(spec_dict: Mapping[str, Any],
               taxonomy_dict: Dict[Tuple[str, str], TaxonNode]
               ) -> Set[Tuple[str, str]]:
    """
    Args:
        spec_dict: dict, contains keys ['taxa', 'dataset_labels']
        taxonomy_dict: dict, maps (taxon_level, taxon_name) to a TaxonNode

    Returns: set of (ds, ds_label), dataset labels requested by the spec

    Raises: ValueError, if specification does not match any dataset labels
    """
    results = set()
    if 'taxa' in spec_dict:
        results |= parse_taxa(spec_dict['taxa'], taxonomy_dict)
    if 'dataset_labels' in spec_dict:
        for ds, ds_labels in spec_dict['dataset_labels'].items():
            for ds_label in ds_labels:
                results.add((ds, ds_label))
    if len(results) == 0:
        raise ValueError('specification matched no dataset labels')
    return results


def validate_json(input_js: Dict[str, Dict[str, Any]],
                  taxonomy_dict: Dict[Tuple[str, str], TaxonNode],
                  allow_multilabel: bool) -> Dict[str, Set[Tuple[str, str]]]:
    """Validates JSON.

    Args:
        input_js: dict, Python dict representation of JSON file
            see classification/README.md
        taxonomy_dict: dict, maps (taxon_level, taxon_name) to a TaxonNode
        allow_multilabel: bool, whether to allow a dataset label to be assigned
            to multiple output labels

    Returns: dict, maps label name to set of (dataset, dataset_label) tuples

    Raises: ValueError, if a classification label specification matches no
        dataset labels, or if allow_multilabel=False but a dataset label is
        included in two or more classification labels
    """
    # maps output label name to set of (dataset, dataset_label) tuples
    label_to_inclusions: Dict[str, Set[Tuple[str, str]]] = {}
    for label, spec_dict in input_js.items():
        include_set = parse_spec(spec_dict, taxonomy_dict)
        if 'exclude' in spec_dict:
            include_set -= parse_spec(spec_dict['exclude'], taxonomy_dict)

        for label_b, set_b in label_to_inclusions.items():
            if not include_set.isdisjoint(set_b):
                print(f'Labels {label} and {label_b} will share images')
                if not allow_multilabel:
                    raise ValueError('Intersection between sets!')

        label_to_inclusions[label] = include_set
    return label_to_inclusions


def get_output_json(label_to_inclusions: Dict[str, Set[Tuple[str, str]]]
                    ) -> Dict[str, Dict[str, Any]]:
    """Queries MegaDB to get image paths matching dataset_labels.

    Args:
        label_to_inclusions: dict, maps label name to set of
            (dataset, dataset_label) tuples, output of validate_json()

    Returns: dict, maps sorted image_path <dataset>/<img_file> to a dict of
        properties
        - 'dataset': str, name of dataset that image is from
        - 'location': str or int, optional
        - 'class': str, class label from the dataset
        - 'label': str, assigned output label
        - 'bbox': list of dicts, optional
    """
    # because MegaDB is organized by dataset, we do the same
    # ds_to_labels = {
    #     'dataset_name': {
    #         'dataset_label': [output_label1, output_label2]
    #     }
    # }
    ds_to_labels: Dict[str, Dict[str, List]] = {}
    for output_label, ds_dslabels_set in label_to_inclusions.items():
        for (ds, ds_label) in ds_dslabels_set:
            if ds not in ds_to_labels:
                ds_to_labels[ds] = {}
            if ds_label not in ds_to_labels[ds]:
                ds_to_labels[ds][ds_label] = []
            ds_to_labels[ds][ds_label].append(output_label)

    # we need the datasets table for getting full image paths
    megadb = megadb_utils.MegadbUtils()
    datasets_table = megadb.get_datasets_table()

    # The line
    #    [img.class[0], seq.class[0]][0] as class
    # selects the image-level class label if available. Otherwise it selects the
    # sequence-level class label. This line assumes the following conditions,
    # expressed in the WHERE clause:
    # - at least one of the image or sequence class label is given
    # - the image and sequence class labels are arrays with length at most 1
    # - the image class label takes priority over the sequence class label
    #
    # In Azure Cosmos DB, if a field is not defined, then it is simply excluded
    # from the result. For example, on the following JSON object,
    #     {
    #         "dataset": "camera_traps",
    #         "seq_id": "1234",
    #         "location": "A1",
    #         "images": [{"file": "abcd.jpeg"}],
    #         "class": ["deer"],
    #     }
    # the array [img.class[0], seq.class[0]] just gives ['deer'] because
    # img.class is undefined and therefore excluded.
    query = '''
    SELECT
        seq.dataset,
        seq.location,
        img.file,
        [img.class[0], seq.class[0]][0] as class,
        img.bbox
    FROM sequences seq JOIN img IN seq.images
    WHERE (ARRAY_LENGTH(img.class) = 1
            AND ARRAY_CONTAINS(@dataset_labels, img.class[0])
        )
        OR (ARRAY_LENGTH(seq.class) = 1
            AND ARRAY_CONTAINS(@dataset_labels, seq.class[0])
            AND (ARRAY_LENGTH(img.class) = 0
                OR NOT IS_DEFINED(img.class)
                OR (ARRAY_LENGTH(img.class) = 1 AND img.class[0] = seq.class[0])
            )
        )
    '''

    output_json = {}  # maps full image path to json object
    for ds in tqdm(sorted(ds_to_labels.keys())):  # sort for determinism
        ds_labels = sorted(ds_to_labels[ds].keys())
        tqdm.write(f'Querying dataset "{ds}" for dataset labels: {ds_labels}')

        start = time.time()
        parameters = [dict(name='@dataset_labels', value=ds_labels)]
        results = megadb.query_sequences_table(
            query, partition_key=ds, parameters=parameters)
        elapsed = time.time() - start
        tqdm.write(f'- query took {elapsed:.0f}s, found {len(results)} images')

        # if no path prefix, set it to the empty string '', because
        #     os.path.join('', x) = x
        img_path_prefix = os.path.join(
            ds, datasets_table[ds].get('path_prefix', ''))
        for result in results:
            # result keys
            # - already has: ['dataset', 'location', 'file', 'class', 'bbox']
            # - add ['label'], remove ['file']
            img_path = os.path.join(img_path_prefix, result['file'])
            del result['file']
            ds_label = result['class']
            result['label'] = ds_to_labels[ds][ds_label]
            output_json[img_path] = result

    # sort keys for determinism
    output_json = {k: output_json[k] for k in sorted(output_json.keys())}
    return output_json


def get_image_sas_uris(img_paths: Iterable[str]) -> List[str]:
    """Converts a image paths to Azure Blob Storage blob URIs with SAS tokens.

    Args:
        img_paths: list of str, <dataset-name>/<image-filename>

    Returns:
        image_sas_uris: list of str, image blob URIs with SAS tokens, ready to
            pass to the batch detection API
    """
    # we need the datasets table for getting SAS keys
    datasets_table = megadb_utils.MegadbUtils().get_datasets_table()

    image_sas_uris = []
    for img_path in img_paths:
        dataset, img_file = img_path.split('/', maxsplit=1)

        # strip leading '?' from SAS token
        sas_token = datasets_table[dataset]['container_sas_key']
        if sas_token[0] == '?':
            sas_token = sas_token[1:]

        image_sas_uri = sas_blob_utils.build_azure_storage_uri(
            account=datasets_table[dataset]['storage_account'],
            container=datasets_table[dataset]['container'],
            blob=img_file,
            sas_token=sas_token)
        image_sas_uris.append(image_sas_uri)
    return image_sas_uris


def remove_bad_images(js: Mapping[str, Dict[str, Any]],
                      output_dir: Optional[str] = None,
                      check_blob_exists: Union[bool, str] = False,
                      num_threads: int = 50
                      ) -> Dict[str, Dict[str, Any]]:
    """Checks if each image path in js:
        1) ends in a supported image file extension
        2) actually exists on Azure Blob Storage (if check_blob_exists=True)

    Args:
        js: dict, keys are image paths <dataset>/<img_file>
        output_dir: optional str, if given saves list of nonexistent images
            to <output_dir>/nonexistent_images.json
        check_blob_exists: bool
        num_threads: int, number of threads to use for checking blob existence

    Returns: copy of js, but with bad images removed
    """
    pool = futures.ThreadPoolExecutor(max_workers=num_threads)
    future_to_img_path = {}

    # only keep images with valid image file extension
    print('Removing images with invalid image file extensions...')
    img_paths = [k for k in js.keys() if path_utils.is_image_file(k)]
    nonimg_paths = set(js.keys()) - set(img_paths)
    print(f'Found {len(nonimg_paths)} files with non-image extensions.')

    def check_local_then_azure(img_path: str, blob_url: str) -> bool:
        assert isinstance(check_blob_exists, str)
        local_path = os.path.join(check_blob_exists, img_path)
        if os.path.exists(local_path):
            return True
        return sas_blob_utils.check_blob_exists(blob_url)

    if check_blob_exists:
        blob_urls = get_image_sas_uris(img_paths)
        total = len(img_paths)
        print(f'Checking {total} images for existence...')
        pbar = tqdm(zip(img_paths, blob_urls), total=total)
        if isinstance(check_blob_exists, bool):
            # only check Azure Blob Storage
            for img_path, blob_url in pbar:
                future = pool.submit(sas_blob_utils.check_blob_exists, blob_url)
                future_to_img_path[future] = img_path
        else:
            # check local directory first before checking Azure Blob Storage
            for img_path, blob_url in pbar:
                future = pool.submit(check_local_then_azure, img_path, blob_url)
                future_to_img_path[future] = img_path

        img_paths = []
        nonexistent_images = []
        print('Fetching results...')
        for future in tqdm(futures.as_completed(future_to_img_path), total=total):
            img_path = future_to_img_path[future]
            try:
                if future.result():  # blob_url exists
                    img_paths.append(img_path)
                    continue
            except Exception as e:  # pylint: disable=broad-except
                exception_type = type(e).__name__
                tqdm.write(f'{img_path} - generated {exception_type}: {e}')
            nonexistent_images.append(img_path)

    pool.shutdown()

    if output_dir is not None:
        filename = 'json_validator_bad_images.json'
        bad_images = {'nonimage_files': sorted(nonimg_paths)}
        if check_blob_exists:
            bad_images['nonexistent_images'] = sorted(nonexistent_images)
        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(bad_images, f, indent=1)

    output_js = {
        img_path: js[img_path] for img_path in sorted(img_paths)
    }
    return output_js


def filter_images(output_js: Mapping[str, Mapping[str, Any]], label: str,
                  datasets: Optional[Container[str]] = None) -> Set[str]:
    """Finds image files from output_js that have a given label and are from
    a set of datasets.

    Args:
        output_js: dict, output of get_output_json()
        label: str, desired label
        datasets: optional list str, dataset names, images from any dataset are
            allowed if datasets=None

    Returns: set of str, image files that match the filtering criteria
    """
    img_files: Set[str] = set()
    for img_file, img_dict in output_js.items():
        cond = (label in img_dict['label']) and (
            datasets is None or img_dict['dataset'] in datasets)
        if cond:
            img_files.add(img_file)
    return img_files


def sample_with_priority(input_js: Mapping[str, Mapping[str, Any]],
                         output_js: Mapping[str, Dict[str, Any]]
                         ) -> Dict[str, Dict[str, Any]]:
    """Uses the optional 'max_count' and 'prioritize' keys from the input
    classification labels specifications JSON file to sample images for each
    classification label.

    Returns: dict, keys are image file names, sorted alphabetically
    """
    filtered_imgs: Set[str] = set()
    for label, spec_dict in input_js.items():
        if 'prioritize' in spec_dict and 'max_count' not in spec_dict:
            raise ValueError('prioritize is invalid without a max_count value.')

        if 'max_count' not in spec_dict:
            filtered_imgs.update(filter_images(output_js, label, datasets=None))
            continue
        quota = spec_dict['max_count']

        # prioritize is a list of prioritization levels
        prioritize = spec_dict.get('prioritize', [])
        prioritize.append(None)

        for level in prioritize:
            img_files = filter_images(output_js, label, datasets=level)

            # number of already matching images
            num_already_matching = len(img_files & filtered_imgs)
            quota = max(0, quota - num_already_matching)
            img_files -= filtered_imgs

            num_to_sample = min(quota, len(img_files))
            sample = random.sample(img_files, k=num_to_sample)
            filtered_imgs.update(sample)

            quota -= num_to_sample
            if quota == 0:
                break

    output_js = {
        img_file: output_js[img_file]
        for img_file in sorted(filtered_imgs)
    }
    return output_js


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Validates JSON.')
    parser.add_argument(
        'label_spec_json',
        help='path to JSON file containing label specification')
    parser.add_argument(
        'taxonomy_csv',
        help='path to taxonomy CSV file')
    parser.add_argument(
        '--allow-multilabel', action='store_true',
        help='flag that allows assigning a (dataset, dataset_label) pair to '
             'multiple output labels')
    parser.add_argument(
        '--single-parent-taxonomy', action='store_true',
        help='flag that restricts the taxonomy to only allow a single parent '
             'for each taxon node')
    parser.add_argument(
        '--check-blob-existence', nargs='?', const=True,
        help='check that the blob for each queried image actually exists. Can '
             'be very slow if reaching throttling limits. Optionally pass in a '
             'local directory to check before checking Azure Blob Storage.')
    parser.add_argument(
        '-o', '--output-dir',
        help='path to directory to save outputs. The output JSON file is saved '
             'at <output-dir>/queried_images.json, and the mapping from '
             'classification labels to dataset labels is saved at '
             '<output-dir>/label_to_dataset_label.txt.')
    parser.add_argument(
        '--json-indent', type=int,
        help='number of spaces to use for JSON indent (default no indent), '
             'only used if --output-dir is given')
    parser.add_argument(
        '--seed', type=int, default=123,
        help='random seed for sampling images, only used if --output-dir is '
             'given and a label specification includes a "max_count" key')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(label_spec_json_path=args.label_spec_json,
         taxonomy_csv_path=args.taxonomy_csv,
         allow_multilabel=args.allow_multilabel,
         single_parent_taxonomy=args.single_parent_taxonomy,
         check_blob_exists=args.check_blob_exists,
         output_dir=args.output_dir,
         json_indent=args.json_indent,
         seed=args.seed)

# main(
#     label_spec_json_path='idfg_classes.json',
#     taxonomy_csv_path='../../camera-traps-private/camera_trap_taxonomy_mapping.csv',
#     output_dir='run_idfg',
#     json_indent=4)
