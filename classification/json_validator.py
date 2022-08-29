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
4) If --allow-multilabel=False, then no dataset label is included in more than
    one classification label.

If --output-dir <output_dir> is given, then we query MegaDB for images
that match the dataset labels identified during the validation step. We filter
out images that have unaccepted file extensions and images that don't actually
exist in Azure Blob Storage. In total, we output the following files:

<output_dir>/
- included_dataset_labels.txt
    lists the original dataset classes included for each classification label
- image_counts_by_label_presample.json
    number of images for each classification label after filtering bad
    images, but before sampling
- image_counts_by_label_sampled.json
    number of images for each classification label in queried_images.json
- json_validator_log_{timestamp}.json
    log of excluded images / labels
- queried_images.json
    main output file, ex:
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
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Container, Iterable, Mapping, MutableMapping
from concurrent import futures
from datetime import datetime
import json
import os
import pprint
import random
from typing import Any

import pandas as pd
import path_utils  # from ai4eutils
import sas_blob_utils  # from ai4eutils
from tqdm import tqdm

from data_management.megadb import megadb_utils
from taxonomy_mapping.taxonomy_graph import (
    build_taxonomy_graph, dag_to_tree, TaxonNode)


def main(label_spec_json_path: str,
         taxonomy_csv_path: str,
         allow_multilabel: bool = False,
         single_parent_taxonomy: bool = False,
         check_blob_exists: bool | str = False,
         min_locs: int | None = None,
         output_dir: str | None = None,
         json_indent: int | None = None,
         seed: int = 123,
         mislabeled_images_dir: str | None = None) -> None:
    """Main function."""
    # input validation
    assert os.path.exists(label_spec_json_path)
    assert os.path.exists(taxonomy_csv_path)
    if mislabeled_images_dir is not None:
        assert os.path.isdir(mislabeled_images_dir)

    random.seed(seed)

    print('Building taxonomy hierarchy')
    taxonomy_df = pd.read_csv(taxonomy_csv_path)
    if single_parent_taxonomy:
        TaxonNode.single_parent_only = True
    graph, taxonomy_dict, _ = build_taxonomy_graph(taxonomy_df)
    dag_to_tree(graph, taxonomy_dict)

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
    output_js = get_output_json(label_to_inclusions, mislabeled_images_dir)
    print(f'In total found {len(output_js)} images')

    # only keep images that:
    # 1) end in a supported file extension, and
    # 2) actually exist in Azure Blob Storage
    # 3) belong to a label with at least min_locs locations
    log: dict[str, Any] = {}
    remove_non_images(output_js, log)
    if isinstance(check_blob_exists, str):
        remove_nonexistent_images(output_js, log, check_local=check_blob_exists)
    elif check_blob_exists:
        remove_nonexistent_images(output_js, log)
    if min_locs is not None:
        remove_images_insufficient_locs(output_js, log, min_locs)

    # write out log of images / labels that were removed
    date = datetime.now().strftime('%Y%m%d_%H%M%S')  # ex: '20200722_110816'
    log_path = os.path.join(output_dir, f'json_validator_log_{date}.json')
    print(f'Saving log of bad images to {log_path}')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=1)

    # save label counts, pre-subsampling
    print('Saving pre-sampling label counts')
    save_path = os.path.join(output_dir, 'image_counts_by_label_presample.json')
    with open(save_path, 'w') as f:
        image_counts_by_label = {
            label: len(filter_images(output_js, label))
            for label in sorted(input_js.keys())
        }
        json.dump(image_counts_by_label, f, indent=1)

    print('Sampling with priority (if needed)')
    output_js = sample_with_priority(input_js, output_js)

    print('Saving queried_images.json')
    output_json_path = os.path.join(output_dir, 'queried_images.json')
    with open(output_json_path, 'w') as f:
        json.dump(output_js, f, indent=json_indent)

    # save label counts, post-subsampling
    print('Saving post-sampling label counts')
    save_path = os.path.join(output_dir, 'image_counts_by_label_sampled.json')
    with open(save_path, 'w') as f:
        image_counts_by_label = {
            label: len(filter_images(output_js, label))
            for label in sorted(input_js.keys())
        }
        json.dump(image_counts_by_label, f, indent=1)


def parse_spec(spec_dict: Mapping[str, Any],
               taxonomy_dict: dict[tuple[str, str], TaxonNode]
               ) -> set[tuple[str, str]]:
    """Gathers the dataset labels corresponding to a particular classification
    label specification.

    Args:
        spec_dict: dict, contains keys ['taxa', 'dataset_labels']
        taxonomy_dict: dict, maps (taxon_level, taxon_name) to a TaxonNode

    Returns: set of (ds, ds_label), dataset labels requested by the spec

    Raises: ValueError, if specification does not match any dataset labels
    """
    results = set()
    if 'taxa' in spec_dict:
        # spec_dict['taxa']: list of dict
        #   [
        #       {'level': 'family', 'name': 'cervidae', 'datasets': ['idfg']},
        #       {'level': 'genus',  'name': 'meleagris'}
        #   ]
        for taxon in spec_dict['taxa']:
            key = (taxon['level'].lower(), taxon['name'].lower())
            datasets = taxon.get('datasets', None)
            results |= taxonomy_dict[key].get_dataset_labels(datasets)

    if 'dataset_labels' in spec_dict:
        # spec_dict['dataset_labels']: dict, dataset => list of dataset_label
        #    {
        #       "idfg": ["deer", "elk", "prong"],
        #       "idfg_swwlf_2019": ["elk", "muledeer", "whitetaileddeer"]
        #    }
        for ds, ds_labels in spec_dict['dataset_labels'].items():
            for ds_label in ds_labels:
                results.add((ds, ds_label))

    if len(results) == 0:
        raise ValueError('specification matched no dataset labels')
    return results


def validate_json(input_js: dict[str, dict[str, Any]],
                  taxonomy_dict: dict[tuple[str, str], TaxonNode],
                  allow_multilabel: bool) -> dict[str, set[tuple[str, str]]]:
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
    label_to_inclusions: dict[str, set[tuple[str, str]]] = {}
    for label, spec_dict in input_js.items():
        include_set = parse_spec(spec_dict, taxonomy_dict)
        if 'exclude' in spec_dict:
            include_set -= parse_spec(spec_dict['exclude'], taxonomy_dict)

        for label_b, set_b in label_to_inclusions.items():
            shared = include_set.intersection(set_b)
            if len(shared) > 0:
                print(f'Labels {label} and {label_b} share images:', shared)
                if not allow_multilabel:
                    raise ValueError('Intersection between sets!')

        label_to_inclusions[label] = include_set
    return label_to_inclusions


def get_output_json(label_to_inclusions: dict[str, set[tuple[str, str]]],
                    mislabeled_images_dir: str | None = None
                    ) -> dict[str, dict[str, Any]]:
    """Queries MegaDB to get image paths matching dataset_labels.

    Args:
        label_to_inclusions: dict, maps label name to set of
            (dataset, dataset_label) tuples, output of validate_json()
        mislabeled_images_dir: str, path to directory of CSVs with known
            mislabeled images

    Returns: dict, maps sorted image_path <dataset>/<img_file> to a dict of
        properties
        - 'dataset': str, name of dataset that image is from
        - 'location': str or int, optional
        - 'class': str, class label from the dataset
        - 'label': list of str, assigned output label
        - 'bbox': list of dicts, optional
    """
    # because MegaDB is organized by dataset, we do the same
    # ds_to_labels = {
    #     'dataset_name': {
    #         'dataset_label': [output_label1, output_label2]
    #     }
    # }
    ds_to_labels: dict[str, dict[str, list[str]]] = {}
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
            AND (NOT IS_DEFINED(img.class))
        )
    '''

    output_json = {}  # maps full image path to json object
    for ds in tqdm(sorted(ds_to_labels.keys())):  # sort for determinism
        mislabeled_images: Mapping[str, Any] = {}
        if mislabeled_images_dir is not None:
            csv_path = os.path.join(mislabeled_images_dir, f'{ds}.csv')
            if os.path.exists(csv_path):
                mislabeled_images = pd.read_csv(csv_path, index_col='file',
                                                squeeze=True)

        ds_labels = sorted(ds_to_labels[ds].keys())
        tqdm.write(f'Querying dataset "{ds}" for dataset labels: {ds_labels}')

        start = datetime.now()
        parameters = [dict(name='@dataset_labels', value=ds_labels)]
        results = megadb.query_sequences_table(
            query, partition_key=ds, parameters=parameters)
        elapsed = (datetime.now() - start).total_seconds()
        tqdm.write(f'- query took {elapsed:.0f}s, found {len(results)} images')

        # if no path prefix, set it to the empty string '', because
        #     os.path.join('', x, '') = '{x}/'
        path_prefix = datasets_table[ds].get('path_prefix', '')
        count_corrected = 0
        count_removed = 0
        for result in results:
            # result keys
            # - already has: ['dataset', 'location', 'file', 'class', 'bbox']
            # - add ['label'], remove ['file']
            img_file = os.path.join(path_prefix, result['file'])

            # if img is mislabeled, but we don't know the correct class, skip it
            # otherwise, update the img with the correct class, but skip the
            #   img if the correct class is not one we queried for
            if img_file in mislabeled_images:
                new_class = mislabeled_images[img_file]
                if pd.isna(new_class) or new_class not in ds_to_labels[ds]:
                    count_removed += 1
                    continue

                count_corrected += 1
                result['class'] = new_class

            img_path = os.path.join(ds, img_file)
            del result['file']
            ds_label = result['class']
            result['label'] = ds_to_labels[ds][ds_label]
            output_json[img_path] = result

        tqdm.write(f'- Removed {count_removed} mislabeled images.')
        tqdm.write(f'- Corrected labels for {count_corrected} images.')

    # sort keys for determinism
    output_json = {k: output_json[k] for k in sorted(output_json.keys())}
    return output_json


def get_image_sas_uris(img_paths: Iterable[str]) -> list[str]:
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


def remove_non_images(js: MutableMapping[str, dict[str, Any]],
                      log: MutableMapping[str, Any]) -> None:
    """Remove images with non-image file extensions. Modifies [js] and [log]
    in-place.

    Args:
        js: dict, img_path => info dict
        log: dict, maps str description to log info
    """
    print('Removing images with invalid image file extensions...')
    nonimg_paths = [k for k in js.keys() if not path_utils.is_image_file(k)]
    for img_path in nonimg_paths:
        del js[img_path]
    print(f'Removed {len(nonimg_paths)} files with non-image extensions.')
    if len(nonimg_paths) > 0:
        log['nonimage_files'] = sorted(nonimg_paths)


def remove_nonexistent_images(js: MutableMapping[str, dict[str, Any]],
                              log: MutableMapping[str, Any],
                              check_local: str | None = None,
                              num_threads: int = 50) -> None:
    """Remove images that don't actually exist locally or on Azure Blob Storage.
    Modifies [js] and [log] in-place.

    Args:
        js: dict, image paths <dataset>/<img_file> => info dict
        log: dict, maps str description to log info
        check_local: optional str, path to local dir
        num_threads: int, number of threads to use for checking blob existence
    """
    def check_local_then_azure(local_path: str, blob_url: str) -> bool:
        return (os.path.exists(local_path)
                or sas_blob_utils.check_blob_exists(blob_url))

    pool = futures.ThreadPoolExecutor(max_workers=num_threads)
    future_to_img_path = {}
    blob_urls = get_image_sas_uris(js.keys())
    total = len(js)
    print(f'Checking {total} images for existence...')
    pbar = tqdm(zip(js.keys(), blob_urls), total=total)
    if check_local is None:
        # only check Azure Blob Storage
        for img_path, blob_url in pbar:
            future = pool.submit(sas_blob_utils.check_blob_exists, blob_url)
            future_to_img_path[future] = img_path
    else:
        # check local directory first before checking Azure Blob Storage
        for img_path, blob_url in pbar:
            local_path = os.path.join(check_local, img_path)
            future = pool.submit(check_local_then_azure, local_path, blob_url)
            future_to_img_path[future] = img_path

    nonexistent_images = []
    print('Fetching results...')
    for future in tqdm(futures.as_completed(future_to_img_path), total=total):
        img_path = future_to_img_path[future]
        try:
            if future.result():  # blob_url exists
                continue
        except Exception as e:  # pylint: disable=broad-except
            exception_type = type(e).__name__
            tqdm.write(f'{img_path} - generated {exception_type}: {e}')
        nonexistent_images.append(img_path)
        del js[img_path]
    pool.shutdown()

    print(f'Found {len(nonexistent_images)} nonexistent blobs.')
    if len(nonexistent_images) > 0:
        log['nonexistent_images'] = sorted(nonexistent_images)


def remove_images_insufficient_locs(js: MutableMapping[str, dict[str, Any]],
                                    log: MutableMapping[str, Any],
                                    min_locs: int) -> None:
    """Removes images that have labels that don't have at least min_locs
    locations. Modifies [js] and [log] in-place.

    Args:
        js: dict, image paths <dataset>/<img_file> => info dict
        log: dict, maps str description to log info
        min_locs: optional int, minimum # of locations that each label must
            have in order to be included
    """
    # 1st pass: populate label_to_locs
    # label (tuple of str) => set of (dataset, location)
    label_to_locs = defaultdict(set)
    for img_path, img_info in js.items():
        label = tuple(img_info['label'])
        loc = (img_info['dataset'], img_info.get('location', ''))
        label_to_locs[label].add(loc)

    bad_labels = set(label for label, locs in label_to_locs.items()
                     if len(locs) < min_locs)
    print(f'Found {len(bad_labels)} labels with < {min_locs} locations.')

    # 2nd pass: eliminate bad images
    if len(bad_labels) > 0:
        log[f'labels with < {min_locs} locs'] = sorted(bad_labels)
        for img_path in list(js.keys()):  # copy keys to modify js in-place
            label = tuple(js[img_path]['label'])
            if label in bad_labels:
                del js[img_path]


def filter_images(output_js: Mapping[str, Mapping[str, Any]], label: str,
                  datasets: Container[str] | None = None) -> set[str]:
    """Finds image files from output_js that have a given label and are from
    a set of datasets.

    Args:
        output_js: dict, output of get_output_json()
        label: str, desired label
        datasets: optional list str, dataset names, images from any dataset are
            allowed if datasets=None

    Returns: set of str, image files that match the filtering criteria
    """
    img_files: set[str] = set()
    for img_file, img_dict in output_js.items():
        cond1 = (label in img_dict['label'])
        cond2 = (datasets is None or img_dict['dataset'] in datasets)
        if cond1 and cond2:
            img_files.add(img_file)
    return img_files


def sample_with_priority(input_js: Mapping[str, Mapping[str, Any]],
                         output_js: Mapping[str, dict[str, Any]]
                         ) -> dict[str, dict[str, Any]]:
    """Uses the optional 'max_count' and 'prioritize' keys from the input
    classification labels specifications JSON file to sample images for each
    classification label.

    Returns: dict, keys are image file names, sorted alphabetically
    """
    filtered_imgs: set[str] = set()
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
        help='allow assigning a (dataset, dataset_label) pair to multiple '
             'output labels')
    parser.add_argument(
        '--single-parent-taxonomy', action='store_true',
        help='flag that restricts the taxonomy to only allow a single parent '
             'for each taxon node')
    parser.add_argument(
        '-c', '--check-blob-exists', nargs='?', const=True,
        help='check that the blob for each queried image actually exists. Can '
             'be very slow if reaching throttling limits. Optionally pass in a '
             'local directory to check before checking Azure Blob Storage.')
    parser.add_argument(
        '--min-locs', type=int,
        help='minimum number of locations that each label must have in order '
             'to be included')
    parser.add_argument(
        '-o', '--output-dir',
        help='path to directory to save outputs. The output JSON file is saved '
             'at <output-dir>/queried_images.json, and the mapping from '
             'classification labels to dataset labels is saved at '
             '<output-dir>/included_dataset_labels.txt.')
    parser.add_argument(
        '--json-indent', type=int,
        help='number of spaces to use for JSON indent (default no indent), '
             'only used if --output-dir is given')
    parser.add_argument(
        '--seed', type=int, default=123,
        help='random seed for sampling images, only used if --output-dir is '
             'given and a label specification includes a "max_count" key')
    parser.add_argument(
        '-m', '--mislabeled-images',
        help='path to `megadb_mislabeled` directory of locally mounted '
             '`classifier-training` Azure Blob Storage container where known '
             'mislabeled images are tracked')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(label_spec_json_path=args.label_spec_json,
         taxonomy_csv_path=args.taxonomy_csv,
         allow_multilabel=args.allow_multilabel,
         single_parent_taxonomy=args.single_parent_taxonomy,
         check_blob_exists=args.check_blob_exists,
         min_locs=args.min_locs,
         output_dir=args.output_dir,
         json_indent=args.json_indent,
         seed=args.seed,
         mislabeled_images_dir=args.mislabeled_images)

# main(
#     label_spec_json_path='idfg_classes.json',
#     taxonomy_csv_path='../../camera-traps-private/camera_trap_taxonomy_mapping.csv',
#     output_dir='run_idfg',
#     json_indent=4)
