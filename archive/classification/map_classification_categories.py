r"""Maps a classifier's output categories to desired target categories.

In this file, we use the following terminology:
- "category": a category output by the classifier
- "target": name of a desired group, comprising >= 1 classifier categories

Takes as input 2 label specification JSON files:
1) desired label specification JSON file
    this should not have a target named "other"
2) label specification JSON file of trained classifier

The mapping is accomplished as follows:
1. For each category in the classifier label spec, find all taxon nodes that
    belong to that category.
2. Given a target in the desired label spec, find all taxon nodes that belong
    to that target. If there is any classifier category whose nodes are a
    subset of the target nodes, then map the classifier category to that target.
    Any partial intersection between a target's nodes and a category's nodes
    is considered an error.
3. If there are any classifier categories that have not yet been assigned a
    target, group them into the "other" target.

This script outputs a JSON file that maps each target to a list of classifier
categories.

Implementation Note: the taxonomy mapping parts of this script are very similar
    to json_validator.py.

Example usage:

    python map_classification_categories.py \
        desired_label_spec.json \
        /path/to/classifier/label_spec.json \
        $HOME/camera-traps-private/camera_trap_taxonomy_mapping.csv
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
import json
import os
from typing import Any, Optional

import networkx as nx
import pandas as pd
from tqdm import tqdm

from taxonomy_mapping.taxonomy_graph import (
    build_taxonomy_graph, dag_to_tree, TaxonNode)


def main(desired_label_spec_json_path: str,
         classifier_label_spec_json_path: str,
         taxonomy_csv_path: str,
         output_json_path: str,
         classifier_label_index_path: Optional[str]) -> None:
    """Main function."""
    print('Reading label spec JSON files')
    with open(desired_label_spec_json_path, 'r') as f:
        target_spec = json.load(f)
    with open(classifier_label_spec_json_path, 'r') as f:
        classifier_spec = json.load(f)

    if classifier_label_index_path is not None:
        with open(classifier_label_index_path, 'r') as f:
            classifier_labels = set(json.load(f).values())
        assert classifier_labels <= set(classifier_spec.keys())
        if len(classifier_labels) < len(classifier_spec):
            classifier_spec = {
                k: v for k, v in classifier_spec.items()
                if k in classifier_labels
            }

    print('Building taxonomy hierarchy')
    taxonomy_df = pd.read_csv(taxonomy_csv_path)
    graph, taxon_to_node, label_to_node = build_taxonomy_graph(taxonomy_df)
    dag_to_tree(graph, taxon_to_node)

    print('Mapping label spec to nodes')
    classifier_label_to_nodes = label_spec_to_nodes(
        classifier_spec, taxon_to_node, label_to_node)
    target_label_to_nodes = label_spec_to_nodes(
        target_spec, taxon_to_node, label_to_node)

    print('Creating mapping from target to classifier categories')
    target_to_classifier_labels = map_target_to_classifier(
        target_label_to_nodes, classifier_label_to_nodes)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(target_to_classifier_labels, f, indent=1)


def map_target_to_classifier(
        target_label_to_nodes: Mapping[str, set[TaxonNode]],
        classifier_label_to_nodes: Mapping[str, set[TaxonNode]]
        ) -> dict[str, list[str]]:
    """For each target, if there is any classifier category whose nodes are a
    subset of the target nodes, then assign the classifier category to that
    target. Any partial intersection between a target's nodes and a category's
    nodes is considered an error.

    Args:
        target_label_to_nodes: dict, maps target to set of nodes,
            all of the sets of nodes should be disjoint
        classifier_label_to_nodes: dict, maps classifier label to set of nodes,
            all of the sets of nodes should be disjoint

    Returns: dict, maps target label to set of classifier labels
    """
    remaining_classifier_labels = set(classifier_label_to_nodes.keys())
    target_to_classifier_labels: defaultdict[str, set[str]] = defaultdict(set)
    for target, target_nodes in tqdm(target_label_to_nodes.items()):
        for label, classifier_nodes in classifier_label_to_nodes.items():
            overlap = classifier_nodes & target_nodes
            if len(overlap) == len(classifier_nodes):
                target_to_classifier_labels[target].add(label)
                remaining_classifier_labels.remove(label)
            elif 0 < len(overlap) < len(classifier_nodes):  # partial overlap
                raise ValueError('Only partial overlap between target '
                                 f'{target} and classifier label {label}')
    if len(remaining_classifier_labels) > 0:
        target_to_classifier_labels['other'] = remaining_classifier_labels
    target_to_sorted_labels = {
        target: sorted(labels_set)
        for target, labels_set in target_to_classifier_labels.items()
    }
    return target_to_sorted_labels


def parse_spec(spec_dict: Mapping[str, Any],
               taxon_to_node: dict[tuple[str, str], TaxonNode],
               label_to_node: dict[tuple[str, str], TaxonNode]
               ) -> set[TaxonNode]:
    """
    Args:
        spec_dict: dict, contains keys ['taxa', 'dataset_labels', 'exclude']
            {
              "taxa": [
                {'level': 'family', 'name': 'cervidae', 'datasets': ['idfg']},
                {'level': 'genus',  'name': 'meleagris'} ],
              "dataset_labels": { "idfg_swwlf_2019": ["bird"] },
              "exclude": {...}
            }
        taxon_to_node: dict, maps (taxon_level, taxon_name) to a TaxonNode
        label_to_node: dict, maps (dataset_name, dataset_label) to the lowest
            TaxonNode node in the tree that contains the label

    Returns: set of TaxonNode, nodes selected by the taxa spec

    Raises: ValueError, if specification does not match any dataset labels
    """
    result = set()
    if 'taxa' in spec_dict:
        for taxon in spec_dict['taxa']:
            key = (taxon['level'].lower(), taxon['name'].lower())
            if key in taxon_to_node:
                node = taxon_to_node[key]
                result.add(node)
                result |= nx.descendants(node.graph, node)
            else:
                print(f'Taxon {key} not found in taxonomy graph. Ignoring.')
    if 'dataset_labels' in spec_dict:
        for ds, ds_labels in spec_dict['dataset_labels'].items():
            ds = ds.lower()
            for ds_label in ds_labels:
                node = label_to_node[(ds, ds_label.lower())]
                result.add(node)
                result |= nx.descendants(node.graph, node)
    if 'exclude' in spec_dict:
        result -= parse_spec(spec_dict['exclude'], taxon_to_node, label_to_node)
    if len(result) == 0:
        raise ValueError(f'specification matched no TaxonNode: {spec_dict}')
    return result


def label_spec_to_nodes(label_spec_js: dict[str, dict[str, Any]],
                        taxon_to_node: dict[tuple[str, str], TaxonNode],
                        label_to_node: dict[tuple[str, str], TaxonNode]
                        ) -> dict[str, set[TaxonNode]]:
    """Convert label spec to a mapping from classification labels to a set of
    nodes.

    Args:
        label_spec_js: dict, Python dict representation of JSON file
            see classification/README.md
        taxon_to_node: dict, maps (taxon_level, taxon_name) to a TaxonNode
        label_to_node: dict, maps (dataset_name, dataset_label) to the lowest
            TaxonNode node in the tree that contains the label

    Returns: dict, maps label name to set of TaxonNode

    Raises: ValueError, if a classification label specification matches no
        TaxonNode, or if a node is included in two or more classification labels
    """
    # maps output label name to set of (dataset, dataset_label) tuples
    seen_nodes: set[TaxonNode] = set()
    label_to_nodes: dict[str, set[TaxonNode]] = {}
    for label, spec_dict in label_spec_js.items():
        include_set = parse_spec(spec_dict, taxon_to_node, label_to_node)
        if include_set.isdisjoint(seen_nodes):
            label_to_nodes[label] = include_set
            seen_nodes |= include_set
        else:
            # find which other label (label_b) has intersection
            for label_b, set_b in label_to_nodes.items():
                shared = include_set.intersection(set_b)
                if len(shared) > 0:
                    print(f'Labels {label} and {label_b} share images:', shared)
                    raise ValueError('Intersection between sets!')
    return label_to_nodes


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create mapping from target categories to classifier '
                    'labels.')
    parser.add_argument(
        'desired_label_spec_json',
        help='path to JSON file containing desired label specification')
    parser.add_argument(
        'classifier_label_spec_json',
        help='path to JSON file containing label specification of a trained '
             'classifier')
    parser.add_argument(
        'taxonomy_csv',
        help='path to taxonomy CSV file')
    parser.add_argument(
        '-o', '--output', required=True,
        help='path to output JSON')
    parser.add_argument(
        '-i', '--classifier-label-index',
        help='(optional) path to label index JSON file for trained classifier, '
             'needed if not all labels from <classifier_label_spec_json> were '
             'actually used (e.g., if some labels were filtered out by the '
             '--min-locs argument for create_classification_dataset.py)')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(desired_label_spec_json_path=args.desired_label_spec_json,
         classifier_label_spec_json_path=args.classifier_label_spec_json,
         taxonomy_csv_path=args.taxonomy_csv,
         output_json_path=args.output,
         classifier_label_index_path=args.classifier_label_index)
