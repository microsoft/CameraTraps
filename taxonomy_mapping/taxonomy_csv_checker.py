"""
Checks the taxonomy CSV file to make sure that for each row:
1) The 'taxonomy_level' column matches the lowest-level taxon level in the
    'taxonomy_string' column.
2) The 'scientific_name' column matches the scientific name from the
    lowest-level taxon level in the 'taxonomy_string' column.

Prints out any mismatches.

Also prints out nodes that have 2 ambiguous parents. See "CASE 2" from the
module docstring of taxonomy_graph.py.
"""

#%% Imports

import argparse
from typing import Optional
import networkx as nx
import pandas as pd

from taxonomy_mapping.taxonomy_graph import TaxonNode, dag_to_tree


#%% Taxnomy checking

def check_taxonomy_csv(csv_path: str) -> None:
    """
    See module docstring.
    """
    
    taxonomy_df = pd.read_csv(csv_path)

    graph = nx.DiGraph()
    taxon_to_node = {}  # maps (taxon_level, taxon_name) to a TaxonNode

    num_taxon_level_errors = 0
    num_scientific_name_errors = 0

    for i, row in taxonomy_df.iterrows():
        
        ds = row['dataset_name']
        ds_label = row['query']
        scientific_name = row['scientific_name']
        level = row['taxonomy_level']
        id_source = row['source']

        taxa_ancestry = row['taxonomy_string']
        if pd.isna(taxa_ancestry):
            # taxonomy CSV rows without 'taxonomy_string' entries are excluded
            # from the taxonomy graph, but can be included in a classification
            # label specification JSON via the 'dataset_labels' key
            continue
        else:
            taxa_ancestry = eval(taxa_ancestry)  # pylint: disable=eval-used

        taxon_child: Optional[TaxonNode] = None
        for j, taxon in enumerate(taxa_ancestry):
            taxon_id, taxon_level, taxon_name, _ = taxon

            key = (taxon_level, taxon_name)
            if key not in taxon_to_node:
                taxon_to_node[key] = TaxonNode(level=taxon_level,
                                               name=taxon_name, graph=graph)
            node = taxon_to_node[key]

            if taxon_child is not None:
                node.add_child(taxon_child)

            node.add_id(id_source, int(taxon_id))  # np.int64 -> int
            if j == 0:
                if level != taxon_level:
                    print(f'row: {i}, {ds}, {ds_label}')
                    print(f'- taxonomy_level column: {level}, '
                          f'level from taxonomy_string: {taxon_level}')
                    print()
                    num_taxon_level_errors += 1

                if scientific_name != taxon_name:
                    print(f'row: {i}, {ds}, {ds_label}')
                    print(f'- scientific_name column: {scientific_name}, '
                          f'name from taxonomy_string: {taxon_name}')
                    print()
                    num_scientific_name_errors += 1

            taxon_child = node
    
    # ...for each row in the taxnomy file        
            
    assert nx.is_directed_acyclic_graph(graph)

    for node in graph.nodes:
        assert len(node.parents) <= 2
        if len(node.parents) == 2:
            p0 = node.parents[0]
            p1 = node.parents[1]
            assert p0 is not p1

            p0_is_ancestor_of_p1 = p1 in nx.descendants(graph, p0)
            p1_is_ancestor_of_p0 = p0 in nx.descendants(graph, p1)
            if not p0_is_ancestor_of_p1 and not p1_is_ancestor_of_p0:
                print('Node with two ambiguous parents:', node)
                print('\t', p0)
                print('\t\t', p0.parents)
                print('\t', p1)
                print('\t\t', p1.parents)

    try:
        dag_to_tree(graph, taxon_to_node)
        print('All ambiguous parents have hard-coded resolution in '
              'dag_to_tree().')
    except AssertionError as e:
        print(f'At least one node has unresolved ambiguous parents: {e}')

    print('num taxon level errors:', num_taxon_level_errors)
    print('num scientific name errors:', num_scientific_name_errors)


#%% Command-line driver
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'taxonomy_csv_path',
        help='path to taxonomy CSV file')
    args = parser.parse_args()

    check_taxonomy_csv(args.taxonomy_csv_path)


#%% Interactive driver
    
if False:
    
    #%%
    
    csv_path = r"G:\git\agentmorrisprivate\lila-taxonomy\lila-taxonomy-mapping-input.csv"
    check_taxonomy_csv(csv_path)
    
