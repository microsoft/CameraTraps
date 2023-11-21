"""
Functions to download the `datasets` and `splits` tables, which are small.

The environment variables COSMOS_ENDPOINT and COSMOS_KEY need to be set
or passed in to the initializer.
"""

import os
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Set

from azure.cosmos.cosmos_client import CosmosClient
from azure.storage.blob import ContainerClient


class Splits(str, Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class MegadbUtils:
    """
    Attributes
    - database: azure.cosmos.DatabaseProxy, 'camera-trap' database
    - container_sequences: azure.cosmos.ContainerProxy, 'sequences' container
    """

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        if not url:
            url = os.environ['COSMOS_ENDPOINT']
        if not key:
            key = os.environ['COSMOS_KEY']
        client = CosmosClient(url, credential=key)
        self.database = client.get_database_client('camera-trap')
        self.container_sequences = self.database.get_container_client(
            'sequences')

    def get_datasets_table(self) -> Dict[str, Any]:
        """Gets the datasets table.

        Returns: dict, keys are dataset names, values are dataset properties
        """
        query = '''SELECT * FROM datasets d'''

        container_datasets = self.database.get_container_client('datasets')
        result_iterable = container_datasets.query_items(
            query=query, enable_cross_partition_query=True)

        datasets = {
            i['dataset_name']: {k: i[k] for k in i if not k.startswith('_')}
            for i in result_iterable
        }
        return datasets

    def get_splits_table(self) -> Dict[str, Dict[Splits, Set[Any]]]:
        """Gets the splits table.

        Returns: dict, each key is a dataset name, and each value is a dict with
            the train, val and test splits by location as *sets*.
        """
        query = '''SELECT * FROM datasets d'''

        container_splits = self.database.get_container_client('splits')
        result_iterable = container_splits.query_items(
            query=query, enable_cross_partition_query=True)

        splits = {
            i['dataset']: {split: set(i[split]) for split in Splits}
            for i in result_iterable
        }
        return splits

    def query_sequences_table(
            self, query: str, partition_key: Optional[str] = None,
            parameters: Optional[List[Dict[str, Any]]] = None
        ) -> List[Dict[str, Any]]:
        """
        Args:
            query: str, SQL query
            partition_key: str, the dataset name is the partition key
                see scheme/sequences_schema.json
            parameters: list of dict

        Returns: list of dict, each dict represents a single sequence
        """
        if partition_key:
            result_iterable = self.container_sequences.query_items(
                query=query, partition_key=partition_key, parameters=parameters)
        else:
            result_iterable = self.container_sequences.query_items(
                query=query, enable_cross_partition_query=True,
                parameters=parameters)

        return result_iterable

    @staticmethod
    def get_storage_client(datasets_table: Mapping[str, Any],
                           dataset_name: str) -> ContainerClient:
        """Gets a ContainerClient for the Azure Blob Storage Container
        corresponding to the given dataset.

        Adds 'container_client' key to datasets_table (in-place update) if a new
        ContainerClient is created for the dataset.

        Args:
            datasets_table: dict, the return value of get_datasets_table()
            dataset_name: str, key in datasets_table

        Returns: azure.storage.blob.ContainerClient, corresponds to the
            requested dataset
        """
        if dataset_name not in datasets_table:
            raise KeyError(f'Dataset {dataset_name} is not in datasets table.')

        entry = datasets_table[dataset_name]
        if 'storage_container_client' not in entry:
            # create a new storage container client for this dataset,
            # and cache it
            if 'container_sas_key' not in entry:
                raise KeyError(f'Dataset {dataset_name} does not have the '
                               'container_sas_key field in the datasets table.')
            entry['storage_container_client'] = ContainerClient(
                account_url=f'{entry["storage_account"]}.blob.core.windows.net',
                container_name=entry['container'],
                credential=entry['container_sas_key'])

        return entry['storage_container_client']

    @staticmethod
    def get_full_path(datasets_table: Mapping[str, Any], dataset_name: str,
                      img_path: str) -> str:
        """Gets the full blob path to an image.

        Args:
            datasets_table: dict, the return value of get_datasets_table()
            dataset_name: str, key in datasets_table
            img_path: str, path in 'file' field of an image from the dataset

        Returns: str, full blob path to image
        """
        entry = datasets_table[dataset_name]
        if 'path_prefix' not in entry or entry['path_prefix'] == '':
            return img_path
        else:
            return os.path.join(entry['path_prefix'], img_path)
