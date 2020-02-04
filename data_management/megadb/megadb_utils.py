"""
Functions to download the `datasets` and `splits` tables, which are small.

The environment variables COSMOS_ENDPOINT and COSMOS_KEY need to be set
or passed in to the initializer.
"""

import os

from azure.cosmos.cosmos_client import CosmosClient


class MegadbUtils:

    def __init__(self, url=None, key=None):
        if not url:
            url = os.environ['COSMOS_ENDPOINT']
        if not key:
            key = os.environ['COSMOS_KEY']
        client = CosmosClient(url, credential=key)
        self.database = client.get_database_client('camera-trap')

    def get_datasets_table(self):
        """

        Returns: a dict where the keys are the `dataset` property in sequences and splits,
                 and the values are properties of the dataset

        """
        query = '''SELECT * FROM datasets d'''

        container_datasets = self.database.get_container_client('datasets')
        result_iterable = container_datasets.query_items(query=query, enable_cross_partition_query=True)

        datasets = {i['dataset_name']: {k: v for k, v in i.items() if not k.startswith('_')} for i in
                    iter(result_iterable)}
        return datasets

    def get_splits_table(self):
        """

        Returns: a dict where the key is the name of the `dataset` and value is a dict with
        the train, val and test splits by location as *sets*.

        """
        query = '''SELECT * FROM datasets d'''

        container_splits = self.database.get_container_client('splits')
        result_iterable = container_splits.query_items(query=query, enable_cross_partition_query=True)

        splits = {i['dataset']: {k: set(v) for k, v in i.items() if not k.startswith('_')} for i in
                    iter(result_iterable)}
        return splits
