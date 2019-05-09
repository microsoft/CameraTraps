# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import uuid
import io
from datetime import datetime, timedelta
from urllib import parse

from azure.storage.blob import (
    BlockBlobService,
    ContainerPermissions,
    BlobPermissions,
    PublicAccess,
    ContentSettings,
    BlobBlock,
    BlockListType,
)


"""
This module contains helper functions for dealing with Shared Access Signatures (SAS) tokens
for Azure Blob Storage.

Documentation for Azure Blob Storage:
https://azure-storage.readthedocs.io/ref/azure.storage.blob.baseblobservice.html
https://azure-storage.readthedocs.io/ref/azure.storage.blob.blockblobservice.html

Documentation for SAS:
https://docs.microsoft.com/en-us/azure/storage/common/storage-dotnet-shared-access-signature-part-1
"""

class SasBlob:
    @staticmethod
    def _get_resource_reference(prefix):
        return '{}{}'.format(prefix, str(uuid.uuid4()).replace('-', ''))

    @staticmethod
    def get_service_from_uri(sas_uri):
        return BlockBlobService(
            account_name=SasBlob.get_account_from_uri(sas_uri),
            sas_token=SasBlob.get_sas_key_from_uri(sas_uri))

    @staticmethod
    def get_service_from_datastore(datastore):
        return BlockBlobService(account_name=datastore['account_name'],
                                account_key=datastore['account_key'])

    @staticmethod
    def get_account_from_uri(sas_uri):
        url_parts = parse.urlsplit(sas_uri)
        loc = url_parts.netloc
        return loc.split('.')[0]

    @staticmethod
    def get_container_from_uri(sas_uri):
        url_parts = parse.urlsplit(sas_uri)

        raw_path = url_parts.path[1:]
        container = raw_path.split('/')[0]

        return container

    @staticmethod
    def get_blob_path_from_uri(sas_uri, unquote_blob=True):
        """Return the path to the blob from the root container if this sas_uri is for an
        individual blob; otherwise returns None.

        Args:
            sas_uri: Azure blob storage SAS token
            unquote_blob: Replace any %xx escapes by their single-character equivalent. Default True.

        Returns: Path to the blob from the root container or None.

        """
        # Get the entire path with all slashes after the container
        url_parts = parse.urlsplit(sas_uri)
        raw_path = url_parts.path[1:]
        container = raw_path.split('/')[0]

        parts = raw_path.split(container + '/')
        if len(parts) < 2:
            return None

        blob = parts[1] # first item is an empty string

        if unquote_blob:
            blob = parse.unquote(blob)

        return blob

    @staticmethod
    def get_sas_key_from_uri(sas_uri):
        """Get the query part of the SAS token that contains permissions, access times and
        signature.

        Args:
            sas_uri: Azure blob storage SAS token

        Returns: Query part of the SAS token.
        """
        url_parts = parse.urlsplit(sas_uri)
        return url_parts.query

    @staticmethod
    def get_resource_type_from_uri(sas_uri):
        """Get the resource type pointed to by this SAS token

        Args:
            sas_uri: Azure blob storage SAS token

        Returns: A string (either 'blob' or 'container') or None.
        """
        url_parts = parse.urlsplit(sas_uri)
        data = parse.parse_qs(url_parts.query)
        if 'sr' in data:
            types = data['sr']
            if 'b' in types:
                return 'blob'
            elif 'c' in types:
                return 'container'
        else:
            return None

    @staticmethod
    def get_permissions_from_uri(sas_uri):
        """Get the permissions given by this SAS token

        Args:
            sas_uri: Azure blob storage SAS token

        Returns: A set containing some of 'read', 'write', 'delete' and 'list'. Empty set
        returned if no permission specified in sas_uri.
        """
        # TODO - fix this for testing out only the query string
        url_parts = parse.urlsplit(sas_uri)
        data = parse.parse_qs(url_parts.query)
        permissions_set = set()
        if 'sp' in data:
            permissions = data['sp'][0]
            if 'r' in permissions:
                permissions_set.add('read')
            if 'w' in permissions:
                permissions_set.add('write')
            if 'd' in permissions:
                permissions_set.add('delete')
            if 'l' in permissions:
                permissions_set.add('list')
        return permissions_set

    @staticmethod
    def get_all_query_parts(sas_uri):
        url_parts = parse.urlsplit(sas_uri)
        return parse.parse_qs(url_parts.query)

    @staticmethod
    def check_blob_exists_in_container(blob_name, container_sas_uri=None, datastore=None):
        if container_sas_uri:
            blob_service = BlockBlobService(
                account_name=SasBlob.get_account_from_uri(container_sas_uri),
                sas_token=SasBlob.get_sas_key_from_uri(container_sas_uri))
            container_name = SasBlob.get_container_from_uri(container_sas_uri)
        elif datastore:
            blob_service = SasBlob.get_service_from_datastore(datastore)
            container_name = datastore['container_name']
        else:
            raise RuntimeError('Error in check_blob_exists_in_container(): one of container_sas_uri or datastore must be provided.')

        return blob_service.exists(container_name, blob_name)

    @staticmethod
    def list_blobs_in_container(max_number_to_list, sas_uri=None, datastore=None,
                                blob_prefix=None, blob_suffix=None):
        """Get a list of blob names/paths in the container specified in either the sas_uri or the datastore.
        This function will request a list of 4000 blobs at a time from the Blob Service (max number
        allowed is 5000), and use the next_marker to retrieve more batches, until max_number_to_list is reached.
        If the max_number_to_list is reached, a list containing that many results will be returned.

        If you're processing the blob information one by one, get a generator and iterate using it instead.

        Args:
            max_number_to_list: Maximum number of blob names/paths to list
            sas_uri: Azure blob storage SAS token
            datastore: dict with fields account_name (of the Blob storage account), account_key and container_name
            blob_prefix: Optional, a string as the prefix to blob names/paths to filter the results to those
                        with this prefix. Case-sensitive!
            blob_suffix: Optional, an all lower case string or a tuple of strings, to filter the results to
                        those with this/these suffix(s).
                        The blob names will be lowercased first before comparing with the suffix(es).
        Returns:
            A list of blob names/paths, of length max_number_to_list or shorter.
        """
        if sas_uri and SasBlob.get_resource_type_from_uri(sas_uri) != 'container':
            raise ValueError('The SAS token provided is not for a container.')

        if blob_prefix and not isinstance(blob_prefix, str):
            raise ValueError('blob_prefix needs to be a str.')

        if blob_suffix and not isinstance(blob_suffix, str) and not isinstance(blob_suffix, tuple):
            raise ValueError('blob_suffix needs to be a str or a tuple of strings')

        if sas_uri:
            blob_service = SasBlob.get_service_from_uri(sas_uri)
            container_name = SasBlob.get_container_from_uri(sas_uri)
        elif datastore:
            try:
                container_name = datastore['container_name']
                blob_service = BlockBlobService(account_name=datastore['account_name'],
                                                account_key=datastore['account_key'])
            except Exception as e:
                raise RuntimeError(
                    'Error occurred while connecting to blob via info provided in a datastore object: {}'.format(str(e)))
        else:
            raise RuntimeError('Error in list_blobs_in_container(): one of sas_uri and datstore must be provided.')

        generator = blob_service.list_blobs(container_name, prefix=blob_prefix, num_results=4000)

        list_blobs = []

        while True:
            for blob in generator:
                if blob_suffix is None or blob.name.lower().endswith(blob_suffix):
                    list_blobs.append(blob.name)

            next_marker = generator.next_marker
            if next_marker == '':
                # exhaustively listed all blobs in the container
                return list_blobs[:max_number_to_list]

            if len(list_blobs) > max_number_to_list:
                return list_blobs[:max_number_to_list]

            generator = blob_service.list_blobs(container_name, prefix=blob_prefix, num_results=4000,
                                                marker=next_marker)
        # list_blobs will have been returned by one of the two stopping conditions

    @staticmethod
    def generate_writable_container_sas(account_name, account_key, container_name, access_duration_hrs):
        block_blob_service = BlockBlobService(account_name=account_name, account_key=account_key)

        block_blob_service.create_container(container_name)

        token = block_blob_service.generate_container_shared_access_signature(
            container_name,
            ContainerPermissions.WRITE + ContainerPermissions.READ + ContainerPermissions.LIST,
            datetime.utcnow() + timedelta(hours=access_duration_hrs))

        return block_blob_service.make_container_url(container_name=container_name, sas_token=token).replace(
            "restype=container", "")

    @staticmethod
    def generate_blob_sas_uri(container_sas_uri, blob_name):
        container_name = SasBlob.get_container_from_uri(container_sas_uri)
        sas_service = BlockBlobService(
            account_name=SasBlob.get_account_from_uri(container_sas_uri),
            sas_token=SasBlob.get_sas_key_from_uri(container_sas_uri))
        blob_uri = sas_service.make_blob_url(container_name, blob_name,
                                             sas_token=SasBlob.get_sas_key_from_uri(container_sas_uri))

        return blob_uri

    @staticmethod
    def create_blob_from_bytes(sas_uri, blob_name, input_bytes):
        sas_service = BlockBlobService(
            account_name = SasBlob.get_account_from_uri(sas_uri),
            sas_token = SasBlob.get_sas_key_from_uri(sas_uri))

        container_name = SasBlob.get_container_from_uri(sas_uri)

        sas_service.create_blob_from_bytes(container_name, blob_name, input_bytes)

        return sas_service.make_blob_url(container_name, blob_name, sas_token=SasBlob.get_sas_key_from_uri(sas_uri))

    @staticmethod
    def create_blob_from_text(sas_uri, blob_name, text):
        sas_service = BlockBlobService(
            account_name=SasBlob.get_account_from_uri(sas_uri),
            sas_token=SasBlob.get_sas_key_from_uri(sas_uri))

        container_name = SasBlob.get_container_from_uri(sas_uri)

        sas_service.create_blob_from_text(container_name, blob_name, text, 'utf-8')

        return sas_service.make_blob_url(container_name, blob_name, sas_token=SasBlob.get_sas_key_from_uri(sas_uri))

    @staticmethod
    def create_blob_from_stream(sas_uri, blob_name, input_stream):
        sas_service = BlockBlobService(
            account_name=SasBlob.get_account_from_uri(sas_uri),
            sas_token=SasBlob.get_sas_key_from_uri(sas_uri))

        container_name = SasBlob.get_container_from_uri(sas_uri)

        sas_service.create_blob_from_stream(container_name, blob_name, input_stream)

        return sas_service.make_blob_url(container_name, blob_name, sas_token=SasBlob.get_sas_key_from_uri(sas_uri))

    @staticmethod
    def download_blob_to_stream(sas_uri):
        sas_service = BlockBlobService(
            account_name=SasBlob.get_account_from_uri(sas_uri),
            sas_token=SasBlob.get_sas_key_from_uri(sas_uri))

        with io.BytesIO() as output_stream:
            blob = sas_service.get_blob_to_stream(SasBlob.get_container_from_uri(sas_uri),
                                                  SasBlob.get_blob_path_from_uri(sas_uri),
                                                  output_stream)
        return output_stream, blob

    @staticmethod
    def download_blob_to_text(sas_uri):
        sas_service = BlockBlobService(
            account_name=SasBlob.get_account_from_uri(sas_uri),
            sas_token=SasBlob.get_sas_key_from_uri(sas_uri))

        blob = sas_service.get_blob_to_text(SasBlob.get_container_from_uri(sas_uri),
                                              SasBlob.get_blob_path_from_uri(sas_uri))
        return blob.content


