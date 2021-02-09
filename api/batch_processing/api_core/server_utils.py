# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Helper functions for the batch processing API.
"""

import logging
import os
from datetime import datetime
from typing import Tuple, Any, Sequence, Optional

import sas_blob_utils  # from ai4eutils


log = logging.getLogger(os.environ['FLASK_APP'])


#%% helper classes and functions

def make_error(error_code: int, error_message: str) -> Tuple[dict, int]:
    # TODO log exception when we have more telemetry
    log.error(f'Error {error_code} - {error_message}')
    return {'error': error_message}, error_code


def check_data_container_sas(input_container_sas: str) -> Optional[Tuple[int, str]]:
    """
    Returns a tuple (error_code, msg) if not a usable SAS URL, else returns None
    """
    # TODO check that the expiry date of input_container_sas is at least a month
    # into the future
    permissions = sas_blob_utils.get_permissions_from_uri(input_container_sas)
    data = sas_blob_utils.get_all_query_parts(input_container_sas)

    msg = ('input_container_sas provided does not have both read and list '
           'permissions.')
    if 'read' not in permissions or 'list' not in permissions:
        if 'si' in data:
            # if no permission specified explicitly but has an access policy, assumes okay
            # TODO - check based on access policy as well
            return None

        return 400, msg

    return None


def get_utc_time() -> str:
    # return current UTC time as a string in the ISO 8601 format (so we can query by
    # timestamp in the Cosmos DB job status table.
    # example: '2021-02-08T20:02:05.699689Z'
    return datetime.utcnow().isoformat(timespec='microseconds') + 'Z'


def get_job_status(request_status: str, message: Any) -> dict:
    return {
        'request_status': request_status,
        'message': message
    }


def validate_provided_image_paths(image_paths: Sequence[Any]) -> Tuple[Optional[str], bool]:
    """Given a list of image_paths (list length at least 1), validate them and
    determine if metadata is available.
    Args:
        image_paths: a list of string (image_id) or a list of 2-item lists
            ([image_id, image_metadata])
    Returns:
        error: None if checks passed, otherwise a string error message
        metadata_available: bool, True if available
    """
    # image_paths will have length at least 1, otherwise would have ended before this step
    first_item = image_paths[0]
    metadata_available = False
    if isinstance(first_item, str):
        for i in image_paths:
            if not isinstance(i, str):
                error = 'Not all items in image_paths are of type string.'
                return error, metadata_available
        return None, metadata_available
    elif isinstance(first_item, list):
        metadata_available = True
        for i in image_paths:
            if len(i) != 2:  # i should be [image_id, metadata_string]
                error = ('Items in image_paths are lists, but not all lists '
                         'are of length 2 [image locator, metadata].')
                return error, metadata_available
        return None, metadata_available
    else:
        error = 'image_paths contain items that are not strings nor lists.'
        return error, metadata_available
