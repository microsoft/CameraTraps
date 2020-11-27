"""
If a request has been sent to AML for batch scoring but the monitoring thread of the API was
interrupted (uncaught exception or having to re-start the API container), we could manually
aggregate results from each shard using this script, assuming all jobs submitted to AML have finished.

1. Need to have set environment variables STORAGE_ACCOUNT_NAME and STORAGE_ACCOUNT_KEY to those of the
storage account backing the API. All containers involved are in this storage account.

2. May need to change the import statement in api_core/orchestrator_api/orchestrator.py
"from sas_blob_utils import SasBlob" to
"from .sas_blob_utils import SasBlob" to not confuse with the module in AI4Eutils;

3. Change "import api_config" to
"from api.batch_processing.api_core.api_instances_config import api_config_internal as api_config",
choosing the config file for the correct instance

4. Need to have azure-storage-blob==2.1.0 in the environment (older version) - we're using something
closer to azure-storage-blob==12.5.0 in the rest of the repo.

Execute this script from the root of the repository. You may need to add the repository to PYTHONPATH.
"""

import argparse
import json

from api.batch_processing.api_core.orchestrator_api.orchestrator import AMLMonitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request_id', type=str,
                        help='the request ID to restart monitoring')
    parser.add_argument('model_version', type=str, help='version of megadetector used; this is used to fill in the meta info section of the output file')
    parser.add_argument('request_name', type=str, help='easy to remember name for that job, optional', default='')
    args = parser.parse_args()

    shortened_request_id = args.request_id.split('-')[0]
    if len(shortened_request_id) > 8:
        shortened_request_id = shortened_request_id[:8]

    # list_jobs_submitted cannot be serialized ("can't pickle _thread.RLock objects "), but
    # do not need it for aggregating results
    aml_monitor = AMLMonitor(
        request_id=args.request_id,
        shortened_request_id=shortened_request_id,
        list_jobs_submitted=None,
        request_name=args.request_name,
        request_submission_timestamp='',
        model_version=args.model_version
    )
    output_file_urls = aml_monitor.aggregate_results()
    output_file_urls_str = json.dumps(output_file_urls)
    print(output_file_urls_str)


if __name__ == '__main__':
    main()