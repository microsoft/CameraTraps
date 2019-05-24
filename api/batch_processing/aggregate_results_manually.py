#
# If a request has been sent to AML for batch scoring but the monitoring thread of the API was
# interrupted (uncaught exception or having to re-start the API container), we could manually
# aggregate results from each shard using this script, assuming all jobs submitted to AML have finished.
#
# Need to have set environment variables STORAGE_ACCOUNT_NAME and STORAGE_ACCOUNT_KEY to those of the
# storage account backing the API.
#
# May need to change the import statement in api/orchestrator_api/orchestrator.py "from sas_blob_utils import SasBlob"
# to "from .sas_blob_utils import SasBlob" to not confuse with the module in AI4Eutils

import argparse
import json
import sys

sys.path.append('./api/detector_batch_processing/api/orchestrator_api')

from api.detector_batch_processing.api.orchestrator_api.orchestrator import AMLMonitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request_id', action='store', type=str,
                        help='the request ID to restart monitoring')
    args = parser.parse_args()

    # list_jobs_submitted cannot be serialized ("can't pickle _thread.RLock objects "), but
    # do not need it for aggregating results
    aml_monitor = AMLMonitor(args.request_id, None)
    output_file_urls = aml_monitor.aggregate_results()
    output_file_urls_str = json.dumps(output_file_urls)
    print(output_file_urls_str)


if __name__ == '__main__':
    main()
