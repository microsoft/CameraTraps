"""
This script will be run when all Tasks in this Job have finished and the Job is marked as
Complete. It aggregates the model outputs from each task into one JSON file.
"""

import json
import os
from datetime import datetime
from typing import Tuple

from azure.appconfiguration import AzureAppConfigurationClient

#%% constants

# copied from TFDetector class in detection/run_tf_detector.py
DEFAULT_DETECTOR_LABEL_MAP = {
    '1': 'animal',
    '2': 'person',
    '3': 'vehicle'
}


#%% small helper functions

def make_error(error_code: int, error_message, str) -> Tuple[dict, int]:
    # TODO log exception
    return ({'error': error_message}, error_code)


#%% helper classes

class AppConfig:
    """Wrapper around the Azure App Configuration client"""

    def __int__(self, api_instance):
        APP_CONFIG_CONNECTION_STR = os.environ['APP_CONFIG_CONNECTION_STR']
        self.client = AzureAppConfigurationClient.from_connection_string(APP_CONFIG_CONNECTION_STR)

        self.api_instance = api_instance

        # sentinel should change if new configurations are available
        self.sentinel = self._get_sentinel()
        self.allowlist = self._get_allowlist()


    def _get_sentinel(self):
        return self.client.get_configuration_setting(key='batch_api:sentinel').value


    def _get_allowlist(self):
        filtered_listed = self.client.list_configuration_settings(key_filter='batch_api_allow:*')
        allowlist = []
        for item in filtered_listed:
            if item.value == self.api_instance:
                allowlist.append(item.key.split('batch_api_allow:')[1])
        return allowlist


    def get_allowlist(self):
        cur_sentinel = self._get_sentinel()
        if cur_sentinel == self.sentinel:
            # configs have not changed
            return self.allowlist
        else:
            self.sentinel = cur_sentinel
            self.allowlist = self._get_allowlist()
            return self.allowlist


def get_utc_time() -> str:
    # return current UTC time in string format, e.g., '2019-05-19 08:57:43'
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

def aggregate_results(task_outputs_folder):
    """Merge output from each scoring task into one JSON"""

    all_detections = []
    task_outputs = 0
    for fn in os.listdir(task_outputs_folder):
        if not fn.endswith('.json'):
            continue

        task_outputs += 1
        # error entries are included too
        task_detections = json.load(os.path.join(task_outputs_folder, fn))
        all_detections.extend(task_detections)

    print(f'complete_job.py, aggregate_results(), {task_outputs} task output files found')
    print(f'complete_job.py, aggregate_results(), total detection entries aggregated: {len(all_detections)}')
    return all_detections


def main():
    print('complete_job.py, main()')

    job_id = os.environ['AZ_BATCH_JOB_ID']
    # https://docs.microsoft.com/en-us/azure/batch/virtual-file-mount
    mount_point = os.environ['AZ_BATCH_NODE_MOUNTS_DIR']

    api_instance_name = os.environ['API_INSTANCE_NAME']
    model_version = os.environ['DETECTOR_VERSION']
    output_format_version = os.environ['OUTPUT_FORMAT_VERSION']
    user_suffix = os.environ['USER_SUFFIX']
    user_submission_time = os.environ['USER_SUBMISSION_TIME']

    assert len(api_instance_name) > 0, 'api_instance_name not found as an env variable'
    assert len(model_version) > 0, 'model_version not found as an env variable'
    assert len(output_format_version) > 0, 'output_format_version not found as an env variable'
    assert len(user_submission_time) > 0, 'user_submission_time not found as an env variable'

    job_folder_mounted = os.path.join(mount_point, 'batch-api', api_instance_name, f'job_{job_id}')
    task_outputs_folder = os.path.join(job_folder_mounted, 'task_outputs')
    all_detections = aggregate_results(task_outputs_folder)

    detection_output_content = {
        'info': {
            'detector': f'megadetector_v{model_version}',
            'detection_completion_time': get_utc_time(),
            'format_version': output_format_version
        },
        'detection_categories': DEFAULT_DETECTOR_LABEL_MAP,
        'images': all_detections
    }

    api_output_path = os.path.join(job_folder_mounted,
                                   f'{job_id}_detections_{user_suffix}_{user_submission_time}.json')
    print(f'complete_job.py, main(), api_output_path: {api_output_path}')

    with open(api_output_path, 'w') as f:
        json.dump(detection_output_content, f, indent=1)

    print('complete_job.py, main(), output written to mounted file. Done!')


if __name__ == '__main__':
    main()
