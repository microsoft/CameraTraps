# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
This script can be run in a separate process to monitor all instances of the batch API.
It sends a digest of submissions within the past day to a Teams channel webhook.

It requires the environment variables TEAMS_WEBHOOK, COSMOS_ENDPOINT and COSMOS_READ_KEY to be set.
"""

import time
import os
import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import requests
from azure.cosmos.cosmos_client import CosmosClient


# Cosmos DB `batch-api-jobs` table for job status
COSMOS_ENDPOINT = os.environ['COSMOS_ENDPOINT']
COSMOS_READ_KEY = os.environ['COSMOS_READ_KEY']

TEAMS_WEBHOOK = os.environ['TEAMS_WEBHOOK']


def send_message():
    cosmos_client = CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_READ_KEY)
    db_client = cosmos_client.get_database_client('camera-trap')
    db_jobs_client = db_client.get_container_client('batch_api_jobs')

    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)

    query = f'''
    SELECT *
    FROM job
    WHERE job.job_submission_time >= "{yesterday.isoformat()}T00:00:00Z"
    '''

    result_iterable = db_jobs_client.query_items(query=query,
                                                 enable_cross_partition_query=True)

    # aggregate the number of images, country and organization names info from each job
    # submitted during yesterday (UTC time)
    instance_num_images = defaultdict(lambda: defaultdict(int))
    instance_countries = defaultdict(set)
    instance_orgs = defaultdict(set)

    total_images_received = 0

    for job in result_iterable:
        api_instance = job['api_instance']
        status = job['status']
        call_params = job['call_params']

        if status['request_status'] == 'completed':
            instance_num_images[api_instance]['num_images_completed'] += status.get('num_images', 0)
        instance_num_images[api_instance]['num_images_total'] += status.get('num_images', 0)
        total_images_received += status.get('num_images', 0)

        instance_countries[api_instance].add(call_params.get('country', 'unknown'))
        instance_orgs[api_instance].add(call_params.get('organization_name', 'unknown'))

    print(f'send_message, number of images received yesterday: {total_images_received}')

    if total_images_received < 1:
        print('send_message, no images submitted yesterday, not sending a summary')
        print('')
        return

    # create the card
    sections = []

    for instance_name, num_images in instance_num_images.items():
        entry = {
            'activityTitle': f'API instance: {instance_name}',
            'facts': [
                {
                    'name': 'Total images',
                    'value': '{:,}'.format(num_images['num_images_total'])
                },
                {
                    'name': 'Images completed',
                    'value': '{:,}'.format(num_images['num_images_completed'])
                },
                {
                    'name': 'Countries',
                    'value': ', '.join(sorted(list(instance_countries[instance_name])))
                },
                {
                    'name': 'Organizations',
                    'value': ', '.join(sorted(list(instance_orgs[instance_name])))
                }
            ]
        }
        sections.append(entry)

    card = {
        '@type': 'MessageCard',
        '@context': 'http://schema.org/extensions',
        'themeColor': 'ffcdb2',
        'summary': 'Digest of batch API activities over the past 24 hours',
        'title': f'Camera traps batch API activities on {yesterday.strftime("%b %d, %Y")}',
        'sections': sections,
        'potentialAction': [
            {
                '@type': 'OpenUri',
                'name': 'View Batch account in Azure Portal',
                'targets': [
                    {
                        'os': 'default',
                        'uri': 'https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/resource/subscriptions/74d91980-e5b4-4fd9-adb6-263b8f90ec5b/resourcegroups/camera_trap_api_rg/providers/Microsoft.Batch/batchAccounts/cameratrapssc/accountOverview'
                    }
                ]
            }
        ]
    }

    response = requests.post(TEAMS_WEBHOOK, data=json.dumps(card))
    print(f'send_message, card to send:')
    print(json.dumps(card, indent=4))
    print(f'send_message, sent summary to webhook, response code: {response.status_code}')
    print('')


def main():
    """
    Wake up at 5 minutes past midnight UTC to send a summary of yesterday's activities if there were any.
    Then goes in a loop to wake up and send a summary every 24 hours.
    """
    current = datetime.utcnow()
    future = current.replace(day=current.day, hour=0, minute=5, second=0, microsecond=0) + timedelta(
        days=1)  # current has been modified

    current = datetime.utcnow()
    duration = future - current

    duration_hours = duration.seconds / (60 * 60)
    print(f'Current time: {current}')
    print(f'Will wake up at {future}, in {duration_hours} hours')
    print('')

    time.sleep(duration.seconds)

    while True:
        print(f'Woke up at {datetime.utcnow()}')
        send_message()
        time.sleep(24 * 60 * 60)


if __name__ == '__main__':
    main()
