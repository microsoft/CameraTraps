import inspect
import json
import os
from datetime import datetime, timedelta

import jsonschema
from urllib import parse


def datasets_schema_check(items_json):
    """
    Checks the datasets table against its schema and that the SAS keys have not expired. 

    Args:
        items_json: a str path to a .json where a copy of the datasets table is stored, or the datasets table as a Python list
    """
    if isinstance(items_json, str):
        with open(items_json) as f:
            items_json = json.load(f)

    assert len(items_json) > 0, 'The .json file you passed in is empty'

    # load the schema
    # https://stackoverflow.com/questions/3718657/how-to-properly-determine-current-script-directory
    this_script = inspect.getframeinfo(inspect.currentframe()).filename
    dir = os.path.dirname(this_script)
    with open(os.path.join(dir, 'datasets_schema.json')) as f:
        schema = json.load(f)

    jsonschema.validate(items_json, schema)

    print('Verified that the dataset items conform to the schema.')

    # checks across all datasets items
    names = set([ds['dataset_name'] for ds in items_json])
    assert len(names) == len(items_json), 'The field dataset_name is not unique.'

    # check for expiry date of the SAS keys
    today = datetime.utcnow()
    today_plus_30_days = today + timedelta(days=30)

    for dataset in items_json:
        fields = parse.parse_qs(dataset['container_sas_key'].split('?')[1])
        expiry_date = datetime.strptime(fields['se'][0], '%Y-%m-%dT%H:%M:%SZ')  # This is UTC ("Z" denotes Zulu time)

        if expiry_date <= today:
            print('SAS token of dataset {} has expired!'.format(dataset['dataset_name']))

        if expiry_date <= today_plus_30_days:
            print('SAS token of dataset {} has will expire in the next 30 days'.format(dataset['dataset_name']))

    print('Finished running other checks.')


