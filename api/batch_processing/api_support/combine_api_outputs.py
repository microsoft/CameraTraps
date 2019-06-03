########
#
# combine_api_outputs.py
#
# Takes the paths to multiple API outputs (JSONs) and combine them into one.
# This is only meant to be used on results from different folders/shards of the same project.
# The 'info', 'detection_categories' and 'classification_categories' fields will be
# taken from the first result file provided.
#
# No remapping of the 'category' field is done and no checks for duplicated result entries.
#
########

import json


#%% Parameters

# a list of outputs to combine
output_paths = [
    'detections1.json',
    'detections2.json'
]
for o in output_paths:
    assert o.endswith('.json'), 'outputs provided need to be the .json outputs'

combined_output_path = 'combined.json'
assert combined_output_path.endswith('.json'), 'output_path needs to be a .json'


#%% Script

outputs = []

for o in output_paths:
    result = json.load(open(o))
    print('File {} has {} result entries'.format(o, len(result['images'])))
    print()
    outputs.append(result)

combined = {}
if 'info' in outputs[0]:
    combined['info'] = outputs[0]['info']

if 'detection_categories' in outputs[0]:
    combined['detection_categories'] = outputs[0]['detection_categories']

if 'classification_categories' in outputs[0]:
    combined['classification_categories'] = outputs[0]['classification_categories']

combined_images = []
for output in outputs:
    combined_images.extend(output['images'])

file_names = []
for i in combined_images:
    file_names.append(i['file'])

print('A total of {} result entries are present, where {} are unique.'.format(
    len(file_names), len(set(file_names))))

combined['images'] = combined_images

with open(combined_output_path, 'w') as f:
    json.dump(combined, f, indent=1)

print('Combined output saved at {}'.format(combined_output_path))