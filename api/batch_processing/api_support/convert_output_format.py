########
#
# convert_output_format.py
#
#
#
########

import argparse
import csv
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    args = parser.parse_args()

    assert args.output_path.endswith('.csv'), 'Only supporting json to csv (new to old) conversion currently'

    print('Loading json results...')
    json_output = json.load(open(args.input_path))

    rows = []

    print('Iterating through results...')
    for i in json_output['images']:
        image_id = i['file']
        max_conf = i['max_detection_conf']
        detections = []
        for d in i['detections']:
            detection = d['bbox']
            detection.append(d['conf'])
            detection.append(int(d['category']))
            detections.append(detection)
        rows.append((image_id, max_conf, json.dumps(detections)))

    print('Writing to csv...')
    with open(args.output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['image_path', 'max_confidence', 'detections'])
        writer.writerows(rows)


if __name__ == '__main__':
    main()