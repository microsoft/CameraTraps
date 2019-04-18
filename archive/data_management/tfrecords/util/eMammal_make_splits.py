import json
import os
from collections import defaultdict
from random import shuffle

# eMammal_make_splits.py
#
# Based on a tfrecords_format json file of the database, creates 3 splits according to
# the specified fractions based on location (images from the same location should be in
# one split) or based on images.
#
# If a previous split is provided (append_to_previous_split is True), the entries in
# each split will be preserved, and new entries will be appended, so that new models
# can warm start with a model trained on the original splits.

# configurations and paths
append_to_previous_split = False  # True if adding to splits from a previous splits file
split_by = 'location'  # 'location' or 'image'

# approximate fraction for the new entries
train_frac = 0.8
val_frac = 0.2
test_frac = 0.0

tfrecords_format_json_path = '/home/yasiyu/yasiyu_temp/eMammal_db/eMammal_20180929_tfrecord_format.json'

previous_split_path = '/home/yasiyu/yasiyu_temp/tf_records_eMammal/eMammal_loc_splits_20180918.json'  # path to the splits json or None
output_path = '/home/yasiyu/yasiyu_temp/tf_records_eMammal/eMammal_loc_splits_20180929.json'


# read in the previous splits of image ID or location ID if available
train, val, test = [], [], []
if append_to_previous_split and os.path.exists(previous_split_path):
    previous_split = json.load(open(previous_split_path, 'r'))
    train = previous_split.get('train', [])
    val = previous_split.get('val', [])
    test = previous_split.get('test', [])

all_pre = []
all_pre.extend(train)
all_pre.extend(val)
all_pre.extend(test)
all_pre = set(all_pre)

data = json.load(open(tfrecords_format_json_path, 'r'))
print('Total number of image entries in tfrecords_format_json: ', len(data))

if split_by == 'location':
    print('Previously had data for {} locations. Train {}, val {}, test {}.'.format(
        len(all_pre), len(train), len(val), len(test)))

    # find new locations and assign them to a split, without reassigning any previous locations
    new_locs = defaultdict(int)  # new location ID : how many new images at that location
    for entry in data:
        loc = entry['location']
        if loc not in all_pre:
            new_locs[loc] += 1
    print('Number of new locations to assign a split: ', len(new_locs))

    new_locs_list = list(new_locs.items())
    print('Before shuffle, 1st new loc: ', new_locs_list[0])
    shuffle(new_locs_list)
    print('After shuffle, 1st new loc: ', new_locs_list[1])

    num_new_imgs = sum(new_locs.values())
    num_train = int(train_frac * num_new_imgs)
    num_val = int(val_frac * num_new_imgs)

    train_imgs_added, val_imgs_added = 0, 0
    for loc, img_count in new_locs_list:
        if train_imgs_added < num_train:
            train.append(loc)
            train_imgs_added += img_count
        elif val_imgs_added < num_val:
            val.append(loc)
            val_imgs_added += img_count
        else:
            test.append(loc)
    print('New train locations has {} images, val had {} images, test had {} images'.format(
        train_imgs_added, val_imgs_added, num_new_imgs - train_imgs_added - val_imgs_added))

elif split_by == 'image':
    assert len(train) + len(val) + len(test) == len(all_pre)  # image IDs should be unique
    print('Appending to {} train imgs, {} val imgs, {} test images - total of {}.'.format(
        len(train), len(val), len(test), len(all_pre)
    ))

    # find out which images are new, shuffle and split them
    new_imgs = []  # image IDs

    for entry in data:
        img_id = entry['id']
        if img_id not in all_pre:
            new_imgs.append(img_id)
    print('Number of new image entries to assign a split: ', len(new_imgs))

    print('Before shuffle, 1st img: ', new_imgs[0])
    shuffle(new_imgs)  # shuffle is in place and returns None
    print('After shuffle, 1st img: ', new_imgs[0])

    num_train = int(train_frac * len(new_imgs))
    num_val = int(val_frac * len(new_imgs))

    new_train = new_imgs[:num_train]
    new_val = new_imgs[num_train:num_train + num_val]
    new_test = new_imgs[num_train + num_val:]

    train.extend(new_train)
    val.extend(new_val)
    test.extend(new_test)


shuffle(train)  # shuffle the old and new
shuffle(val)
shuffle(test)

print('Now have {} train entries, {} val entries, {} test entries'.format(
    len(train), len(val), len(test)
))

# do NOT sort the IDs to keep the shuffled order
updated_splits = {
    'train': train,
    'val': val,
    'test': test
}

with open(output_path, 'w') as f:
    json.dump(updated_splits, f, indent=4)

print('Updated splits saved at ', output_path)
