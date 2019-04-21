#
# create_splits.py
#
# Based on a tfrecords_format json file of the database, creates 3 splits according to
# the specified fractions based on location (images from the same location should be in
# one split) or based on images.
#
# If a previous split is provided (append_to_previous_split is True), the entries in
# each split will be preserved, and new entries will be appended, so that new models
# can warm start with a model trained on the original splits.
#

#%% Imports
from collections import defaultdict
from random import shuffle


#%% Main function

def create_splits(tfrecord_format_json, split_by, split_frac, previous_split=None):

    print('Creating train/val/test splits...')
    train = previous_split.get('train', [])
    val = previous_split.get('val', [])
    test = previous_split.get('test', [])
    prev_split_by = previous_split.get('split_by', '')

    if prev_split_by == '':
        print("In previous split, split_by is not specified. Using '{}'".format(split_by))
    else:
        assert prev_split_by == split_by, 'Specified split_by {}, but it is {} in the previous split provided'.format(
            split_by, prev_split_by)

    all_previous = train + val + test

    print('Previously had data for {} {}s. Train {}, val {}, test {}.'.format(
        len(all_previous), split_by, len(train), len(val), len(test)))

    # code below references 'locatioin' as the attribute to split on, but it works for any split_by attribute
    # present in the image entryes

    # find new locations and assign them to a split, without reassigning any previous locations
    new_locs = defaultdict(int)  # new location ID : how many new images at that location
    for entry in tfrecord_format_json:
        loc = entry[split_by]
        if loc not in all_previous:
            new_locs[loc] += 1
    print('Number of new locations to assign a split: ', len(new_locs))

    new_locs_list = list(new_locs.items())
    print('Before shuffle, 1st new loc: ', new_locs_list[0])
    shuffle(new_locs_list)
    print('After shuffle, 1st new loc: ', new_locs_list[1])

    num_new_imgs = sum(new_locs.values())
    num_train = int(split_frac['train'] * num_new_imgs)
    num_val = int(split_frac['val'] * num_new_imgs)

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

    print('Number of new images added is {} for train, {} for val, and {} for test'.format(
        train_imgs_added, val_imgs_added, num_new_imgs - train_imgs_added - val_imgs_added))

    print('The splits have {} train entries, {} val entries, {} test entries'.format(
        len(train), len(val), len(test)
    ))

    # do NOT sort the IDs to keep the shuffled order
    updated_splits = {
        'train': sorted(train),
        'val': sorted(val),
        'test': sorted(test),
        'split_by': split_by
    }
    return updated_splits