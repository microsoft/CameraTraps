# 
# create_new_annotation_json.py
#
# Creates a subset of a larger .json database, in this case specifically to pick some images
# from Snapshot Serengeti.
#

import json
import pickle
import numpy as np
import random
#from utils import get_db_dicts

prev_ann_file = '/datadrive/snapshotserengeti/databases/already_annotated_1.p'
this_ann_file = '/datadrive/snapshotserengeti/databases/already_annotated_2.p'
db_name = 'SnapshotSerengeti'
db_file = '/datadrive/snapshotserengeti/databases/'+db_name+'.json'
num_seqs_per_cat_per_loc = 1
output_file = '/datadrive/snapshotserengeti/databases/imerit_annotation_images_ss_2.json'
seasons_to_keep = ['S2', 'S3', 'S4']
already_annotated = pickle.load(open(prev_ann_file,'rb'))

print('Already annotated: ', len(already_annotated))

with open(db_file,'r') as f:
    data = json.load(f)

images = data['images']
annotations = data['annotations']
categories = data['categories']

print('This database has:')
print(str(len(images)) + ' images')
print(str(len(annotations)) + ' annotations')

im_id_to_im = {im['id']: im for im in images}
im_id_to_cat = {ann['image_id']:ann['category_id'] for ann in annotations}
cat_id_to_name = {cat['id']:cat['name'] for cat in categories}
cat_to_id = {cat['name']:cat['id'] for cat in categories}

im_id_to_seq = {im['id']: im['seq_id'] for im in images}
seq_to_ims = {}
seq_to_season = {}
for im in images:
    if im['seq_id'] not in seq_to_ims:
        seq_to_ims[im['seq_id']] = []
    seq_to_ims[im['seq_id']].append(im['id'])
    seq_to_season[im['seq_id']] = im['season']

loc_to_seqs = {}
for im in images:
    if im['location'] not in loc_to_seqs:
        loc_to_seqs[im['location']] = []
    loc_to_seqs[im['location']].append(im_id_to_seq[im['id']])
loc_to_seqs = {loc:list(set(loc_to_seqs[loc])) for loc in loc_to_seqs}

print(str(len(loc_to_seqs)) + ' locations')

cats_per_location = {loc:{cat['id']:[] for cat in categories} for loc in loc_to_seqs}
for im in images:
    cats_per_location[im['location']][im_id_to_cat[im['id']]].append(im['seq_id'])
cats_per_location = {loc:{cat:list(set(cats_per_location[loc][cat])) for cat in cats_per_location[loc]} for loc in loc_to_seqs}

cat_to_seqs = {}
for ann in annotations:
    if ann['category_id'] not in cat_to_seqs:
        cat_to_seqs[ann['category_id']] = []
    cat_to_seqs[ann['category_id']].append(im_id_to_seq[ann['image_id']])
cat_to_seqs = {cat:list(set(cat_to_seqs[cat])) for cat in cat_to_seqs}

print(str(len(cat_to_seqs)) + ' categories')

# for seq in already_annotated:
#     seq_to_ims.pop(seq)

seqs_to_annotate = []
for loc in loc_to_seqs:
    for cat in cats_per_location[loc]:
        if cat == cat_to_id['empty']:
            continue
        if len(cats_per_location[loc][cat]) > num_seqs_per_cat_per_loc:
            seqs = cats_per_location[loc][cat]
            seqs = [seq for seq in seqs if seq_to_season[seq] in seasons_to_keep]
            seqs_to_keep = []
            seqs_to_keep = random.sample(seqs,min(len(seqs),num_seqs_per_cat_per_loc))
            #remove already annotated images
            seqs_to_keep = [seq for seq in seqs_to_keep if seq not in already_annotated]
            seqs_to_annotate.extend(seqs_to_keep) 
        else:
            seqs = cats_per_location[loc][cat]
            seqs = [seq for seq in seqs if seq_to_season[seq] in seasons_to_keep]
            seqs_to_annotate.extend(seqs)
print('Seqs to annotate:',len(seqs_to_annotate))
ims_to_annotate = []
for seq in seqs_to_annotate:
    ims_to_annotate.extend(seq_to_ims[seq])

print('Ims to annotate: ',len(ims_to_annotate))
already_annotated.extend(seqs_to_annotate)

#add lion images
num_lions = 50
lion_seqs_to_annotate = []
for loc in loc_to_seqs:
    lion_seqs = cats_per_location[loc][cat_to_id['lionMale']]+cats_per_location[loc][cat_to_id['lionFemale']]
    lion_seqs = [seq for seq in lion_seqs if seq not in already_annotated]
    lion_seqs = [seq for seq in lion_seqs if seq_to_season[seq] in seasons_to_keep]
    lion_seqs_to_annotate.extend(random.sample(lion_seqs, min(len(lion_seqs),num_lions)))

# lion_seqs = cat_to_seqs[cat_to_id['lionMale']] + cat_to_seqs[cat_to_id['lionFemale']]
# #print(len(lion_seqs))
# lion_seqs = [seq for seq in lion_seqs if seq not in already_annotated]
# #print(len(lion_seqs))
# lion_seqs_to_annotate.extend(random.sample(lion_seqs, min(len(lion_seqs),num_lions)))
#print(len(lion_seqs_to_annotate))

lion_ims_to_annotate = []
for seq in lion_seqs_to_annotate:
    lion_ims_to_annotate.extend(seq_to_ims[seq])

print('Lion ims: ',len(lion_ims_to_annotate))

already_annotated.extend(lion_seqs_to_annotate)

num_elephants = 10
elephant_seqs_to_annotate = []
for loc in loc_to_seqs:
    elephant_seqs = cats_per_location[loc][cat_to_id['elephant']]
    elephant_seqs = [seq for seq in elephant_seqs if seq not in already_annotated]
    elephant_seqs = [seq for seq in elephant_seqs if seq_to_season[seq] in seasons_to_keep]
    elephant_seqs_to_annotate.extend(random.sample(elephant_seqs, min(len(elephant_seqs),num_elephants)))

# num_elephants = 1000
# elephant_seqs = cat_to_seqs[cat_to_id['elephant']]
# #print(len(lion_seqs))
# elephant_seqs = [seq for seq in elephant_seqs if seq not in already_annotated]
# #print(len(lion_seqs))
# elephant_seqs_to_annotate = random.sample(elephant_seqs, num_elephants)
elephant_ims_to_annotate = []
for seq in elephant_seqs_to_annotate:
    elephant_ims_to_annotate.extend(seq_to_ims[seq])

already_annotated.extend(elephant_ims_to_annotate)

print('Elephant ims: ',len(elephant_ims_to_annotate))

#num_empty = 10
#empty_seqs_to_annotate = []
#for loc in loc_to_seqs:
#    empty_seqs = cats_per_location[loc][cat_to_id['empty']]
#    empty_seqs = [seq for seq in empty_seqs if seq not in already_annotated]
#    empty_seqs = [seq for seq in empty_seqs if seq_to_season[seq] in seasons_to_keep]
#    empty_seqs_to_annotate.extend(random.sample(empty_seqs, min(len(empty_seqs),num_empty)))

ims_to_annotate.extend(lion_ims_to_annotate)
ims_to_annotate.extend(elephant_ims_to_annotate)
#ims_to_annotate.extend(empty_ims_to_annotate)

print('Total num ims: ',len(ims_to_annotate))

keep = {im['id']:False for im in images}
for im_id in ims_to_annotate:
    keep[im_id] = True

new_images = [im for im in images if keep[im['id']]]
new_anns = [ann for ann in annotations if keep[ann['image_id']]]
data['images'] = new_images
data['annotations'] = new_anns
data['info']['description'] = '2nd iMerit annotation database for Snapshot Serengeti'

json.dump(data, open(output_file,'w'))

pickle.dump(already_annotated, open(this_ann_file, 'wb'))
