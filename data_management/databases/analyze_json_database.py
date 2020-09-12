#
# analyze_json_database.py
#
# Plots location/class/etc. distributions for classes in a coco-camera-traps .json file.
#
# Currently includes some one-off code for specific species.
#

#%% Constants and imports

import colorsys
import json
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


#%% Path configuration

db_name = 'imerit_annotation_images_ss_2'
db_file = '/datadrive/snapshotserengeti/databases/'+db_name+'.json'
plot_directory = '/datadrive/snapshotserengeti/databases/Plots/'+db_name

if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)


#%% Load source data
    
with open(db_file,'r') as f:
    data = json.load(f)

images = data['images']
annotations = data['annotations']
categories = data['categories']

print('This database has:')
print(str(len(images)) + ' images')
print(str(len(annotations)) + ' annotations')


#%% Build image/category dictionaries

im_id_to_im = {im['id']: im for im in images}
im_id_to_cat = {ann['image_id']:ann['category_id'] for ann in annotations}
cat_id_to_name = {cat['id']:cat['name'] for cat in categories}
cat_to_id = {cat['name']:cat['id'] for cat in categories}
loc_to_ims = {}
for im in images:
    if im['location'] not in loc_to_ims:
        loc_to_ims[im['location']] = []
    loc_to_ims[im['location']].append(im['id'])

print(str(len(loc_to_ims)) + ' locations')

cat_to_ims = {}
for ann in annotations:
    if ann['category_id'] not in cat_to_ims:
        cat_to_ims[ann['category_id']] = []
    cat_to_ims[ann['category_id']].append(ann['image_id'])
print(str(len(cat_to_ims)) + ' categories')

season_to_ims = {}
for im in images:
    if im['season'] not in season_to_ims:
        season_to_ims[im['season']] = []
    season_to_ims[im['season']].append(im['id'])


#%% Make plot of category distribution
    
sortedCats = sorted(zip([len(cat_to_ims[cat]) for cat in cat_to_ims],[cat for cat in cat_to_ims]), key = lambda t: t[0], reverse = True)
plt.bar(range(len(sortedCats)),[cat[0] for cat in sortedCats], log = True)
plt.ylabel('Number of images')
plt.title('Number of images per category')
plt.ylabel('Categories')
plt.xticks(range(len(sortedCats)), [cat_id_to_name[cat[1]] for cat in sortedCats], rotation = 90)
plt.tight_layout()
plt.savefig(plot_directory + '/ims_per_cat.jpg')  
plt.clf()


#%% make plots of location distribution

sortedLocs = sorted(zip([len(loc_to_ims[loc]) for loc in loc_to_ims],[loc for loc in loc_to_ims]), key = lambda t: t[0], reverse = True)
plt.bar(range(len(loc_to_ims)),[loc[0] for loc in sortedLocs])
plt.ylabel('Number of images')
plt.title('Number of images per location')
plt.ylabel('Locations')
plt.xticks(range(len(loc_to_ims)), [loc[1] for loc in sortedLocs], rotation = 90, fontsize = 2)
plt.tight_layout()
plt.savefig(plot_directory + '/ims_per_loc.jpg', dpi = 500)   
plt.clf()

cat_count_per_location = {loc[1]:{cat['id']:0 for cat in categories} for loc in sortedLocs}
for im in images:
    cat_count_per_location[im['location']][im_id_to_cat[im['id']]] += 1
colors = []
maxd = len(sortedCats)
for depth in range(maxd):
    (r,g,b) = colorsys.hsv_to_rgb(float(depth)/maxd, 1.0, 1.0)
    colors.append((r,g,b))
ind = np.arange(len(loc_to_ims))
bottom = np.zeros(len(loc_to_ims))
catNames = []
count = 0
sortedCats.reverse()
for ims,cat_id in sortedCats:
    cat_name = cat_id_to_name[cat_id]
    catNames.append(cat_name)
    catLocCounts = [cat_count_per_location[loc[1]][cat_id] for loc in sortedLocs]
    plt.bar(ind,catLocCounts,log=True,bottom = bottom, color = colors[count])
    count += 1
    bottom = [bottom[i] + catLocCounts[i] for i in range(len(bottom))]

plt.xticks(ind,[loc[1] for loc in sortedLocs], rotation=90,fontsize=2)
plt.ylabel('Number of images')
plt.xlabel('Location')
plt.legend(catNames, loc='center left', bbox_to_anchor=(1.005,0.5), edgecolor=None, fontsize=4)
#plt.title('Number of images per location, by category')
plt.tight_layout()
#plt.tight_layout(rect=[0,0,1,0.9])
plt.savefig(plot_directory + '/cats_per_loc.jpg', dpi=500)
plt.clf()
# #make plot of images per season
sortedLocs = list(zip([len(season_to_ims[loc]) for loc in season_to_ims],[loc for loc in season_to_ims]))
plt.bar(range(len(sortedLocs)),[loc[0] for loc in sortedLocs])
plt.ylabel('Number of images')
plt.title('Number of images per season')
plt.ylabel('Seasonss')
plt.xticks(range(len(sortedLocs)), [loc[1] for loc in sortedLocs], rotation = 90)
plt.tight_layout()
plt.savefig(plot_directory + '/ims_per_season.jpg')   
plt.clf()


#%% Make plot of lions per location

lion_ids = [cat_to_id['lionMale'], cat_to_id['lionFemale']]
loc_to_lion_ims = {loc:[i for i in loc_to_ims[loc] if im_id_to_cat[i] in lion_ids] for loc in loc_to_ims}
loc_to_lion_ims = {loc:loc_to_lion_ims[loc] for loc in loc_to_lion_ims if len(loc_to_lion_ims[loc]) > 0}
sortedLocs = sorted(zip([len(loc_to_lion_ims[loc]) for loc in loc_to_lion_ims],[loc for loc in loc_to_lion_ims]), key = lambda t: t[0], reverse = True)
plt.bar(range(len(loc_to_lion_ims)),[loc[0] for loc in sortedLocs])
plt.ylabel('Number of images')
plt.title('Number of lion images per location')
plt.ylabel('Locations')
plt.xticks(range(len(loc_to_lion_ims)), [loc[1] for loc in sortedLocs], rotation = 90, fontsize = 6)
plt.tight_layout()
plt.savefig(plot_directory + '/lion_ims_per_loc.jpg')   
plt.clf()


#%% Make plot of elephants per location

elephant_ids = [cat_to_id['elephant']]
loc_to_elephant_ims = {loc:[i for i in loc_to_ims[loc] if im_id_to_cat[i] in elephant_ids] for loc in loc_to_ims}
loc_to_elephant_ims = {loc:loc_to_elephant_ims[loc] for loc in loc_to_elephant_ims if len(loc_to_elephant_ims[loc]) > 0}
sortedLocs = sorted(zip([len(loc_to_elephant_ims[loc]) for loc in loc_to_elephant_ims],[loc for loc in loc_to_elephant_ims]), key = lambda t: t[0], reverse = True)
plt.bar(range(len(loc_to_elephant_ims)),[loc[0] for loc in sortedLocs])
plt.ylabel('Number of images')
plt.title('Number of elephant images per location')
plt.ylabel('Locations')
plt.xticks(range(len(loc_to_elephant_ims)), [loc[1] for loc in sortedLocs], rotation = 90, fontsize = 6)
plt.tight_layout()
plt.savefig(plot_directory + '/elephant_ims_per_loc.jpg')   
plt.clf()

lions_and_elephants = list(set(loc_to_elephant_ims.keys()).intersection(set(loc_to_lion_ims.keys())))
total = []
for loc in lions_and_elephants:
    total.append(len(loc_to_elephant_ims[loc]+ loc_to_lion_ims[loc]))

sorted_by_total = sorted(zip(lions_and_elephants,total),reverse=True, key = lambda t: t[1])

# for loc in sorted_by_total[:25]:
#     print('Location:' + loc[0] +', Lions: ' + str(len(loc_to_lion_ims[loc[0]])) + ', Elephants: ' + str(len(loc_to_elephant_ims[loc[0]])) + ', Total ims: ' + str(len(loc_to_ims[loc[0]])))
