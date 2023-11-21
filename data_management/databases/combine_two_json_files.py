#
# combine_two_json_files.py
#
# Merges two coco-camera-traps .json files. In particular, categories are combined and re-numbered.
#

import datetime, json

file_1 = '/ai4edevfs/databases/emammal/emammal_kays_20190409.json'
with open(file_1,'r') as f:
    data_1 = json.load(f)

file_2 = '/ai4edevfs/databases/emammal/emammal_mcshea_20190409.json'
with open(file_2,'r') as f:
    data_2 = json.load(f)

# Combined Info
info_1 = data_1['info']
info_2 = data_2['info']
info_new = dict.fromkeys(info_1.keys())
info_new['contributor'] = info_1['contributor']+', '+info_2['contributor']
desc_2 = ' '.join([info_2['description'].split()[0].lower()] + info_2['description'].split()[1:])
info_new['description'] = info_1['description'][:-1]+', combined with '+desc_2
info_new['year'] = max(info_1['year'], info_2['year'])
info_new['date_created'] = datetime.date.today().strftime('%Y-%m-%d')
info_new['version'] = info_1['version']

# Combined Images
images_1 = data_1['images']
images_2 = data_2['images']
images_new = images_1 + images_2

# Combined Categories
categories_1 = data_1['categories']
categories_2 = data_2['categories']
### categories to merge
categories_to_merge = [['elk aka red deer', 'elk_red deer'], ['other bird species', 'unknown bird'], 
     ['blue eared pheasant', 'blue eared-pheasant'], ['reeves\' muntjac', 'reeve\'s muntjac'],
     ['tragopan pheasant', 'temminck\'s tragopan', 'tragopan temminckii']]
category_remapping = dict()
for c2m in categories_to_merge:
    cat_group_name = c2m[0]
    for cat in c2m[1:]:
        category_remapping[cat] = cat_group_name
new_category_names = ['empty'] + sorted(list(set([c['name'] for c in categories_1] + [c['name'] for c in categories_2]) - {'empty'} - set(category_remapping.keys())))
categories_new = []
for i, cname in enumerate(new_category_names):
    categories_new.append({'id':i, 'name':cname})
new_cat_id_lookup = {v['name']:v['id'] for v in categories_new}
cat_1_name_lookup = {v['id']:v['name'] for v in categories_1}
cat_2_name_lookup = {v['id']:v['name'] for v in categories_2}


# Combined Annotations
annotations_1 = data_1['annotations']
for ann in annotations_1:
    a1_cid = ann['category_id']
    a1_cname = cat_1_name_lookup[a1_cid]
    if a1_cname in category_remapping.keys():
        ann['category_id'] = new_cat_id_lookup[category_remapping[a1_cname]]
    else:
        ann['category_id'] = new_cat_id_lookup[a1_cname]

annotations_2 = data_2['annotations']
for ann in annotations_2:
    a2_cid = ann['category_id']
    a2_cname = cat_2_name_lookup[a2_cid]
    if a2_cname in category_remapping:
        ann['category_id'] = new_cat_id_lookup[category_remapping[a2_cname]]
    else:
        ann['category_id'] = new_cat_id_lookup[a2_cname]

annotations_new = annotations_1 + annotations_2

new_data = {}
new_data['info'] = info_new
new_data['images'] = images_new
new_data['categories'] = categories_new
new_data['annotations'] = annotations_new
output_file = '/datadrive/emammal/emammal_mcshea_kays_20190409.json'
json.dump(new_data, open(output_file,'w'))

