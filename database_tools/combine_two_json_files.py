import json

file_1 = '/datadrive/iwildcam/annotations/eccv_18_annotation_files/train_annotations.json'
with open(file_1,'r') as f:
    data_1 = json.load(f)

file_2 = '/datadrive/iwildcam/imerit/updated_imerit_iwildcam_annotations_2.json'
with open(file_2,'r') as f:
    data_2 = json.load(f)

output_file = '/datadrive/iwildcam/annotations/combined_annotations/eccv_train_and_imerit_2.json'
version = 'ECCV train file and imerit annotations set 2'

new_images = data_1['images']
new_images.extend(data_2['images'])
#print(data_1['images'][0])
#print(data_2['images'][0])
print(len([im['id'] for im in new_images]),len(list(set([im['id'] for im in new_images]))))
for im in new_images:
    im['file_name'] = im['id'] + '.jpg'
new_anns = data_1['annotations']
new_anns.extend(data_2['annotations'])

for ann in new_anns:
    ann['category_id'] = int(ann['category_id'])

print(len(new_anns))

#print(data_1['annotations'][0:5])
#print(data_2['annotations'][0:5])

for cat in data_1['categories']:
    cat['id'] = int(cat['id'])

for cat in data_2['categories']:
    cat['id'] = int(cat['id'])


new_cats = data_1['categories']
print(len(new_cats))
cat_names = [cat['name'] for cat in new_cats]
cat_ids = [cat['id'] for cat in new_cats]
new_cats.extend([cat for cat in data_2['categories'] if cat['id'] not in cat_ids])
print(len(new_cats))
#print(new_cats)

ann_cats = []
for ann in new_anns:
    if ann['category_id'] not in ann_cats:
        ann_cats.append(ann['category_id'])
#print(ann_cats)
new_cats = [cat for cat in new_cats if cat['id'] in ann_cats]
print(len(new_cats),len(ann_cats))
#print(new_cats)

new_data = {}
new_data['categories'] = new_cats
new_data['annotations'] = new_anns
new_data['images'] = new_images
new_data['info'] = data_1['info']
new_data['info']['version'] = version

json.dump(new_data, open(output_file,'w'))
#print(new_data['info'])
