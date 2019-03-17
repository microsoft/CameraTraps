import csv
import json
import uuid
import datetime

output_file = 'NACTI_FULL_INFO.json'
csv_file_name = 'NACTI_FULL_INFO.csv'
annot_map= {}
data_dicts = {}
with open(csv_file_name,'r') as f:
    reader = csv.reader(f, dialect = 'excel')
    for i,row in enumerate(reader):
      if i==0:
        data_fields= row
      else:
        if row[0] not in data_dicts:
          data_dicts[row[0]]={data_fields[i]:row[i] for i in (1,2,3,4)}
          data_dicts[row[0]].update({data_fields[i]:int(row[i]) for i in (0,5,6)})
          annot_map[row[1]]=row[7]

json_data = {}
json_data['images'] = list(data_dicts.values())
#del data_dicts
info = {}
info['year'] = 2018
info['version'] = 1.0
info['description'] = 'North American Camera Trap Images'
info['contributor'] = 'USDA NWRC'
info['date_created'] = str(datetime.date.today())
json_data['info'] = info
#del info
cat_dict={}
with open('label_map_correct.csv') as f:
    reader = csv.reader(f, dialect = 'excel')
    for i,row in enumerate(reader):
      if i==0:
        category_fields= row
      else:
        if row[0] not in cat_dict:
          cat_dict[row[0]]={category_fields[i]:row[i] for i in (0,1,3,4,5,6,7)}
json_data['categories']=list(cat_dict.values())
print(cat_dict)
#del cat_dict
annotation_dict={}
for key in annot_map.keys():
  annotation_dict[key]={"id":str(uuid.uuid1()), "image_id":key, "category_id": int(annot_map[key])}

#print(len(annotation_dict))
#print(list(annotation_dict.values())[0:10])
#print(list(data_dicts.values())[0:10])
json_data['annotations']= list(annotation_dict.values())
json.dump(json_data,open(output_file,'w'))
