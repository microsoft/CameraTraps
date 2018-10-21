import csv
import pickle

output_file = '/datadrive/snapshotserengeti/databases/already_annotated.p'
csv_file_name = '/datadrive/snapshotserengeti/databases/gold_standard_data.csv'
data = []
with open(csv_file_name,'r') as f:
    reader = csv.reader(f, dialect = 'excel')
    for row in reader:
        data.append(row)

data_fields = data[0]
print(data_fields)

data_dicts = {}
for event in data[1:]:
    data_dicts[event[0]] = {data_fields[i]:event[i] for i in range(len(data_fields))}

already_annotated_list = list(data_dicts.keys())
print(len(already_annotated_list))
print(already_annotated_list[0])

pickle.dump(already_annotated_list,open(output_file,'wb'))
