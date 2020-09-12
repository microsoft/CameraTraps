import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

file_1 = '/ai4efs/models/object_detection/faster_rcnn_inception_resnet_v2_atrous/train_on_ss/predictions/eccv_train_per_cat_prec_recall_data.npz'
data_1 = np.load(open(file_1,'r'))

file_2 = '/ai4efs/models/object_detection/faster_rcnn_inception_resnet_v2_atrous/train_on_ss_and_inat/predictions/eccv_train_per_cat_prec_recall_data.npz'
data_2 = np.load(open(file_2,'r'))

file_3 = '/ai4efs/models/object_detection/faster_rcnn_inception_resnet_v2_atrous/train_on_ss_no_deer/predictions/eccv_train_per_cat_prec_recall_data.npz'
data_3 = np.load(open(file_3,'r'))

file_4 = '/ai4efs/models/object_detection/faster_rcnn_inception_resnet_v2_atrous/train_on_ss_no_deer_and_inat/predictions/eccv_train_per_cat_prec_recall_data.npz'
data_4 = np.load(open(file_4,'r'))

ap = data_1['ap'].tolist()
cat_id_to_cat = data_1['cat_id_to_cat'].tolist()

cat_ids = [i for i in ap if not np.isnan(ap[i])]
print(cat_ids)

N = len(cat_ids)
ind = np.arange(N)
width = 0.15


fig = plt.figure()
ax = fig.add_subplot(111)
aps = [ap[i] for i in cat_ids]
print(aps)

print(len(ind),len(aps))
rects1 = ax.bar(ind, aps, width, color='royalblue')

ap = data_2['ap'].tolist()
rects2 = ax.bar(ind+width, [ap[i] for i in cat_ids], width, color='seagreen')

ap = data_3['ap'].tolist()
rects3 = ax.bar(ind+width*2, [ap[i] for i in cat_ids], width, color='red')

ap = data_4['ap'].tolist()
rects4 = ax.bar(ind+width*3, [ap[i] for i in cat_ids], width, color='orange')



ax.set_ylabel('mAP per class')
ax.set_title('mAP per class with and without iNat and deer-like animals')
ax.set_xticks(ind + 3*width / 2)
ax.set_xticklabels([cat_id_to_cat[i] for i in cat_ids])
plt.xticks(rotation=90)

ax.legend((rects1[0],rects2[0], rects3[0], rects4[0]),('w/deer, w/o iNat','w/ deer, w/ iNat','w/o deer, w/o iNat','w/o deer, w/iNat'), loc='lower center')

plt.tight_layout()

plt.savefig('/ai4efs/models/object_detection/faster_rcnn_inception_resnet_v2_atrous/train_on_ss_no_deer_and_inat/predictions/compare_per_seq_mAP_w_deer_and_no_deer.jpg')



















