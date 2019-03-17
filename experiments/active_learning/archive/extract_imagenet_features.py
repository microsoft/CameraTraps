from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
import random

model = VGG16(weights='imagenet', include_top=False)
model.summary()
co =0
images_list = []
for subdir, dirs, files in os.walk('/data/dataD/snapshot/S1'):
  for file in files:
    #print(os.path.join(subdir, file))
    img_path = subdir + os.sep + file
    if img_path.endswith(".JPG"):
      images_list.append(img_path)
      co+=1
vgg16_feature_list=[]
samples_list= random.sample(images_list, 5000)
for x in samples_list:
  img = image.load_img(x, target_size=(224, 224))
  img_data = image.img_to_array(img)
  img_data = np.expand_dims(img_data, axis=0)
  img_data = preprocess_input(img_data)
  vgg16_feature = np.array(model.predict(img_data))
  vgg16_feature_list.append(vgg16_feature.flatten())

np.save('features.npy',vgg16_feature_list)
np.save('images_paths.npy', samples_list)
