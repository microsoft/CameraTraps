import os
from shutil import copyfile
root= 'crops_train'
images_folder='crops'
os.mkdir(root)
co =0
with open('train_crops.csv') as f:
  for line in f:
    filename,species= line.split(",")
    print(filename, species)
    dest= os.path.join(root,species.rstrip())
    if not os.path.exists(dest):
      os.mkdir(dest)
    copyfile(os.path.join(images_folder,filename),os.path.join(dest,filename))
    co+=1
    print(co)
    
