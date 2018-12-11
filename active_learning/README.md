# Contents

This directory constitutes an ongoing project to employ active learning for semi-automated camera trap image labeling.

## DL
Core learning code shared by the offline (i.e., simulated-oracle) and online (i.e., human-oracle) versions of the project.

## UIComponents
The UI that presents images for labeling as part of the active learning process and iteratively trains a classifier.


# Training an embedding on a new data set

## Run the detector
	
Use archive/parallel_run_tf_detector.py . This is basically the code that you sent me, I just changed it a little bit to:

a. Be more GPU efficient, It can use one GPU for two parallel processes speeding up the task two times
b. Be more robust to corrupted images
c. Save the detection results for reuses
	
To use the scripts:

a. Specify your model on line 126.
b. Specify a folder containing all your images in line 128
c. Specify number of parallel processes in line 128, each two processed use one GPU. So if you set to 6, the code will run 6 different sub process and they will consume 3 GPUs
	
It will automatically read all the files inside the subfolders. Divide them between sub processes and run the detector on them. For each subprocess the code saves a csv file. The CSV file contains 6 columns:

`full path of the image, confidence,bbox_X1,bbox_Y1,bbox_X2,bbox_Y2`
	
Currently I just save the detections having more than 0.9 confidence, you may want to change it on line 121.
	

## Prepare your (cropped) images in a folder

Inside the main folder, each species should have its own subfolder.


## Train the embedding

Run main.py (filebased_main in the master branch) with appropriate command-line arguments. Here is the list of options with a very short description:
	
	--train_data,  path to train dataset
	--val_data, path to validation dataset, default=None, the code does not do validation if this is not provided
	--arch, Architecture of the model, default='resnet18'
	--workers, number of data loading workers (default: 4), if you have enough RAM and CPUs, you may increase this for faster training
	--epochs, default=20, number of total epochs for training
	--batch_size, default=256, mini-batch size (default: 256)
	--lr or  --learning_rate, default=0.001, initial learning rate, use 0.001 for training from scratch and use 0.0001 for finetuning
	--weight_decay or --wd, default=5e-4, type=float, L2 weight decay (default: 5e-4)'
	--print_freq or -p, default=10, type=int, print progress every n batches (default: 10)
	--resume, path to the base checkpoint, this option loads a pretrained model for finetuning
	--checkpoint_prefix, The scripts will add this as a prefix to saved checkpoints, this is handy when you want to run different experiments in the same folder
	--pretrained, use a pre-trained model on imagenet as a starting weights. I recommend to always use this
	--seed, random seed for being able to reproduce results
	--loss_type, default='triplet', either triplet,siamese, or center 
	--margin, default=1.0, margin for siamese or triplet loss, leave the default value
	-f or --feat_dim, default=256, number of features of the embedding (the embedding size)
	--raw_size, The width, height and number of channels of images for loading from disk, leave the default value
	--processed_size, The width and height of images after preprocessing, leave the default value
	--balanced_P,  The number of classes in each balanced batch, this will be set to number of classes by default
	--balanced_K, The number of examples from each class in each balanced batch, default is 10 but you may change it based on the number of classes you have, anything from 4 to 64, if your GPU memory allows
	
Here is an example of training resnet50 with triplet loss from scratch on the emammal dataset (many default values look good):  
	
`python filebased_main.py --train_data ~/all_crops/emammal_crops --arch resnet50  --epochs 55 --balanced_P 32 --balanced_K 8 --pretrained`
	
This will save 55 .tar files into the current folder. One after each epoch. The name of the last snapshot should look like this : triplet_model_0054.tar
	
To finetune an embedding model on a new dataset (SS for example) use a command like this:

`python filebased_main.py --train_data ~/all_crops/SS_crops_train_old --arch resnet50  --epochs 110 --balanced_P 8  --balanced_K 16 --resume triplet_model_0054.tar --lr 0.0001`
	
Note that epochs will start at 55, and here 110 means 55 more epochs for fine tuning.

	
# Installing DB, creating tables, adding images, etc.

The database can  be initialized using this code:
	
```
from peewee import *
from UIComponents.DBObjects import *
	
db = SqliteDatabase('SS.db')
proxy.initialize(db)
db.create_tables([Info, Image, Detection, Category])
```
	
This will create empty tables and set up primary keys, foreign keys, and indices. You need to only run this code once.
	
To import images, I make a csv file of images and then import the csv file to the image table.
	
The archive/init.py contains two parts.

To import detections, The first part (it has been commented out in the file) assumes you have already imported the results of your detection model into the database (Table S6 here). Then this code will assign a UUID to each crop and save them in the detection table:

```	
import sqlite3
import uuid
	
conn = sqlite3.connect('SS.db')
	
c = conn.cursor()
	
c.execute("select * from S6")
	
rows = c.fetchall()
	
for row in rows:
	#print(row)
	c.execute("insert into model_detection(id,image_id,category_id, bbox_confidence,bbox_x1,bbox_y1,bbox_x2,bbox_y2) values(?,?,?,?,?,?,?,?)",(str(uuid.uuid1()),row[0][27:],-1,
	float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5])))
	
	conn.commit()
```
	
The second part, actually crops the bounding boxes and save the crops in a specified directory.  This code uses parallel CPUs.
	

# Pointing UI code to a trained embedding

Change line 121 of UI.py 
