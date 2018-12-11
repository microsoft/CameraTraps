"""from peewee import *
from DB_types import *

db = SqliteDatabase('SS.db')
proxy.initialize(db)
db.create_tables([ModelDetection,OracleDetection])"""


"""import sqlite3
import uuid

conn = sqlite3.connect('SS.db')

c = conn.cursor()

c.execute("select * from S6")

rows= c.fetchall()

for row in rows:
  #print(row)
  c.execute("insert into model_detection(id,image_id,category_id, bbox_confidence,bbox_x1,bbox_y1,bbox_x2,bbox_y2) values(?,?,?,?,?,?,?,?)",(str(uuid.uuid1()),row[0][27:],-1,
float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5])))

conn.commit()"""


import os
from shutil import copyfile
import sqlite3
from PIL import Image
import sys
from multiprocessing import Process

def do_chunk(pid,todo):
  print(pid,"Started")
  co=0
  for line in todo:
    try:
      dest= os.path.join(images_folder,line[0]+".JPG")
      src= os.path.join(root,line[1])
      image = Image.open(src)
      dpi = 100
      s = image.size; imageHeight = s[1]; imageWidth = s[0]
      figsize = imageWidth / float(dpi), imageHeight / float(dpi)
      topRel = float(line[4])
      leftRel = float(line[5])
      bottomRel = float(line[6])
      rightRel = float(line[7])
      #unq_id= "crops_"+str(uuid.uuid1())
      #print(line,imageWidth,imageHeight)
      #print("%s,%s,%s,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f"%(line[0], line[1],line[2],float(line[3]),topRel,leftRel,bottomRel,rightRel))
      sys.stdout.flush()
      x1 = int(leftRel * imageWidth)
      y1 = int(topRel * imageHeight)
      x2 = int(rightRel* imageWidth)
      y2 = int(bottomRel * imageHeight)
      crop= image.crop((x1,y1,x2,y2)).resize((256,256),Image.BILINEAR)

      #if not os.path.exists(dest):
      #  os.mkdir(dest)

      crop.save(dest)
      image.close()
      co+=1
      if co%1000==0:
        print(pid,co)
    except:
      #raise
      pass
  #out.close()


def divide(t,n,i):
    length=t/(n+0.0)
    #print length,(i-1)*length,i*length
    return int(round((i-1)*length)),int(round(i*length))



conn = sqlite3.connect('SS.db')
c = conn.cursor()
root= '/datadrive0/dataD/snapshot'
images_folder='SS_crops'
c.execute('SELECT * FROM model_detection')
rows = c.fetchall()
os.mkdir(images_folder)
total_records=len(rows)
total_processors=12
print(total_records)
for i in range(1,total_processors+1):
  st,ln=divide(total_records,total_processors,i)
  p1 = Process(target=do_chunk, args=(i,rows[st:ln]))
  p1.start()
 
