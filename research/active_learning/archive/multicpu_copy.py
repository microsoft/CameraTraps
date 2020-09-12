import os
from shutil import copyfile
import sqlite3
from PIL import Image
import uuid
from multiprocessing import Process

def do_chunk(pid,todo):
  out= open("all_detections_"+str(pid)+".csv","w")
  co=0
  for line in todo:
    try:
      dest= os.path.join(root,line[2].replace(" ","_"))
      src= os.path.join('/datadrive0/emammal',line[1])
      image = Image.open(src)
      dpi = 100
      s = image.size; imageHeight = s[1]; imageWidth = s[0]
      figsize = imageWidth / float(dpi), imageHeight / float(dpi)
      topRel = float(line[4])
      leftRel = float(line[5])
      bottomRel = float(line[6])
      rightRel = float(line[7])
      unq_id= "crops_"+str(uuid.uuid1())
      #print(line,imageWidth,imageHeight)
      print("%s,%s,%s,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f"%(unq_id, line[0],line[2],float(line[3]),topRel,leftRel,bottomRel,rightRel), file=out)
      x1 = int(leftRel * imageWidth)
      y1 = int(topRel * imageHeight)
      x2 = int(rightRel* imageWidth)
      y2 = int(bottomRel * imageHeight)
      crop= image.crop((x1,y1,x2,y2)).resize((256,256),Image.BILINEAR)

      if not os.path.exists(dest):
        os.mkdir(dest)

      crop.save(os.path.join(dest,unq_id+".JPG"))
      image.close()
      co+=1
      if co%1000==0:
        print(pid,co)
    except:
      pass
  out.close()


def divide(t,n,i):
    length=t/(n+0.0)
    #print length,(i-1)*length,i*length
    return int(round((i-1)*length)),int(round(i*length))



conn = sqlite3.connect('emammal.db')
c = conn.cursor()
root= 'all_crops/emammal_crops'
images_folder='crops'
c.execute('SELECT * FROM detections where label in (select label from species)')
rows = c.fetchall()
os.mkdir(root)
total_records=len(rows)
total_processors=12
print(total_records)
for i in range(1,total_processors+1):
  st,ln=divide(total_records,total_processors,i)
  p1 = Process(target=do_chunk, args=(i,rows[st:ln]))
  p1.start()
    
