from DL.sqlite_data_loader import SQLDataLoader
from DL.losses import *
from DL.utils import *
from DL.networks import *
from DL.Engine import Engine
from UIComponents.DBObjects import *

import os
import time
import sys
import matplotlib.pyplot as plt
import numpy as np

if __name__=='__main__':
    db_path = "datasets/SS/SS.db"
    print(db_path)
    db = SqliteDatabase(db_path)
    proxy.initialize(db)
    db.connect()

    trainset_query = Detection.select(Detection.id,Oracle.label).join(Oracle).where(Detection.kind==DetectionKind.UserDetection.value).order_by(fn.Random()).limit(5000)
    train_dataset = SQLDataLoader(trainset_query, os.path.join("datasets/SS", 'crops'), is_training= False)
    
    checkpoint = load_checkpoint("triplet_resnet50_1499.tar")
    embedding_net = EmbeddingNet(checkpoint['arch'], checkpoint['feat_dim'], False)
    #embedding_net = EmbeddingNet('resnet50', 256, True)
    model = torch.nn.DataParallel(embedding_net).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    sys.stdout.flush()
    start= time.time()
    for i in range(4000):
        x = train_dataset[i][0]
        x = x.unsqueeze(0)
        x = x.cuda(non_blocking=True)
        # compute output
        output = model(x)
        print(time.time()-start)
        sys.stdout.flush()
        start= time.time()
    #val_loader = train_dataset.getSingleLoader(batch_size = 8)
    #for a, b , c in val_loader:
        #print(b[0])
        #plt.imshow(np.rollaxis(np.rollaxis(a[0].numpy(), 1, 0), 2, 1))
        #plt.show()
        #print(np.rollaxis(a[0].numpy() , 1, 0).shape)
