'''
One-off script to look at embedding results for a set of script
'''

import argparse, os, pickle, random, sys, time
import numpy as np
import torch
import scipy.io
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

sys.path.append("..")
from DL.utils import *
from DL.networks import *
from Database.DB_models import *
from DL.sqlite_data_loader import SQLDataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', default='missouricameratraps', type=str, help='Name of the training (target) data Postgres DB.')
    parser.add_argument('--db_user', default='user', type=str, help='Name of the user accessing the Postgres DB.')
    parser.add_argument('--db_password', default='password', type=str, help='Password of the user accessing the Postgres DB.')
    parser.add_argument('--num', default=2500, type=int, help='Number of samples to draw from dataset to get embedding features.')
    parser.add_argument('--crop_dir', type=str, help='Path to directory with cropped images to get embedding features for.')
    parser.add_argument('--base_model', type=str, help='Path to latest embedding model checkpoint.')
    parser.add_argument('--random_seed', default=1234, type=int, help='Random seed to get same samples from database.')
    parser.add_argument('--output_dir', type=str, help='Output directory for subset of crops')
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    BASE_MODEL = args.base_model
    DB_NAME = args.db_name
    USER = args.db_user
    PASSWORD = args.db_password

    # Connect to database and sample a dataset
    target_db = PostgresqlDatabase(DB_NAME, user=USER, password=PASSWORD, host='localhost')
    target_db.connect(reuse_if_open=True)
    db_proxy.initialize(target_db)
    dataset_query = Detection.select(Detection.image_id, Oracle.label, Detection.kind).join(Oracle).limit(args.num)
    dataset = SQLDataLoader(args.crop_dir, query=dataset_query, is_training=False, kind=DetectionKind.ModelDetection.value, num_workers=8, limit=args.num)
    imagepaths = dataset.getallpaths()

    # Load the saved embedding model from the checkpoint
    checkpoint = load_checkpoint(BASE_MODEL)
    if checkpoint['loss_type'].lower() == 'center' or checkpoint['loss_type'].lower() == 'softmax':
        embedding_net = SoftmaxNet(checkpoint['arch'], checkpoint['feat_dim'], checkpoint['num_classes'], False)
    else:
        embedding_net = NormalizedEmbeddingNet(checkpoint['arch'], checkpoint['feat_dim'], False)
    model = torch.nn.DataParallel(embedding_net).cuda()
    model.load_state_dict(checkpoint['state_dict'])

    # Update the dataset embedding
    dataset.updateEmbedding(model)
    X_train = dataset.em[range(len(dataset))]
    y_train = np.asarray(dataset.getalllabels())
    imagepaths = dataset.getallpaths()

    datasetindices = list(range(len(dataset)))
    
    sample_features = np.array([]).reshape(0, 256)
    sample_labels = []
    sample_images = []

    for idx in datasetindices:
        sample_features = np.vstack([sample_features, X_train[idx]])
        sample_labels.append(y_train[idx])
        img_path = imagepaths[idx].split('.JPG')[0]
        image = dataset.loader(img_path)
        sample_images.append(image)
    
    # save the images
    for idx in datasetindices:
        img_path = imagepaths[idx].split('.JPG')[0]
        image = dataset.loader(img_path)
        os.makedirs(os.path.join(args.output_dir, 'crops'), exist_ok=True)
        image.save(os.path.join(args.output_dir, 'crops', '%d.JPG'%idx))
    
    # save the features
    # with open(os.path.join(args.output_dir, 'lastlayer_features.mat'), 'wb') as f:
        # pickle.dump(sample_features, f)
    
    # with open(os.path.join(args.output_dir, 'labels.mat'), 'wb') as f:
        # pickle.dump(sample_labels, f)
    
    with open(os.path.join(args.output_dir, 'lastlayer_features_and_labels.mat'), 'wb') as f:
        scipy.io.savemat(f, mdict={'features': sample_features, 'labels': sample_labels})
        
    


if __name__ == '__main__':
    main()