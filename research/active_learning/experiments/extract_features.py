'''
extract_features.py

Loads a pre-trained embedding model, uses it for inference to obtain embedded feature representations on input images, and explores the embedded feature space.

'''

import argparse, os, random, sys, time
import numpy as np
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')

from DL.utils import *
from DL.networks import *
from DL.sqlite_data_loader import *
from Database.DB_models import *

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

# class EmbeddingNet(nn.Module):
#     def __init__(self, architecture, feat_dim, use_pretrained=False):
#         super(EmbeddingNet, self).__init__()
#         self.feat_dim= feat_dim
#         self.inner_model = models.__dict__[architecture](pretrained=use_pretrained)
#         if architecture.startswith('resnet'):
#             in_feats= self.inner_model.fc.in_features
#             self.inner_model.fc = nn.Linear(in_feats, feat_dim)
#         elif architecture.startswith('inception'):
#             in_feats= self.inner_model.fc.in_features
#             self.inner_model.fc = nn.Linear(in_feats, feat_dim)
#         if architecture.startswith('densenet'):
#             in_feats= self.inner_model.classifier.in_features
#             self.inner_model.classifier = nn.Linear(in_feats, feat_dim)
#         if architecture.startswith('vgg'):
#             in_feats= self.inner_model.classifier._modules['6'].in_features
#             self.inner_model.classifier._modules['6'] = nn.Linear(in_feats, feat_dim)
#         if architecture.startswith('alexnet'):
#             in_feats= self.inner_model.classifier._modules['6'].in_features
#             self.inner_model.classifier._modules['6'] = nn.Linear(in_feats, feat_dim)

#     def forward(self, x):
#         return self.inner_model.forward(x)

# class NormalizedEmbeddingNet(EmbeddingNet):
#     def __init__(self, architecture, feat_dim, use_pretrained=False):
#         EmbeddingNet.__init__(self, architecture, feat_dim, use_pretrained = use_pretrained)

#     def forward(self, x):
#         embedding =  F.normalize(self.inner_model.forward(x))*10.0
#         return embedding, embedding

# def get_random_images(num, image_dir, test_transforms):
#     data = datasets.ImageFolder(image_dir, transform=test_transforms) # slight abuse; this expects subfolders corresponding to classes but we have no classes here
#     indices = list(range(len(data)))
#     np.random.shuffle(indices)
#     idx = indices[:num]
#     from torch.utils.data.sampler import SubsetRandomSampler
#     sampler = SubsetRandomSampler(idx)
#     loader = torch.utils.data.DataLoader(data, 
#                     sampler=sampler, batch_size=num)
#     dataiter = iter(loader)
#     images, labels = dataiter.next()
#     return images, labels

# def predict_image(image, model, test_transforms):
#     device = torch.device("cuda" if torch.cuda.is_available() 
#                                     else "cpu")
#     image_tensor = test_transforms(image).float()
#     image_tensor = image_tensor.unsqueeze_(0)
#     input = Variable(image_tensor)
#     input = input.to(device)
#     output = model(input)[0]
#     return output.data.cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', default='missouricameratraps', type=str, help='Name of the training (target) data Postgres DB.')
    parser.add_argument('--db_user', default='new_user', type=str, help='Name of the user accessing the Postgres DB.')
    parser.add_argument('--db_password', default='new_user_password', type=str, help='Password of the user accessing the Postgres DB.')
    parser.add_argument('--base_model', type=str, help='Path to latest embedding model checkpoint.')
    parser.add_argument('--crop_dir', type=str, help='Path to directory with cropped images to get embedding features for.')
    parser.add_argument('--num', type=int, help='Number of samples to draw from dataset to get embedding features.')
    parser.add_argument('--random_seed', type=int, help='Random seed to get same samples from database.')
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Connect to database and initialize db_proxy
    ## database connection credentials
    DB_NAME = args.db_name
    USER = args.db_user
    PASSWORD = args.db_password
    target_db = PostgresqlDatabase(DB_NAME, user=USER, password=PASSWORD, host='localhost')
    target_db.connect(reuse_if_open=True)
    db_proxy.initialize(target_db)
    ## load the dataset
    dataset_query = Detection.select(Detection.image_id, Oracle.label, Detection.kind).join(Oracle).limit(args.num)
    dataset = SQLDataLoader(args.crop_dir, query=dataset_query, is_training=False, kind=DetectionKind.ModelDetection.value, num_workers=8, limit=args.num)

    
    # Load the saved embedding model from the checkpoint
    checkpoint = load_checkpoint(args.base_model)
    if checkpoint['loss_type'].lower() == 'center' or checkpoint['loss_type'].lower() == 'softmax':
        embedding_net = SoftmaxNet(checkpoint['arch'], checkpoint['feat_dim'], checkpoint['num_classes'], False)
    else:
        embedding_net = NormalizedEmbeddingNet(checkpoint['arch'], checkpoint['feat_dim'], False)
    model = torch.nn.DataParallel(embedding_net).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    ## update the dataset embedding
    dataset.updateEmbedding(model)

    print('len(dataset)', len(dataset))
    random_anchor_idx = np.random.randint(len(dataset))
    print(random_anchor_idx)



    # # Create a folder for saving embedding visualizations with this model checkpoint
    # model_emb_dirname = os.path.basename(args.base_model).split('.')[0]
    # os.makedirs(model_emb_dirname, exist_ok=True)
    # plot_embedding_images(dataset.em[:], np.asarray(dataset.getlabels()) , dataset.getpaths(), {}, model_emb_dirname+'/embedding_plot.png')
    model_emb_dirname = os.path.basename(args.base_model).split('.')[0]+'_temp'
    os.makedirs(model_emb_dirname, exist_ok=True)

    # dataset.embedding_mode()
    X_train = dataset.em[range(len(dataset))]
    y_train = np.asarray(dataset.getalllabels())
    imagepaths = dataset.getallpaths()
    random_anchor_img = dataset.loader(imagepaths[random_anchor_idx].split('.JPG')[0])
    random_anchor_img.save(model_emb_dirname+"/anchor_img.png")
    random_anchor_img_np = np.asarray(random_anchor_img)
    print(random_anchor_img_np.shape)
    # assert 2==3, 'break'

    # datasetindices = list(range(len(dataset)))
    # np.random.shuffle(datasetindices)
    # random_indices = datasetindices[:args.num]
    # print(random_indices)
    
    # selected_sample_features = np.array([]).reshape(0, 256)
    # selected_sample_labels = []
    selected_sample_images = []

    # for idx in random_indices:
    #     selected_sample_features = np.vstack([selected_sample_features, X_train[idx]])
    #     selected_sample_labels.append(y_train[idx])
    #     img_path = imagepaths[idx].split('.JPG')[0]
    #     image = dataset.loader(img_path)
    #     selected_sample_images.append(image)
    
    
    # # TRY NEAREST NEIGHBORS WALK THROUGH EMBEDDING
    # nbrs = NearestNeighbors(n_neighbors=args.num).fit(selected_sample_features)
    # distances, indices = nbrs.kneighbors(selected_sample_features)
    timer = time.time()
    nbrs = NearestNeighbors(n_neighbors=args.num).fit(X_train)
    print('Finished fitting nearest neighbors for whole dataset in %0.2f seconds'%(float(time.time() - timer)))
    distances, indices = nbrs.kneighbors(X_train)
    ten_closest_to_anchor = indices[random_anchor_idx, 1:11]
    print(distances[random_anchor_idx, 0:11])

    for idx in ten_closest_to_anchor:
        img_path = imagepaths[idx].split('.JPG')[0]
        image = dataset.loader(img_path)
        selected_sample_images.append(image)

    # plot_embedding_images(dataset.em[:], np.asarray(dataset.getlabels()) , dataset.getpaths(), {}, model_emb_dirname+'/embedding_plot.png')

    # idx_w_closest_nbr = np.where(distances[:,1] == min(distances[:,1]))[0][0]
    # order = [idx_w_closest_nbr]
    # for ii in range(len(distances)):
    #     distances[ii, 0] = np.inf

    # while len(order)<args.num:
    #     curr_idx = order[-1]
    #     curr_neighbors = indices[curr_idx]
    #     curr_dists = list(distances[curr_idx])
    #     # print(min(curr_dists))
    #     next_closest_pos = curr_dists.index(min(curr_dists))
    #     next_closest = curr_neighbors[next_closest_pos]
    #     order.append(next_closest)
    #     # make sure you can't revisit past nodes
    #     for vi in order:
    #         vi_pos = list(indices[next_closest]).index(vi)
    #         distances[next_closest, vi_pos] = np.inf
    
    # for ii in range(len(order)):
    #     imgidx = order[ii]
    #     image = selected_sample_images[imgidx]
    #     image.save(model_emb_dirname+"/img"+str(ii)+"_"+str(selected_sample_labels[imgidx])+".png")

    for ii in range(len(selected_sample_images)):
        image = selected_sample_images[ii]
        image.save(model_emb_dirname+"/img"+str(ii)+".png")







    # # Specify the transformations on the input images before inference
    # # test_transforms = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    # test_transforms = transforms.Compose([transforms.Resize([256, 256]), transforms.RandomCrop([224, 224]), transforms.RandomHorizontalFlip(), transforms.ColorJitter(), transforms.ToTensor(), transforms.Normalize([0.407328, 0.407328, 0.407328], [0.118641, 0.118641, 0.118641])])

    
    # images, labels = get_random_images(num, image_dir, test_transforms)
    # all_features = np.array([]).reshape(0, 256)
    # for ii in range(len(images)):
    #     image = to_pil(images[ii])
    #     features = predict_image(image, model, test_transforms)
    #     all_features = np.vstack([all_features, features])

    
    

    

    # # for ii in range(len(images)):
    # #     image = to_pil(images[ii])
    # #     image.save("img"+str(ii)+".png")

    # # TRY CLUSTERING
    # kmeans1 = KMeans(n_clusters=5).fit(StandardScaler().fit_transform(all_features))
    # print(kmeans1.labels_)
    # for ii in range(len(images)):
    #     image = to_pil(images[ii])
    #     filename = str(kmeans1.labels_[ii])+"/img"+str(ii)+".png"
    #     if not os.path.exists(os.path.dirname(filename)):
    #         os.makedirs(os.path.dirname(filename))
    #     image.save(filename)
    

if __name__ == '__main__':
    main()