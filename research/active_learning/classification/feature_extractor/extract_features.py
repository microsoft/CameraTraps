'''
extract_features.py

Loads a pre-trained embedding model, uses it for inference to obtain embedded feature representations on input images, and explores the embedded feature space.

'''

import argparse, os, sys, time
import numpy as np
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

class EmbeddingNet(nn.Module):
    def __init__(self, architecture, feat_dim, use_pretrained=False):
        super(EmbeddingNet, self).__init__()
        self.feat_dim= feat_dim
        self.inner_model = models.__dict__[architecture](pretrained=use_pretrained)
        if architecture.startswith('resnet'):
            in_feats= self.inner_model.fc.in_features
            self.inner_model.fc = nn.Linear(in_feats, feat_dim)
        elif architecture.startswith('inception'):
            in_feats= self.inner_model.fc.in_features
            self.inner_model.fc = nn.Linear(in_feats, feat_dim)
        if architecture.startswith('densenet'):
            in_feats= self.inner_model.classifier.in_features
            self.inner_model.classifier = nn.Linear(in_feats, feat_dim)
        if architecture.startswith('vgg'):
            in_feats= self.inner_model.classifier._modules['6'].in_features
            self.inner_model.classifier._modules['6'] = nn.Linear(in_feats, feat_dim)
        if architecture.startswith('alexnet'):
            in_feats= self.inner_model.classifier._modules['6'].in_features
            self.inner_model.classifier._modules['6'] = nn.Linear(in_feats, feat_dim)

    def forward(self, x):
        return self.inner_model.forward(x)

class NormalizedEmbeddingNet(EmbeddingNet):
    def __init__(self, architecture, feat_dim, use_pretrained=False):
        EmbeddingNet.__init__(self, architecture, feat_dim, use_pretrained = use_pretrained)

    def forward(self, x):
        embedding =  F.normalize(self.inner_model.forward(x))*10.0
        return embedding, embedding

def get_random_images(num, image_dir, test_transforms):
    data = datasets.ImageFolder(image_dir, transform=test_transforms) # slight abuse; this expects subfolders corresponding to classes but we have no classes here
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, 
                    sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels

def predict_image(image, model, test_transforms):
    device = torch.device("cuda" if torch.cuda.is_available() 
                                    else "cpu")
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)[0]
    return output.data.cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str,
                        help='Path to latest embedding model checkpoint')
    parser.add_argument('--crop_dir', type=str,
                        help='Path to directory with cropped images for classification')
    args = parser.parse_args()

    image_dir = args.crop_dir
    num = 50

    # Try to load the embedding model from the checkpoint
    checkpoint = torch.load(args.base_model)
    if checkpoint['loss_type'].lower() == 'center' or checkpoint['loss_type'].lower() == 'softmax':
        embedding_net = SoftmaxNet(checkpoint['arch'], checkpoint['feat_dim'], checkpoint['num_classes'], False)
    else:
        embedding_net = NormalizedEmbeddingNet(checkpoint['arch'], checkpoint['feat_dim'], False)
    model = torch.nn.DataParallel(embedding_net).cuda()
    model.load_state_dict(checkpoint['state_dict'])

    # Specify the transformations on the input images before inference
    # test_transforms = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.Resize([256, 256]), transforms.RandomCrop([224, 224]), transforms.RandomHorizontalFlip(), transforms.ColorJitter(), transforms.ToTensor(), transforms.Normalize([0.407328, 0.407328, 0.407328], [0.118641, 0.118641, 0.118641])])

    to_pil = transforms.ToPILImage()
    images, labels = get_random_images(num, image_dir, test_transforms)
    all_features = np.array([]).reshape(0, 256)
    for ii in range(len(images)):
        image = to_pil(images[ii])
        features = predict_image(image, model, test_transforms)
        all_features = np.vstack([all_features, features])

    
    # TRY NEAREST NEIGHBORS WALK THROUGH EMBEDDING
    nbrs = NearestNeighbors(n_neighbors=num).fit(all_features)
    distances, indices = nbrs.kneighbors(all_features)

    idx_w_closest_nbr = np.where(distances[:,1] == min(distances[:,1]))[0][0]
    order = [idx_w_closest_nbr]
    for ii in range(len(distances)):
        distances[ii, 0] = np.inf

    while len(order)<num:
        curr_idx = order[-1]
        curr_neighbors = indices[curr_idx]
        curr_dists = list(distances[curr_idx])
        print(min(curr_dists))
        next_closest_pos = curr_dists.index(min(curr_dists))
        next_closest = curr_neighbors[next_closest_pos]
        order.append(next_closest)
        # make sure you can't revisit past nodes
        for vi in order:
            vi_pos = list(indices[next_closest]).index(vi)
            distances[next_closest, vi_pos] = np.inf

    for ii in range(len(order)):
        imgidx = order[ii]
        image = to_pil(images[imgidx])
        image.save("img"+str(ii)+".png")

    # for ii in range(len(images)):
    #     image = to_pil(images[ii])
    #     image.save("img"+str(ii)+".png")

    # TRY CLUSTERING
    kmeans1 = KMeans(n_clusters=5).fit(StandardScaler().fit_transform(all_features))
    print(kmeans1.labels_)
    for ii in range(len(images)):
        image = to_pil(images[ii])
        filename = str(kmeans1.labels_[ii])+"/img"+str(ii)+".png"
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        image.save(filename)
    

if __name__ == '__main__':
    main()