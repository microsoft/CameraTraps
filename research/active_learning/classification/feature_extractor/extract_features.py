'''
extract_features.py

Loads a pre-trained embedding model, uses it for inference to obtain embedded feature representations on input images, and explores the embedded feature space.
'''

import argparse, os, sys, time
import numpy as np

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.nn.functional as F

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str,
                        help='Path to latest embedding model checkpoint')
    parser.add_argument('--crop_dir', type=str,
                        help='Path to directory with cropped images for classification')
    args = parser.parse_args()

    # Try to load the embedding model from the checkpoint
    checkpoint = torch.load(args.base_model)
    if checkpoint['loss_type'].lower() == 'center' or checkpoint['loss_type'].lower() == 'softmax':
        embedding_net = SoftmaxNet(checkpoint['arch'], checkpoint['feat_dim'], checkpoint['num_classes'], False)
    else:
        embedding_net = NormalizedEmbeddingNet(checkpoint['arch'], checkpoint['feat_dim'], False)
    model = torch.nn.DataParallel(embedding_net).cuda()
    model.load_state_dict(checkpoint['state_dict'])

    # Get random images
    test_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    data = datasets.ImageFolder(args.crop_dir, transform=test_transforms)
    classes = data.classes
    print(classes)


if __name__ == '__main__':
    main()