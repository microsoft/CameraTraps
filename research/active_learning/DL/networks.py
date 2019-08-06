'''
networks.py

Specifies architectures for different embedding networks.

'''

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EmbeddingNet(nn.Module):
    def __init__(self, architecture, feat_dim, use_pretrained=False):
        super(EmbeddingNet, self).__init__()
        self.feat_dim = feat_dim
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


class SoftmaxNet(nn.Module):
    def __init__(self, architecture, feat_dim, num_classes, use_pretrained = False):
        super(SoftmaxNet, self).__init__()
        self.embedding = EmbeddingNet(architecture, feat_dim, use_pretrained = use_pretrained)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        embed = self.embedding(x)
        x = F.relu(embed)
        x = self.classifier(x)
        return x, embed


class ClassificationNet(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super(ClassificationNet, self).__init__()
        self.fc1 = nn.Linear(feat_dim, 128)
        self.fc12 = nn.Linear(128, 64)
        self.bn1 = nn.BatchNorm1d(64)
        #self.fc13 = nn.Linear(128, 64)
        #self.bn2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p = 0.2, training=self.training)
        x = F.relu(self.bn1(self.fc12(x)))
        #x = F.relu(self.fc12(x))

        #x = F.relu(self.bn1(self.fc13(x)))
        #x = F.relu(self.fc13(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class CombinedNet(nn.Module):
    def __init__(self, embedding_net, classification_net):
        super(CombinedNet, self).__init__()
        self.embedding_net = embedding_net
        self.classification_net = classification_net

    def train(self, mode=True):
        self.embedding_net.train()
        self.classification_net.train()

    def eval(self, mode=False):
        self.embedding_net.eval()
        self.classification_net.eval()

    def forward(self, x):
        # save features last FC layer
        x = self.embedding_net(x)
        #x = F.relu(x)
        x = self.classification_net(x)
        return x

    def get_embedding(self, x):
        # save features last FC layer
        return self.embedding_net(x)
