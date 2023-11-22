import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import BasicBlock
import torch.utils.model_zoo as model_zoo
import torch
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
        embedding = self.inner_model.forward(x)
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