#
# networks.py
#
# Network architectures for active learning: embedding and classification.
#

#%% Constants and imports

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


#%% Embedding architecture

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
        # save features last FC layer
        feat = F.normalize(self.inner_model.forward(x))
        return feat


#%% Classification architecture
        
class ClassificationNet(nn.Module):
    
    def __init__(self, feat_dim, num_classes):
        super(ClassificationNet, self).__init__()
        self.fc1 = nn.Linear(feat_dim, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


#%% Combined embedding and classification architecture
        
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
