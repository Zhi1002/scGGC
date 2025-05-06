import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class InnerProductDecoder(nn.Module):
    def __init__(self, activation=torch.sigmoid, dropout=0.1):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.activation = activation

    def forward(self, z):
        z = F.dropout(z, self.dropout)
        adj = self.activation(torch.mm(z, z.t()))
        return adj

class GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, norm=None):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.norm = norm

    def forward(self, graph, features):
        h = torch.matmul(graph, features)
        h = self.linear(h)
        if self.activation:
            h = self.activation(h)
        if self.norm:
            h = self.norm(h)
        return h

class GCNAE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation=None, norm=None, dropout=0.1, hidden=None):
        super(GCNAE, self).__init__()
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation, norm=norm))
        for _ in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation, norm=norm))
        self.decoder = InnerProductDecoder(activation=lambda x: x)
        self.hidden = hidden
        if hidden is not None:
            enc = []
            for i, _ in enumerate(hidden):
                if i == 0:
                    enc.append(nn.Linear(n_hidden, hidden[i]))
                else:
                    enc.append(nn.Linear(hidden[i-1], hidden[i]))
            self.encoder = nn.Sequential(*enc)

    def forward(self, A, features):
        x = features
        for layer in self.layers:
            if self.dropout is not None:
                x = self.dropout(x)
            x = layer(A, x)
        x = x.view(x.shape[0], -1)
        if self.hidden is not None:
            x = self.encoder(x)
        adj_rec = self.decoder(x)
        return adj_rec, x






class Generator(nn.Module):
    def __init__(self, latent_dim, feature_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, feature_dim),
            nn.Tanh()  # 假设特征值在[-1, 1]
        )

    def forward(self, x):
        return self.model(x)

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, num_classes + 1),  # 增加一个类别用于假样本
        )

    def forward(self, x):
        return self.model(x)


class CellDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]