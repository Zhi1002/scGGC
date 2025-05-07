import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class InnerProductDecoder(nn.Module):
    def __init__(self, activation=torch.sigmoid, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.activation = activation
    def forward(self, z):
        z = F.dropout(z, self.dropout)
        return self.activation(z @ z.t())

class GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, norm=None):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.norm = norm
    def forward(self, graph, features):
        h = graph @ features
        h = self.linear(h)
        if self.activation: h = self.activation(h)
        if self.norm: h = self.norm(h)
        return h

class GCNAE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation=None, norm=None, dropout=0.1, hidden=None, hidden_relu=False, hidden_bn=False):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([GraphConv(in_feats if i==0 else n_hidden, n_hidden, activation=activation, norm=norm) for i in range(n_layers)])
        self.decoder = InnerProductDecoder(activation=lambda x:x)
        if hidden:
            enc = []
            for i, h in enumerate(hidden):
                enc.append(nn.Linear(n_hidden if i==0 else hidden[i-1], h))
                if hidden_bn: enc.append(nn.BatchNorm1d(h))
                if hidden_relu: enc.append(nn.LeakyReLU(0.01))
            self.encoder = nn.Sequential(*enc)
        else:
            self.encoder = None
    def forward(self, A, features):
        x = features
        for conv in self.layers:
            x = self.dropout(x)
            x = conv(A, x)
        x = x.reshape(x.size(0), -1)
        if self.encoder: x = self.encoder(x)
        return self.decoder(x), x



class Generator(nn.Module):
    def __init__(self, noise_dim, embed_dim, num_classes, output_dim):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(noise_dim + embed_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
    def forward(self, noise, labels):
        emb = self.label_emb(labels)
        return self.net(torch.cat([noise, emb], dim=1))

class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, True),
        )
        self.adv_head   = nn.Sequential(nn.Linear(256,1), nn.Sigmoid())
        self.class_head = nn.Linear(256, num_classes)
    def forward(self, x):
        f = self.feature(x)
        return self.adv_head(f), self.class_head(f), f

def train_ac_gan(X_train, y_train, device, num_classes, input_dim, n_epochs=30):
    noise_dim = 100
    embed_dim = 50
    lr = 1e-4

    G = Generator(noise_dim, embed_dim, num_classes, input_dim).to(device)
    D = Discriminator(input_dim, num_classes).to(device)

    adv_loss = nn.BCELoss()
    cls_loss = nn.CrossEntropyLoss()
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    for epoch in range(1, n_epochs+1):
        D.train(); G.train()
        d_loss_acc = g_loss_acc = 0.0
        for real_x, real_y in train_loader:
            bs = real_x.size(0)
            real_src = torch.ones(bs,1,device=device)
            fake_src = torch.zeros(bs,1,device=device)

            opt_D.zero_grad()
            out_r_src, out_r_cls, _ = D(real_x)
            loss_r = adv_loss(out_r_src, real_src) + cls_loss(out_r_cls, real_y)
            z = torch.randn(bs, noise_dim, device=device)
            fake_y = torch.randint(0, num_classes, (bs,), device=device)
            fake_x = G(z, fake_y)
            out_f_src, _, _ = D(fake_x.detach())
            loss_f = adv_loss(out_f_src, fake_src)
            d_loss = 0.5*(loss_r + loss_f)
            d_loss.backward(); opt_D.step()

            opt_G.zero_grad()
            out_f_src2, out_f_cls2, _ = D(fake_x)
            g_loss = adv_loss(out_f_src2, real_src) + cls_loss(out_f_cls2, fake_y)
            g_loss.backward(); opt_G.step()

            d_loss_acc += d_loss.item()
            g_loss_acc += g_loss.item()

        print(f"Epoch {epoch}/{n_epochs} | D_loss: {d_loss_acc/len(train_loader):.4f} | G_loss: {g_loss_acc/len(train_loader):.4f}")

    return G, D

def extract_embeddings_and_predict_labels(D, X_all, device, num_classes):
    D.eval()
    embeddings, probs_list = [], []
    with torch.no_grad():
        for i in range(0, X_all.size(0), 64):
            batch = X_all[i:i+64]
            _, out_cls, feat = D(batch)
            embeddings.append(feat.cpu().numpy())
            prob = torch.softmax(out_cls, dim=1)
            probs_list.append(prob.cpu().numpy())

    embeddings = np.vstack(embeddings)
    probabilities = np.vstack(probs_list)
    predicted_idx = np.argmax(probabilities, axis=1)

    return embeddings, predicted_idx, probabilities  # 返回 probabilities