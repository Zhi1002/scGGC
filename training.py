import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

def train_gan(generator, discriminator, data_loader, optimizer_g, optimizer_d, criterion, num_epochs, latent_dim, num_classes):
    for epoch in range(num_epochs):
        for inputs, labels in data_loader:
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.LongTensor)
            batch_size = inputs.size(0)

            optimizer_d.zero_grad()

            real_samples = inputs
            real_outputs = discriminator(real_samples)
            real_loss = criterion(real_outputs[:, :-1], labels)

            noise = torch.randn(batch_size, latent_dim)
            fake_samples = generator(noise)
            fake_outputs = discriminator(fake_samples)
            fake_labels = torch.full((batch_size,), num_classes, dtype=torch.long)
            fake_loss = criterion(fake_outputs, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            noise = torch.randn(batch_size, latent_dim)
            fake_samples = generator(noise)
            outputs = discriminator(fake_samples)
            g_loss = criterion(outputs[:, :-1], labels)
            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")

def visualize_tsne(features, labels, title, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    tsne_df = pd.DataFrame({
        'tsne_dim1': tsne_results[:, 0],
        'tsne_dim2': tsne_results[:, 1],
        'label': labels
    })
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='tsne_dim1', y='tsne_dim2', hue='label', palette='viridis', data=tsne_df, legend='full', alpha=0.8, s=50)
    plt.title(title)
    plt.savefig(save_path)
    plt.show()

def plot_clusters(data, labels, title='Clusters', save_path=None):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(data)
    plt.figure(figsize=(12, 8))
    num_clusters = len(set(labels))
    colors = sns.color_palette('viridis', num_clusters)
    scatter = sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette=colors, s=50)
    scatter.legend(loc='upper right', markerscale=2)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def enforce_label_consistency(predicted_labels, num_classes):
    unique, counts = np.unique(predicted_labels, return_counts=True)
    label_count = dict(zip(unique, counts))

    while len(label_count) < num_classes:
        min_label = min(label_count, key=label_count.get)
        min_count = label_count[min_label]
        for i in range(num_classes):
            if i not in label_count:
                predicted_labels[np.where(predicted_labels == min_label)[0][:min_count]] = i
                label_count[i] = min_count
                break

    while len(label_count) > num_classes:
        max_label = max(label_count, key=label_count.get)
        max_count = label_count[max_label]
        for i in range(num_classes):
            if i in label_count and label_count[i] < max_count:
                predicted_labels[np.where(predicted_labels == max_label)[0][:max_count - label_count[i]]] = i
                label_count[i] += max_count - label_count[i]
                del label_count[max_label]
                break

    return predicted_labels
