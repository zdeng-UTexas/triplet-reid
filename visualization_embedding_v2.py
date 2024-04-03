
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Replace 'your_file.csv' with the path to your CSV file
file_path = '/home/zhiyundeng/aeroplan/experiment/20240402_lejeune_emount_training/embedding_of_patch_part_5.csv'
# file_path = '/home/dengzy/triplet-reid/embeddings.csv'

# Read the CSV file, assuming the first row is data, not header
df = pd.read_csv(file_path, header=None)

# Extract the 128-dimensional vectors, excluding the first column which contains the names
data = df.iloc[:, 1:].values

# Define the labels for each class based on the description
labels = ['building'] * 30 + ['dirt'] * 30 + ['grass'] * 30 + ['smooth_trail'] * 30

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=0)
tsne_results = tsne.fit_transform(data)

# Perform PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(data)

# Color mapping with updated colors for better visual distinction
colors = {
    'building': 'orange',  # Using gold for a realistic representation of dry grass
    'dirt': 'lime',  # Bright green for fresh grass
    'grass': 'greenyellow',  # Darker green for shrubbery
    'smooth_trail': 'slategrey',  # Slate grey for concrete, providing contrast
}

# Improved Plotting with customized axis labels
fig, axes = plt.subplots(1, 2, figsize=(24, 10), dpi=100)  # Increased figure size and DPI for clarity

# Function to plot PCA and t-SNE with customizable axis labels
def plot_results(ax, results, title, xlabel, ylabel):
    ax.set_title(title, fontsize=18)
    for label in set(labels):
        indices = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(results[indices, 0], results[indices, 1], label=label, color=colors[label], s=50, alpha=0.8, edgecolors='k', linewidth=0.5)
    ax.legend(markerscale=2., fontsize=12, loc='best')
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

# Plot PCA with "Component 1" and "Component 2" labels
plot_results(axes[0], pca_results, 'PCA Visualization of the 128-dimensional Vectors', 'Component 1', 'Component 2')

# Plot t-SNE with "Feature 1" and "Feature 2" labels
plot_results(axes[1], tsne_results, 't-SNE Visualization of the 128-dimensional Vectors', 'Feature 1', 'Feature 2')

plt.tight_layout()  # Adjust layout to not overlap
plt.savefig('visualization.png')

