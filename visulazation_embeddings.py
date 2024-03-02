
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Replace 'your_file.csv' with the path to your CSV file
file_path = '/home/dengzy/triplet-reid/embeddings_old.csv'
# file_path = '/home/dengzy/triplet-reid/embeddings.csv'

# Read the CSV file, assuming the first row is data, not header
df = pd.read_csv(file_path, header=None)

# Extract the 128-dimensional vectors, excluding the first column which contains the names
data = df.iloc[:, 1:].values

# Define the labels for each class based on the description
labels = ['dry_grass'] * 20 + ['fresh_grass'] * 20 + ['shrubbery'] * 20 + ['smooth_concrete'] * 20 + ['tree'] * 20

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=0)
tsne_results = tsne.fit_transform(data)

# Perform PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(data)

# Color mapping with updated colors for better visual distinction
colors = {
    'dry_grass': 'orange',  # Using gold for a realistic representation of dry grass
    'fresh_grass': 'lime',  # Bright green for fresh grass
    'shrubbery': 'greenyellow',  # Darker green for shrubbery
    'smooth_concrete': 'slategrey',  # Slate grey for concrete, providing contrast
    'tree': 'darkgreen'  # Peru color for trees, simulating bark color
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
plt.show()


# import pandas as pd
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import numpy as np

# # Replace 'your_file.csv' with the path to your CSV file
# file_path = '/home/dengzy/triplet-reid/embeddings.csv'

# # Read the CSV file
# df = pd.read_csv(file_path,header=None)
# # print(df.head())
# # print(df.tail())

# # Assuming 'df' is your DataFrame from the CSV file
# # Extract the 128-dimensional vectors, excluding the first column which contains the names
# data = df.iloc[:, 1:].values
# # print(data.shape)

# # Define the labels for each class based on the description
# labels = ['dry_grass'] * 20 + ['fresh_grass'] * 20 + ['shrubbery'] * 20 + ['smooth_concrete'] * 20 + ['tree'] * 20

# # Initialize t-SNE
# tsne = TSNE(n_components=2, random_state=0)

# # Perform t-SNE on the data
# tsne_results = tsne.fit_transform(data)
# print(tsne_results.shape)

# # Perform PCA
# pca = PCA(n_components=2)
# pca_results = pca.fit_transform(data)

# # Plotting
# plt.figure(figsize=(12, 10))

# # Color mapping
# # Updated color mapping with more realistic representations
# colors = {
#     'dry_grass': 'orange',  # Dry grass is often a golden color
#     'fresh_grass': 'lime',  # Fresh grass is a vibrant green
#     'shrubbery': 'seagreen',  # Shrubbery can be represented by a darker green to differentiate from fresh grass
#     'smooth_concrete': 'gray',  # Smooth concrete is typically gray, using a light shade for visibility
#     'tree': 'tomato'  # Trees can be represented by a brown, considering the trunk color
# }

# # Plotting both PCA and t-SNE results
# fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# # PCA Plot
# axes[0].set_title('PCA visualization of the 128-dimensional vectors')
# for label in set(labels):
#     indices = [i for i, l in enumerate(labels) if l == label]
#     axes[0].scatter(pca_results[indices, 0], pca_results[indices, 1], label=label, color=colors[label], alpha=0.7)
# axes[0].legend()
# axes[0].set_xlabel('PCA feature 1')
# axes[0].set_ylabel('PCA feature 2')

# # t-SNE Plot
# axes[1].set_title('t-SNE visualization of the 128-dimensional vectors')
# for label in set(labels):
#     indices = [i for i, l in enumerate(labels) if l == label]
#     axes[1].scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=label, color=colors[label], alpha=0.7)
# axes[1].legend()
# axes[1].set_xlabel('t-SNE feature 1')
# axes[1].set_ylabel('t-SNE feature 2')

# plt.show()

# # Set the background color
# plt.gca().set_facecolor('whitesmoke')  # Light gray background for contrast
# plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)  # Add grid for better readability

# for label in set(labels):
#     # Find indices of the current label
#     indices = [i for i, l in enumerate(labels) if l == label]
    
#     # Scatter plot for each class, with increased marker size and edge color for clarity
#     plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=label, color=colors[label], 
#                 s=100,  # Increase marker size
#                 edgecolors='w',  # White edges around the markers to make them stand out more
#                 alpha=0.7)  # Slightly transparent markers

# plt.legend(frameon=True, facecolor='white', framealpha=0.8, edgecolor='gray', fontsize=14)  # Styled legend for better visibility
# plt.title('t-SNE visualization of the 128-dimensional vectors', fontsize=16, fontweight='bold')
# plt.xlabel('t-SNE feature 1', fontsize=14)
# plt.ylabel('t-SNE feature 2', fontsize=14)
# plt.xticks(fontsize=12)  # Adjust font size for x-ticks
# plt.yticks(fontsize=12)  # Adjust font size for y-ticks
# plt.show()