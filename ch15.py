from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

digits = load_digits()
tsne = TSNE(n_components=3, random_state=11)
reduced_data = tsne.fit_transform(digits.data)

axes = plt.figure().add_subplot(projection='3d')
dots = axes.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], 
    c=digits.target, cmap=plt.cm.get_cmap('nipy_spectral_r', 10))
colorbar = plt.colorbar(dots)

plt.show()