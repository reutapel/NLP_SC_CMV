import pandas as pd
import umap
from ivis import Ivis
import numpy as np

model = Ivis(embedding_dims=2, k=15)

embeddings = model.fit_transform(X_scaled)

# dimension reduction
clusterable_embedding = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(mnist.data)
# plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],
#             c=mnist.target, s=0.1, cmap='Spectral');

# cluster
labels = hdbscan.HDBSCAN(
    min_samples=10,
    min_cluster_size=500,
).fit_predict(clusterable_embedding)

# visualize
clustered = (labels >= 0)
plt.scatter(standard_embedding[~clustered, 0],
            standard_embedding[~clustered, 1],
            c=(0.5, 0.5, 0.5),
            s=0.1,
            alpha=0.5)
plt.scatter(standard_embedding[clustered, 0],
            standard_embedding[clustered, 1],
            c=labels[clustered],
            s=0.1,
            cmap='Spectral');

# quantitive assesment of clustering with HDNSCAN
adjusted_rand_score(mnist.target, labels), adjusted_mutual_info_score(mnist.target, labels)
# only where it is confident
clustered = (labels >= 0)
(
    adjusted_rand_score(mnist.target[clustered], labels[clustered]),
    adjusted_mutual_info_score(mnist.target[clustered], labels[clustered])
)

# check how much of the data we are clustering
np.sum(clustered) / mnist.data.shape[0]