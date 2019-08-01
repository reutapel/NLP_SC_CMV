import pandas as pd
import umap
from ivis import Ivis

model = Ivis(embedding_dims=2, k=15)

embeddings = model.fit_transform(X_scaled)