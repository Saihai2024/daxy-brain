"""
Drop-in shim for pysparnn.cluster_index.MultiClusterIndex.

The original library is unavailable on PyPI; here we provide a minimal
cosine-similarity-based replacement compatible with the API used in
recall/recall.py.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MultiClusterIndex:
    def __init__(self, features, records_data, *_args, **_kwargs):
        self._features = features
        self._records = list(records_data)

    def search(self, query_features, k=10, k_clusters=10, return_distance=False, **_kwargs):
        sims = cosine_similarity(query_features, self._features)
        results = []
        for row in sims:
            top_idx = np.argsort(-row)[:k]
            if return_distance:
                results.append([(1.0 - float(row[i]), self._records[i]) for i in top_idx])
            else:
                results.append([self._records[i] for i in top_idx])
        return results
