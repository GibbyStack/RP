from scipy.spatial import distance

vector_norm = lambda x, m: - 2 * (x.T @ m) + (m.T @ m)

euclidean = lambda x, m: distance.euclidean(x, m)

cosine_similarity = lambda x, m: distance.cosine(x, m)

manhattan = lambda x, m: distance.minkowski(x, m, 1)

minkowski = lambda x, m: distance.minkowski(x, m, 2)

correlation = lambda x, m: distance.correlation(x, m)

chebyshev = lambda x, m: distance.chebyshev(x, m)