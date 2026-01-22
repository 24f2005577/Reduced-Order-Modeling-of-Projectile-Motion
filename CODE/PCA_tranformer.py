import numpy as np
def pca_tranformer(ar):
    mean = np.mean(ar, axis=0)
    std = np.std(ar, axis=0)

    ar_std = (ar - mean) / std   # 🔥 THIS IS THE FIX

    cov = np.cov(ar_std.T)
    eigval, eigvec = np.linalg.eigh(cov)

    idx = np.argsort(eigval)[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]

    compo = eigvec[:, :2]
    x_t = ar_std @ compo

    return x_t, eigval, compo
