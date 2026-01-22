from dataset_genarator import generator
import numpy as np
def pca_tranformer(ar):
    mean=np.mean(ar,axis=0)
    ar=ar-mean
    cov=np.cov(ar.T)
    eigval , eigvec = np.linalg.eigh(cov)
    id=np.argsort(eigval)[::-1]
    eigval=eigval[id]
    eigvec=eigvec[:,id]
    compo=eigvec[:,0:2]
    x_t=np.dot(ar,compo)
    return(x_t,eigval,compo)