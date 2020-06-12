import numpy as np
import scipy
import sklearn
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from numpy.linalg import cholesky, pinv
import scipy
from clustNP import gauss_kernal_mat
import tensorly as tl
from scipy.spatial.distance import pdist, squareform

def constr_spec_clust(A, Q, beta=1, n_clust=2):
    n = Q.shape[0]
    vol = np.sum(A)
    Dneg12 = np.sqrt(1/np.sum(A, axis=0))
    Lbar = np.eye(n)-Dneg12.T*A*Dneg12
    Qbar = Dneg12.T*Q*Dneg12
    lam_max, _ = scipy.sparse.linalg.eigs(Qbar, k=1, which='LR')
    lam_max = np.real(lam_max)
    if beta >= lam_max*vol:
        return []
    w, V = scipy.linalg.eig(Lbar, b=Qbar - (beta/vol)*np.eye(n))
    mask = w>0
    V = V[:, mask]*np.sqrt(vol)
    mags = np.diag(V.T@Lbar@V)
    idx = np.argpartition(mags, range(n_clust))
    idx = idx[:n_clust-1]
    v_star = V[:,idx]
    u_star = (v_star.T*Dneg12).T
    if n_clust>2:
        kmeans = sklearn.cluster.KMeans(n_clusters=n_clust) #return this for out of sample pred
        kmeans.fit(u_star)
        labels = kmeans.labels_
    else:
        kmeans = sklearn.cluster.KMeans(n_clusters=n_clust) #return this for out of sample pred
        kmeans.fit(u_star)
        labels = np.sign(u_star)  
    return u_star, labels, kmeans 



def NPMIX(X, M=2, M_over=2):
    probs = np.zeros((X.shape[0],M))
    #overfit a GMM
    gmm = GaussianMixture(n_components=M_over, random_state=0, covariance_type='diag')
    gmm_labs = gmm.fit_predict(X)
    gmm_probs = gmm.predict_proba(X)
    #do single linkage agglomerative clustering on the obtained means
    ac = AgglomerativeClustering(n_clusters=M, linkage='single')
    links = ac.fit_predict(gmm.means_)
    for i in range(M):
        probs[:,i] = np.sum(gmm_probs[:,links==i], axis=1)
    
    #return labels, overfitterd gmm, and assignments of overfitted components
    labels = np.argmax(probs,axis=1)
    ofit_gmm = gmm
    gmm_assn = links
    
    #overall class probabilities can be obtained by
    #for i in range(M):
    #    probs[:,i] = np.sum(ofit_gmm.predict_proba(X)[:,gmm_assn==i], axis=1)
    
    return labels, ofit_gmm, gmm_assn


"""
implementation of the kernel spectral algorithm from
Song, Le, Animashree Anandkumar, Bo Dai, and Bo Xie. 
"Nonparametric estimation of multi-view latent variable models." 
In International Conference on Machine Learning, pp. 640-648. 2014.
"""

def MVLVM(X1,X2,X3,k=2,sigma=0.2, n_iter=100, reg=0):
    rank = k
    Xa = np.vstack((X1,X2))
    Xb = np.vstack((X2,X1))
    K = gauss_kernal_mat(Xa,Xa,sigma) # [1st elmts of pair, 2nd elmts of pair]
    K = 0.5*(K + K.T) + reg*np.eye(Xa.shape[0])
    L = gauss_kernal_mat(Xb,Xb,sigma)  # [2nd elmts of pair, 1st elmts of pair]
    L = 0.5*(L + L.T) + reg*np.eye(Xa.shape[0])
    n = K.shape[0]//2
    R = scipy.linalg.cholesky(K) ### todo: why is K not full rank??
    R = R.T #account for their convention
    s, beta_tilde = scipy.sparse.linalg.eigs((1/(4*n**2))*R@L@R.T, k=rank)
    s = np.real(s)
    beta_tilde = np.real(beta_tilde)
    S12 = np.diag(s**0.5)
    Sn12 = np.diag(s**-0.5)
    beta = pinv(R)@beta_tilde
    
    #form the tensor
    z1 = (Sn12@beta.T)@K[:,:n]
    z2 = (Sn12@beta.T)@K[:,n:]
    z3 = (Sn12@beta.T)@gauss_kernal_mat(Xa,X3,sigma)
    weights = np.ones((n,))
    factors = [z3, z1, z2] #different order to make tensor power method
    T = tl.kruskal_to_tensor((weights, factors))
    T = (1/(3*n))*tl.to_numpy(T)
    
    #tensor power method
    M = np.zeros((rank,rank))
    lams = np.zeros((rank,))
    for j in range(rank):
        v = np.random.randn(rank,)
        v = v / np.inner(v,v)**0.5
        vold = v
        for i in range(n_iter):
            v = np.dot(T,v)@v
            lam = np.inner(v,v)**0.5
            v = v / lam
#             print(np.linalg.norm(v-vold))
            vold = v
        M[:,j] = v
        lams[j] = lam
        ws = np.ones((1,))
        facs = [v,v,v]
        V = np.einsum("i,j,k ->ijk", v,v,v)
        T = T - lam*V
    A = beta@S12@M@np.diag(lams)
    pi = lams**-2

    return A, pi

"""
def gkern(X, sigma):
    pairwise_dists = squareform(pdist(X, 'euclidean'))
    K = scipy.exp(-pairwise_dists ** 2 / sigma ** 2)
    return K
"""
