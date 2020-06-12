import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

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