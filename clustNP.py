#!/usr/bin/env python

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.sparse.linalg import svds
import scipy.optimize as sopt
from copy import copy

def clustNP(X, pair_ids, A0, w0, Z, G, Ci=None, stepsize=1, ss_decr=1, epoch_decr = 20, method='pgd', 
            max_iter=100, f_tol=1e-10, grad_tol=1e-5, R=10, sigma=1.0, 
            batch_size=16, backtrack=True, decay=0, momentum=0, large=False):
    # A0 (M,K): matrix with initial alpha_i as rows
    # w0 (M,): vector of initial mixture proportions
    
    #define some necessary variables
    fs = np.ones(5,)*np.inf; #pad 
    c1 = 0.001
    n = X.shape[0] // 2
    A = copy(A0)
    w = copy(w0)
    R, M = A0.shape
    not_converged = True
    n_iter = 0
    epoch = 1
    if not large:
        # then precompute the Ci for speed
        if Ci is None:
            Ci = gen_Ci(X, pair_ids, Z, R, sigma)
        Cfull = np.sum(Ci, axis=2)
    delta_w = 0
    delta_A = 0
    order = np.random.permutation(n)
    f_star = 0 
    A_star = 0 
    w_star = 0
    n_iter = 0 
    #generate G, C, Z
    
    if method in 'psgd':
        #solve with projected stochastic gradient descent
        
        #generate the full training sequence now
        f_prev = 1e10 #dummy previous function value
        batch_num = 0
        while not_converged:
            #create batch indices for paired examples
            batch_num += 1
            pairs = order[(batch_num-1)*batch_size:min(batch_num*batch_size, n)]
            
            #pairs stored next to each other, faster than list comprehension
            idxs = 2*(np.maximum(pairs-1, 0))
            temp = np.append(np.insert(idxs, slice(1, None), 0), 0)
            temp[1::2] = idxs+1
            idxs = temp
            if not large:
                C = np.sum(Ci[:,:,idxs], axis=2)
            else:
                C = gen_C(X[idxs,:], pair_ids[idxs], Z, R, sigma)
                if np.mod(batch_num, 500) == 0:
                    print("processed observations this epoch: ", batch_num*batch_size)
            for i in range(2):
                #calculate objective and correct partial subgradient
                _, df = clustNP_obj(A, w, G, C, batch_size, i)
                if i==0:
                    #update w
                    if n_iter>1 and momentum>0:
                        delta_w = -(1-momentum)*(stepsize/(1+decay*n_iter))*df + momentum*delta_w
                        w = proj_simplex(w + delta_w);
                else:
                    #update alphas
                    delta_A = -(1-momentum)*(stepsize/(1+decay*n_iter))*df + momentum*delta_A
                    for i in range(M):
                        A[:,i] = proj_simplex(A[:,i] + delta_A[:,i])        
            n_iter += 1

            #print details every epochs
            if batch_size*batch_num >= n:
                #reset batches for next epoch
                batch_num = 0
                order = np.random.permutation(n)
                if not large:
                    f, _ = clustNP_obj(A, w, G, Cfull, n, 0)
                else:
                    f, _ = clustNP_obj(A, w, G, C, n, 0)
                    print(f)
                epoch += 1
                fs = np.append(fs,f)
                #store best params so far
                if f < np.min(fs[:-1]):
                    f_star = f
                    A_star = A
                    w_star = w
                # reduce stepsize if objective goes up by 1% or more
                if  n_iter>10 and np.mean(fs[-1:-11:-1]) >= np.min(fs[:-1]):
                    stepsize = stepsize*ss_decr
                #only end upon finishing an epoch
                if ( np.abs(f_prev-f)<f_tol or epoch > max_iter):
                    #stop iterating
                    not_converged = False
                else:
                    #do another round of updates with decreasing stepsize
                    f_prev = f

    elif method in 'pgd':
        print('using pgd')
        #solve with projected gradient descent
        G, C, _ = gen_mats_clustNP(X, pair_ids, R, sigma, Z)

        f_prev = 1e10 #dummy previous function value
        while not_converged:
            for i in range(2):
                #calculate objective and correct partial subgradient
                _, df = clustNP_obj(A, w, G, C, n, i)
                if i==0:
                    #update w
                    if n_iter>1 and momentum>0:
                        delta_w = -(1-momentum)*(stepsize/(1+decay*n_iter))*df + momentum*delta_w
                        w = proj_simplex(w + delta_w);
                else:
                    #update alphas
                    delta_A = -(1-momentum)*(stepsize/(1+decay*n_iter))*df + momentum*delta_A
                    for i in range(M):
                        A[:,i] = proj_simplex(A[:,i] + delta_A[:,i])        
            n_iter += 1

            #print details every few iterations
            if np.mod(n_iter, 100) == 0:
                #reset batches for next epoch
                f, _ = clustNP_obj(A, w, G, C, n, 0)
                print(f, (stepsize/(1+decay*n_iter)) )
                epoch += 1
                fs = np.append(fs,f)
                #store best params so far
                if f < np.min(fs[:-1]):
                    f_star = f
                    A_star = A
                    w_star = w
                if ( n_iter>10 and np.mean(fs[-1:-11:-1])-f<f_tol) or n_iter > max_iter:
                    #stop iterating
                    not_converged = False
                else:
                    #do another round of updates with decreasing stepsize
                    f_prev = f

    return f_star, A_star, w_star, Z, n_iter        
 
    
def gen_Z(X, R, sigma, method='k-means', random_state=0):
    if method is 'k-centers':
        Z = k_centers(X, R)
    else:
        kmeans = KMeans(n_clusters=R, n_init=100, random_state=random_state).fit(X)
        Z = kmeans.cluster_centers_
    return Z
    
def gen_ZG(X, R, sigma, Z=None, method='k-means', random_state=0):
    if Z is None:
        if R == X.shape[0]:
            Z = X
        else:
#             if method in 'k-centers':
#                 Z = k_centers(X, R)
#             else:
            if X.shape[0]>5000:
                kmeans = MiniBatchKMeans(n_clusters=R, random_state=random_state).fit(X)
                Z = kmeans.cluster_centers_
            else:
                kmeans = KMeans(n_clusters=R, n_init=100, random_state=random_state).fit(X)
                Z = kmeans.cluster_centers_
    G = gauss_kernal_mat(Z, Z, sigma=(2**0.5)*sigma);
    return Z, G

def k_centers(X, R, random_state=0):
    np.random.seed(random_state)
    n = X.shape[0]//2
    try:
        d = X.shape[1]
    except:
        #in case X is just a vector with no second dimension
        d = 1
    print (R,d)
    Z = np.zeros((R,d))
    idx1 = np.random.randint(0, 2*n)
    Z[0,:] = X[idx1,:]
    d = np.zeros((2*n,R))
    d[:,0] = np.linalg.norm(X - Z[0,:], axis=1)
    for i in range(R-1):
        idx = np.argmax(np.sum(d, axis=1).flatten())
        Z[i+1,:] = X[idx,:]
        d[:,i+1] = np.linalg.norm((X - Z[i+1,:]), axis=1)
    return Z
        

def gen_C(X, pair_ids, Z, R, sigma):
    C = np.zeros((R, R));
    for i in  set(pair_ids):
        idx = pair_ids==i
        Xpair = X[idx,:];
        K = gauss_kernal_mat(Xpair, Z, sigma)
        C += np.outer(K[0,:],K[1,:]);
    return C

def gen_Ci(X, pair_ids, Z, R, sigma):
    Ci = np.zeros((R, R, X.shape[0]));
    for i in  set(pair_ids):
        idx = pair_ids==i
        Xpair = X[idx,:];
        K = gauss_kernal_mat(Xpair, Z, sigma)
        Ci[:,:,i] += np.outer(K[0,:],K[1,:]);
    return Ci

def gen_mats_clustNP(X, pair_ids, R, sigma, Z):
    G = gauss_kernal_mat(Z, Z, sigma=(2**0.5)*sigma);
    C = np.zeros((R, R));
    for i in  set(pair_ids):
        idx = pair_ids==i
        KernMat = gauss_kernal_mat(X[idx,], Z, sigma)
        C += KernMat.T@KernMat;
    return G, C, Z

def clustNP_obj(A, w, G, C, n, deriv_idx):
    M, K = A.shape
    #calculate Gprime and Cprime
    Gprime = (A.T @ G @ A) ** 2
    Cprime = np.diag(A.T@C@A)
    wwt = np.outer(w,w)

    f = w.T@Gprime@w - (2/n)*np.inner(Cprime,w)

    #calculate specified derivative
    if deriv_idx == 0:
        #calculate partial w.r.t w
        partial_f = Gprime@w + np.diag(Gprime)*w - (2/n)*Cprime
    else:
        #calculate partial w.r.t all alpha_i
        GA = G@A
        AdiagW = A*w
        partial_f = 4*( GA@(AdiagW.T@(GA*w)) - (C@AdiagW)/n )
    return f, partial_f


def proj_simplex(v, z=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert z > 0, "Radius s must be strictly positive (%d <= 0)"
    n = v.shape[0]
    # check if we are already on the simplex
    if v.sum() == z and np.alltrue(v >= 0):
        return v
    
    U = np.arange(n)
    s=0
    p=0
    while U.shape[0] > 0:
        #print(p)
        # says pick k at random but just take k=0
        # and go through vector sequentially
        k = U[0]
        vk = v[k]
        G = np.where(v>=vk)[0]
        G = np.intersect1d(G,U)
        L = np.where(v<vk)[0]
        L = np.intersect1d(L,U)
        dp = G.shape[0]
        ds = np.sum(v[G])
        snew = s+ds
        pnew = p+dp
        if snew - pnew*vk < z:
            s = snew
            p = pnew
            U = L
        else:
            U = G[1:]
    theta = (s-z)/p
    return (v-theta).clip(0)

def gauss_kernal_mat(x1, x2, sigma=1):
    xp1 = np.sum(x1**2,axis=1);
    xp2 = np.sum(x2**2,axis=1);
    if (len(x1.shape) < 2) or (len(x2.shape)) < 2:
        Rn_Kernel = (-2*np.outer(x1,x2).T + xp1).T + xp2
    else:
        Rn_Kernel = (-2*(x1@x2.T).T + xp1).T + xp2
    RBF_Kernel = np.exp(-Rn_Kernel/(2*sigma**2))

    return RBF_Kernel





