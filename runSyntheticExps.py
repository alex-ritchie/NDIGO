#!/usr/bin/env python

import argparse
import numpy as np
import sklearn.mixture
import scipy
from sklearn.preprocessing import KernelCenterer
import itertools
from sklearn import datasets, metrics
from sklearn.mixture import GaussianMixture
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans

from sample_distributions import sample_intersecting_moons, sample_nonintersecting_moons
from sample_distributions import sample_olympic_rings, sample_half_disks
from clustNP import clustNP, gauss_kernal_mat, gen_ZG, gen_C, clustNP_obj, proj_simplex
from clustering_methods import constr_spec_clust, NPMIX, MVLVM
from constr_gmm import constr_gmm

def main(dataset, n, propHold, propTest, nRuns, M, outOfSample=False):
    np.random.seed(0)
    nhold = int(n*propHold)
    ntest = int(n*propTest)
    
    NDIGO_aris = []
    NDIGO_bcas = []
    CSC_aris = []
    CSC_bcas = []
    PCKM_aris = []
    PCKM_bcas = []
    constr_GMM_aris = []
    constr_GMM_bcas = []
    NPMIX_aris = []
    NPMIX_bcas = []
    MVLVM_aris = []
    MVLVM_bcas = []
    
    for run in range(nRuns):
        print("run number: %i"%(run+1))
        
        #generate data for this run
        if dataset in 'moons':
            print("Testing moons...")
            Rs = [10,20,30,40,50]
            noise_sd = 0.07
            X, component_ids, pair_ids = sample_nonintersecting_moons(n, noise_sd=noise_sd)
            Xh, cid_h, pid_h = sample_nonintersecting_moons(nhold, noise_sd=noise_sd)
            Xte, cid_te, pid_te = sample_nonintersecting_moons(ntest, noise_sd=noise_sd)
            X1 = np.vstack((X[:n//3,:], X[n:n+n//3,:])) 
            cid1 = np.hstack((component_ids[:n//3],component_ids[n:n+n//3]));
            X2 = np.vstack((X[n//3:2*(n//3),:], X[n+(n//3):n+2*(n//3),:])); 
            cid2 = np.hstack((component_ids[n//3:2*(n//3)],component_ids[n+(n//3):n+2*(n//3)]));
            X3 = np.vstack((X[2*(n//3):3*(n//3),:], X[n+2*(n//3):n+3*(n//3),:])); 
            cid3 = np.hstack((component_ids[2*(n//3):3*(n//3)],component_ids[n+2*(n//3):n+3*(n//3)]));
            ss = 2
            mom = 0.2
            decay = 0.1
        elif dataset in 'overlapping_moons':
            print("Testing overlapping moons...")
            Rs = [10,20,30,40,50]
            noise_sd = 0.07
            X, component_ids, pair_ids = sample_intersecting_moons(n, noise_sd=noise_sd, rot_rad = np.pi/2)
            Xh, cid_h, pid_h = sample_intersecting_moons(nhold, noise_sd=noise_sd, rot_rad = np.pi/2)
            Xte, cid_te, pid_te = sample_intersecting_moons(ntest, noise_sd=noise_sd, rot_rad = np.pi/2)
            X1 = np.vstack((X[:n//3,:], X[n:n+n//3,:])) 
            cid1 = np.hstack((component_ids[:n//3],component_ids[n:n+n//3]));
            X2 = np.vstack((X[n//3:2*(n//3),:], X[n+(n//3):n+2*(n//3),:])); 
            cid2 = np.hstack((component_ids[n//3:2*(n//3)],component_ids[n+(n//3):n+2*(n//3)]));
            X3 = np.vstack((X[2*(n//3):3*(n//3),:], X[n+2*(n//3):n+3*(n//3),:])); 
            cid3 = np.hstack((component_ids[2*(n//3):3*(n//3)],component_ids[n+2*(n//3):n+3*(n//3)]));
            ss = 1
            mom = 0.1
            decay = 0.005
        elif dataset in 'half_disks':
            print("Testing half disks...")
            Rs = [50,60,70,80,90,100]
            X, component_ids, pair_ids = sample_half_disks(n)
            Xh, cid_h, pid_h = sample_half_disks(nhold)
            Xte, cid_te, pid_te = sample_half_disks(ntest)
            X1 = np.vstack((X[:n//3,:], X[n:n+n//3,:])) 
            cid1 = np.hstack((component_ids[:n//3],component_ids[n:n+n//3]));
            X2 = np.vstack((X[n//3:2*(n//3),:], X[n+(n//3):n+2*(n//3),:])); 
            cid2 = np.hstack((component_ids[n//3:2*(n//3)],component_ids[n+(n//3):n+2*(n//3)]));
            X3 = np.vstack((X[2*(n//3):3*(n//3),:], X[n+2*(n//3):n+3*(n//3),:])); 
            cid3 = np.hstack((component_ids[2*(n//3):3*(n//3)],component_ids[n+2*(n//3):n+3*(n//3)]));
            ss = 3
            mom = 0.3
            decay = 0.01
        elif dataset in 'rings':
            print("Testing rings...")
            Rs = [50,60,70,80,90,100]
            noise_sd = 0.1
            X, component_ids, pair_ids = sample_olympic_rings(n, noise_sd=noise_sd)
            Xh, cid_h, pid_h = sample_olympic_rings(nhold, noise_sd=noise_sd)
            Xte, cid_te, pid_te = sample_olympic_rings(ntest, noise_sd=noise_sd)
            X1 = np.vstack((X[:n//3,:], X[n:n+n//3,:])) 
            cid1 = np.hstack((component_ids[:n//3],component_ids[n:n+n//3]));
            X2 = np.vstack((X[n//3:2*(n//3),:], X[n+(n//3):n+2*(n//3),:])); 
            cid2 = np.hstack((component_ids[n//3:2*(n//3)],component_ids[n+(n//3):n+2*(n//3)]));
            X3 = np.vstack((X[2*(n//3):3*(n//3),:], X[n+2*(n//3):n+3*(n//3),:])); 
            cid3 = np.hstack((component_ids[2*(n//3):3*(n//3)],component_ids[n+2*(n//3):n+3*(n//3)]));
            ss = 10
            mom = 0.1
            decay = 0.01
        else:
            print("invalid dataset selection")
            return None
        d = X.shape[1]
        
        #NDIGO
        
        #initialize
        sd = np.min(np.std(X, axis=0))
        best_ISE = 1e10
        for R in Rs:
            sigma = sd*(2*n)**(-1/(d+4))
            Z, G = gen_ZG(X, R, sigma)
            C = gen_C(X, pair_ids, Z, R, sigma)
            w, A  = scipy.sparse.linalg.eigs(G, k=M)
            w = np.real(w)
            A = np.real(A)
            for i in range(M):
                A[:,i] = proj_simplex(A[:,i])
            w = w/np.sum(w)
            A = A / np.sum(A,axis=0)
            f, _ = clustNP_obj(A, w, G, C, n, 0)
            if f < best_ISE:
                best_ISE = f
                best_sigma = sigma
                best_R = R
                best_w = w
                best_Z = Z
                best_C = C
                best_G = G
                best_A = A
        #solve
        sigma = best_sigma
        f_star, A_star, w_star, Z, n_iter = clustNP(X, pair_ids, best_A, best_w, best_Z, best_G, stepsize=ss, ss_decr=1, 
                                            epoch_decr = 1 ,method='psgd',max_iter=200, f_tol=1e-16, 
                                            grad_tol=1e-8, R=R, sigma=sigma, batch_size=64, backtrack=False,
                                            decay=decay, momentum=mom)
        print(best_R, best_sigma, f_star)
        #test with no pairwise info
        if outOfSample: 
            phats_te = np.zeros((Xte.shape[0], M))
            for i in range(Xte.shape[0]):
                kxte = gauss_kernal_mat(Xte[i,np.newaxis], best_Z, best_sigma)
                phats_te[i, :] = kxte@A_star
            lte = w_star.T*phats_te
            decisions_te = np.argmax(lte, 1)
            #compute best accuracy of clustering
            cm = sklearn.metrics.confusion_matrix(cid_te, decisions_te, normalize='all')
            perms = np.array(tuple((itertools.permutations(range(cm.shape[0])))))
            acc = 0
            for perm in range(perms.shape[0]):
                tr = np.trace(cm[perms[perm,:],:])
                if tr > acc:
                    acc = tr
            NDIGO_bcas.append(acc)
            #compute ARI
            NDIGO_aris.append(sklearn.metrics.adjusted_rand_score(cid_te, decisions_te))
        else:
            phats = np.zeros((X.shape[0], M))
            for i in range(X.shape[0]):
                kx = gauss_kernal_mat(X[i,np.newaxis], best_Z, best_sigma)
                phats[i, :] = kx@A_star
            ltr = w_star.T*phats
            for i in set(pair_ids):
                ltr[pair_ids==i, :] = np.tile(w_star.T*np.prod(phats[pair_ids==i,np.newaxis], axis=0), (2,1))
            labels = np.argmax(ltr, 1)
            #compute best accuracy of clustering
            cm = sklearn.metrics.confusion_matrix(component_ids, labels, normalize='all')
            perms = np.array(tuple((itertools.permutations(range(cm.shape[0])))))
            acc = 0
            for perm in range(perms.shape[0]):
                tr = np.trace(cm[perms[perm,:],:])
                if tr > acc:
                    acc = tr
            NDIGO_bcas.append(acc)
            #compute ARI
            ari = sklearn.metrics.adjusted_rand_score(component_ids,labels)
            print(ari) 
            NDIGO_aris.append(ari)
            
         
        # CSC
        Aff = gauss_kernal_mat(X,X,best_sigma)
        connectivity = np.zeros((X.shape[0], X.shape[0]))
        for i in range(n):
            try:
                a, b = tuple(np.nonzero(pair_ids==i)[0])
                connectivity[a,b] = 1
                connectivity[b,a] = 1
            except:
                print(i, X.shape)
        best_ari_h = -2
        best_ari_te = -2
        for beta in [1, 10, 20, 30, 40, 50]:
            u_star, labels, spclust = constr_spec_clust(Aff, connectivity, beta=beta, n_clust=M)
            Kh = gauss_kernal_mat(Xh, X, best_sigma)
            emb_h = Kh@u_star
            labels_h = spclust.predict(emb_h)
            ari_h = sklearn.metrics.adjusted_rand_score(cid_h, labels_h)
            if ari_h > best_ari_h:
                best_labels = labels.flatten()
                best_ari = sklearn.metrics.adjusted_rand_score(component_ids, best_labels)
                Kte = gauss_kernal_mat(Xte, X, best_sigma)
                emb_te = Kte@u_star
                labels_te = spclust.predict(emb_te)
                ari_te = sklearn.metrics.adjusted_rand_score(cid_te, labels_te)
                best_ari_te = ari_te
                best_labels_te = labels_te
        # compute bac, ari
        if outOfSample:
            cm = sklearn.metrics.confusion_matrix(cid_te, best_labels_te, normalize='all')
            perms = np.array(tuple((itertools.permutations(range(cm.shape[0])))))
            acc = 0
            for perm in range(perms.shape[0]):
                tr = np.trace(cm[perms[perm,:],:])
                if tr > acc:
                    acc = tr
            CSC_bcas.append(acc)
            CSC_aris.append(best_ari_te)
        else: 
            cm = sklearn.metrics.confusion_matrix(component_ids, best_labels, normalize='all')
            perms = np.array(tuple((itertools.permutations(range(cm.shape[0])))))
            acc = 0
            for perm in range(perms.shape[0]):
                tr = np.trace(cm[perms[perm,:],:])
                if tr > acc:
                    acc = tr
            CSC_bcas.append(acc)
            CSC_aris.append(best_ari)
        """
        # PCK-means
        pck = PCKMeans(n_clusters=M)
        # generate list of tuples for pairwise constraints
        ml = [tuple(np.nonzero(pair_ids==i)[0]) for i in range(X.shape[0]//2)]
        pck.fit(X, ml=ml)
        centers = pck.cluster_centers_
        labels_te = np.argmax(Xte@centers.T, axis=1)
        # compute bac, ari
        if outOfSample:
            cm = sklearn.metrics.confusion_matrix(cid_te, labels_te, normalize='all')
            perms = np.array(tuple((itertools.permutations(range(cm.shape[0])))))
            acc = 0
            for perm in range(perms.shape[0]):
                tr = np.trace(cm[perms[perm,:],:])
                if tr > acc:
                    acc = tr
            PCKM_bcas.append(acc)
            PCKM_aris.append(sklearn.metrics.adjusted_rand_score(cid_te, labels_te))
        else:
            cm = sklearn.metrics.confusion_matrix(component_ids, pck.labels_, normalize='all')
            perms = np.array(tuple((itertools.permutations(range(cm.shape[0])))))
            acc = 0
            for perm in range(perms.shape[0]):
                tr = np.trace(cm[perms[perm,:],:])
                if tr > acc:
                    acc = tr
            PCKM_bcas.append(acc)
            PCKM_aris.append(sklearn.metrics.adjusted_rand_score(component_ids, pck.labels_))
        """    

        # NPMIX
        probs = np.zeros((X.shape[0],M))
        probs_h = np.zeros((Xh.shape[0],M))
        probs_te = np.zeros((Xte.shape[0],M))
        ofit_levels = range(2,11,1)
        best_ari_h = -2
        for level in ofit_levels:
            try:
                labels, ofit_gmm, gmm_assn = NPMIX(X, M=M, M_over=level*M)
                #issue with logsumexp in sklearn gaussianmixture
            except:
                continue
            for i in range(M):
                probs[:,i] = np.sum(ofit_gmm.predict_proba(X)[:,gmm_assn==i], axis=1)
            labels = np.argmax(probs_h, axis=1)
            for i in range(M):
                probs_h[:,i] = np.sum(ofit_gmm.predict_proba(Xh)[:,gmm_assn==i], axis=1)
            labels_h = np.argmax(probs_h, axis=1)
            ari_h = sklearn.metrics.adjusted_rand_score(cid_h, labels_h)
            if ari_h > best_ari_h:
                best_labels = np.argmax(probs,axis=1)
                best_ari = sklearn.metrics.adjusted_rand_score(component_ids, best_labels)
                for i in range(M):
                    probs_te[:,i] = np.sum(ofit_gmm.predict_proba(Xte)[:,gmm_assn==i], axis=1)
                best_labels_te = np.argmax(probs_te, axis=1)
                best_ari_te = sklearn.metrics.adjusted_rand_score(cid_te, best_labels_te)
        # compute bac, ari
        if outOfSample:
            cm = sklearn.metrics.confusion_matrix(cid_te, best_labels_te, normalize='all')
            perms = np.array(tuple((itertools.permutations(range(cm.shape[0])))))
            acc = 0
            for perm in range(perms.shape[0]):
                tr = np.trace(cm[perms[perm,:],:])
                if tr > acc:
                    acc = tr
            NPMIX_bcas.append(acc)
            NPMIX_aris.append(best_ari_te)
        else: 
            cm = sklearn.metrics.confusion_matrix(component_ids, best_labels, normalize='all')
            perms = np.array(tuple((itertools.permutations(range(cm.shape[0])))))
            acc = 0
            for perm in range(perms.shape[0]):
                tr = np.trace(cm[perms[perm,:],:])
                if tr > acc:
                    acc = tr
            NPMIX_bcas.append(acc)
            NPMIX_aris.append(best_ari)
        

        #constrained-GMM
    
        cgmm = constr_gmm(X,pair_ids,M,100,0)     
        cgmm.run()
        labels= np.argmax(cgmm.predict(X), axis=1)
        labels_te = np.argmax(cgmm.predict(Xte), axis=1)
        # compute bac, ari
        if outOfSample:
            cm = sklearn.metrics.confusion_matrix(cid_te, labels_te, normalize='all')
            perms = np.array(tuple((itertools.permutations(range(cm.shape[0])))))
            acc = 0
            for perm in range(perms.shape[0]):
                tr = np.trace(cm[perms[perm,:],:])
                if tr > acc:
                    acc = tr
            constr_GMM_bcas.append(acc)
            constr_GMM_aris.append(sklearn.metrics.adjusted_rand_score(cid_te, labels_te))
        else:
            cm = sklearn.metrics.confusion_matrix(component_ids, labels, normalize='all')
            perms = np.array(tuple((itertools.permutations(range(cm.shape[0])))))
            acc = 0
            for perm in range(perms.shape[0]):
                tr = np.trace(cm[perms[perm,:],:])
                if tr > acc:
                    acc = tr
            constr_GMM_bcas.append(acc)
            constr_GMM_aris.append(sklearn.metrics.adjusted_rand_score(component_ids, labels))
   
        # MVLVM
        Xa = np.vstack((X1,X2))
        Xc = np.vstack((X1,X2,X3))
        probs = np.zeros((Xc.shape[0],M))
        probs_te = np.zeros((Xte.shape[0],M))
        sd = np.min(np.std(Xa, axis=0))
        sigma = sd*(2*n)**(-1/(d+4))
        try:
            A, w = MVLVM(X1,X2,X3,k=M,sigma=sigma, n_iter=50)
        except:
            A, w = MVLVM(X1,X2,X3,k=M,sigma=sigma, n_iter=50, reg=0.01*sigma)
        if outOfSample:
            for i in range(M):
                kde = sklearn.neighbors.KernelDensity(bandwidth=sigma);
                kde.fit(Xa, y=None, sample_weight=(A[:,i]).clip(1e-16))
                probs_te[:,i] = kde.score_samples(Xte).T
            labels = np.argmax(probs_te,axis=1)
            cm = sklearn.metrics.confusion_matrix(cid_te, labels, normalize='all')
            perms = np.array(tuple((itertools.permutations(range(cm.shape[0])))))
            acc = 0
            for perm in range(perms.shape[0]):
                tr = np.trace(cm[perms[perm,:],:])
                if tr > acc:
                    acc = tr
            MVLVM_bcas.append(acc)
            MVLVM_aris.append(sklearn.metrics.adjusted_rand_score(cid_te, labels))

        else:
            for i in range(M):
                kde = sklearn.neighbors.KernelDensity(bandwidth=sigma);
                kde.fit(Xa, y=None, sample_weight=(A[:,i]).clip(1e-16))
                probs[:,i] = kde.score_samples(Xc).T
            labels = np.argmax(probs,axis=1)
            cm = sklearn.metrics.confusion_matrix(np.hstack((cid1,cid2,cid3)), labels, normalize='all')
            perms = np.array(tuple((itertools.permutations(range(cm.shape[0])))))
            acc = 0
            for perm in range(perms.shape[0]):
                tr = np.trace(cm[perms[perm,:],:])
                if tr > acc:
                    acc = tr
            MVLVM_bcas.append(acc)
            MVLVM_aris.append(sklearn.metrics.adjusted_rand_score(np.hstack((cid1,cid2,cid3)), labels))
    
    NDIGO_res = "NDIGO: BAC = %.3f \pm %.3f; ARI = %.3f \pm %.3f" %(np.mean(NDIGO_bcas), np.std(NDIGO_bcas),
                                                                    np.mean(NDIGO_aris), np.std(NDIGO_aris))
    
    CSC_res = "CSC: BAC = %.3f \pm %.3f; ARI = %.3f \pm %.3f" %(np.mean(CSC_bcas), 
                                                                np.std(CSC_bcas),
                                                                np.mean(CSC_aris),
                                                                np.std(CSC_aris))
    """                                                            
    PCKM_res = "PCK-means: BAC = %.3f \pm %.3f; ARI = %.3f \pm %.3f" %(np.mean(PCKM_bcas), 
                                                                       np.std(PCKM_bcas),
                                                                       np.mean(PCKM_aris),
                                                                       np.std(PCKM_aris))
    """
    NPMIX_res = "NPMIX: BAC = %.3f \pm %.3f; ARI = %.3f \pm %.3f" %(np.mean(NPMIX_bcas), 
                                                                    np.std(NPMIX_bcas),
                                                                    np.mean(NPMIX_aris),
                                                                    np.std(NPMIX_aris))
    constr_GMM_res = "constrained-GMM: BAC = %.3f \pm %.3f; ARI = %.3f \pm %.3f"%(np.mean(constr_GMM_bcas), 
                                                                                  np.std(constr_GMM_bcas),
                                                                                  np.mean(constr_GMM_aris),
                                                                                  np.std(constr_GMM_aris))
    
    MVLVM_res = "MVLVM: BAC = %.3f \pm %.3f; ARI = %.3f \pm %.3f"%(np.mean(MVLVM_bcas), 
                                                                   np.std(MVLVM_bcas),
                                                                   np.mean(MVLVM_aris),
                                                                   np.std(MVLVM_aris))
    print("Results: \n", NDIGO_res, "\n", CSC_res, "\n", #PCKM_res, "\n",
            NPMIX_res, "\n", constr_GMM_res, "\n",  MVLVM_res, "\n")
    
    
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
        action   = 'store',
        required = True,
        type     = str,
        choices  = ['moons', 'overlapping_moons', 'rings', 'half_disks'],
        default = 'moons',
        help     = 'dataset')

    parser.add_argument('--n',
        action   = 'store',
        required = False,
        type     = int,
        default  = 500,
        help     = 'number of data pairs')

    parser.add_argument('--propHold',
        action   = 'store',
        required = False,
        type     = float,
        default  = 0.2,
        help     = """proportion of training data to use for Holdout. Generated separately does not / 
                        decrease trainng data. Not used by all methods.""")

    parser.add_argument('--propTest',
        action   = 'store',
        required = False,
        type     = float,
        default  = 0.2,
        help     = 'proportion of training data to use for Test. Generated separately does not decrease training data.')
    
    parser.add_argument('--nRuns',
        action   = 'store',
        required = False,
        type     = int,
        default  = 20,
        help     = 'number of tests to average over.')
    
    parser.add_argument('--M',
        action   = 'store',
        required = False,
        type     = int,
        default  = 2,
        help     = 'number of tests to average over.')
    
    parser.add_argument('--outOfSample',
        action   = 'store',
        required = False,
        type     = bool,
        default  = False,
        help     = 'return out of sample results.')

    args = parser.parse_args()
    main(args.dataset, args.n, args.propHold, args.propTest, args.nRuns, args.M, args.outOfSample) 
