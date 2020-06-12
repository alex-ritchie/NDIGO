import numpy as np
from sklearn.datasets import make_moons, make_circles
from copy import copy

def pair_sample_mixture_of_GMs(mus, sigmas, w_outer, w_inner, n=100):
    #mus is array of means (Mprime, d, M)
    #sigmas is array of covariance matrices (Mprime, d , d, M)
    #w_outer is array of mixture proportions (M,) must sum to one
    #w_inner is array of mixture proportions (M, Mprime) must sum to one
    #n is number of samples desired
    #
    # output: samples is an array of samples (n, d)

    #make sure ws sum to one, and sample categorical variables from w_outer
    cond1 = ( ((np.sum(w_outer)-1)**2>1e-8) or (w_outer<0).any())
    cond2 = ( ((np.sum(w_inner, axis=1)-1)**2>1e-8).any() or (w_inner<0).any())
    #TODO: stopped here writin this function
    if cond1 or cond2:
        print('error: w_outer and/or w_inner does not sum to one or has negative entries')
        return
    component_ids = np.random.choice(range(w_outer.shape[0]), size=(n,), replace=True, p=w_outer)
    samples = [ sample_GM(mus[:,:,c], sigmas[:,:,:,c], w_inner[c,:], 2) for c in component_ids]
    samples = np.array(samples).flatten()
    pair_ids = [ [c for i in range(2)] for c in component_ids]
    component_ids = [each for (idx, pair) in enumerate(pair_ids) for each in pair]
    component_ids = np.array(component_ids).flatten()
    pair_ids = [idx  for (idx, pair) in enumerate(pair_ids) for each in pair]
    pair_ids = np.array(pair_ids).flatten()
    return samples[:,None], component_ids, pair_ids

def sample_mixture_of_GMs(mus, sigmas, w_outer, w_inner, n=100):
    #mus is array of means (Mprime, d, M)
    #sigmas is array of covariance matrices (Mprime, d , d, M)
    #w_outer is array of mixture proportions (M,) must sum to one
    #w_inner is array of mixture proportions (M, Mprime) must sum to one
    #n is number of samples desired
    #
    # output: samples is an array of samples (n, d)

    #make sure ws sum to one, and sample categorical variables from w_outer
    cond1 = ( ((np.sum(w_outer)-1)**2>1e-8) or (w_outer<0).any())
    cond2 = ( ((np.sum(w_inner, axis=1)-1)**2>1e-8).any() or (w_inner<0).any())
    #TODO: stopped here writin this function
    if cond1 or cond2:
        print('error: w_outer and/or w_inner does not sum to one or has negative entries')
        return
    component_ids = np.random.choice(range(w_outer.shape[0]), size=(n,), replace=True, p=w_outer)
    samples = [ sample_GM(mus[:,:,c], sigmas[:,:,:,c], w_inner[c,:], 1) for c in component_ids]
    samples = np.array(samples).flatten()
    return samples, component_ids

def sample_GM(mus, sigmas, w, n=100):
    #mus is array of means (Mprime, d)
    #sigmas is array of covariance matrices (Mprime, d , d)
    #w is array of mixture proportions (Mprime,) must sum to one
    #n is number of samples desired
    #
    # output: samples is an array of samples (n, d)

    #make sure w sums to one, and sample categorical variables from it

    if ( (((np.sum(w)-1)**2)>1e-8) or (w<0).any()):
        print(w)
        print('error: w does not sum to one or has negative entries')
        return
    component_ids = np.random.choice(range(w.shape[0]), size=(n,), replace=True, p=w)
    samples = [ np.random.multivariate_normal(mus[c,:], sigmas[c,:,:]) for c in component_ids]
    samples = np.array(samples).flatten()
    return samples

def sample_intersecting_moons(n = 100, noise_sd=0.1, rot_rad = np.pi/2, rand_state=None):
    #must be even to assign pairs
    np.random.seed(rand_state)
    n = n + (n % 2)
    samples, cid = make_moons(n_samples=2*n, shuffle=False, noise=noise_sd, random_state=rand_state)
    perm = np.append(np.random.permutation(n),np.random.permutation(n)+n) #preserves classes
    samples = samples[perm,:]
    cid = cid[perm]
    pid = np.array([ c for c in range(n) for i in range(2)])
    samples[cid==1,:] = samples[cid==1,:]@np.array(((np.cos(rot_rad),-np.sin(rot_rad)), 
                                                    (np.sin(rot_rad), np.cos(rot_rad))))
    samples[cid==1,:] = samples[cid==1,:] + np.mean(samples[cid==0,:],0) - np.mean(samples[cid==1,:],0)
    return samples, cid, pid

def sample_nonintersecting_moons(n = 100, noise_sd=0.1, rot_rad = 0, rand_state=None):
    #must be even to assign pairs
    samples, cid = make_moons(n_samples=2*n, shuffle=False, noise=noise_sd, random_state=rand_state)
    samples = samples[:2*n,:]; cid=cid[:2*n,:]
    perm = np.append(np.random.permutation(n),np.random.permutation(n)+n) #preserves classes
    samples = samples[perm,:]
    cid = cid[perm]
    pid = np.array([ c for c in range(n) for i in range(2)])
    return samples, cid, pid

def sample_olympic_rings(n = 100, noise_sd=0.1, rand_state=None):
    #must be even to assign pairs
    np.random.seed(rand_state)
    n = n + (n % 5)
    n = n//5
    samples = np.zeros((5*2*n,2))
    cids = np.zeros((5*2*n,))
    pids = np.zeros((5*2*n,))
    for i in range(5):
        s, c = make_circles(n_samples=4*n, shuffle=True, noise=noise_sd)#4n because we are not using inner circle
        s = s[c==0,:]
        samples[2*n*i:2*n*(i+1), :] = s
        cids[2*n*i:2*n*(i+1)] = i
        pids[2*n*i:2*n*(i+1)] = np.array([ c for c in np.linspace(n*i, n*(i+1)-1, num=n) for j in range(2)])
    samples[cids==1,0] = samples[cids==1,0] + 2.2
    samples[cids==2,0] = samples[cids==2,0] - 2.2
    samples[cids==3,0] = samples[cids==3,0] - 1.1
    samples[cids==3,1] = samples[cids==3,1] - 1.1
    samples[cids==4,0] = samples[cids==4,0] + 1.1
    samples[cids==4,1] = samples[cids==4,1] - 1.1
    return samples, cids.astype("int"), pids.astype("int")

def sample_half_disks(n = 100, n_disks=3, rand_state=None):
    #must be even to assign pairs
    np.random.seed(rand_state)
    n = n + (n % n_disks)
    n = n//n_disks
    rot_rad = 2*np.pi/n_disks
    samples = np.zeros((n_disks*2*n,2))
    cids = np.zeros((n_disks*2*n,))
    pids = np.zeros((n_disks*2*n,))
    for i in range(n_disks):
        s = np.random.rand(8*n,2) - 0.5 #center at  origin, oversample so there  are enough points in half  disk
        mask = (np.sum(s*s, axis=1)**0.5<=0.5)*(s[:,1]>=0)
        s = s[mask,:] #take top half of center disk
        s = s[range(2*n),:] #throw away extras
        samples[2*n*i:2*n*(i+1), :] = s@np.array(((np.cos(i*rot_rad),-np.sin(i*rot_rad)), 
                                                    (np.sin(i*rot_rad), np.cos(i*rot_rad))))
        cids[2*n*i:2*n*(i+1)] = i
        pids[2*n*i:2*n*(i+1)] = np.array([ c for c in np.linspace(n*i, n*(i+1)-1, num=n) for j in range(2)])
    return samples, cids.astype("int"), pids.astype("int")


