# code originally from https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php
# altered to include constraints
import numpy as np
from scipy.stats import multivariate_normal
class constr_gmm:

    def __init__(self,X,pair_ids,number_of_sources,iterations,reg):
        self.iterations = iterations
        self.number_of_sources = number_of_sources
        self.X = X
        self.pair_ids = pair_ids
        self.mu = None
        self.pi = None
        self.cov = None
        self.XY = None
        self.reg = reg
        
    def mult_probs(self, probs):
        n = self.X.shape[0]//2
        p = np.zeros((n,))
        for i in range(n):
            p[i] = np.prod(probs[self.pair_ids==i], axis=0)
        return p
            
    def avg_pairs(self):
        n = self.X.shape[0]//2
        avg = np.zeros((n,self.X.shape[1]))
        for i in range(n):
            avg[i,:] = np.sum(self.X[self.pair_ids==i], axis=0)/2
        return avg
            
            

    """Define a function which runs for iterations, iterations"""
    def run(self):
        self.reg_cov = self.reg*np.identity(len(self.X[0]))
        x,y = np.meshgrid(np.sort(self.X[:,0]),np.sort(self.X[:,1]))
        self.XY = np.array([x.flatten(),y.flatten()]).T
           
                    
        """ 1. Set the initial mu, covariance and pi values"""
        self.mu = np.random.randint(np.floor(min(self.X[:,0])),np.ceil(max(self.X[:,0])),size=(self.number_of_sources,len(self.X[0]))) # This is a nxm matrix since we assume n sources (n Gaussians) where each has m dimensions
        self.cov = np.zeros((self.number_of_sources,self.X.shape[1],self.X.shape[1])) # We need a nxmxm covariance matrix for each source since we have m features --> We create symmetric covariance matrices with ones on the digonal
        for dim in range(len(self.cov)):
            np.fill_diagonal(self.cov[dim],5)


        self.pi = np.ones(self.number_of_sources)/self.number_of_sources # Are "Fractions"
        log_likelihoods = [] # In this list we store the log likehoods per iteration and plot them in the end to check if
                             # if we have converged
            
        
        for i in range(self.iterations):               

            """E Step"""
            r_ic = np.zeros((self.X.shape[0]//2,len(self.cov)))
            for m,co,p,r in zip(self.mu,self.cov,self.pi,range(len(r_ic[0]))):
                mn = multivariate_normal(mean=m,cov=co)
                r_ic[:,r] = p*(self.mult_probs(mn.pdf(self.X)))/np.sum(
                    [pi_c*self.mult_probs(multivariate_normal(mean=mu_c,cov=cov_c).pdf(self.X)) 
                     for pi_c,mu_c,cov_c in zip(self.pi,self.mu,self.cov)],axis=0)


            """M Step"""

            # Calculate the new mean vector and new covariance matrices, based on the probable membership of the single x_i to classes c --> r_ic
            self.mu = []
            self.cov = []
            self.pi = []
            log_likelihood = []

            for c in range(len(r_ic[0])):
                m_c = np.sum(r_ic[:,c],axis=0)
                mu_c = (1/m_c)*np.sum(self.avg_pairs()*r_ic[:,c].reshape(self.X.shape[0]//2,1),axis=0)
                self.mu.append(mu_c)

                # Calculate the covariance matrix per source based on the new mean
                covariance = np.dot((np.array([r/2 for dummy in range(2) for r in r_ic[:,c]]).reshape(self.X.shape[0],1)
                                                 *(self.X-mu_c)).T,(self.X-mu_c)) + self.reg_cov
                self.cov.append((1/m_c)*covariance)
                # Calculate pi_new which is the "fraction of points" respectively the fraction of the probability assigned to each source 
                self.pi.append(m_c/np.sum(r_ic)) # Here np.sum(r_ic) gives as result the number of instances. This is logical since we know 
                                                # that the columns of each row of r_ic adds up to 1. Since we add up all elements, we sum up all
                                                # columns per row which gives 1 and then all rows which gives then the number of instances (rows) 
                                                # in X --> Since pi_new contains the fractions of datapoints, assigned to the sources c,
                                                # The elements in pi_new must add up to 1

            
            
            """Log likelihood"""
            log_likelihoods.append(np.log(np.sum(
                [k*self.mult_probs(multivariate_normal(self.mu[i],self.cov[j]).pdf(self.X)) 
                 for k,i,j in zip(self.pi,range(len(self.mu)),range(len(self.cov)))])))


    
    """Predict the membership of an unseen, new datapoint"""
    def predict(self,Y):
        prediction = []        
        for m,c in zip(self.mu,self.cov):  
            prediction.append(multivariate_normal(mean=m,cov=c).pdf(Y)/np.sum([multivariate_normal(mean=mean,cov=cov).pdf(Y) for mean,cov in zip(self.mu,self.cov)]))
        prediction = np.array(prediction).T
        return prediction
    
    
    