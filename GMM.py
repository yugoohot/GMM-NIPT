import pandas as pd
import numpy as np
from scipy import stats
import os

class GMM(object):
    def __init__(self, k,p,means,covs,iteration): 
        self.iteration = iteration
        self.K = k
        self.p = p
        self.p = self.p / self.p.sum()      
        self.means = means
        self.covs = covs
        
    def fit(self, data):
        old_p = self.p - 1000
        old_means = self.means - 1000
        old_covs = self.covs - 1000
        for i in range(self.iteration):            
            if np.linalg.norm(means-old_means) < 0.01 and np.linalg.norm(covs-old_covs) < 0.01 and np.linalg.norm(p-old_p) < 0.0001:
                print(i)
                print(self.means)
                print(self.p)
                print(self.covs)
                break
            old_p = self.p
            old_means = self.means
            old_covs = self.covs
            density = np.ones((len(data), self.K))
            for i in range(self.K):
                norm = stats.multivariate_normal(self.means[0,i], (self.covs[0,i]) ** 2)
                density[:,i] = np.array([[norm.pdf(data)]])
            posterior = density * self.p
            posterior = posterior / posterior.sum(axis=1, keepdims=True) 
            p_hat = posterior.sum(axis=0)  
            mean_hat = np.ones((1,self.K))
            for g in range(self.K):
                mean_hat[0,g] = (posterior[:,g].reshape(len(data),1) * data).sum()
            self.means = mean_hat / p_hat
            self.p = (p_hat / len(data)).reshape(1,self.K)
            cov_hat = np.ones((1,self.K))
            for f in range(self.K):
                cov_hat[0,f] = np.sqrt(np.sum((posterior[:,f].reshape(len(data),1))*np.square(data-self.means[0,f]))/(np.sum(posterior[:,f])))
            self.covs = cov_hat
            if i % 1000 == 0:
                print(i)
                print(self.means)
                print(self.p)
                print(self.covs)
       
    def fit_samples(self, path):
        name = path.split(".")[0]
        data = pd.read_csv(path.strip(), sep=',')
        data = np.array(data)
        print(name)
        old_p = self.p - 1000
        old_means = self.means - 1000
        old_covs = self.covs - 1000
        for i in range(self.iteration):
            if np.linalg.norm(means-old_means) < 0.01 and np.linalg.norm(covs-old_covs) < 0.01 and np.linalg.norm(p-old_p) < 0.0001:
                print(i)
                print(self.means)
                print(self.p)
                print(self.covs)
                break
            old_p = self.p
            old_means = self.means
            old_covs = self.covs
            density = np.ones((len(data), self.K))
            for k in range(self.K):
                norm = stats.multivariate_normal(self.means[0,k], (self.covs[0,k]) ** 2)
                density[:,k] = np.array([[norm.pdf(data)]])
            posterior = density * self.p
            posterior = posterior / posterior.sum(axis=1, keepdims=True) 
            p_hat = posterior.sum(axis=0) 
            mean_hat = np.ones((1,self.K))
            for g in range(self.K):
                mean_hat[0,g] = (posterior[:,g].reshape(len(data),1) * data).sum()
            self.means = mean_hat / p_hat
            self.p = p_hat / len(data)
            cov_hat = np.ones((1,self.K))
            for f in range(self.K):
                cov_hat[0,f] = np.sqrt(np.sum((posterior[:,f].reshape(len(data),1))*np.square(data-self.means[0,f]))/(np.sum(posterior[:,f])))
            self.covs = cov_hat
        print(i)
        self.means = (self.means).reshape(self.K,)
        self.p = (self.p).reshape(self.K,)
        self.covs = (self.covs).reshape(self.K,)
        parameter = [] 
        parameter.append(self.means)
        parameter.append(self.p)
        parameter.append(self.covs)
        parameter = pd.DataFrame(parameter)
        parameter.to_csv(name.strip()+"_parameter.csv")
        

p=np.array([[1,1,1,5,5,5,30,10,10,5]])
means=np.array([[101,111,123,133,152,160,167,180,190,220]])
covs=np.array([[10,10,10,10,10,10,10,10,10,10]])

val = os.system('ls *mother.csv > csvnames.txt')
print(val)

with open("csvnames.txt", "r") as f:
    csvnames = f.readlines()
    for csvname in csvnames:
        g=GMM(10,p,means,covs,30000)
        g.fit_samples(csvname.strip()) 
