# Variational Inference
# CAVI
# GMM

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm


class gmm():
    def __init__(self, data, num_clusters, sigma=2):
        '''initialization
            model parameters(m,s2,phi)--need update
        '''
        self.data = data
        self.K = num_clusters
        self.n = data.shape[0]
        self.sigma = sigma
        # phi is the the probability that case nth comes from Kth distribution.
        self.phi = np.random.random([self.n, self.K])  # phi(matrix n*k)
        self.m = np.random.randint(np.min(data), np.max(data), self.K).astype(float)  # m(matrix 1*k)
        self.s2 = np.random.random(self.K)  # s2(matrix 1*k)

    def compute_elbo(self):
        '''calculate ELOB '''
        p1 = -np.sum((self.m ** 2 + self.s2) / (2 * self.sigma ** 2))
        p2 = (-0.5 * np.add.outer(self.data ** 2, self.m ** 2 + self.s2) + np.outer(self.data, self.m)) * (self.phi)
        p3 = -np.sum(np.log(self.phi))
        p4 = np.sum(0.5 * np.sum(np.log(self.s2)))
        elbo_c = p1 + np.sum(p2) + p3 + p4
        return elbo_c

    def update_bycavi(self):
        '''update (m,s2,phi)'''
        # phi update
        e = np.outer(self.data, self.m) + (-0.5 * (self.m ** 2 + self.s2))[np.newaxis,
                                          :]  # [np.newaxis, :] is to matrix
        self.phi = np.exp(e) / np.sum(np.exp(e), axis = 1)[:, np.newaxis]  # normalization  K*N matrix
        # m update
        self.m = np.sum(self.data[:, np.newaxis] * self.phi, axis = 0) / (
                1.0 / self.sigma ** 2 + np.sum(self.phi, axis = 0))
        # s2 update
        self.s2 = 1.0 / (1.0 / self.sigma ** 2 + np.sum(self.phi, axis = 0))

    def trainmodel(self, epsilon, iters):
        '''train model
             epsilon: epsilon-convergence
             iters: iteration number
        '''
        elbo = []
        elbo.append(self.compute_elbo())
        # use cavi to update elbo until epsilon-convergence
        for i in range(iters):
            self.update_bycavi()
            elbo.append(self.compute_elbo())
            print("elbo is: ", elbo[i])
            if np.abs(elbo[-1] - elbo[-2]) <= epsilon:
                print("iter of convergence:", i)
                break
        return elbo

    def plot(self, size):
        sns.set_style("whitegrid")
        for i in range(int(self.n / size)):
            sns.distplot(data[size * i: (i + 1) * size], rug = True)
            x = np.linspace(self.m[i] - 3 * self.sigma, self.m[i] + 3 * self.sigma, 100)
            plt.plot(x, norm.pdf(x, self.m[i], self.sigma), color = 'black')
        plt.show()


if __name__ == "__main__":
    # generate data
    number = 1000
    clusters = 3
    sigma = 1
    mu = np.array([x*2 for x in range(clusters)])
    data = []
    # x-N(mu, sigma)
    for i in range(clusters):
        data.append(np.random.normal(mu[i], sigma, number))
    # concatenate data
    data = np.concatenate(np.array(data))
    model = gmm(data, clusters, sigma = sigma)
    model.trainmodel(1e-3, 1000)
    print("converged_means:", sorted(model.m))
    model.plot(number)
