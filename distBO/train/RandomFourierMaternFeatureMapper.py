from __future__ import print_function, division

import tensorflow as tf
import numpy as np
from scipy.stats import norm, t, uniform
from math import pi, sqrt
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import invgamma

def multivariate_t_rvs(m, S, rs, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = rs.chisquare(df, n)/df
    z = rs.multivariate_normal(np.zeros(d),S,(n,))
    return m + z/np.sqrt(x)[:,None]

class RandomFourierMaternFeatureMapper():
    def __init__(self, in_dim, n_rff=500, stddev=1.0, seed=23):
        self.in_dim = in_dim
        self.n_rff = n_rff
        self.seed = seed
        self.stddev = stddev
        self.rs = check_random_state(self.seed)

    def sample(self, nu=1.5):
        #Z = self.rs.normal(size=(self.in_dim, self.n_rff))
        #U = invgamma.rvs(nu, loc=0.0, scale=nu, size=1)
        #r = np.sqrt(U) * Z
        r = multivariate_t_rvs(np.zeros(self.in_dim), np.eye(self.in_dim), self.rs, df=2.0*nu, n=self.n_rff)
        return r 

    def map(self, inputs, nu=1.5, tf_version=True):
        dtype = inputs.dtype
        samples = np.transpose(self.sample(nu))
        bias = 2.0 * pi * uniform.rvs(size=self.n_rff, random_state=self.rs)
        if tf_version:
            bias = tf.constant(bias, dtype=dtype)
            samples = tf.constant(samples, dtype=dtype)
            samples = tf.divide(samples, self.stddev)
            phi = tf.cos(tf.matmul(inputs, samples) + bias) # Check this broadcasting
        else:
            inputs = np.divide(inputs, self.stddev)
            phi = np.cos(np.matmul(inputs, samples) + bias)
        scaled_phi = sqrt(2.0 / self.n_rff) * phi
        return scaled_phi
    # If matern, it considers a seperable kernel on the dimensions.
    def check(self, inputs, nu=1.5):
        size = len(inputs)
        scaled_phi = self.map(inputs, nu, tf_version=False)
        rff_kernel = np.matmul(scaled_phi, scaled_phi.T)
        kernel_class = Matern(length_scale=self.stddev, nu=nu)
        kernel = kernel_class(inputs)
        print('Approx Kernel:', rff_kernel, np.sum(rff_kernel))
        print('True Kernel:', kernel, np.sum(kernel))
        return rff_kernel, kernel

if __name__ == '__main__':
    x = np.random.normal(size=(1000, 5))
    rff = RandomFourierMaternFeatureMapper(5, 100000)
    rff.check(x)
