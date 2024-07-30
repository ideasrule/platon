import numpy as np
import scipy.stats
import sys

import dill as dl

import pandas as pd

class _Param:
    def __init__(self, best_guess, low_guess=None, high_guess=None):
        self.best_guess = best_guess
        self.low_guess = low_guess
        self.high_guess = high_guess

    def ln_prior(self, value):
        raise NotImplementedError

    def from_unit_interval(self, u):
        # u is between 0 and 1
        raise NotImplementedError

    def within_limits(self, value):
        raise NotImplementedError

    def get_random_value(self):
        return np.random.uniform(self.low_guess, self.high_guess)

    def __repr__(self):
        return "Guess, low, high: {}, {}, {}".format(
            self.best_guess, self.low_guess, self.high_guess)

class _UniformParam(_Param):
    def __init__(self, best_guess, low_lim, high_lim, low_guess, high_guess):
        _Param.__init__(self, best_guess, low_guess, high_guess)
        self.low_lim = low_lim
        self.high_lim = high_lim

    def within_limits(self, value):
        return value > self.low_lim and value < self.high_lim

    def ln_prior(self, value):
        if not self.within_limits(value):
            return -np.inf
        return 0

    def from_unit_interval(self, u):
        assert(u >= 0 and u <= 1)
        if np.isinf(self.low_lim) or np.isinf(self.high_lim):
            raise ValueError("Limit cannot be infinity")
        return self.low_lim + (self.high_lim - self.low_lim) * u

    def __repr__(self):
        return "Guess, low, high: {}, {}, {}".format(
            self.best_guess, self.low_guess, self.high_guess)

class _GaussianParam(_Param):
    def __init__(self, best_guess, std, low_guess, high_guess):
        _Param.__init__(self, best_guess, low_guess, high_guess)

        self.std = std

    def ln_prior(self, value):
        # print(np.log(scipy.stats.norm.pdf(value, self.best_guess, self.std)))
        # sys.exit()
        return np.log(scipy.stats.norm.pdf(value, self.best_guess, self.std))

    def from_unit_interval(self, u):
        return scipy.stats.norm.ppf(u, self.best_guess, self.std)

    def within_limits(self, value):
        return True

    def __repr__(self):
        return "Guess, STD: {}, {}".format(self.best_guess, self.std)


class _CLRParam(_Param):
    def __init__(self, best_guess, ng, low_lim = None, high_lim = None):
        _Param.__init__(self, best_guess, low_lim, high_lim)
        self.ppf = dl.load(open('/Users/kimparagas/desktop/research/main/jwst_project/clr_stuff/clr_prior_p3.pkl', 'rb'))
        
        clr_df = pd.read_csv(f'/Users/kimparagas/Desktop/research/main/jwst_project/clr_stuff/clr_priors/ng_{ng}_prior.csv',
             sep = '\t')
        
        self.lower_bin = clr_df['lower bin'].to_numpy()
        self.upper_bin = clr_df['upper bin'].to_numpy()
        self.clr_prior_vals = clr_df['clr prior value'].to_numpy()
        
        self.bins = np.array([self.lower_bin, self.upper_bin]).T
        
        self.low_lim = self.lower_bin.min()
        self.high_lim = self.upper_bin.max()

    def ln_prior(self, value):
        """generate a center log ratio distrubtion, corresponding to the log-uniform distribution of [fmin, 1]"""
        # generate uniform distributions in log space between 1e-12 and 1
        
        if not self.within_limits(value):
            return -np.inf

        if value <= self.low_lim:
            return np.log(self.clr_prior_vals[0])
        
        if value >= self.high_lim:
            return np.log(self.clr_prior_vals[-1])
        else: 
            for i,b in enumerate(self.bins): 
                if value <= b[1] and value >= b[0]:
                    prior_value = (np.log(self.clr_prior_vals[i]))
            return prior_value

    def from_unit_interval(self, u, ng = 1):
        # print(ppf[str(ng)](u))
        assert(u >= 0 and u <= 1)
        return self.ppf[str(ng)](u)

    def within_limits(self, value):
        return value > self.low_lim and value < self.high_lim
        # return True

    def __repr__(self):
         return "Guess, low, high: {}, {}, {}".format(
            self.best_guess, self.low_guess, self.high_guess)
