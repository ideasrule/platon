import numpy as np
import scipy.stats

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
        return self.from_unit_interval(np.random.uniform())

    
class _UniformParam(_Param):
    def __init__(self, best_guess, low_lim, high_lim, low_guess, high_guess):
        _Param.__init__(self, best_guess, low_guess, high_guess)        
        self.low_lim = low_lim
        self.high_lim = high_lim

    def within_limits(self, value):
        return value > self.low_lim and value < self.high_lim
        
    def ln_prior(self, value):
        if not self.within_limits(value): return -np.inf
        return 0

    def from_unit_interval(self, u):
        assert(u >= 0 and u <= 1)
        if np.isinf(self.low_lim) or np.isinf(self.high_lim):
            raise ValueError("Limit cannot be infinity")       
        return self.low_lim + (self.high_lim - self.low_lim)*u


class _GaussianParam(_Param):
    def __init__(self, best_guess, std):
        _Param.__init__(self, best_guess,
                       best_guess - 2*std, best_guess + 2*std)        

        self.std = std

    def ln_prior(self, value):
        return np.log(scipy.stats.norm.pdf(value, self.best_guess, self.std))

    def from_unit_interval(self, u):
        return scipy.stats.norm.ppf(u, self.best_guess, self.std)

    def within_limits(self, value):
        return True
        
    
