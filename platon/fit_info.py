import numpy as np
from ._params import _UniformParam, _GaussianParam, _Param

class FitInfo:
    def __init__(self, guesses_dict):
        self.fit_param_names = []
        self.all_params = dict()
        
        for key in guesses_dict:
            self.all_params[key] = _Param(guesses_dict[key])

    def add_uniform_fit_param(self, name, low_lim, high_lim,
                      low_guess=None, high_guess=None):
        
        if name in self.fit_param_names:
            raise ValueError("Already fitting for {0}".format(name))
        
        best_guess = self.all_params[name].best_guess
            
        self.fit_param_names.append(name)
        self.all_params[name] = _UniformParam(best_guess, low_lim, high_lim,
                                             low_guess, high_guess)

    def add_gaussian_fit_param(self, name, std):
        if name in self.fit_param_names:
            raise ValueError("Already fitting for {0}".format(name))
        
        mean = self.all_params[name].best_guess
        self.fit_param_names.append(name)
        self.all_params[name] = _GaussianParam(mean, std)

        
    def freeze_fit_param(self, name):
        if name not in self.fit_param_names:
            raise ValueError("{0} not being fit".format(name))
        
        self.fit_param_names.remove(name)

        
    def interpret_param_array(self, array):
        if len(array) != len(self.fit_param_names):
            raise ValueException("Fit array invalid")

        result = dict()
        for i, key in enumerate(self.fit_param_names):
            result[key] = array[i]

        for key in self.all_params:
            if key not in result:
                result[key] = self.all_params[key].best_guess
                
        return result

    def within_limits(self, array):
        if len(array) != len(self.fit_param_names):
            raise ValueException("Fit array invalid")

        for i, key in enumerate(self.fit_param_names):
            if not self.all_params[key].within_limits(array[i]):
                return False

        return True

    def generate_rand_param_arrays(self, num_arrays):
        result = []
        
        for i in range(num_arrays):
            row = []
            for name in self.fit_param_names:
                if i == 0:
                    #Have one walker with fiducial value
                    row.append(self.all_params[name].best_guess)
                else:
                    row.append(self.all_params[name].get_random_value())
            result.append(row)
            
        return np.array(result)

    def get(self, name):
        return self.all_params[name].best_guess

    def get_num_fit_params(self):
        return len(self.fit_param_names)

    
    def from_unit_interval(self, index, u):
        name = self.fit_param_names[index]
        return self.all_params[name].from_unit_interval(u)

    def ln_prior(self, array):
        result = 0
        for i, name in enumerate(self.fit_param_names):
            result += self.all_params[name].ln_prior(array[i])
            
        return result
