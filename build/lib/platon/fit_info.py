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
        '''Fit for the parameter `name` using a uniform prior between `low_lim`
        and `high_lim`.  If using emcee, the walkers' initial values for this
        parameter are randomly selected to be between `low_guess` and
        `high_guess`.  If not specified, `low_guess` is set to `low_lim`, and
        similarly with `high_guess`.'''

        if name in self.fit_param_names:
            raise ValueError("Already fitting for {0}".format(name))

        if low_guess is None:
            low_guess = low_lim
        if high_guess is None:
            high_guess = high_lim
        best_guess = self.all_params[name].best_guess

        self.fit_param_names.append(name)
        self.all_params[name] = _UniformParam(best_guess, low_lim, high_lim,
                                              low_guess, high_guess)

    def add_gaussian_fit_param(self, name, std, low_guess=None, high_guess=None):
        '''Fit for the parameter `name` using a Gaussian prior with standard
        deviation `std`.  If using emcee, the walkers' initial values for this
        parameter are randomly selected to be between `low_guess` and
        `high_guess`.  If `low_guess` is None, it is set to mean-2*std; if
        `high_guess` is None, it is set to mean+2*std.'''

        if name in self.fit_param_names:
            raise ValueError("Already fitting for {0}".format(name))

        mean = self.all_params[name].best_guess
        if low_guess is None:
            low_guess = mean - 2 * std
        if high_guess is None:
            high_guess = mean + 2 * std

        self.fit_param_names.append(name)
        self.all_params[name] = _GaussianParam(
            mean, std, low_guess, high_guess)

    def _interpret_param_array(self, array):
        if len(array) != len(self.fit_param_names):
            raise ValueException("Fit array invalid")

        result = dict()
        for i, key in enumerate(self.fit_param_names):
            result[key] = array[i]

        for key in self.all_params:
            if key not in result:
                result[key] = self.all_params[key].best_guess

        return result

    def _within_limits(self, array):
        if len(array) != len(self.fit_param_names):
            raise ValueException("Fit array invalid")

        for i, key in enumerate(self.fit_param_names):
            if not self.all_params[key].within_limits(array[i]):
                return False

        return True

    def _generate_rand_param_arrays(self, num_arrays):
        result = []

        for i in range(num_arrays):
            row = []
            for name in self.fit_param_names:
                if i == 0:
                    # Have one walker with fiducial value
                    row.append(self.all_params[name].best_guess)
                else:
                    row.append(self.all_params[name].get_random_value())
            result.append(row)

        return np.array(result)

    def _get(self, name):
        return self.all_params[name].best_guess

    def _get_num_fit_params(self):
        return len(self.fit_param_names)

    def _from_unit_interval(self, index, u):
        name = self.fit_param_names[index]
        return self.all_params[name].from_unit_interval(u)

    def _ln_prior(self, array):
        result = 0
        for i, name in enumerate(self.fit_param_names):
            result += self.all_params[name].ln_prior(array[i])

        return result

    def __repr__(self):
        return "Params to fit: {}; all params: {}".format(self.fit_param_names,
                                                          self.all_params)
