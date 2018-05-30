import numpy as np

class FitParam:
    def __init__(self, value, low_guess=None, high_guess=None, low_lim=None, high_lim=None):
        self.value = value
        self.low_guess = low_guess
        self.high_guess = high_guess
        self.low_lim = low_lim
        self.high_lim = high_lim

    def within_limits(self, value):
        return value > self.low_lim and value < self.high_lim

class FitInfo:
    def __init__(self, guesses_dict):
        self.fit_param_names = []
        self.all_params = dict()
        
        for key in guesses_dict:
            self.all_params[key] = FitParam(guesses_dict[key])

    def add_fit_param(self, name, low_guess, high_guess, low_lim=None, high_lim=None, value=None):
        if value is None:
            value = self.all_params[name].value
        if low_lim is None:
            low_lim = low_guess
        if high_lim is None:
            high_lim = high_guess
            
        self.fit_param_names.append(name)
        self.all_params[name] = FitParam(value, low_guess, high_guess, low_lim, high_lim)    

    def freeze_fit_param(self, name, value=None):
        if value is not None:
            self.all_params[name].value = value
        self.fit_param_names.remove(name)

        
    def get_param_array(self):
        result = []
        for name in self.fit_param_names:
            result.append(self.all_params[name].value)
        return np.array(result)

        
    def interpret_param_array(self, array):
        if len(array) != len(self.fit_param_names):
            raise ValueException("Fit array invalid")

        result = dict()
        for i, key in enumerate(self.fit_param_names):
            result[key] = array[i]

        for key in self.all_params:
            if key not in result:
                result[key] = self.all_params[key].value
                
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
                    row.append(self.all_params[name].value)
                else:
                    row.append(np.random.uniform(self.all_params[name].low_guess, self.all_params[name].high_guess))
            result.append(row)
            
        return np.array(result)

    def get(self, name):
        return self.all_params[name].value

    def get_num_fit_params(self):
        return len(self.fit_param_names)
        
    def get_guess_bounds(self, index):
        name = self.fit_param_names[index]
        return self.all_params[name].low_guess, self.all_params[name].high_guess
    
