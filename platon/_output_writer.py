import numpy as np
from sys import stdout
import os, glob

def write_param_estimates_file(samples, best_params, best_lnprob, fit_labels,
                               filename="BestFit.txt"):
    output = "#Parameter Lower_error Median Upper_error Best_fit\n"
    output += "Max_lnprob {}\n".format(best_lnprob)
    
    for i, name in enumerate(fit_labels):    
        lower = np.percentile(samples[:, i], 16)
        median = np.median(samples[:, i])
        upper = np.percentile(samples[:, i], 84)
        best = best_params[i]
        output += "{} {} {} {} {}\n".format(
            name, median - lower, median, upper - median, best_params[i])

    print(output)
    with open(filename, "w") as f:
        f.write(output)
    
    
