import numpy as np

def convert_clr_to_vmr(clrs):
    clr_bkg = - np.sum(clrs)
    clrs_with_bkg = np.append(clrs, clr_bkg)
    # print(clrs_with_bkg)
    geometric_mean = 1 / np.sum(np.exp(clrs_with_bkg))
    # vmrs = np.exp(clrs + np.log(geometric_mean))
    vmrs_with_bkg = np.exp(clrs_with_bkg + np.log(geometric_mean))
    if np.around(np.sum(vmrs_with_bkg), decimals = 5) == 1.0:
        return vmrs_with_bkg
    else:
        print(np.around(np.sum(vmrs_with_bkg), decimals = 5))
        raise ValueError(
            'VMRs did not sum to unity. Something is wrong.')