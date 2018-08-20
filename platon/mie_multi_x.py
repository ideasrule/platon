import numpy as np
import scipy
import scipy.special as spl
import copy
import pdb

def aa2(a , ri, num, ru):
    s = a / ri
    wave_indices = np.arange(len(a))
    ru[[num-1],[wave_indices]] = num * s
    #ru[num-1] = (num) * s #try (num + 1)*s instead
    counter = copy.deepcopy(num)
    while np.any(np.greater_equal(counter, 2)):
        wave_indices = wave_indices[counter>=2]
        counter = counter[counter>=2]
        s1 = counter * s[wave_indices]
        ru[[counter-2],[wave_indices]] = s1 - 1.0/ (ru[[counter-1],[wave_indices]] + s1 )
        counter = counter - 1
    return ru

def shexqnn2 (ri, x):
    #ri = 1.33 + 0.5j
    #x = 100000.0
    nterms = 1000000
    #nterms = 550000000
    eps = 1.0 * 10**(-15)
    xmin = 1.0 * 10**(-6)

    ru = np.zeros((nterms,len(x)),dtype=complex)

    ier     = 0
    Qext    = np.zeros(len(x))
    Qsca    = np.zeros(len(x))
    Qabs    = np.zeros(len(x))
    Qbk     = np.zeros(len(x))
    Qpr     = np.zeros(len(x))
    albedo  = np.zeros(len(x))
    g       = np.zeros(len(x))

    fact = np.array([1.0,1.0e+150])
    factor = 1.0e+150

    if np.any(np.less_equal(x,xmin)):
        ier = 1
        print('<!> Error in subroutine shexqnn2')
        print('Mie scattering limit exceeded')
        print('current size parameter: '+str(x))
        return ier

    ax = 1.0 / x
    b = 2.0 * ax**2
    ss = np.zeros(len(x),dtype=complex)
    s3 = 0.0 - 1.0j
    an = 3.0

    y = np.sqrt(np.real(ri * np.conj(ri))) * x
    num = 1.25 * y + 15.5

    num[y<1.0] = 7.5 * y[y<1.0] + 9.0

    num[(y>100.0) & (y<50000.0)] = 1.0625 * y[(y>100.0) & (y<50000.0)] + 28.5

    num[y>50000.0] = 1.005 * y[y>50000.0] + 50.5

    num = num.astype(int)

    if np.any(np.greater(num,nterms)):
        ier = 2
        print('<!> Error in subroutine shexqnn2')
        print('Maximum number of terms: '+str(nterms))
        print('Number of terms required: '+str(num))
        print('Solution: increase default value of the variable nterms')
        return ier

    ru = aa2(ax,ri,num,ru)

    iterm = 0 #1

    ass = np.sqrt(np.pi / 2.0 * ax)
    w1 = 2.0 / np.pi * ax
    Si = np.sin(x) / x
    Co = np.cos(x) / x

    besJ0 = Si / ass
    besY0 = - Co / ass
    iu0 = 0

    besJ1 = ( Si * ax - Co) / ass
    besY1 = (-Co * ax - Si) / ass
    iu1 = 0
    iu2 = 0

    #Mie Coefficients
    s = ru[iterm] / ri + ass
    s1 = s * besJ1 - besJ0
    s2 = s * besY1 - besY0
    ra0 = s1 / (s1 - s3 * s2)

    s   = ru[iterm] * ri + ax
    s1  = s * besJ1 - besJ0
    s2  = s * besY1 - besY0
    rb0 = s1 / (s1 - s3 * s2)

    r = -1.5 * (ra0 - rb0)
    Qext = np.real(an * (ra0 + rb0))
    Qsca = np.real(an * (ra0 * np.conj(ra0) + rb0 * np.conj(rb0)))

    iterm = 1

    z = -1.0

    while np.any(np.less(iterm, num)):
        indices = np.less(iterm,num)
        an = an + 2.0
        an2 = an - 2.0

        besY2 = np.zeros(len(x))
        besJ2 = np.zeros(len(x))

        if iu1 == iu0:
            besY2[indices] = an2 * ax[indices] * besY1[indices] - besY0[indices]
        else:
            besY2[indices] = an2 * ax[indices] * besY1[indices] - besY0[indices] / factor

        if np.any(np.greater(np.abs(besY2),1.0e+200)):
            besY2[np.abs(besY2)>1.0e+200] = besY2[np.abs(besY2)>1.0e+200] / factor
            iu2 = iu1 + 1

        besJ2[indices] = (w1[indices] + besY2[indices] * besJ1[indices]) / besY1[indices]
        r_iterm = iterm + 1

        s = ru[iterm,indices] / ri + r_iterm * ax[indices]
        if iu1 > 1:
            ier = 1
            return
        if iu2 > 1:
            ier = 1
            return

        s1 = s * besJ2[indices] / fact[iu2] - besJ1[indices] / fact[iu1]
        s2 = s * besY2[indices] / fact[iu2] - besY1[indices] / fact[iu1]
        ra1 = s1 / (s1 - s3 * s2)

        s = ru[iterm,indices] * ri + r_iterm * ax[indices]
        s1 = s * besJ2[indices] / fact[iu2] - besJ1[indices] / fact[iu1]
        s2 = s * besY2[indices] / fact[iu2] - besY1[indices] / fact[iu1]
        rb1 = s1 / (s1 - s3 * s2)

        z  = -z
        rr = z * (r_iterm + 0.5) * (ra1 - rb1)
        r[indices] = r[indices] + rr
        ss[indices] = ss[indices] + (r_iterm - 1.0) * (r_iterm + 1.0) / r_iterm * np.real(ra0[indices] * np.conj(ra1) + rb0[indices] * np.conj(rb1)) + an2 / r_iterm / (r_iterm - 1.0) * np.real(ra0[indices] * np.conj(rb0[indices]))

        qq = np.real(an * (ra1 + rb1))
        Qext[indices] = Qext[indices] + qq
        Qsca[indices] = Qsca[indices] + np.real(an * (ra1 * np.conj(ra1) + rb1 * np.conj(rb1)))

        if np.all(np.less(np.abs(qq / Qext[indices]), eps)):
            break

        besJ0 = copy.deepcopy(besJ1)
        besJ1 = copy.deepcopy(besJ2)
        besY0 = copy.deepcopy(besY1)
        besY1 = copy.deepcopy(besY2)
        iu0   = copy.deepcopy(iu1)
        iu1   = copy.deepcopy(iu2)
        ra0[indices]   = copy.deepcopy(ra1)
        rb0[indices]   = copy.deepcopy(rb1)

        iterm = iterm + 1

    Qext = b * Qext
    Qsca   = b * Qsca
    Qbk    = np.real(2.0 * b * r * np.conj(r))
    Qpr    = Qext - np.real(2.0 * b * ss)
    Qabs   = Qext - Qsca
    albedo = Qsca / Qext
    g      = (Qext - Qpr) / Qsca

    ier = 0
    return (Qext, Qsca, Qabs, Qbk, Qpr, albedo, g, ier)
    #print(Qext, Qsca, Qabs, Qbk, Qpr, albedo, g, ier)
