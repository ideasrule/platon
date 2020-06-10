Install
*******

Before installing PLATON, it is highly recommended to have a fast linear
algebra library (BLAS) and verify that numpy is linked to it.  This is because
the heart of the radiative transfer code is a matrix multiplication operation
conducted through numpy.dot, which in turn calls a BLAS library if it can find
one.  If it can't find one, your code will be many times slower.

We recommend using Anaconda, which automatically installs BLAS libraries.
If you don't want to use Anaconda, a good BLAS library to install on Linux is
OpenBLAS.  You can install it on Ubuntu with::
  
  sudo apt install libopenblas-dev

On OS X, a good choice is Accelerate/vecLib, which should already be installed
by default.

To check if your numpy is linked to BLAS, do::

  numpy.__config__.show()

If blas_opt_info mentions OpenBLAS or vecLib, that's a good sign.  If it says
"NOT AVAILABLE", that's a bad sign.

Once you have a BLAS installed and linked to numpy, download PLATON,
install the requirements, and install PLATON itself.  Although it is possible
to install PLATON using pip (pip install platon), the recommended method is to
clone the GitHub repository and install from there.  This is because the
repository includes examples, which you don't get when pip installing.

To install from GitHub::

  git clone https://github.com/ideasrule/platon.git
  cd platon/
  python setup.py install

You can run unit tests to make sure everything works::
  
  nosetests -v 

The unit tests should also give you a good idea of how fast the code will be.
On a decent Ubuntu machine with OpenBLAS, it takes 3 minutes.

The default data files (in platon/data) have a wavelength resolution of R=1000,
but if you want higher resolution, you can download R=10,000 and
R=375,000 data from `this webpage <http://astro.caltech.edu/~mz/absorption.html>`_
