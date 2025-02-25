Install
*******
We highly recommend installing PLATON in a conda environment, which typically comes with optimized BLAS libraries.  Once in the conda environment::
  
  git clone https://github.com/ideasrule/platon.git
  cd platon/
  pip install -e .

Doing "pip install -e ." instead of "pip install ." installs PLATON in-place instead of copying the source files somewhere else.  This way, you can modify the source code and have the changes reflected instantly.

PLATON 6.2 uses the GPU by default, and falls back on the CPU if it can't find one.  If you have a Nvidia GPU, we highly recommend that you install CUDA and cupy, which will speed up PLATON many-fold::
  
  conda install -c conda-forge cupy

PLATON will automatically detect the existence of cupy and use the GPU.  You can force it to use the CPU by setting FORCE_CPU = True in _cupy_numpy.py.

PLATON supports both dynesty and pymultinest for nested sampling.  dynesty is installed by default.  To install pymultinest::
  
  conda install -c conda-forge mpi4py pymultinest

After installing PLATON, run one of the examples so that the data files are automatically downloaded::
  cd examples/
  python transit_depth_example.py
  
You can also run unit tests to make sure everything works::
  
  nosetests -v 

The default data files (in platon/data) have a wavelength resolution of R=20k.
If you want higher resolution, you can download higher-resolution opacities from the `DACE opacity database <https://dace.unige.ch/opacityDatabase>`_ and interpolate to PLATON's temperature and pressure grid.
