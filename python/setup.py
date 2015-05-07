from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize(
           "malis_loss.pyx",                 # our Cython source
           sources=["malis_loss.cpp"],  # additional source file(s)
           language="c++",             # generate C++ code
      ))