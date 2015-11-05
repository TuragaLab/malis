from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("malisLoss", ["malis_loss.pyx", "malis_loss_cpp.cpp"], language='c++',)]

setup(cmdclass = {'build_ext': build_ext}, ext_modules = ext_modules)