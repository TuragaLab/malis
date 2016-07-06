from distutils.sysconfig import get_config_vars, get_config_var, get_python_inc
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import os


include_dirs = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "malis"),
    os.path.dirname(get_python_inc()),
    get_python_inc()
]
library_dirs = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "malis"),
    get_config_var("LIBDIR")
]

# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
cfg_vars = get_config_vars()
if "CFLAGS" in cfg_vars:
    cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace("-Wstrict-prototypes", "")

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

setup(name='malis',
      version='0.0',
      description='MALIS segmentation loss function',
      url='https://github.com/TuragaLab/malis',
      author='Srinivas Turaga',
      author_email='turagas@janelia.hhmi.org',
	  cmdclass=dict(
            build_ext=build_ext
    	),
      license='MIT',
      install_requires=['cython','numpy','h5py','scipy'],	
	  setup_requires=['cython','numpy','scipy'],	
      packages=['malis'],
         ext_modules = [Extension("malis.malis",
                         ["malis/malis.pyx", "malis/malis_cpp.cpp"],
                         include_dirs=include_dirs,
                         library_dirs=library_dirs,
                         language='c++',
                         # std= 'c++11',
                         extra_link_args=["-std=c++11"],
                         extra_compile_args=["-std=c++11", "-w"])],
      zip_safe=False)
