def setup_cython():
	from distutils.core import setup
	from distutils.extension import Extension
	from Cython.Distutils import build_ext

	import numpy

	ext_modules = [Extension("malis", ["malis.pyx", "malis_cpp.cpp"], language='c++',extra_link_args=["-std=c++11"],
                         extra_compile_args=["-std=c++11", "-w"])]

	setup(cmdclass = {'build_ext': build_ext}, include_dirs=[numpy.get_include()], ext_modules = ext_modules)
if __name__=='__main__':
	setup_cython()