from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("opti_parallel_cython.pyx", compiler_directives={
        'language_level': "3", 
        "boundscheck": False, 
        "wraparound": False
    }),
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-O3", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)
