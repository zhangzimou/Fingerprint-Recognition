
from distutils.core import setup

from Cython.Build import cythonize

setup(name="_preprocess", ext_modules=cythonize('_preprocess.pyx'))
setup(name="_match", ext_modules=cythonize('_match.pyx'))
