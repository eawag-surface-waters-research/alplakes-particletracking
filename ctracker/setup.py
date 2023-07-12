from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(
  name = "writetools module",
    ext_modules = cythonize([
    Extension("ctracker.modules.writetools",
              sources=["ctracker/modules/writetools.pyx"]) ])) 
setup(
  name = "loop module",
    ext_modules = cythonize([
    Extension("ctracker.modules.loop",
              sources=["ctracker/modules/loop.pyx"],
              libraries=["m"]) ])) # Math library
setup(
  name = "Cartesian loop module",
    ext_modules = cythonize([
    Extension("ctracker.modules.loop_cartesian",
              sources=["ctracker/modules/loop_cartesian.pyx"],
              libraries=["m"]) ])) # Math library
