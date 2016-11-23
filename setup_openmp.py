from distutils.core import setup, Extension
import numpy

include_dirs_numpy = [numpy.get_include()]

correlation = Extension('dynaphopy.power_spectrum.correlation',
                        extra_compile_args=['-std=c99', '-fopenmp'],
                        extra_link_args=['-lgomp'],
                        include_dirs = include_dirs_numpy,
                        sources=['Extensions/correlation.c'])


mem = Extension('dynaphopy.power_spectrum.mem',
                extra_compile_args=['-std=c99', '-fopenmp'],
                extra_link_args=['-lgomp'],
                include_dirs = include_dirs_numpy,
                sources=['Extensions/mem.c'])

displacements = Extension('dynaphopy.displacements',
                extra_compile_args=['-std=c99', '-fopenmp'],
                extra_link_args=['-lgomp'],
                include_dirs = include_dirs_numpy,
                sources=['Extensions/displacements.c'])

setup(name='dynaphopy',
      version='1.14',
      description='dynaphopy module',
      author='Abel Carreras',
      url='https://github.com/abelcarreras/DynaPhoPy',
      author_email='abelcarreras83@gmail.com',
      packages=['dynaphopy',
                'dynaphopy.orm',
                'dynaphopy.power_spectrum',
                'dynaphopy.analysis',
                'dynaphopy.analysis.fitting',
                'dynaphopy.interface',
                'dynaphopy.interface.iofile'],
      scripts=['scripts/dynaphopy'],
      ext_modules=[correlation, mem, displacements])


exit()