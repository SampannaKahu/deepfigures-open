#!/usr/bin/python

import os
import sys
from setuptools import setup, Extension, find_packages

tf_include = '/'.join(sys.executable.split('/')[:-2]) + \
             '/lib/python%d.%d/site-packages/tensorflow/include' % sys.version_info[:2]

extra_defs = []
if os.uname().sysname == 'Darwin':
    extra_defs.append('-D_GLIBCXX_USE_CXX11_ABI=0')
else:
    os.environ['CC'] = 'g++'
    os.environ['CXX'] = 'g++'

requirements_file = os.path.join(
    os.path.dirname(__file__),
    'requirements.txt')
requirements = open(requirements_file).read().split('\n')
requirements = [r for r in requirements if not '-e' in r]

setup(
    name='deepfigures-open',
    version='0.0.1',
    url='http://github.com/SampannaKahu/deepfigures-open',
    packages=find_packages(),
    setup_requires=['Cython'],
    install_requires=requirements,
    tests_require=[],
    zip_safe=False,
    test_suite='py.test',
    entry_points='',
    cffi_modules=['deepfigures/utils/stringmatch/stringmatch_builder.py:ffibuilder'],
    ext_modules=[
        Extension(
            name='tensorboxresnet.utils.stitch_wrapper',
            sources=[
                './vendor/tensorboxresnet/tensorboxresnet/utils/stitch_wrapper.pyx',
                './vendor/tensorboxresnet/tensorboxresnet/utils/stitch_rects.cpp',
                './vendor/tensorboxresnet/tensorboxresnet/utils/hungarian/hungarian.cpp'
            ],
            language='c++',
            extra_compile_args=[
                                   '-std=c++11',
                                   '-Itensorbox/utils',
                                   '-I%s' % tf_include
                               ] + extra_defs,
        )
    ],
    package_data={
        'vendor.tensorboxresnet.tensorboxresnet': [
            'logging.conf'
        ]
    },
    include_package_data=True
)
