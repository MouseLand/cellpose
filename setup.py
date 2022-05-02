import setuptools
from setuptools import setup

install_deps = ['numpy>=1.20.0', 'scipy', 'natsort',
                'tifffile', 'tqdm', 'numba', 
                'torch>=1.6',
                'opencv-python-headless',
                'fastremap'
                ]

gui_deps = [
        'pyqtgraph>=0.11.0rc0', 
        'pyqt5', 
        'pyqt5.sip',
        'google-cloud-storage'
        ]

docs_deps = [
        'sphinx>=3.0',
        'sphinxcontrib-apidoc',
        'sphinx_rtd_theme',
      ]

distributed_deps = [
        'dask',
        'dask_image',
        'scikit-learn',
]

try:
    import torch
    a = torch.ones(2, 3)
    version = int(torch.__version__[2])
    if version >= 6:
        install_deps.remove('torch')
except:
    pass

with open("README.md", "r") as fh:
    long_description = fh.read()
    
    
setup(
    name="cellpose",
    license="BSD",
    author="Marius Pachitariu and Carsen Stringer",
    author_email="stringerc@janelia.hhmi.org",
    description="anatomical segmentation algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MouseLand/cellpose",
    setup_requires=[
      'pytest-runner',
      'setuptools_scm',
    ],
    packages=setuptools.find_packages(),
    use_scm_version=True,
    install_requires = install_deps,
    tests_require=[
      'pytest'
    ],
    extras_require = {
      'docs': docs_deps,
      'gui': gui_deps,
      'distributed': distributed_deps,
      'all': gui_deps + distributed_deps,
    },
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ),
     entry_points = {
        'console_scripts': [
          'cellpose = cellpose.__main__:main']
     }
)
