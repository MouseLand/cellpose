import setuptools
from setuptools import setup

install_deps = [
    'numpy>=1.20.0,<2.1',
    'scipy',
    'natsort',
    'tifffile',
    'tqdm',
    'numba>=0.53.0',
    'llvmlite',
    'torch>=1.6',
    'opencv-python-headless',
    'fastremap',
    'imagecodecs',
    'roifile',
]

image_deps = ['nd2', 'pynrrd']

gui_deps = [
    'pyqtgraph>=0.11.0rc0', "pyqt6", "pyqt6.sip", 'qtpy', 'superqt',
]

docs_deps = [
    'sphinx>=3.0',
    'sphinxcontrib-apidoc',
    'sphinx_rtd_theme',
    'sphinx-argparse',
]

distributed_deps = [
    'dask',
    'dask_image',
    'scikit-learn',
]

bioimageio_deps = [
    'bioimageio.core',
]

try:
    import torch
    a = torch.ones(2, 3)
    from importlib.metadata import version
    ver = version("torch")
    major_version, minor_version, _ = ver.split(".")
    if major_version == "2" or int(minor_version) >= 6:
        install_deps.remove("torch>=1.6")
except:
    pass

try:
    import PyQt5
    gui_deps.remove("pyqt6")
    gui_deps.remove("pyqt6.sip")
    gui_deps.append("pyqt5")
    gui_deps.append("pyqt5.sip")
except:
    pass

try:
    import PySide2
    gui_deps.remove("pyqt6")
    gui_deps.remove("pyqt6.sip")
except:
    pass

try:
    import PySide6
    gui_deps.remove("pyqt6")
    gui_deps.remove("pyqt6.sip")
except:
    pass

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="cellpose", license="BSD", author="Marius Pachitariu and Carsen Stringer",
    author_email="stringerc@janelia.hhmi.org",
    description="anatomical segmentation algorithm", long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MouseLand/cellpose", setup_requires=[
        'pytest-runner',
        'setuptools_scm',
    ], packages=setuptools.find_packages(), use_scm_version=True,
    install_requires=install_deps, tests_require=['pytest'], extras_require={
        'docs': docs_deps,
        'gui': gui_deps,
        'distributed': distributed_deps,
        'bioimageio': bioimageio_deps,
        'all': gui_deps + distributed_deps + image_deps + bioimageio_deps,
    }, include_package_data=True, classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ), entry_points={'console_scripts': ['cellpose = cellpose.__main__:main']})
