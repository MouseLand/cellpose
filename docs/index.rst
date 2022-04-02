.. cellpose master

cellpose
===================================

cellpose is an anatomical segmentation algorithm written in Python 3 
by Carsen Stringer and Marius Pachitariu. For support, please open 
an `issue`_.

We make pip installable releases of cellpose, here is the `pypi`_. You
can install it as ``pip install cellpose[gui]``.

You can try it out without installing at `cellpose.org`_. 
Also check out these resources:

Cellpose 2.0

- `paper <https://youtu.be/3Y1VKcxjNy4>`_ on biorxiv
- twitter `paper <https://youtu.be/3Y1VKcxjNy4>`_
- human-in-the-loop training protocol `video <https://youtu.be/3Y1VKcxjNy4>`_

Cellpose 1.0 

- `paper <https://www.biorxiv.org/content/10.1101/2020.02.02.931238v1>`_ on biorxiv (see figure 1 below) and in `nature methods <https://www.nature.com/articles/s41592-020-01018-x>`_
- twitter `thread`_
- Marius's `talk`_

.. image:: _static/fig1.PNG
    :width: 1200px
    :align: center
    :alt: fig1

.. _cellpose.org: http://www.cellpose.org
.. _thread: https://twitter.com/computingnature/status/1224477812763119617
.. _issue: https://github.com/MouseLand/cellpose/issues
.. _talk: https://www.youtube.com/watch?v=7y9d4VIKiS8
.. _pypi: https://pypi.org/project/cellpose/


.. toctree::
   :maxdepth: 3
   :caption: Basics:

   installation
   gui
   inputs
   settings
   outputs
   models
   train
   

.. toctree::
   :maxdepth: 3
   :caption: Examples:

   notebook
   command
   

.. toctree::
   :caption: API Reference:

   api