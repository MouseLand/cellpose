OpenVINO
------------------------------

`OpenVINO <https://github.com/openvinotoolkit/openvino>`_ is an optional backend for Cellpose which optimizes deep learning inference for Intel Architectures.

It should be installed in the same environment with Cellpose by the following command :

::

    pip install --no-deps openvino

Using ``openvino_utils.to_openvino``, convert PyTorch model to OpenVINO one:

::

    from cellpose.contrib import openvino_utils

    model = models.CellposeModel(...)

    model = openvino_utils.to_openvino(model)
