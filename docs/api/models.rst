PytorchWildlife.models
==============================

.. module:: PytorchWildlife.models
This module contains the PytorchWildlife models. The models are currently divided between :ref:`Classification <classification>` and :ref:`Detection <detection>`.

.. _Detection:

Detection models
------------------

Yolov5
^^^^^^

.. autoclass:: YOLOV5Base
   :members:
   :undoc-members:
   :show-inheritance:

.. _Classification:

Classification models
----------------------

ResNet
^^^^^^

.. autoclass:: PlainResNetInference
   :members:
   :undoc-members:
   :show-inheritance:

Pretrained Classification Weights
---------------------------------

This section provides the pretrained weights that are currently available for the classification models. Below is a table detailing the available pretrained weights:

.. csv-table:: Pretrained Weights
   :file: classification_weights.csv
   :header-rows: 1
   :widths: 20, 60, 30, 20, 40

AI4GOpossum
^^^^^^

.. autoclass:: AI4GOpossum
   :members:
   :undoc-members:
   :show-inheritance:

AI4GAmazonRainforest
^^^^^^

.. autoclass:: AI4GAmazonRainforest
   :members:
   :undoc-members:
   :show-inheritance: