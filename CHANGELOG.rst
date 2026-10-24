nneuroutil 0.1.0 (October 24, 2026)
-----------------------------------

This is the first official release of the project. It currently contains some
minimal helpers and is intentionally small in scope.

All functionality is based on the `Array API <https://data-apis.org/array-api>`__,
so all algorithms work across different backends. Only ``numpy``, ``jax`` and
``torch`` are actively tested at the moment.

Features
^^^^^^^^

* Support for basic classifiers: ``LDA``, ``SVM``.
* Support for basic DMD methods: classic DMD, EDMD, Forward-Backward DMD,
  total least-squares de-biasing. classic DMDc, symmetric DMDc.
* Timeseries focused data loaders: ``SlicedDataLoader``, ``SlidingWindowDataset``.
* Basic scores: DICE, F1, IoU, volumetric similarity.
* Basic thresholding: Otsu's Method.
* Complex activation functions: modReLU, LeakyModReLU, Cardioid, zReLU, phase
  preserving ``log`` and ``exp``, etc.
* Custom Kaiming initializers for all new activations functions.
* Custom layers: ``ComplexLinear``, ``SymmetricLinear``, etc.
