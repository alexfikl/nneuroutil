.. |badge-ci| image:: https://github.com/alexfikl/nneuroutil/actions/workflows/ci.yml/badge.svg
    :alt: Build Status
    :target: https://github.com/alexfikl/nneuroutil/actions/workflows/ci.yml

.. |badge-rtd| image:: https://readthedocs.org/projects/nneuroutil/badge/?version=latest
    :alt: Documentation
    :target: https://nneuroutil.readthedocs.io/en/latest/?badge=latest

.. |badge-reuse| image:: https://api.reuse.software/badge/github.com/alexfikl/nneuroutil
    :alt: REUSE
    :target: https://api.reuse.software/info/github.com/alexfikl/nneuroutil

.. |badge-mit| image:: https://img.shields.io/badge/License-MIT-blue.svg
    :alt: MIT license
    :target: https://spdx.org/licenses/MIT.html

|badge-ci| |badge-rtd| |badge-reuse| |badge-mit|

nneuroutil
==========

This is a bag of random utilities that can be used to analyse and visualize
data from neuroscience. The focus is mainly on AI implementations of various
models and pipelines. It will mainly grow as various needs arise and it does
not aim to be a general framework for such problems, as the field is too vast.

There are many frameworks and libraries that can be used for this sort of
stuff. For some Python examples, see

* `MNE-Python <https://github.com/mne-tools/mne-python>`__: Package for
  exploring, visualizing, and analyzing human neurophysiological data such as
  MEG, EEG, sEEG, ECoG, and more.
* `Braindecode <https://github.com/braindecode/braindecode>`__: Deep learning
  software to decode EEG, ECG or MEG signals. It builds on MNE-Python.
* `Nilearn <https://github.com/nilearn/nilearn>`__: Machine learning and
  decoding for neuroimaging (e.g. fMRI).
* `NeuroKit2 <https://github.com/neuropsychology/NeuroKit>`__: Generic package
  for advanced biosignal processing routines.
* `PyRiemann <https://github.com/pyRiemann/pyRiemann>`__: Generic multivariate
  data analysis, with a focus on biosignals (EEG, MEG, EMG).
* `PyDMD <https://github.com/PyDMD/PyDMD>`__ and `PySINDy
  <https://github.com/dynamicslab/pysindy>`__: More generic dynamics libraries.

Scope
-----

The main driving philosophy of ``nneuroutil`` is that it should be
backend-agnostic. All current algorithms are implemented against the `Array API
Standard <https://data-apis.org/array-api/latest>`__ and work with both mutable
backends (e.g. ``numpy`` or ``torch``) and immutable backends (e.g. ``jax``).

On top of that, the scope is not to overthrow other libraries. The algorithms
that are added here should be either (a) standard and not worth an extra
dependency or (b) novel / with different tradeoffs.

An example of this is the ``LDA`` algorithm implemented in
``nneuroutil.classifier``, which uses a GSVD-based formulation for
rank-deficient problems where the within-class scatter matrix is singular.
Unlike ``scikit-learn``'s default SVD-based LDA, it can retain discriminative
directions in the null space of that matrix, which can be useful when there are
more features than samples.

Install
-------

.. code-block:: bash

    pip install nneuroutil
    # or: uv add nneuroutil

Optional backends are extras: ``nneuroutil[jax]``, ``nneuroutil[torch]``, and
``nneuroutil[visualization]``.

Quick Start
-----------

The library contains a few different algorithms and tools for general analysis
of (mainly) timeseries data, classification, constructing neural network
models, etc.

It's generally pretty straightforward. For example, the code below shows a
standard LDA classifier based on the GSVD algorithm of `Howland & Park (2004)
<https://doi.org/10.1109/tpami.2004.46>`__ that uses ``jax``.

.. code-block:: python

    import jax
    import jax.numpy as jnp

    from nneuroutil.classifier import classify_linear_discriminant_analysis

    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    features = [
        jax.random.normal(key1, (50, 2)) * 0.5 + jnp.array([-2.0, 0.0]),
        jax.random.normal(key2, (50, 2)) * 0.5 + jnp.array([2.0, 0.0]),
    ]

    clf = classify_linear_discriminant_analysis(features, [0, 1])
    print(clf.predict(jnp.array([[1.8, 0.2], [-1.9, -0.1]])))  # [1. 0.]

Links
-----

* `Code <https://github.com/alexfikl/nneuroutil>`__.
* `Documentation <https://nneuroutil.readthedocs.io>`__.
* License: MIT (see ``LICENSES/MIT.txt``).
