nneuroutil: utilities for neuroscience
======================================

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: API Reference

    api/classifier
    api/dmd
    api/flax_extras
    api/torch_extras
    api/helpers
    api/typing

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Notes

    notes/initializers.rst

.. warning::

   This package is currently in development and very experimental (the API
   can and will change frequently for the foreseeable future). This is also meant
   as a personal bag of tricks and will not necessarily be consistent or complete.

This is a bag of random utilities that can be used to analyse and visualize
data from neuroscience. The focus is mainly on AI implementations of various
models and pipelines. It will mainly grow as various needs arise and it does
not aim to be a general framework for such problems, as the field is too vast.

There are many frameworks and libraries that can be used for this sort of stuff.
For some Python examples, see

* `MNE <https://github.com/mne-tools/mne-python>`__: Package for exploring,
  visualizing, and analyzing human neurophysiological data such as MEG, EEG,
  sEEG, ECoG, and more.
* `Braindecode <https://github.com/braindecode/braindecode>`__: Deep learning
  software to decode EEG, ECG or MEG signals. This uses ``mne-python`` as a
  dependency.
* `Nilearn <https://github.com/nilearn/nilearn>`__: machine learning and
  decoding for neuroimaging (e.g. fMRI).
* `NeuroKit <https://github.com/neuropsychology/NeuroKit>`__: Generic package
  for advanced biosignal processing routines.
* `PyRiemann <https://github.com/pyRiemann/pyRiemann>`__: Generic multivariate
  data analysis, with a focus on biosignals (EEG, MEG, EMG).
* `PyDMD <https://github.com/PyDMD/PyDMD>`__ and `PySINDy
  <https://github.com/dynamicslab/pysindy>`__: More generic dynamics libraries.

The main driving philosophy of ``nneuroutil`` is that it should be backend
agnostic. All current algorithms are implemented against the `Array API
Standard <https://data-apis.org/array-api/latest>`__ and work both with
mutable backends (e.g. ``numpy`` or ``pytorch``) or immutable backends (e.g.
``jax``).

On top of that, the scope is not to overthrow other libraries. The algorithms
that are added here should be either (a) standard and not worth an extra
dependency or (b) novel or with different tradeoffs. An example for the is the
``LDA`` algorithm implemented in ``nneuroutil.classifier``, that focuses on
cases where the number of features is larger than the number of samples (i.e.
the system is singular), unlike the standard one implemented in
``scikit-learn``.

Acknowledgments
===============

Work on ``nneuroutil`` was sponsored, in part, by

* the CNCS-UEFISCDI (Romania), under Project No. ROSUA-2024-0002.
* the "Romanian Hub for Artificial Intelligence - HRIA", Smart Growth,
  Digitization and Financial Instruments Program, 2021-2027, MySMIS no. 351416.

The views and opinions expressed herein do not necessarily reflect those of the
funding agencies.
