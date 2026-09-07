.. _notes-initializers:

Initializers
============

We have several non-standard activation functions that require custom
initialization. Below, we explain the general framework used to derive the
initializer parameters used in :mod:`nneuroutil.torch_extras` and
:mod:`nneuroutil.flax_extras`. For details, see classic works such as
[Glorot2010]_ (worked on sigmoid activations) or [He2015]_ (worked on ReLU
activations).

.. [Glorot2010] X. Glorot, Y. Bengio,
    *Understanding the difficulty of training deep feedforward neural networks*,
    2010,
    `URL <http://proceedings.mlr.press/v9/glorot10a.html>`__.

.. [He2015] K. He, X. Zhang, S. Ren, J. Sun,
    *Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification*,
    2015,
    `URL <https://arxiv.org/abs/1502.01852>`__.

The general idea
----------------

To showcase the ideas, we consider a neural network with linear layers and
activation functions of the form

.. math::

    f_n(\cdots f_3(W_3 f_2(W_2 f_1(W_1 x + b_1) + b_2) + b_3) \cdots)

For some notation, let :math:`y_l = W_l x_l + b_1` be the pre-activation value
at layer :math:`l` and :math:`x_{l + 1} = f_l(y_l)`, with :math:`x_0 = x`, be
the post-activation that feeds into the next layer. We will assume that
:math:`b_j = 0` and that :math:`x` is a Gaussian with zero mean and unit
variance.

The goal is to initialize the weights :math:`W_l` such that
:math:`\operatorname{Var}(y_l) = \operatorname{Var}(y_{l - 1}) = 1`, i.e. we
want to keep the pre-activation variance stable across layers. We will assume
that the weights :math:`W_l` are also i.i.d. with zero mean and a variance
:math:`\sigma_w^2` to determine, depending on :math:`f_{l - 1}` activations.

Consider a single layer with :math:`n = \text{fan\_in}` inputs,

.. math::

    y_l = \sum_{i = 1}^{n} w_{l, i} x_{l, i}.

Since both :math:`w_{l, i}` and :math:`x_{l, i}` have zero mean, the
pre-activation variance is

.. math::
    :label: variance

    \operatorname{Var}(y_l) = n \sigma_{w_l}^2 \mathbb{E}[x_l^2].

i.e. it factorizes into a term that we control (the weight variance times the
fan-in or fan-out) and a term that is fixed by the activation function (the
*second moment* of its output). The convention used throughout this package
(and ``jax``, ``pytorch``, etc.) is to impose a *unit second moment*

.. math::
    :label: init_unit_variance

    \operatorname{Var}(y_l)
        = n \sigma_{w_l}^2 \mathbb{E}[x_l^2]
        = n \sigma_{w_l}^2 \mathbb{E}\bigl[f_{l - 1}(y_{l - 1})^2\bigr] = 1.

So, given :math:`\operatorname{Var}(y_l)`, we have that

.. math::

   \sigma_{w_l}^2 = \frac{1}{n \mathbb{E}\bigl[f_{l - 1}(y_{l - 1})^2\bigr]}.

Knowing the mean and variance of :math:`W_l`, we can sample from any
distribution of interest. Usually, packages use the uniform distribution or the
normal distribution (sometimes truncated). For the uniform, distribution, we
have that the variance is given by

.. math::
    :label: uniform_variance

    \operatorname{Var}(\mathcal{U}([-b, b])) = \frac{b^2}{3}.

Therefore, for each activation function, we just need to compute
:math:`\mathbb{E}[x_l^2]` to obtain :math:`\sigma_{w_l}^2`.

Quadratic Activation
---------------------

We first look at the simplest added nonlinearity: a quadratic activation
function :math:`f(y) = y^2`. Then, we have that

.. math::

   \mathbb{E}[x_l^2] = \mathbb{E}[y_{l - 1}^4] = 3 \sigma_{y, l - 1}^4 = 3,

assuming :math:`\sigma_{y_l}^2 = 1`, so

.. math::

    \sigma_w^2 = \frac{1}{3 n},

which is exactly the value computed in :func:`~nneuroutil.torch_extras.kaiming_uniform_`
when using using ``nonlinearity="quadratic"``.

.. note::

    For ``flax`` we use :func:`~flax.nnx.initializers.variance_scaling` to
    generate the appropriate initializer function. There, the convention is
    that

    .. math::

        \sigma_w^2 = \frac{\text{scale}}{n},

    where ``scale`` is the value that is being passed in. The function then
    takes care of scaling it by :math:`n` and computing the appropriate
    parameter for the chosen distribution (e.g. :math:`b` for uniform).

.. warning::

    This activation is not depth-stable: deviations from the unit variance
    fixed point are amplified by a factor of :math:`2` per layer.

Complex Quadratic Activation
-----------------------------

In the complex case, we assume that the real and imaginary parts of :math:`y_l`
satisfy

.. math::

    \operatorname{Var}(\Re y_l) = \operatorname{Var}(\Im y_l) = \frac{\sigma_{y_l}^2}{2}
    \implies
    \operatorname{Var}(\|y_l\|^2) = \sigma_{y_l}^2.

Then,

.. math::

    \mathbb{E}[\|x_l\|^2] = \mathbb{E}[\|y_{l - 1}\|^4]
        = \mathbb{E}[(\Re y_{l - 1})^4]
          + 2 \mathbb{E}[(\Re y_{l - 1})^2 (\Im y_{l - 1})^2]
          + \mathbb{E}[(\Im y_{l - 1})^4]
        = 2 \sigma_{y_l}^4 = 2

and

.. math::

   \sigma_w^2 = \frac{1}{2 n}.

.. warning::

    This activation is not depth-stable: deviations from the unit second
    moment fixed point are amplified by a factor of :math:`2` per layer.

Blended Quadratic Activation
-----------------------------

The blended quadratic function is given by :math:`f(y) = \alpha y + (1 -
\alpha) y^2`. In the case of :math:`\alpha = 1`, this degenerates to
essentially having no activation function. Then, we have that

.. math::

    \sigma_w^2 = \frac{1}{n}.

The more interesting case is :math:`\alpha \in [0, 1)`. There, we have that

.. math::

   \mathbb{E}[x_l^2] = \mathbb{E}[(\alpha y_{l - 1} + (1 - \alpha) y_{l - 1}^2)^2]
        = \alpha^2 \sigma_{y_{l - 1}}^2 + 3 (1 - \alpha)^2 \sigma_{y_{l - 1}}^4
        = \alpha^2 + 3 (1 - \alpha)^2,

under the assumption that :math:`y` is Gaussian with unit variance. Then, we
have that

.. math::

    \sigma_w^2 = \frac{1}{(\alpha^2 + 3 (1 - \alpha)^2) n}.

.. warning::

    For :math:`\alpha < 1`, this activation is not depth-stable: deviations
    from the unit variance fixed point are amplified by a factor between
    :math:`1` and :math:`2` per layer, decreasing with :math:`\alpha`.

Complex Blended Quadratic Activation
--------------------------------------

We follow the same derivation here to obtain

.. math::

    \mathbb{E}[\|x_l\|^2]
        = \mathbb{E}[\|\alpha y_{l - 1} + (1 - \alpha) y_{l - 1}^2\|^2]
        = \alpha^2 \sigma_{y_{l - 1}}^2 + 2 (1 - \alpha)^2 \sigma_{y_{l - 1}}^4,

so we have that

.. math::

    \sigma_w^2 = \frac{1}{(\alpha^2 + 2 (1 - \alpha)^2) n}.

.. warning::

    For :math:`\alpha < 1`, this activation is not depth-stable: deviations
    from the unit second moment fixed point are amplified by a factor between
    :math:`1` and :math:`2` per layer, decreasing with :math:`\alpha`.

ModReLU Activation
------------------

The ``modReLU`` activation function is given by :math:`f(y) =
\operatorname{ReLU}(\|y\| + b) \operatorname{sgn}(y)` for a complex :math:`y`
(see [Arjovsky2015]_). We have two cases of interest here, when :math:`b \ge 0`
and when it is negative. If :math:`b` is positive, then we have that

.. math::

   f(y) = (\|y\| + b) \operatorname{sgn}(y) = y + b \operatorname{sign}(y)

and

.. math::

   \mathbb{E}[\|x_l\|^2]
    = \mathbb{E}[\|f_{l - 1}(y_{l - 1})\|^2]
    = \mathbb{E}[\|y_{l - 1}\|^2] + 2 b \mathbb{E}[\|y_{l - 1}\|] + b^2

For :math:`y` complex with normally distributed real and imaginary parts, we
know that :math:`r = \|y\|` is `Rayleigh distributed
<https://en.wikipedia.org/wiki/Rayleigh_distribution>`__ with scale
:math:`\sigma_r = \sigma_y / \sqrt{2}`, i.e. :math:`p(r) = \frac{2
r}{\sigma_y^2} \exp(-r^2 / \sigma_y^2)` and :math:`\mathbb{E}[r] = \sqrt{\pi
\sigma_y^2} / 2`. Therefore, we have that

.. math::

   \mathbb{E}[\|x_l\|^2] = 1 + b \sqrt{\pi} + b^2.


On the other hand, if :math:`b < 0`, the ReLU part does not simplify away
directly. We actually have to compute the integral here, since the result is
not known. We have that

.. math::

    \mathbb{E}[\|x_l\|^2] = \mathbb{E}[\operatorname{ReLU}(r + b)^2]
         = \int_{-b}^\infty (r + b)^2 \frac{2 r}{\sigma_y^2}
             \exp\left(-\frac{r^2}{\sigma_y^2}\right) \,\mathrm{d} r.

Without going into details, we get that

.. math::

    \mathbb{E}[\|x_l\|^2] = e^{-b^2} + b \sqrt{\pi} \operatorname{erfc}(-b).

.. warning::

    For :math:`b < 0`, this activation is not depth-stable: deviations from
    the unit second moment fixed point are amplified by a factor of
    :math:`\approx 1.6` per layer for :math:`b = -0.5` (:math:`\approx 2.6`
    for :math:`b = -1`), worsening as :math:`b` decreases.

Leaky ModReLU Activation
-------------------------

We consider the following leaky ``modReLU`` activation function:

.. math::

    f(y; b, \alpha) = \operatorname{sgn}(y) \times
    \begin{cases}
    \|y\| + b, & \quad \|y\| + b \ge 0, \\
    \alpha \|y\|, & \quad \text{otherwise}.
    \end{cases}

In the case :math:`b \ge 0`, the second branch never triggers, so this is
equivalent to the modReLU result above:

.. math::

   \mathbb{E}[\|x_l\|^2] = 1 + b \sqrt{\pi} + b^2.


In the case :math:`b < 0`, we once again resort to computing the integrals
directly. We have that

.. math::

   \mathbb{E}[\|x_l\|^2] = \mathbb{E}[\operatorname{LeakyModReLU}(y_{l - 1})^2]
    = \int_{0}^{-b} (\alpha r)^2
        \frac{2 r}{\sigma_y^2} \exp\left(-\frac{r^2}{\sigma_y^2}\right) \,\mathrm{d} r
    + \int_{-b}^\infty (r + b)^2
        \frac{2 r}{\sigma_y^2} \exp\left(-\frac{r^2}{\sigma_y^2}\right) \,\mathrm{d} r.

The second integral is the same as in the previous case, so we just compute the
first integral. This gives

.. math::

    \mathbb{E}[\|x_l\|^2] =
        \alpha^2 \left[1 - (b^2 + 1) e^{-b^2}\right]
        + e^{-b^2} + b \sqrt{\pi} \operatorname{erfc}(-b).

.. warning::

    For :math:`b < 0`, this activation is not depth-stable: deviations grow by
    roughly the same factor as the ``modReLU`` above, since the leak only
    enters the statistics at :math:`O(\alpha^2)`.

Cardioid Activation
--------------------

The complex cardioid activation from [Virtue2017]_ is given by

.. math::

    f(y) = \frac{1}{2} (1 + \cos \theta(y)) y,

where :math:`\theta(y) = \operatorname{arg} y` is the phase of the input.
Assuming that :math:`y` is normally distributed and circularly symmetric as
before, we have that :math:`r` is Rayleigh distributed and :math:`\theta` is
uniformly distributed in :math:`[0, 2 \pi]`.

Therefore, we have that

.. math::

   \mathbb{E}[\|x_l\|^2] =
    \frac{1}{4} \mathbb{E}[(1 + \cos \theta_{l - 1})^2] \mathbb{E}[r_{l - 1}^2]
    = \frac{3}{8} \sigma_y^2.

zReLU
-----

The zReLU activation from [Guberman2016]_ is given by

.. math::

    f(y) =
    \begin{cases}
    y, & \quad \Re(y) \geq 0, \Im(y) \geq 0, \\
    0, & \quad \text{otherwise}.
    \end{cases}

We can rewrite this as

.. math::

   \mathbb{E}[\|x_l\|^2] = \mathbb{E}[r^2] \mathbb{E}[\delta_{[0, \pi / 2]}(\theta)]
    = \sigma_y^2 \frac{1}{4} = \frac{1}{4},

where we have used the fact that :math:`\theta` is uniform and the probability
of it landing in the first quadrant is just :math:`1/4`.
