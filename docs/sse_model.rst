Mathematical Model for ``class_sse.py``
=======================================

Overview
--------

The ``SSE`` class propagates a complex coefficient vector
:math:`\mathbf{C}(t) = (C_0, C_1, \ldots, C_{N-1})^T` under:

- coherent field-free dynamics,
- dipole coupling with external electric fields,
- dissipative channels encoded in ``Gamma``,
- stochastic couplings generated trajectory by trajectory.

Governing equation
------------------

At each time step, the implemented structure is:

.. math::

   d\mathbf{C} =
   \left(-i \mathbf{H}(t) - \frac{1}{2}\mathbf{\Gamma}\right)\mathbf{C}\,dt
   - i\,\mathbf{R}(t)\mathbf{C}\,\sqrt{dt},

where:

- :math:`\mathbf{H}(t) = \mathbf{H}_0 + \mathbf{H}_I(t)`,
- :math:`\mathbf{H}_0 = \mathrm{diag}(E_0, E_1, \ldots, E_{N-1})`,
- :math:`\mathbf{H}_I(t) = -(\mu_x E_x(t) + \mu_y E_y(t) + \mu_z E_z(t))`.

The dissipation matrix :math:`\mathbf{\Gamma}` is diagonal and built from all
outgoing rates per initial state.

Stochastic couplings
--------------------

``Gamma`` rows are interpreted as ``[final_state, initial_state, gamma]``.
From this, a base matrix :math:`\mathbf{R}_0` is built with:

.. math::

   (\mathbf{R}_0)_{f,i} = \sqrt{\gamma_{f \leftarrow i}}.

At each time step, random normal entries scale this base matrix to generate
:math:`\mathbf{R}(t)`.

Numerical propagation
---------------------

The class offers three step rules:

- Euler-Maruyama (``EM``),
- Heun predictor-corrector (``Heun``),
- deterministic RK4 plus stochastic correction (``RK4``).

Each trajectory is normalized with the weighted normalization strategy already
implemented in ``SSE._apply_weighted_normalization``.

Ensemble observables
--------------------

For ``n_paths`` trajectories, populations are:

.. math::

   P_k(t) = |C_k(t)|^2.

The class returns:

- mean population ``population_average``,
- trajectory-wise spread ``population_std``.
