Notebook Examples
=================

The ``notebooks/`` directory contains exploratory workflows that validate the
``SSE`` implementation against analytic population dynamics in small state
models. Each notebook:

- builds a Gaussian electric-field pulse with ``ElectricFieldPulse`` (the field
  couples to the system but these toy models use minimal dipole data from
  ``data_minimal/dummy_data/``),
- defines jump channels in the ``gamma`` table (see :doc:`sse_model`),
- runs an ensemble of stochastic trajectories,
- plots ``population_average`` with uncertainty bands and overlays analytic
  predictions as dashed red curves.

Common settings
---------------

- Time step: :math:`\Delta t = 5 \times 10^{-3}` a.u. (set via ``dt`` in the
  pulse and inherited by ``SSE``).
- Integration scheme: set ``propagation_method`` to ``"EM"`` (Euler--Maruyama),
  ``"Heun"``, or ``"RK4"`` to compare the three solvers documented in
  :doc:`sse_model`.
- Notebook path: ``notebooks/SSE_<name>_test.ipynb``.

The ``gamma`` array uses zero-based state indices with rows
``[final_state, initial_state, rate]``, matching
:math:`\Gamma_{\lambda_2,\lambda_1}` and
:math:`\hat{S}_{\lambda_1,\lambda_2}` in :doc:`sse_model`.

First test: two-state model
---------------------------

**Notebook:** ``SSE_first_test.ipynb``

**States:** :math:`\{|0\rangle, |1\rangle\}`.

**Jump operator:** a single relaxation channel from :math:`|1\rangle` to
:math:`|0\rangle`,

.. math::

   \hat{S}_{1,0} = \sqrt{\Gamma_{0,1}}\,|0\rangle\langle 1|,
   \qquad \Gamma_{0,1} = 2.0.

In the notebook:

.. code-block:: python

   gamma = np.array([[0, 1, 2]])  # final, initial, rate

**Ensemble:** 30 independent trajectories (``n_paths = 30``).

**Initial populations:** :math:`p_0(0) = 0`, :math:`p_1(0) = 1`.

**Analytic populations:**

.. math::

   \begin{aligned}
       p_0(t) &= 1 - p_1(0)\,e^{-\Gamma_{0,1} t}, \\
       p_1(t) &= p_1(0)\,e^{-\Gamma_{0,1} t}.
   \end{aligned}

Run the notebook with ``propagation_method`` set to ``"EM"``, ``"Heun"``, or
``"RK4"`` and compare the ensemble mean of :math:`|C_k|^2` to these curves.

Second test: three-state model (two transition operators)
---------------------------------------------------------

**Notebook:** ``SSE_second_test.ipynb``

**States:** :math:`\{|0\rangle, |1\rangle, |2\rangle\}`.

**Jump operators:** relaxation from :math:`|1\rangle` and :math:`|2\rangle` into
:math:`|0\rangle`,

.. math::

   \begin{aligned}
       \hat{S}_{1,0} &= \sqrt{\Gamma_{0,1}}\,|0\rangle\langle 1|,
       \qquad \Gamma_{0,1} = 1.0, \\
       \hat{S}_{2,0} &= \sqrt{\Gamma_{0,2}}\,|0\rangle\langle 2|,
       \qquad \Gamma_{0,2} = 2.0.
   \end{aligned}

In the notebook:

.. code-block:: python

   gamma = np.array([
       [0, 1, 1],
       [0, 2, 2],
   ])

**Ensemble:** 30 trajectories.

**Initial populations:** :math:`p_0(0) = 0`, :math:`p_1(0) = 0.5`,
:math:`p_2(0) = 0.5`.

**Analytic populations:**

.. math::

   \begin{aligned}
       p_0(t) &= 1 - p_1(0)\,e^{-\Gamma_{0,1} t} - p_2(0)\,e^{-\Gamma_{0,2} t}, \\
       p_1(t) &= p_1(0)\,e^{-\Gamma_{0,1} t}, \\
       p_2(t) &= p_2(0)\,e^{-\Gamma_{0,2} t}.
   \end{aligned}

Again, repeat the run with ``"EM"``, ``"Heun"``, and ``"RK4"`` to assess each
integration scheme at the same :math:`\Delta t`.

Third test: three-state model (single transition operator)
----------------------------------------------------------

**Notebook:** ``SSE_third_test.ipynb``

**States:** :math:`\{|0\rangle, |1\rangle, |2\rangle\}`.

**Jump operator:** relaxation from :math:`|2\rangle` to :math:`|1\rangle`,

.. math::

   \hat{S}_{2,1} = \sqrt{\Gamma_{1,2}}\,|1\rangle\langle 2|,
   \qquad \Gamma_{1,2} = 2.0.

In the notebook:

.. code-block:: python

   gamma = np.array([[1, 2, 2]])

**Ensemble:** 100 trajectories are used in the reference calculations described
here; increase ``n_paths`` in the notebook if you need a tighter statistical
average.

**Initial populations:** :math:`p_0(0) = 0.9`, :math:`p_1(0) = 0`,
:math:`p_2(0) = 0.1`. State :math:`|0\rangle` is not connected by any jump, so
its population stays fixed.

**Analytic populations:**

.. math::

   \begin{aligned}
       p_0(t) &= p_0(0) = 0.9, \\
       p_1(t) &= 1 - p_0(0) - p_2(0)\,e^{-\Gamma_{1,2} t}, \\
       p_2(t) &= p_2(0)\,e^{-\Gamma_{1,2} t}.
   \end{aligned}

This case exercises the **weighted normalization** scheme: only :math:`|1\rangle`
appears as a final state of a jump, while :math:`|0\rangle` remains a fixed
state (see :eq:`weighted-normalization` in :doc:`sse_model`).

Compare results for ``"EM"``, ``"Heun"``, and ``"RK4"`` as in the previous
examples.
