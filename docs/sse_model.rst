Mathematical Model for ``class_sse.py``
=======================================

This page summarizes the stochastic open-system model implemented in ``SSE``.
For the full derivation (TDSE in an excitation basis, jump operators, Itô
discretization, weighted normalization, and integration schemes), see the
technical PDF bundled with this documentation:
:download:`Stochastic Schrödinger equation — numerical methods (PDF) <_static/sse_ito_discretization.pdf>`.
The notation below is aligned with that document where possible.

Overview
--------

The code propagates a complex coefficient vector

.. math::

   \mathbf{C}(t) = (C_0(t), C_1(t), \ldots, C_{N-1}(t))^{\mathsf{T}}

in a fixed orthonormal basis :math:`\{|\lambda\rangle\}_{\lambda=0}^{N-1}` of
electronic states (for example, eigenstates of a field-free Hamiltonian from a
linear-response calculation). The dynamics combine:

- coherent evolution under a time-dependent Hamiltonian matrix
  :math:`\mathbf{H}(t)`,
- Markovian dissipation through jump operators encoded by rates supplied in the
  ``gamma`` table,
- stochastic terms driven by independent Wiener increments at each step,
- a **weighted normalization** of :math:`\mathbf{C}` after each step so that
  states that never appear as *final* states of a jump are not spuriously fed by
  noise.

From the TDSE to the matrix formulation
---------------------------------------

The time-dependent Schrödinger equation with an electric field
:math:`\mathbf{F}(t)` reads, in abstract form,

.. math::

   i\frac{d}{dt}|\Psi(t)\rangle
   = \bigl(\hat H_0 - \mathbf{F}(t)\cdot\hat{\boldsymbol{\mu}}\bigr)|\Psi(t)\rangle.

Expanding :math:`|\Psi(t)\rangle = \sum_\lambda C_\lambda(t)|\lambda\rangle` and
projecting onto :math:`\langle\lambda'|` yields coupled equations for the
coefficients. After all spatial integrals are evaluated (dipole matrix in the
working basis), one obtains the **same abstract matrix form** used in the code:

.. math::

   i\frac{d\mathbf{C}(t)}{dt} = \mathbf{H}(t)\,\mathbf{C}(t),

with :math:`\mathbf{H}(t) = \mathbf{H}_0 + \mathbf{H}_I(t)`,
:math:`\mathbf{H}_0 = \mathrm{diag}(E_0,\ldots,E_{N-1})`, and a dipole coupling
term encoded from transition dipoles and field components :math:`E_x,E_y,E_z`.
In ``SSE``, :math:`\mathbf{H}(t)` is built each step from the stored dipole tensor
and the tabulated field ``electric_field``.

Markovian stochastic Schrödinger equation
-----------------------------------------

In the Markovian limit, an SSE for the stochastic state :math:`|\Psi_s(t)\rangle`
can be written (schematic operator form) as

.. math::

   i\frac{d}{dt}|\Psi_s(t)\rangle
   = \hat H(t)|\Psi_s(t)\rangle
   + \sum_m W_m(t)\,\hat S_m|\Psi_s(t)\rangle
   - \frac{i}{2}\sum_m \hat S_m^\dagger \hat S_m\,|\Psi_s(t)\rangle,

where :math:`W_m(t)` are Wiener processes and :math:`\hat S_m` are jump operators
coupling the system to the environment. Expanding again in the same
:math:`\{|\lambda\rangle\}` basis and collecting matrices
:math:`(\mathbf{S}_m)_{\lambda'\lambda} = \langle\lambda'|\hat S_m|\lambda\rangle`
and :math:`(\boldsymbol{\Gamma}_m)_{\lambda'\lambda} = \langle\lambda'|\hat S_m^\dagger
\hat S_m|\lambda\rangle`, one arrives at the **Stratonovich/Itô bookkeeping**
used in discrete solvers. For a single relaxation channel from an initial state
:math:`|\lambda_1\rangle` to a final state :math:`|\lambda_2\rangle` with rate
:math:`\Gamma_{\lambda_2\leftarrow\lambda_1}`, a minimal jump operator is

.. math::

   \hat S_{\lambda_1,\lambda_2}
   = \sqrt{\Gamma_{\lambda_2\leftarrow\lambda_1}}\,|\lambda_2\rangle\langle\lambda_1|,

so that :math:`\mathbf{S}_{\lambda_1,\lambda_2}` has a single non-zero entry
:math:`\sqrt{\Gamma_{\lambda_2\leftarrow\lambda_1}}` in row :math:`\lambda_2` and
column :math:`\lambda_1`, and :math:`\boldsymbol{\Gamma}_{\lambda_1,\lambda_2}
= \mathbf{S}_{\lambda_1,\lambda_2}^\dagger \mathbf{S}_{\lambda_1,\lambda_2}` is
non-zero only on the diagonal position :math:`(\lambda_1,\lambda_1)` with value
:math:`\Gamma_{\lambda_2\leftarrow\lambda_1}`. Summing all channels yields a
**diagonal dissipation matrix** :math:`\boldsymbol{\Gamma}` with entries given by
the total decay rate out of each initial state.

Connection to the ``gamma`` table in the code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The constructor argument ``gamma`` is a real array with one row per open channel,
each row meaning

.. code-block:: text

   [final_state, initial_state, rate]

with **zero-based** state indices, matching the ordering of :math:`\mathbf{C}`.
This is the same information as the rates :math:`\Gamma_{\lambda_{\mathrm{final}}
\leftarrow\lambda_{\mathrm{initial}}}` in the presentation. The code builds a
sparse stochastic **base** matrix :math:`\mathbf{R}_0` with entries

.. math::

   (R_0)_{f,i} = \sqrt{\Gamma_{f\leftarrow i}}

for each supplied channel, and a diagonal **dissipation matrix**
:math:`\boldsymbol{\Gamma}` (in the code: ``gamma_matrix``) whose diagonal
elements are the sums of all outgoing rates from each initial state.

Itô SDE and Euler–Maruyama discretization
-----------------------------------------

Collecting deterministic and stochastic parts, an Itô stochastic differential
equation for the coefficient vector can be written as

.. math::

   d\mathbf{C}_t
   = \Bigl(-i\mathbf{H}_t - \tfrac{1}{2}\boldsymbol{\Gamma}\Bigr)\mathbf{C}_t\,dt
   - i\sum_{\lambda_1,\lambda_2}
     \mathbf{S}_{\lambda_1,\lambda_2}\,\mathbf{C}_t\,dW_{\lambda_1,\lambda_2,t},

where :math:`dW` denotes Wiener increments. The **Euler–Maruyama** scheme with
time step :math:`\Delta t` uses independent standard normals :math:`\xi` for each
channel increment :math:`\Delta W = \sqrt{\Delta t}\,\xi`. With all channels
absorbed into a single random matrix :math:`\mathbf{R}_n` (zero mean, entry-wise
product structure inherited from the :math:`\mathbf{S}` matrices and the
:math:`\xi_{fi,n}`), one step reads

.. math::

   \mathbf{C}_{n+1}
   = \mathbf{C}_n
   + \Delta t\Bigl(-i\mathbf{H}_n - \tfrac{1}{2}\boldsymbol{\Gamma}\Bigr)\mathbf{C}_n
   - i\sqrt{\Delta t}\,\mathbf{R}_n\,\mathbf{C}_n,

which is the structure implemented for ``propagation_method="EM"`` (with
:math:`\mathbf{R}_n` sampled as ``R_base *`` independent Gaussians each step).

Heun and RK4 variants
^^^^^^^^^^^^^^^^^^^^^

The **Heun** and **RK4** options split the right-hand side into a deterministic
drift :math:`D(\mathbf{C},t)=(-i\mathbf{H}(t)-\tfrac{1}{2}\boldsymbol{\Gamma})\mathbf{C}`
and a stochastic increment
:math:`S(\mathbf{C},t)=-i\sqrt{\Delta t}\,\mathbf{R}_n\mathbf{C}`:

- **Heun** averages drift and noise at the endpoints of a predictor step
  :math:`\tilde{\mathbf{C}} = \mathbf{C}_n + \Delta t\,D(\mathbf{C}_n,t)
  + S(\mathbf{C}_n,t)`.
- **RK4** applies fourth-order Runge–Kutta to the deterministic part and appends
  the stochastic correction evaluated at the advanced deterministic state, as in
  the presentation (deterministic substeps :math:`K_1,\ldots,K_4`).

Weighted normalization
----------------------

After each integration substep, coefficients are renormalized so that
:math:`\|\mathbf{C}\|^2 = \sum_\lambda |C_\lambda|^2` remains consistent with the
probabilistic interpretation, **without** injecting noise into states that never
receive jumps as *final* states. Let :math:`\mathcal{M}_f` be “fixed” indices and
:math:`\mathcal{M}_a` “active” indices (states that appear as final states in at
least one channel). With populations
:math:`p_f=\sum_{\lambda\in\mathcal{M}_f}|C_\lambda|^2` and
:math:`p_a=\sum_{\lambda\in\mathcal{M}_a}|C_\lambda|^2`, the scheme rescales only
the active block:

.. math::

   \{C_{\lambda}\}_{\lambda\in\mathcal{M}_f} &\to \{C_{\lambda}\}_{\lambda\in\mathcal{M}_f},\\
   \{C_{\lambda}\}_{\lambda\in\mathcal{M}_a}
   &\to \Bigl\{\sqrt{\frac{1-p_f}{p_a}}\,C_{\lambda}\Bigr\}_{\lambda\in\mathcal{M}_a},

when :math:`p_a>0` and :math:`p_f\le 1`. If all states are active, this reduces to
ordinary normalization of the full vector. This matches
``SSE._apply_weighted_normalization``.

Governing equation (implemented increment)
------------------------------------------

At each accepted step, the code applies the same split as in the discrete
formulas above: a deterministic matrix propagator involving
:math:`\mathbf{H}_n` and :math:`\boldsymbol{\Gamma}`, plus a stochastic term
linear in :math:`\sqrt{\Delta t}` and :math:`\mathbf{R}_n`. In the documentation
of the continuous-time limit it is convenient to write

.. math::

   d\mathbf{C} =
   \Bigl(-i \mathbf{H}(t) - \tfrac{1}{2}\boldsymbol{\Gamma}\Bigr)\mathbf{C}\,dt
   - i\,\mathbf{R}(t)\mathbf{C}\,\sqrt{dt},

where :math:`\mathbf{R}(t)` denotes the instantaneous rescaled noise matrix whose
discrete realizations are :math:`\mathbf{R}_n`.

Here:

- :math:`\mathbf{H}(t) = \mathbf{H}_0 + \mathbf{H}_I(t)`,
- :math:`\mathbf{H}_0 = \mathrm{diag}(E_0, E_1, \ldots, E_{N-1})`,
- :math:`\mathbf{H}_I(t) = -(\mu_x E_x(t) + \mu_y E_y(t) + \mu_z E_z(t))` with
  dipole tensors ``matrix[:,:,q]`` for :math:`q\in\{x,y,z\}`.

Numerical propagation flags
-----------------------------

The class exposes three integrators, matching the presentation:

- ``EM`` — Euler–Maruyama,
- ``Heun`` — predictor–corrector as above,
- ``RK4`` — fourth-order deterministic substeps plus stochastic correction.

Ensemble observables
--------------------

For ``n_paths`` independent trajectories, state populations are
:math:`P_k(t)=|C_k(t)|^2`. The code returns:

- ``population_average`` — mean over trajectories at each time,
- ``population_std`` — spread used in the notebooks as a simple uncertainty band
  (standard error of the mean when :math:`n_{\mathrm{paths}}>1`).
