Mathematical Model for ``class_sse.py``
=======================================

This page documents the time-domain propagation model, the stochastic
Schrödinger equation (SSE), the weighted normalization scheme, and the
integration methods implemented in ``SSE``. A companion PDF with the same
material is also available:
:download:`Stochastic Schrödinger equation — numerical methods (PDF) <_static/sse_ito_discretization.pdf>`.

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

Time-domain propagation
-----------------------

The time-dependent Schrödinger equation (TDSE) including the perturbation due
to an external electric field :math:`\mathbf{F}(t)` reads

.. math::

   \begin{aligned}
       i \frac{d}{dt}|\Psi(t)\rangle
       &= \hat{H}(t)|\Psi(t)\rangle \\
       &= \bigl(\hat{H}_0 + \hat{H}_I(t)\bigr)|\Psi(t)\rangle \\
       &= \bigl(\hat{H}_0 - \mathbf{F}(t)\cdot\hat{\boldsymbol{\mu}}\bigr)|\Psi(t)\rangle,
   \end{aligned}

where :math:`\hat{H}_0` is the field-free Hamiltonian, :math:`\hat{H}_I(t)` is
the perturbation Hamiltonian, :math:`\hat{\boldsymbol{\mu}}` is the dipole moment
operator, and :math:`|\Psi(t)\rangle` is the time-dependent electronic
wavefunction at time :math:`t`.

Expanding the wavefunction in the basis of eigenstates :math:`|\lambda\rangle` of
the field-free Hamiltonian (as obtained, e.g., from linear-response TDDFT), we
write

.. math::
   :label: tdse-expansion

   |\Psi(t)\rangle = \sum_{\lambda}^{N} C_{\lambda}(t)|\lambda\rangle.

Substituting :eq:`tdse-expansion` into the TDSE yields

.. math::

   \sum_{\lambda}^{N} i \frac{dC_{\lambda}(t)}{dt}|\lambda\rangle
   = \sum_{\lambda}^{N}
   \bigl(\hat{H}_0 - \mathbf{F}(t)\cdot\hat{\boldsymbol{\mu}}\bigr)
   C_{\lambda}(t)|\lambda\rangle.

Projecting onto :math:`\langle\lambda'|` yields

.. math::

   \begin{aligned}
       i \frac{dC_{\lambda'}(t)}{dt}
       &= \sum_{\lambda}^{N}
       \langle\lambda'|\hat{H}_0 - \mathbf{F}(t)\cdot\hat{\boldsymbol{\mu}}|\lambda\rangle
       C_{\lambda}(t) \\
       &= E_{\lambda'}C_{\lambda'}(t)
       - \sum_{\lambda}^{N}
       \langle\lambda'|\mathbf{F}(t)\cdot\hat{\boldsymbol{\mu}}|\lambda\rangle
       C_{\lambda}(t),
   \end{aligned}

where :math:`E_{\lambda'}` is the energy of :math:`|\lambda'\rangle`. In the
one-hole--one-particle (1h--1p) space each :math:`|\lambda\rangle` expands as

.. math::

   |\lambda\rangle
   = \sum_{i}^{N_{\mathrm{occ}}}\sum_{a}^{N_{\mathrm{virt}}}
   d_{i,\lambda}^{a}|\phi_i^a\rangle,

where :math:`N_{\mathrm{occ}}` and :math:`N_{\mathrm{virt}}` denote the numbers
of occupied and virtual orbitals, and :math:`d_{i,\lambda}^{a}` expands
:math:`|\lambda\rangle` in the singly excited determinants :math:`|\phi_i^a\rangle`
(promotion from occupied :math:`i` to virtual :math:`a`). The TDSE in the 1h--1p
space then becomes

.. math::

   \begin{aligned}
       i \frac{dC_{\lambda'}(t)}{dt}
       &= E_{\lambda'}C_{\lambda'}(t)
       - \sum_{\lambda}^{N}\sum_{i,j}^{N_{\mathrm{occ}}}\sum_{a,b}^{N_{\mathrm{virt}}}
       d_{i,\lambda'}^{a} d_{j,\lambda}^{b}
       \langle\phi_i^a|\mathbf{F}(t)\cdot\hat{\boldsymbol{\mu}}|\phi_j^b\rangle
       C_{\lambda}(t) \\
       &= E_{\lambda'}C_{\lambda'}(t)
       - \sum_{q=x,y,z}\sum_{\lambda}^{N}\sum_{i,j}^{N_{\mathrm{occ}}}\sum_{a,b}^{N_{\mathrm{virt}}}
       d_{i,\lambda'}^{a} d_{j,\lambda}^{b} F_q(t)
       \langle\phi_i^a|\hat{\mu}_q|\phi_j^b\rangle C_{\lambda}(t),
   \end{aligned}

where :math:`F_q(t)` is the :math:`q`-th component of :math:`\mathbf{F}(t)` and
:math:`\hat{\mu}_q` is that of :math:`\hat{\boldsymbol{\mu}}`. The
Slater--Condon rules give

.. math::

   \langle\phi_i^a|\hat{\boldsymbol{\mu}}|\phi_j^b\rangle
   = \begin{cases}
       \mathbf{D}_0 - \boldsymbol{\mu}_{ii} + \boldsymbol{\mu}_{aa}
       & \text{if } i = j,\; a = b, \\
       \boldsymbol{\mu}_{ab} & \text{if } i = j,\; a \neq b, \\
       -\boldsymbol{\mu}_{ij} & \text{if } i \neq j,\; a = b, \\
       0 & \text{otherwise},
   \end{cases}

where :math:`\boldsymbol{\mu}_{lm}` is the dipole matrix element in the
molecular-orbital basis (as supplied by standard quantum-chemistry codes), and
:math:`\mathbf{D}_0 = \sum_{l}\boldsymbol{\mu}_{ll}`.

In matrix form, the TDSE in the 1h--1p space reads

.. math::

   i \frac{d\mathbf{C}(t)}{dt} = \mathbf{H}(t)\mathbf{C}(t),

where :math:`\mathbf{C}(t)` collects the expansion coefficients and
:math:`\mathbf{H}(t)` is the Hamiltonian matrix in the 1h--1p basis, with
elements

.. math::
   :label: hamiltonian-matrix

   (\mathbf{H}(t))_{\lambda',\lambda}
   = E_{\lambda}\delta_{\lambda',\lambda}
   - \sum_{q=x,y,z}\sum_{i,j}^{N_{\mathrm{occ}}}\sum_{a,b}^{N_{\mathrm{virt}}}
   d_{i,\lambda'}^{a} d_{j,\lambda}^{b} F_q(t)
   \langle\phi_i^a|\hat{\mu}_q|\phi_j^b\rangle.

In ``SSE``, :math:`\mathbf{H}(t)` is built each step from the stored dipole tensor
and the tabulated field ``electric_field``.

Stochastic Schrödinger equation
-------------------------------

In the Markovian limit, the stochastic Schrödinger equation (SSE) reads

.. math::

   i \frac{d}{dt}|\Psi_s(t)\rangle
   = \hat{H}(t)|\Psi_s(t)\rangle
   + \sum_{m}^{M} W_m(t)\hat{S}_m|\Psi_s(t)\rangle
   - \frac{i}{2}\sum_{m}^{M}\hat{S}_m^\dagger\hat{S}_m|\Psi_s(t)\rangle,

where :math:`\hat{S}_m` describes coupling to the environment through :math:`M`
channels indexed by :math:`m`. Dissipation enters via the anti-Hermitian Lindblad
contribution :math:`-\frac{i}{2}\sum_{m}^{M}\hat{S}_m^\dagger\hat{S}_m`, while
the stochastic driving :math:`\sum_{m}^{M} W_m(t)\hat{S}_m` is modeled with
Wiener processes :math:`W_m(t)`.

Expanding :math:`|\Psi_s(t)\rangle` in the same eigenbasis
:math:`\{|\lambda\rangle\}` as in :eq:`tdse-expansion`,

.. math::
   :label: sse-expansion

   |\Psi_s(t)\rangle = \sum_{\lambda}^{N} C_{\lambda}(t)|\lambda\rangle.

Substituting :eq:`sse-expansion` into the SSE yields

.. math::

   \sum_{\lambda}^{N} i \frac{dC_{\lambda}(t)}{dt}|\lambda\rangle
   = \sum_{\lambda}^{N} C_{\lambda}(t)\hat{H}(t)|\lambda\rangle
   + \sum_{\lambda}^{N}\sum_{m}^{M} W_m(t) C_{\lambda}(t)\hat{S}_m|\lambda\rangle
   - \frac{i}{2}\sum_{\lambda}^{N}\sum_{m}^{M}
   C_{\lambda}(t)\hat{S}_m^\dagger\hat{S}_m|\lambda\rangle.

Hence, projecting onto :math:`\langle\lambda'|` gives

.. math::

   i\frac{dC_{\lambda'}(t)}{dt}
   = \sum_{\lambda}^{N}(\mathbf{H}(t))_{\lambda',\lambda} C_{\lambda}(t)
   + \sum_{\lambda}^{N}\sum_{m}^{M} W_m(t)(\mathbf{S}_m)_{\lambda',\lambda} C_{\lambda}(t)
   - \frac{i}{2}\sum_{\lambda}^{N}\sum_{m}^{M}
   (\mathbf{\Gamma}_m)_{\lambda',\lambda} C_{\lambda}(t),

with :math:`(\mathbf{S}_m)_{\lambda',\lambda} = \langle\lambda'|\hat{S}_m|\lambda\rangle`,
:math:`(\mathbf{\Gamma}_m)_{\lambda',\lambda}
= \langle\lambda'|\hat{S}_m^\dagger\hat{S}_m|\lambda\rangle`, and
:math:`(\mathbf{H}(t))_{\lambda',\lambda}` as in :eq:`hamiltonian-matrix`.

In compact matrix notation,

.. math::

   \frac{d\mathbf{C}(t)}{dt}
   = \Bigl(-i\mathbf{H}(t) - \frac{1}{2}\sum_{m}^{M}\mathbf{\Gamma}_m\Bigr)\mathbf{C}(t)
   - i\sum_{m}^{M} W_m(t)\mathbf{S}_m\mathbf{C}(t).

If excited-state relaxation from a state :math:`|\lambda_1\rangle` to a state
:math:`|\lambda_2\rangle` is to be considered in this framework, the operator
:math:`\hat{S}_m` should be replaced by :math:`\hat{S}_{\lambda_1,\lambda_2}`,
defined by

.. math::
   :label: sse-operator

   \hat{S}_{\lambda_1,\lambda_2}
   = \sqrt{\Gamma_{\lambda_2,\lambda_1}}\,|\lambda_2\rangle\langle\lambda_1|,

where :math:`\Gamma_{\lambda_2,\lambda_1}` is the transition rate from
:math:`|\lambda_1\rangle` to :math:`|\lambda_2\rangle`. The matrix
:math:`\mathbf{S}_{\lambda_1,\lambda_2}` is therefore

.. math::

   \begin{aligned}
       (\mathbf{S}_{\lambda_1,\lambda_2})_{\lambda',\lambda}
       &= \langle\lambda'|\hat{S}_{\lambda_1,\lambda_2}|\lambda\rangle \\
       &= \sqrt{\Gamma_{\lambda_2,\lambda_1}}
       \langle\lambda'|\lambda_2\rangle\langle\lambda_1|\lambda\rangle \\
       &= \sqrt{\Gamma_{\lambda_2,\lambda_1}}
       \delta_{\lambda',\lambda_2}\delta_{\lambda_1,\lambda}.
   \end{aligned}

In matrix form, :math:`\mathbf{S}_{\lambda_1,\lambda_2}` is zero everywhere
except :math:`\sqrt{\Gamma_{\lambda_2,\lambda_1}}` in row :math:`\lambda_2` and
column :math:`\lambda_1`.

Similarly, :math:`\mathbf{\Gamma}_m` is replaced by
:math:`\mathbf{\Gamma}_{\lambda_1,\lambda_2}`, defined by

.. math::

   \begin{aligned}
       (\mathbf{\Gamma}_{\lambda_1,\lambda_2})_{\lambda',\lambda}
       &= \langle\lambda'|\hat{S}_{\lambda_1,\lambda_2}^\dagger
       \hat{S}_{\lambda_1,\lambda_2}|\lambda\rangle \\
       &= \Gamma_{\lambda_2,\lambda_1}
       \langle\lambda'|\lambda_1\rangle\langle\lambda_2|\lambda_2\rangle
       \langle\lambda_1|\lambda\rangle \\
       &= \Gamma_{\lambda_2,\lambda_1}
       \langle\lambda'|\lambda_1\rangle\langle\lambda_1|\lambda\rangle \\
       &= \Gamma_{\lambda_2,\lambda_1}
       \delta_{\lambda',\lambda_1}\delta_{\lambda_1,\lambda}.
   \end{aligned}

In matrix form, :math:`\mathbf{\Gamma}_{\lambda_1,\lambda_2}` is zero everywhere
except :math:`\Gamma_{\lambda_2,\lambda_1}` in row :math:`\lambda_1` and column
:math:`\lambda_1`.

The SSE then becomes

.. math::

   \frac{d\mathbf{C}(t)}{dt}
   = \Bigl(-i\mathbf{H}(t)
   - \frac{1}{2}\sum_{\lambda_1,\lambda_2}\mathbf{\Gamma}_{\lambda_2,\lambda_1}\Bigr)
   \mathbf{C}(t)
   - i\sum_{\lambda_1,\lambda_2}
   W_{\lambda_1,\lambda_2}(t)\mathbf{S}_{\lambda_1,\lambda_2}\mathbf{C}(t),

where :math:`W_{\lambda_2,\lambda_1}(t)` is the Wiener process for the
transition from :math:`|\lambda_1\rangle` to :math:`|\lambda_2\rangle`.

In differential form, the SSE reads

.. math::

   \mathrm{d}\mathbf{C}(t)
   = \Bigl(-i\mathbf{H}(t)
   - \frac{1}{2}\sum_{\lambda_1,\lambda_2}\mathbf{\Gamma}_{\lambda_2,\lambda_1}\Bigr)
   \mathbf{C}(t)\,\mathrm{d}t
   - i\sum_{\lambda_1,\lambda_2}
   \mathbf{S}_{\lambda_1,\lambda_2}\mathbf{C}(t)\,
   \mathrm{d}W_{\lambda_1,\lambda_2}(t).

In Itô notation, the SSE reads

.. math::
   :label: sse-ito

   \mathrm{d}\mathbf{C}_t
   = \Bigl(-i\mathbf{H}_t
   - \frac{1}{2}\sum_{\lambda_1,\lambda_2}\mathbf{\Gamma}_{\lambda_2,\lambda_1}\Bigr)
   \mathbf{C}_t\,\mathrm{d}t
   - i\sum_{\lambda_1,\lambda_2}
   \mathbf{S}_{\lambda_1,\lambda_2}\mathbf{C}_t\,
   \mathrm{d}W_{\lambda_1,\lambda_2,t},

where the subscript :math:`t` indicates that the quantities are evaluated at time
:math:`t`.

Euler--Maruyama discretization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Euler--Maruyama approximation to the true solution :math:`\mathbf{C}` of the
Itô differential equation in :eq:`sse-ito` is the Markov chain
:math:`\bar{\mathbf{C}}` given by

.. math::

   \bar{\mathbf{C}}_{n+1}
   = \bar{\mathbf{C}}_n
   + \Bigl(-i\mathbf{H}_n
   - \frac{1}{2}\sum_{\lambda_1,\lambda_2}\mathbf{\Gamma}_{\lambda_2,\lambda_1}\Bigr)
   \bar{\mathbf{C}}_n\,\Delta t
   - i\sum_{\lambda_1,\lambda_2}
   \mathbf{S}_{\lambda_1,\lambda_2}\bar{\mathbf{C}}_n\,
   \Delta W_{\lambda_1,\lambda_2,n},

where :math:`\Delta t` is the time step, :math:`n` is the time-step index, and
:math:`\Delta W_{\lambda_1,\lambda_2,n}` is the increment of the Wiener process.
This increment is given by

.. math::

   \Delta W_{\lambda_1,\lambda_2,n}
   = \sqrt{\Delta t}\,\xi_{\lambda_1,\lambda_2,n},

where :math:`\xi_{\lambda_1,\lambda_2,n} \sim \mathcal{N}(0,1)` are independent
standard normal random variables. Hence,

.. math::

   \bar{\mathbf{C}}_{n+1}
   = \bar{\mathbf{C}}_n
   + \Bigl(-i\mathbf{H}_n
   - \frac{1}{2}\sum_{\lambda_1,\lambda_2}\mathbf{\Gamma}_{\lambda_2,\lambda_1}\Bigr)
   \bar{\mathbf{C}}_n\,\Delta t
   - i\sqrt{\Delta t}\sum_{\lambda_1,\lambda_2}
   \mathbf{S}_{\lambda_1,\lambda_2}\xi_{\lambda_1,\lambda_2,n}\bar{\mathbf{C}}_n.

In matrix form, the term
:math:`\sum_{\lambda_1,\lambda_2}\mathbf{S}_{\lambda_1,\lambda_2}
\xi_{\lambda_1,\lambda_2,n}` is given by

.. math::

   \mathbf{R}_n
   = \sum_{\lambda_1,\lambda_2}
   \mathbf{S}_{\lambda_1,\lambda_2}\xi_{\lambda_1,\lambda_2,n}
   = \begin{pmatrix}
       \sqrt{\Gamma_{0,0}}\xi_{0,0,n} & \cdots
       & \sqrt{\Gamma_{0,\lambda_1}}\xi_{0,\lambda_1,n} & \cdots
       & \sqrt{\Gamma_{0,N}}\xi_{0,N,n} \\
       \vdots & \vdots & \vdots & \vdots & \vdots \\
       \sqrt{\Gamma_{\lambda_2,0}}\xi_{\lambda_1,0,n} & \cdots
       & \sqrt{\Gamma_{\lambda_2,\lambda_1}}\xi_{\lambda_1,\lambda_2,n} & \cdots
       & \sqrt{\Gamma_{\lambda_2,N}}\xi_{\lambda_1,N,n} \\
       \vdots & \vdots & \vdots & \vdots & \vdots \\
       \sqrt{\Gamma_{N,0}}\xi_{N,0,n} & \cdots
       & \sqrt{\Gamma_{N,\lambda_1}}\xi_{N,\lambda_1,n} & \cdots
       & \sqrt{\Gamma_{N,N}}\xi_{N,N,n}
   \end{pmatrix}.

The term :math:`\sum_{\lambda_1,\lambda_2}\mathbf{\Gamma}_{\lambda_2,\lambda_1}`
is given by

.. math::

   \mathbf{\Gamma}
   = \sum_{\lambda_1,\lambda_2}\mathbf{\Gamma}_{\lambda_2,\lambda_1}
   = \begin{pmatrix}
       \sum_{\lambda_2}\Gamma_{\lambda_2,0} & 0 & \cdots & 0 \\
       0 & \sum_{\lambda_2}\Gamma_{\lambda_2,1} & \cdots & 0 \\
       \vdots & \vdots & \ddots & \vdots \\
       0 & \cdots & \sum_{\lambda_2}\Gamma_{\lambda_2,N-1} & 0 \\
       0 & \cdots & 0 & \sum_{\lambda_2}\Gamma_{\lambda_2,N}
   \end{pmatrix}.

Consequently, the Euler--Maruyama approximation to the SSE is

.. math::
   :label: euler-maruyama-sse

   \bar{\mathbf{C}}_{n+1}
   = \bar{\mathbf{C}}_n
   + \Delta t\Bigl(-i\mathbf{H}_n - \frac{1}{2}\mathbf{\Gamma}\Bigr)\bar{\mathbf{C}}_n
   - i\sqrt{\Delta t}\mathbf{R}_n\bar{\mathbf{C}}_n,

which is the structure implemented for ``propagation_method="EM"`` (with
:math:`\mathbf{R}_n` sampled as ``R_base *`` independent Gaussians each step).

Connection to the ``gamma`` table in the code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The constructor argument ``gamma`` is a real array with one row per open channel,
each row meaning

.. code-block:: text

   [final_state, initial_state, rate]

with **zero-based** state indices, matching the ordering of :math:`\mathbf{C}`.
This is the same information as the rates :math:`\Gamma_{\lambda_2,\lambda_1}` in
the presentation. The code builds a sparse stochastic **base** matrix
:math:`\mathbf{R}_0` with entries

.. math::

   (R_0)_{f,i} = \sqrt{\Gamma_{f,i}}

for each supplied channel, and a diagonal **dissipation matrix**
:math:`\mathbf{\Gamma}` (in the code: ``gamma_matrix``) whose diagonal elements
are the sums of all outgoing rates from each initial state.

Weighted normalization scheme
-----------------------------

When the typical wavefunction normalization scheme (i.e., the squared moduli of
the coefficients are rescaled so that their sum equals~1) is applied at each time
step of the Euler--Maruyama approximation, noise can enter coefficients that
would otherwise evolve deterministically (i.e., coefficients of states that never
appear as the final state of a transition). Such contamination is undesirable
because it can produce artificial population oscillations. To avoid this, one
may use a weighted normalization scheme. At each time step, the scheme
normalizes the coefficients of the active states (i.e., states that participate
in at least one transition as a final state) and the coefficients of the fixed
states (i.e., states that do not participate in any transition as a final state)
as follows:

.. math::
   :label: weighted-normalization

   \begin{aligned}
       \{\bar{C}_{\lambda_f}(t)\}_{\lambda_f \in \mathcal{M}_{f}}
       &\to \{\bar{C}_{\lambda_f}(t)\}_{\lambda_f \in \mathcal{M}_{f}},\\
       \{\bar{C}_{\lambda_a}(t)\}_{\lambda_a \in \mathcal{M}_{a}}
       &\to \Bigl\{\sqrt{\frac{1 - p_{f}(t)}{p_a(t)}}
       \bar{C}_{\lambda_a}(t)\Bigr\}_{\lambda_a \in \mathcal{M}_{a}}
   \end{aligned}

where :math:`\mathcal{M}_{f}` and :math:`\mathcal{M}_{a}` denote the manifolds of
fixed and active states, respectively, and
:math:`p_f(t) = \sum_{\lambda_f \in \mathcal{M}_{f}} |\bar{C}_{\lambda_f}(t)|^2`
and
:math:`p_a(t) = \sum_{\lambda_a \in \mathcal{M}_{a}} |\bar{C}_{\lambda_a}(t)|^2`
are the populations of the fixed and active states at time :math:`t`. When all
states are involved in transitions as final states, the weighted normalization
scheme is equivalent to the typical normalization scheme. This matches
``SSE._apply_weighted_normalization``.

Integration schemes
-------------------

The integration schemes used for the implementation of the SSE are the
Euler--Maruyama scheme, the Heun scheme, and the Runge--Kutta (4th-order) scheme.

The Euler--Maruyama scheme is the simplest one and is obtained by directly
implementing :eq:`euler-maruyama-sse`.

The Heun integration scheme is given by

.. math::

   \bar{\mathbf{C}}_{n+1}
   = \bar{\mathbf{C}}_{n}
   + \frac{\Delta t}{2}\bigl(D(\bar{\mathbf{C}}_n, t) + D(\tilde{\mathbf{C}}, t + \Delta t)\bigr)
   + \frac{1}{2}\bigl(S(\bar{\mathbf{C}}_n, t) + S(\tilde{\mathbf{C}}, t + \Delta t)\bigr),

where

.. math::

   \begin{aligned}
       D(\bar{\mathbf{C}}_n, t)
       &= \Bigl(-i\mathbf{H}_n - \frac{1}{2}\mathbf{\Gamma}\Bigr)\bar{\mathbf{C}}_n, \\
       S(\bar{\mathbf{C}}_n, t) &= -i\sqrt{\Delta t}\mathbf{R}_n\bar{\mathbf{C}}_n, \\
       \tilde{\mathbf{C}}
       &= \bar{\mathbf{C}}_n + \Delta t D(\bar{\mathbf{C}}_n, t) + S(\bar{\mathbf{C}}_n, t).
   \end{aligned}

This corresponds to ``propagation_method="Heun"``.

The Runge--Kutta (4th-order) scheme is given by

.. math::

   \bar{\mathbf{C}}_{n+1}
   = \bar{\mathbf{C}}_{\mathrm{det}} + S(\bar{\mathbf{C}}_{\mathrm{det}}, t + \Delta t),

where

.. math::

   \begin{aligned}
       \bar{\mathbf{C}}_{\mathrm{det}}
       &= \bar{\mathbf{C}}_{n}
       + \frac{\Delta t}{6}(K_1 + 2K_2 + 2K_3 + K_4), \\
       K_1 &= D(\bar{\mathbf{C}}_n, t), \\
       K_2 &= D\Bigl(\bar{\mathbf{C}}_n + \frac{\Delta t}{2}K_1, t + \frac{\Delta t}{2}\Bigr), \\
       K_3 &= D\Bigl(\bar{\mathbf{C}}_n + \frac{\Delta t}{2}K_2, t + \frac{\Delta t}{2}\Bigr), \\
       K_4 &= D(\bar{\mathbf{C}}_n + \Delta t K_3, t + \Delta t).
   \end{aligned}

This corresponds to ``propagation_method="RK4"``.

Governing equation (implemented increment)
------------------------------------------

At each accepted step, the code applies the same split as in the discrete
formulas above: a deterministic matrix propagator involving
:math:`\mathbf{H}_n` and :math:`\mathbf{\Gamma}`, plus a stochastic term linear
in :math:`\sqrt{\Delta t}` and :math:`\mathbf{R}_n`. In the documentation of the
continuous-time limit it is convenient to write

.. math::

   \mathrm{d}\mathbf{C}
   = \Bigl(-i\mathbf{H}(t) - \tfrac{1}{2}\mathbf{\Gamma}\Bigr)\mathbf{C}\,\mathrm{d}t
   - i\,\mathbf{R}(t)\mathbf{C}\,\sqrt{\mathrm{d}t},

where :math:`\mathbf{R}(t)` denotes the instantaneous rescaled noise matrix whose
discrete realizations are :math:`\mathbf{R}_n`. Here:

- :math:`\mathbf{H}(t) = \mathbf{H}_0 + \mathbf{H}_I(t)`,
- :math:`\mathbf{H}_0 = \mathrm{diag}(E_0, E_1, \ldots, E_{N-1})`,
- :math:`\mathbf{H}_I(t) = -(\mu_x E_x(t) + \mu_y E_y(t) + \mu_z E_z(t))` with
  dipole tensors ``matrix[:,:,q]`` for :math:`q\in\{x,y,z\}`.

Numerical propagation flags
---------------------------

The class exposes three integrators:

- ``EM`` — Euler--Maruyama,
- ``Heun`` — predictor--corrector as above,
- ``RK4`` — fourth-order deterministic substeps plus stochastic correction.

Ensemble observables
--------------------

For ``n_paths`` independent trajectories, state populations are
:math:`P_k(t)=|C_k(t)|^2`. The code returns:

- ``population_average`` — mean over trajectories at each time,
- ``population_std`` — spread used in the notebooks as a simple uncertainty band
  (standard error of the mean when :math:`n_{\mathrm{paths}}>1`).
