Pulse Modeling in ``class_field.py``
====================================

Overview
--------

``Electric_Field_Pulse`` creates time-domain pulses and computes their spectral
representation for later use in ``SSE`` propagation.

Physical conversions
--------------------

The implementation performs:

- intensity to field amplitude conversion:

  .. math::

     E_{\max} = \sqrt{I / 3.51\times10^{16}},

- attoseconds to atomic units conversion for pulse widths.

Supported pulse families
------------------------

1. ``gaussian``

   .. math::

      E(t) = E_{\max}\exp\left[-\frac{(t-t_0)^2}{2\sigma^2}\right]

2. ``gaussian_sin``

   .. math::

      E(t) = E_{\max}\exp\left[-\frac{(t-t_0)^2}{2\sigma^2}\right]\sin(\omega_c t)

3. ``sinc``

   .. math::

      E(t) = E_{\max}\,
      \frac{\sin\left[\tfrac{T}{2}(t-t_0)\right]}{\tfrac{T}{2}(t-t_0)}
      \sin(\omega_c t)\exp\left(-\frac{|t-t_0|}{\tau_{\mathrm{ap}}}\right)

Spectral analysis
-----------------

After pulse construction, the class computes FFT magnitude and frequency axis.
It also estimates:

- dominant spectral frequency from peaks,
- crossing energies at a user-defined relative threshold,
- plotting windows in time and frequency.

These quantities provide practical input checks before launching SSE
trajectories.
